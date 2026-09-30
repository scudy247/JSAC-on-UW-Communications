"""Bandit agents for Proposal B (step 26a: naive baselines + BCL truncated-mean robust UCB).

All agents speak the TableEnv interface (envs/table_env.py): select(now_s) -> arm,
observe(arm, tx_time_s, reward: dict), optional observe_noise(now_s, samples). They learn
from the reward field named `field` (e.g. 'gamma_db', or 'ack' for ACK-only agents) and never
see anything else (project.MD §7.2). Time t counts decisions (1, 2, ...).

TruncatedMeanUCB follows Bubeck, Cesa-Bianchi & Lugosi (2013), checked against the PDF:
  Lemma 1: with E|X|^(1+eps) <= u, the truncated mean with per-sample threshold
           B_j = (u j / log(1/delta))^(1/(1+eps)) satisfies, w.p. >= 1 - delta,
           mu_hat <= mu + 4 u^(1/(1+eps)) (log(1/delta)/n)^(eps/(1+eps))  (and symmetrically);
  Prop. 1 / Fig. 1: robust UCB uses delta = t^-2, i.e. log(1/delta) = 2 log t.
BCL bound RAW moments (THEORY-B T1 detail): u must bound E|X|^(1+eps) including the mean.
Because the thresholds depend on t, the exact estimator keeps each arm's samples (O(n) memory);
the O(1) question belongs to NIR-UCB (step 26b).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gamma as _G


def sas_abs_moment(p: float, alpha: float, c: float) -> float:
    """E|X|^p for X ~ SaS(alpha, c) (CF exp(-|c theta|^alpha)), -1 < p < alpha.

    c^p 2^p Gamma((1+p)/2) Gamma(1 - p/alpha) / (sqrt(pi) Gamma(1 - p/2)); checked against the
    Gaussian (alpha = 2) and Cauchy (alpha = 1) closed forms and by Monte Carlo (p < alpha/2).
    """
    if not (-1.0 < p < alpha) or not 0.0 < alpha <= 2.0 or not c > 0:
        raise ValueError("need -1 < p < alpha <= 2 and c > 0")
    return float(c ** p * 2 ** p * _G((1 + p) / 2) * _G(1 - p / alpha)
                 / (math.sqrt(math.pi) * _G(1 - p / 2)))


class _Agent:
    def __init__(self, K: int, field: str):
        if K < 2:
            raise ValueError("need K >= 2 arms")
        self.K, self.field = int(K), field
        self.t = 0
        self.n = np.zeros(K, dtype=int)

    def _first_unpulled(self):
        z = np.flatnonzero(self.n == 0)
        return int(z[0]) if z.size else None


class EmpiricalUCB(_Agent):
    """Naive: empirical mean + sigma * sqrt(2 log t / n) (Gaussian widths, sigma assumed)."""

    def __init__(self, K, field="gamma_db", sigma=1.0):
        super().__init__(K, field)
        self.sigma = float(sigma)
        self.s = np.zeros(K)
        self._pending = np.zeros(K, dtype=int)

    def select(self, now_s):
        self.t += 1
        k = self._first_unpulled_or_pending()
        if k is not None:
            return k
        idx = self.s / self.n + self.sigma * np.sqrt(2 * math.log(self.t) / self.n)
        return int(np.argmax(idx))

    def _first_unpulled_or_pending(self):
        # before any feedback exists for an arm, play it once (delay-safe initialisation)
        z = np.flatnonzero((self.n == 0) & (self._pending == 0))
        if z.size:
            self._pending[z[0]] += 1
            return int(z[0])
        if np.any(self.n == 0):            # waiting for delayed first feedback: round-robin
            return int(np.argmin(self.n + self._pending))
        return None

    def observe(self, arm, tx_time_s, reward):
        self.n[arm] += 1
        self._pending[arm] = max(self._pending[arm] - 1, 0)
        self.s[arm] += reward[self.field]


class GaussianTS(EmpiricalUCB):
    """Naive Thompson sampling: mu_k ~ N(mean_k, sigma^2 / n_k)."""

    def __init__(self, K, field="gamma_db", sigma=1.0, rng=None):
        super().__init__(K, field, sigma)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator (explicit seeding)")
        self.rng = rng

    def select(self, now_s):
        self.t += 1
        k = self._first_unpulled_or_pending()
        if k is not None:
            return k
        return int(np.argmax(self.rng.normal(self.s / self.n, self.sigma / np.sqrt(self.n))))


class AckUCB1(EmpiricalUCB):
    """ACK-only UCB1 on a reward in [0, 1] (bounded, safe, slow): width sqrt(2 log t / n)."""

    def __init__(self, K, field="ack"):
        super().__init__(K, field, sigma=1.0)


class BernoulliTS(EmpiricalUCB):
    """ACK-only Thompson sampling with Beta(1, 1) priors."""

    def __init__(self, K, field="ack", rng=None):
        super().__init__(K, field)
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator (explicit seeding)")
        self.rng = rng

    def select(self, now_s):
        self.t += 1
        return int(np.argmax(self.rng.beta(1 + self.s, 1 + self.n - self.s)))


def truncated_mean(x, u: float, eps: float, log_inv_delta: float) -> float:
    """BCL Lemma 1 estimator: mean of X_j 1{|X_j| <= (u j / log(1/delta))^(1/(1+eps))}."""
    x = np.asarray(x, dtype=float)
    j = np.arange(1, x.size + 1)
    b = (u * j / log_inv_delta) ** (1.0 / (1.0 + eps))
    return float(np.sum(np.where(np.abs(x) <= b, x, 0.0)) / x.size)


def bcl_width(u: float, eps: float, log_inv_delta: float, n: int) -> float:
    """BCL Lemma 1 deviation: 4 u^(1/(1+eps)) (log(1/delta) / n)^(eps/(1+eps))."""
    return 4.0 * u ** (1.0 / (1.0 + eps)) * (log_inv_delta / n) ** (eps / (1.0 + eps))


class TruncatedMeanUCB(_Agent):
    """BCL robust UCB with the truncated mean (known eps and raw-moment bound u)."""

    def __init__(self, K, field="gamma_db", eps=1.0, u=1.0):
        super().__init__(K, field)
        if not 0.0 < eps <= 1.0 or not u > 0:
            raise ValueError("need 0 < eps <= 1 and u > 0")
        self.eps, self.u = float(eps), float(u)
        self.x = [[] for _ in range(K)]
        self._pending = np.zeros(K, dtype=int)

    _first_unpulled_or_pending = EmpiricalUCB._first_unpulled_or_pending

    def select(self, now_s):
        self.t += 1
        k = self._first_unpulled_or_pending()
        if k is not None:
            return k
        return int(np.argmax(self.index()))

    def index(self) -> np.ndarray:
        """BCL index per arm at the current t (all arms must have >= 1 sample)."""
        L = 2.0 * math.log(max(self.t, 2))
        return np.array([truncated_mean(self.x[a], self.u, self.eps, L)
                         + bcl_width(self.u, self.eps, L, self.n[a]) for a in range(self.K)])

    def observe(self, arm, tx_time_s, reward):
        self.n[arm] += 1
        self._pending[arm] = max(self._pending[arm] - 1, 0)
        self.x[arm].append(float(reward[self.field]))


def median_of_means(x, log_term: float) -> float:
    """BCL Lemma 2: k = floor(min(8 log(e^(1/8)/delta), n/2)) blocks of N = floor(n/k) points
    (the first kN samples), median of the block means. `log_term` = log(e^(1/8)/delta)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    k = int(max(1, math.floor(min(8.0 * log_term, n / 2.0))))
    N = n // k
    return float(np.median(x[: k * N].reshape(k, N).mean(axis=1)))


def mom_width(v: float, eps: float, log_term: float, n: int) -> float:
    """BCL Lemma 2 deviation: (12 v)^(1/(1+eps)) (16 log(e^(1/8)/delta) / n)^(eps/(1+eps)),
    with v bounding the CENTRED moment E|X - mu|^(1+eps) (Theorem 3)."""
    return (12.0 * v) ** (1.0 / (1.0 + eps)) * (16.0 * log_term / n) ** (eps / (1.0 + eps))


class MedianOfMeansUCB(TruncatedMeanUCB):
    """BCL robust UCB with the median-of-means estimator (Theorem 3; centred moment bound v)."""

    def __init__(self, K, field="gamma_db", eps=1.0, v=1.0):
        super().__init__(K, field, eps=eps, u=v)
        self.v = self.u

    def index(self) -> np.ndarray:
        log_term = 0.125 + 2.0 * math.log(max(self.t, 2))           # log(e^(1/8) / t^-2)
        return np.array([median_of_means(self.x[a], log_term)
                         + mom_width(self.v, self.eps, log_term, self.n[a]) for a in range(self.K)])


# --- AdaR-UCB (Genalti, Marsigli, Gatti, Metelli, COLT 2024) -----------------------------------
ADAR_C = (1.0 + math.sqrt(2.0)) ** 2


def adar_threshold(xp, log_inv_delta: float, c: float = ADAR_C) -> float | None:
    """Empirical trimming threshold: positive root M of sum_j min(X'_j^2 / M^2, 1) = c log(1/delta)
    (Genalti Eq. 13). The left side decreases from #nonzero (M -> 0) to 0 (M -> inf), so a root
    exists iff #{X'_j != 0} > c log(1/delta) (Proposition 9); otherwise None.
    Exact solve: with x2 sorted and j values below M^2, the equation is (n - j) + S_j / M^2 = target
    (S_j = sum of the j smallest), valid for M^2 in (x2[j-1], x2[j]]; O(n log n)."""
    x2 = np.sort(np.asarray(xp, dtype=float) ** 2)
    n = x2.size
    target = c * log_inv_delta
    if np.count_nonzero(x2) <= target:
        return None
    S = np.concatenate([[0.0], np.cumsum(x2)])
    j = np.arange(n + 1)
    denom = target - (n - j)
    with np.errstate(divide="ignore", invalid="ignore"):
        M2 = np.where(denom > 0, S / denom, np.nan)
    lower = np.concatenate([[0.0], x2])
    upper = np.concatenate([x2, [np.inf]])
    ok = np.flatnonzero((denom > 0) & (M2 > lower) & (M2 <= upper))
    if ok.size == 0:                       # numerically on a boundary: fall back to bisection
        return _adar_threshold_bisect(x2, target)
    return float(math.sqrt(M2[ok[0]]))


def _adar_threshold_bisect(x2, target):
    if np.count_nonzero(x2) <= target:
        raise ValueError("no positive root (Proposition 9 condition not met)")
    g = lambda M: float(np.sum(np.minimum(x2 / M ** 2, 1.0)))
    hi = math.sqrt(float(x2.max())) + 1.0
    while g(hi) > target:
        hi *= 2.0
    lo = hi
    while g(lo) < target:
        lo /= 2.0
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if g(mid) > target else (lo, mid)
        if hi / lo < 1 + 1e-12:
            break
    return math.sqrt(lo * hi)


class AdaRUCB(_Agent):
    """AdaR-UCB, theory-consistent version (owner decision 2026-09-30, THEORY-B T-7).

    Differences from the printed Algorithm 1, which contradicts the paper's own text/theory:
    - the threshold is computed from X' and the trimmed mean/variance from X (text of §5.1 and
      Theorem 6 need independent halves; lines 7-8 print X for both);
    - forced exploration while #{X' != 0} <= c log tau^3 with c = (1+sqrt 2)^2 (Proposition 9:
      the threshold exists iff the count exceeds c log(1/delta), delta = tau^-3; line 4 prints
      '4 log tau^-3', which is negative).
    The chosen arm is played twice per round (line 12): the first observation goes to X, the
    second to X'. Index (line 9): mu + sqrt(2 V log tau^3 / N) + 10 M log tau^3 / N.
    Guarantee requires Genalti's Assumption 1 (truncated non-positivity of the optimal arm).
    """

    def __init__(self, K, field="gamma_db"):
        super().__init__(K, field)
        self.X = [[] for _ in range(K)]
        self.Xp = [[] for _ in range(K)]
        self.tau = 0
        self._second = None                   # arm owing its second pull in this round
        self._toggle = np.zeros(K, dtype=int)

    def index(self) -> np.ndarray:
        L = 3.0 * math.log(max(self.tau, 2))  # log tau^3
        B = np.full(self.K, np.inf)
        for i in range(self.K):
            N = len(self.X[i])
            if N < 2:
                continue
            M = adar_threshold(self.Xp[i], L)
            if M is None:
                continue
            x = np.asarray(self.X[i])
            y = np.where(np.abs(x) <= M, x, 0.0)
            mu = float(y.mean())
            V = float(np.sum((y - mu) ** 2) / (N - 1))
            B[i] = mu + math.sqrt(2.0 * V * L / N) + 10.0 * M * L / N
        return B

    def select(self, now_s):
        self.t += 1
        if self._second is not None:
            k, self._second = self._second, None
            return k
        self.tau += 1
        B = self.index()
        k = int(np.argmax(B))                 # +inf (forced exploration) wins; ties -> first
        self._second = k
        return k

    def observe(self, arm, tx_time_s, reward):
        self.n[arm] += 1
        (self.X if self._toggle[arm] == 0 else self.Xp)[arm].append(float(reward[self.field]))
        self._toggle[arm] ^= 1


# --- NIR-UCB v0 (Proposal B; DRAFT pending group sign-off, THEORY-B T-8) ------------------------
class NIRUCB(_Agent):
    """Noise-Informed Robust UCB, version 0.

    Side information: every observe_noise() feeds a NoiseState (log-moment tracker, THEORY-B D-B3
    recommendation) giving (alpha_hat, c_hat) and regime alarms. `noise_to_reward` maps the noise
    law to the reward-noise law; default identity = the Layer-1 assumption that reward noise and
    noise stream share one law (the real mapping is the gate's M3 calibration).
    Estimator (plan §2.4, owner decision T-8): BCL truncated mean on y = X - a_k, with a_k the
    running median of arm k's samples (HEURISTIC: BCL's proof assumes a data-independent shift;
    it makes the raw-moment bound u ~ centred moment of the noise).
    Moment order (owner decision T-8): per arm and decision, eps in (0, min(1, alpha_r - 1)) that
    minimises the predicted BCL width 4 u(eps)^(1/(1+eps)) (L/n)^(eps/(1+eps)), with
    u(eps) = E|X|^(1+eps) of SaS(alpha_r, c_r); alpha_r >= 2 - 1e-3 -> eps = 1, u = 2 c_r^2.
    Regime alarm (D-B5 default, widths only): widths x `inflate` for `inflate_window` decisions.
    Before the noise state is available: round-robin (no side information yet).
    """

    EPS_GRID = np.linspace(0.02, 1.0, 50)

    def __init__(self, K, field="gamma_db", noise_state=None, noise_to_reward=None,
                 inflate=2.0, inflate_window=100, min_noise_samples=1000):
        super().__init__(K, field)
        from ..estimation.noise_state import NoiseState
        from ..estimation.regime import RegimeDetector
        self.ns = noise_state or NoiseState("log_moment", detector=RegimeDetector())
        self.map = noise_to_reward or (lambda a, c: (a, c))
        self.inflate, self.inflate_window = float(inflate), int(inflate_window)
        self.min_noise = int(min_noise_samples)
        self.x = [[] for _ in range(K)]
        self._noise_seen = 0
        self._inflate_until = -1
        self.state = {}

    def observe_noise(self, now_s, samples):
        out = self.ns.update(samples)
        self._noise_seen += np.size(samples)
        self.state = out
        if out.get("regime_change"):
            self._inflate_until = self.t + self.inflate_window

    def observe(self, arm, tx_time_s, reward):
        self.n[arm] += 1
        self.x[arm].append(float(reward[self.field]))

    def choose_eps(self, alpha_r, c_r, n, L) -> tuple[float, float]:
        if alpha_r >= 2.0 - 1e-3:
            return 1.0, 2.0 * c_r ** 2
        grid = self.EPS_GRID[self.EPS_GRID < min(1.0, alpha_r - 1.0) - 1e-6]
        if grid.size == 0:
            raise ValueError(f"alpha_r = {alpha_r} too close to 1 for a finite (1+eps) moment")
        us = np.array([sas_abs_moment(1.0 + e, alpha_r, c_r) for e in grid])
        w = 4.0 * us ** (1.0 / (1.0 + grid)) * (L / n) ** (grid / (1.0 + grid))
        j = int(np.argmin(w))
        return float(grid[j]), float(us[j])

    def index(self) -> np.ndarray:
        L = 2.0 * math.log(max(self.t, 2))
        alpha_r, c_r = self.map(self.state["alpha"], self.state["c"])
        mult = self.inflate if self.t <= self._inflate_until else 1.0
        out = np.empty(self.K)
        for a in range(self.K):
            x = np.asarray(self.x[a])
            shift = float(np.median(x))
            eps, u = self.choose_eps(alpha_r, c_r, x.size, L)
            out[a] = shift + truncated_mean(x - shift, u, eps, L) + mult * bcl_width(u, eps, L, x.size)
        return out

    def select(self, now_s):
        self.t += 1
        ready = self._noise_seen >= self.min_noise and np.isfinite(self.state.get("alpha", np.nan))
        if not ready or np.any(self.n < 2):
            return (self.t - 1) % self.K          # round-robin until side information and 2 samples/arm
        return int(np.argmax(self.index()))
