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
