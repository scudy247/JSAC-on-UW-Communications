"""Predictive-UCB v2 and Predictive Thompson Sampling (THEORY.md §5).

Two-state scalar Kalman filter per arm, state x_k = [m_k ; d_k]:
  m_k  static arm mean      (drift model: random walk with rate q_m, ~0)
  d_k  OU deviation, forgets at rate alpha = exp(-gap / Tc_hat)

Observation model r = H x + eps, H = [1, 1], Var(eps) = sigma_n^2. Updates are
O(1) per arm (three covariance floats, one division); no matrix inversion, no GP
-- this is the embedded-friendly recursion that is the whole point of the paper.

Timeline (THEORY.md D2, simplified). A packet transmitted at slot s measures the
channel at reception time s + tau_ow; the decision at slot t targets a packet
received at t + tau_ow. The prediction gap is therefore (t+tau_ow)-(s+tau_ow) =
t - s, i.e. the *transmit-slot* age. So the agent lives entirely in the
transmit-slot timeline and never sees tau_ow: it needs only Tc_hat. The staleness
floor (age >= tau_rt) is produced by the environment's feedback queue, not here.
Consequently no agent method takes tau -- that would be oracle-adjacent bookkeeping
the env already owns.

Inputs that are physics, not tuning (THEORY.md §5): Tc_hat (from Doppler spread),
sigma2 = stationary deviation variance, sigma_n2 = observation-noise variance.
Free knobs: beta (UCB width) and q_m (mean-drift rate, default 0).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

_VAR_FLOOR = 0.0            # variances clipped to this; negativity is a bug
_NEG_TOL = 1e-9            # tolerance before we assert a real sign error


class PredictiveBanditBase:
    """Shared 2-state Kalman filter bank; subclasses define the policy."""

    def __init__(self, K: int, Tc_hat_s: float, sigma2: float, sigma_n2: float,
                 T_slot_s: float = 1.0, q_m: float = 0.0,
                 prior_m_mean: float = 0.0, prior_m_var: float = 1.0):
        if K < 1:
            raise ValueError("K must be >= 1")
        if not Tc_hat_s > 0:
            raise ValueError("Tc_hat_s must be > 0")
        if sigma2 < 0 or sigma_n2 < 0 or prior_m_var < 0 or q_m < 0:
            raise ValueError("variances and q_m must be >= 0")
        self.K = int(K)
        self.Tc_hat_s = float(Tc_hat_s)
        self.sigma2 = float(sigma2)         # stationary Var(d_k)
        self.sigma_n2 = float(sigma_n2)     # observation noise Var
        self.T_slot_s = float(T_slot_s)
        self.q_m = float(q_m)

        # Filter state, referenced to each arm's last-observed transmit slot.
        self.m_hat = np.full(self.K, float(prior_m_mean))
        self.d_hat = np.zeros(self.K)
        self.Pmm = np.full(self.K, float(prior_m_var))
        self.Pmd = np.zeros(self.K)
        self.Pdd = np.full(self.K, float(sigma2))     # stationary prior on d
        self.last_tx = np.full(self.K, -np.inf)        # transmit slot of filter ref
        self.seen = np.zeros(self.K, dtype=bool)
        self.n_obs = np.zeros(self.K, dtype=int)
        self._t = 0                                     # decisions taken (for beta_t)

    # -- alpha over a gap expressed in seconds --------------------------------
    def _alpha(self, gap_s):
        return np.exp(-np.asarray(gap_s, dtype=float) / self.Tc_hat_s)

    # -- prediction (pure; does not mutate state) -----------------------------
    def _predict_forward(self, now_s: float):
        """Predict every arm's reward-mean posterior to transmit-time `now_s`.

        Returns (mean, var) arrays of shape (K,), where mean = E[mu_k(now)|data]
        and var = Var of that posterior mean (EXCLUDING observation noise --
        THEORY.md §5 index uses sqrt(Pmm + 2 Pmd + Pdd)).
        """
        gap = now_s - self.last_tx                      # inf for unseen arms
        a = self._alpha(gap)                            # 0 for unseen (exp(-inf))
        Pdd_p = a * a * self.Pdd + self.sigma2 * (1.0 - a * a)
        Pmd_p = a * self.Pmd
        # Mean random-walk drift only accrues for seen arms over a finite gap.
        gap_safe = np.where(self.seen, np.where(np.isfinite(gap), gap, 0.0), 0.0)
        Pmm_p = self.Pmm + self.q_m * gap_safe
        mean = self.m_hat + a * self.d_hat
        var = Pmm_p + 2.0 * Pmd_p + Pdd_p
        return mean, self._clip_var(var)

    @staticmethod
    def _clip_var(v):
        v = np.asarray(v, dtype=float)
        if np.any(v < -_NEG_TOL):
            raise AssertionError(f"negative predictive variance: min={v.min():.3e}")
        return np.maximum(v, _VAR_FLOOR)

    # -- measurement update ---------------------------------------------------
    def observe(self, arm: int, tx_time_s: float, reward: float) -> None:
        """Fold one delayed feedback sample into arm's filter.

        `tx_time_s` is the transmit slot-time of the packet (its age is enforced
        >= tau_rt by the environment, never here). The predict step bridges from
        the arm's previous transmit-time reference to this one.
        """
        k = int(arm)
        if self.seen[k]:
            gap = tx_time_s - self.last_tx[k]
            if gap < 0:
                raise ValueError(
                    f"out-of-order feedback on arm {k}: tx {tx_time_s} < "
                    f"last {self.last_tx[k]}")
            a = math.exp(-gap / self.Tc_hat_s)
            # Predict to tx_time.
            self.d_hat[k] *= a
            self.Pdd[k] = a * a * self.Pdd[k] + self.sigma2 * (1.0 - a * a)
            self.Pmd[k] = a * self.Pmd[k]
            self.Pmm[k] = self.Pmm[k] + self.q_m * gap
        else:
            # First sample: reference the deviation state at this transmit slot.
            self.d_hat[k] = 0.0
            self.Pdd[k] = self.sigma2

        # Scalar-observation Kalman update, Joseph form for PSD safety.
        Pmm, Pmd, Pdd = self.Pmm[k], self.Pmd[k], self.Pdd[k]
        S = Pmm + 2.0 * Pmd + Pdd + self.sigma_n2      # innovation variance
        if S <= 0:
            raise AssertionError(f"non-positive innovation variance {S}")
        # Kalman gain K = P H^T / S, H=[1,1]  ->  K = [Pmm+Pmd, Pmd+Pdd]/S
        Km = (Pmm + Pmd) / S
        Kd = (Pmd + Pdd) / S
        innov = reward - (self.m_hat[k] + self.d_hat[k])
        self.m_hat[k] += Km * innov
        self.d_hat[k] += Kd * innov
        # Joseph: P <- (I-KH) P (I-KH)^T + K R K^T,  with H=[1,1].
        a11 = 1.0 - Km            # (I-KH) row m: [1-Km, -Km]
        a12 = -Km
        a21 = -Kd                 # row d: [-Kd, 1-Kd]
        a22 = 1.0 - Kd
        # M = (I-KH) P
        m11 = a11 * Pmm + a12 * Pmd
        m12 = a11 * Pmd + a12 * Pdd
        m21 = a21 * Pmm + a22 * Pmd
        m22 = a21 * Pmd + a22 * Pdd
        # P' = M (I-KH)^T + K R K^T
        nPmm = m11 * a11 + m12 * a12 + Km * Km * self.sigma_n2
        nPmd = m11 * a21 + m12 * a22 + Km * Kd * self.sigma_n2
        nPdd = m21 * a21 + m22 * a22 + Kd * Kd * self.sigma_n2
        self.Pmm[k] = max(nPmm, _VAR_FLOOR)
        self.Pdd[k] = max(nPdd, _VAR_FLOOR)
        self.Pmd[k] = nPmd
        if self.Pmm[k] < -_NEG_TOL or self.Pdd[k] < -_NEG_TOL:
            raise AssertionError("Joseph update produced negative variance")

        self.last_tx[k] = tx_time_s
        self.seen[k] = True
        self.n_obs[k] += 1

    # -- policy hook ----------------------------------------------------------
    def select(self, now_s: float, rng: np.random.Generator | None = None) -> int:
        raise NotImplementedError

    # -- introspection (for tests / metrics; NOT used to drive the policy) ----
    def posterior(self, now_s: float):
        """(mean, std) of each arm's predicted reward-mean at transmit-time now."""
        mean, var = self._predict_forward(now_s)
        return mean, np.sqrt(var)


class PredictiveUCB(PredictiveBanditBase):
    """Index = mean + beta_t * sqrt(Pmm + 2 Pmd + Pdd)  (THEORY.md §5)."""

    def __init__(self, *args, beta: float | Callable[[int], float] = 1.5, **kw):
        super().__init__(*args, **kw)
        self._beta = beta

    def beta_t(self) -> float:
        if callable(self._beta):
            return float(self._beta(self._t))
        return float(self._beta)

    def select(self, now_s: float, rng: np.random.Generator | None = None) -> int:
        self._t += 1
        mean, var = self._predict_forward(now_s)
        idx = mean + self.beta_t() * np.sqrt(var)
        return int(np.argmax(idx))


class PredictiveThompson(PredictiveBanditBase):
    """Sample mu_k(now) ~ N(mean, var) per arm, play the argmax.

    Parameter-free; delayed feedback is unproblematic for TS (THEORY.md D5).
    """

    def select(self, now_s: float, rng: np.random.Generator | None = None) -> int:
        self._t += 1
        if rng is None:
            raise ValueError("PredictiveThompson.select needs an rng")
        mean, var = self._predict_forward(now_s)
        sample = mean + np.sqrt(var) * rng.standard_normal(self.K)
        return int(np.argmax(sample))


def beta_theory(c: float = 1.0) -> Callable[[int], float]:
    """UCB width sqrt(2 ln t) (THEORY.md D5), scaled by c. t counted from 1."""
    return lambda t: c * math.sqrt(2.0 * math.log(max(t, 2)))
