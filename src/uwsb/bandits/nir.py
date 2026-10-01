"""NIR-UCB v1 (THEORY-B G-0/G-5): naive-UCB structure, confidence widths calibrated online from the noise
stream. DRAFT method pending the mid-October checkpoint.

Side information: a NoiseState (log-moment tracker, D-B3) gives (alpha_hat, c_hat); a reward-law map
(estimation/reward_law.py) turns them into the reward-noise scale:
  width="gauss"      w = sigma(alpha_hat) sqrt(2 L / n)       (equals naive UCB with the oracle sigma when
                                                                alpha_hat is exact; L = log t as in EmpiricalUCB)
  width="bernstein"  w = min_j max(sqrt(2 nu2_j L / n), 2 b_j L / n)   (sub-exponential widths, valid for every
                                                                        tabulated pair; the minimising pair is
                                                                        used, like v0's minimum-width eps rule)
Index: empirical mean + w (x `inflate` for `inflate_window` decisions after a regime alarm). Regime alarm
(detector: RegimeDetector.REAL_NOISE by default, G-4): the tracker restarts so alpha_hat follows the new regime;
the last calibrated parameters are kept until the new tracker has `min_noise_samples` samples. Before any
side information: round-robin. Delay-safe initialisation as EmpiricalUCB.

debias=True (regime-switching noise, 2026-10-01): the reward-noise mean moves with the noise law (branch E: a 12 dB
drop from alpha 1.9 to 1.3, while arm gaps are ~1 dB), so raw averages mix regimes. Each observation is debiased with
E[X](alpha-hat, c-hat) from the law (`law.bias`) at the time it is received, and the width uses the noise scale of
each observation: gauss w = sqrt(2 L sum_i sigma_i^2) / n; bernstein w = min_j max(sqrt(2 L sum_i nu2_ij) / n,
2 B_j L / n) with B_j = max_i b_ij. Observations received before the first calibration are debiased retroactively
when it arrives. Off by default (stationary runs unchanged).
"""

from __future__ import annotations

import math

import numpy as np

from .robust import EmpiricalUCB


class NIRUCBv1(EmpiricalUCB):
    def __init__(self, K, field="gamma_db", law=None, width="gauss", forget=1.0, detector_params=None,
                 inflate=2.0, inflate_window=100, min_noise_samples=1000, debias=False):
        super().__init__(K, field, sigma=1.0)
        self.debias = bool(debias)
        self._c = None                           # c-hat at the current calibration
        self._early = []                         # (arm, raw reward) received before the first calibration
        self.v = np.zeros(K)                     # debias mode: sum of sigma_i^2 per arm
        self.vj = None                           # debias mode: sum of nu2_ij per arm (K, n_factors)
        self.bj = None                           # debias mode: max b_ij per arm
        from ..estimation.noise_state import LogMomentTracker, NoiseState
        from ..estimation.regime import RegimeDetector
        if law is None:
            raise ValueError("a reward-law map is required (estimation/reward_law.py)")
        if width not in ("gauss", "bernstein"):
            raise ValueError("width must be 'gauss' or 'bernstein'")
        # detector_params: None -> REAL_NOISE preset; dict -> RegimeDetector(**dict); False -> no detector (ablation)
        dp = RegimeDetector.REAL_NOISE if detector_params is None else detector_params
        self.ns = NoiseState("log_moment", forget=forget, detector=RegimeDetector(**dp) if dp is not False else None)
        self._new_tracker = lambda: LogMomentTracker(forget)
        self.law, self.width = law, width
        self.inflate, self.inflate_window = float(inflate), int(inflate_window)
        self.min_noise = int(min_noise_samples)
        self._since_reset = 0
        self._inflate_until = -1
        self.params = None                       # (sigma, nu2[j], b[j]) in use
        self.alarms = 0

    def observe_noise(self, now_s, samples):
        out = self.ns.update(samples)
        self._since_reset += np.size(samples)
        if out.get("regime_change"):
            self.alarms += 1
            self._inflate_until = self.t + self.inflate_window
            self.ns.tracker = self._new_tracker()    # follow the new regime; keep old params meanwhile
            self._since_reset = 0
            return
        if self._since_reset >= self.min_noise and np.isfinite(out["alpha"]):
            first = self.params is None
            self.params = self.law(out["alpha"], out["c"])
            self._alpha, self._c = float(out["alpha"]), float(out["c"])
            if self.debias and first and self._early:
                early, self._early = self._early, []
                for arm, r in early:                 # undo the raw accumulation, redo it debiased
                    self.n[arm] -= 1
                    self.s[arm] -= r
                    self._accumulate(arm, r)

    def _accumulate(self, arm, r):
        sigma, nu2, b = self.params
        self.n[arm] += 1
        self.s[arm] += r - self.law.bias(self._alpha, self._c)
        self.v[arm] += sigma ** 2
        if nu2 is not None:
            if self.vj is None:
                self.vj = np.zeros((self.K, len(nu2)))
                self.bj = np.zeros((self.K, len(nu2)))
            self.vj[arm] += nu2
            self.bj[arm] = np.maximum(self.bj[arm], b)

    def observe(self, arm, tx_time_s, reward):
        if not self.debias:
            return super().observe(arm, tx_time_s, reward)
        self._pending[arm] = max(self._pending[arm] - 1, 0)
        r = float(reward[self.field])
        if self.params is None:                      # no calibration yet: keep raw, fix later
            self.n[arm] += 1
            self.s[arm] += r
            self._early.append((arm, r))
            return
        self._accumulate(arm, r)

    def widths(self) -> np.ndarray:
        sigma, nu2, b = self.params
        L = math.log(max(self.t, 2))
        n = self.n.astype(float)
        if self.debias:
            if self.width == "gauss":
                w = np.sqrt(2.0 * L * self.v) / n
            else:
                w = np.min(np.maximum(np.sqrt(2.0 * L * self.vj) / n[:, None], 2.0 * self.bj * L / n[:, None]), axis=1)
            return w * (self.inflate if self.t <= self._inflate_until else 1.0)
        if self.width == "gauss":
            w = sigma * np.sqrt(2.0 * L / n)
        else:
            if nu2 is None:
                raise ValueError("bernstein widths need a law with sub-exponential pairs (branch E)")
            w = np.min(np.maximum(np.sqrt(2.0 * np.outer(1.0 / n, nu2) * L), 2.0 * np.outer(1.0 / n, b) * L), axis=1)
        return w * (self.inflate if self.t <= self._inflate_until else 1.0)

    def select(self, now_s):
        self.t += 1
        k = self._first_unpulled_or_pending()
        if k is not None:
            return k
        if self.params is None:
            return (self.t - 1) % self.K            # no side information yet
        return int(np.argmax(self.s / self.n + self.widths()))
