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
"""

from __future__ import annotations

import math

import numpy as np

from .robust import EmpiricalUCB


class NIRUCBv1(EmpiricalUCB):
    def __init__(self, K, field="gamma_db", law=None, width="gauss", forget=1.0, detector_params=None,
                 inflate=2.0, inflate_window=100, min_noise_samples=1000):
        super().__init__(K, field, sigma=1.0)
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
            self.params = self.law(out["alpha"], out["c"])

    def widths(self) -> np.ndarray:
        sigma, nu2, b = self.params
        L = math.log(max(self.t, 2))
        n = self.n.astype(float)
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
