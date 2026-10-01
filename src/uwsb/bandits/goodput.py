"""Agents for the Layer-1 goodput model (envs/goodput.py; arms a = j * M + m, sub-band j, MCS m).

Unstructured, ACK-based (one arm per sub-band x MCS; feedback = success):
  GoodputAckTS     Beta(1, 1) per arm on success; plays argmax rate_a * theta_a.
  GoodputAckUCB1   UCB1 on goodput / max rate (in [0, 1]).
Structured on gamma-hat (gamma-hat of sub-band j is observed whatever MCS is played on j):
  StructuredGoodputUCB  per sub-band location S_j = E[gamma_j] - E[X]; optimistic S_j + w_j; plays the best
                        (j, m) under P_m(S; alpha) from a SuccessTable.
      calibration = <alpha>         fixed: the table at that alpha (2.0 = AWGN-calibrated, what a modem uses);
      calibration = "noise_stream"  NIR-UCB v1-goodput: alpha-hat from the noise stream sets the gamma-hat bias
                                    E[X], the width and the success curves (REAL_NOISE detector by default).
      width "gauss": sigma_X sqrt(2 log t / n_j); "bernstein": min_j max(sqrt(2 nu2 L / n), 2 b L / n).
      fading_aware=True: estimates the fading variance as the pooled variance of gamma-hat beyond the noise law,
      sf2 = max(0, var - sigma_X^2(alpha)), averages the success curves over N(0, sf2) (9-node Gauss-Hermite) and
      widens the location width to sqrt(sigma_X^2 + sf2) (gauss). Off by default (no-fading runs unchanged).
  ThresholdAMC     comms-native: per sub-band median of the last `window` gamma-hats vs AWGN thresholds (the S
                   where P_m(S; 2) >= target, shifted by E[X; 2]); highest feasible rate, greedy; one initial probe
                   per sub-band at the lowest MCS.
"""

from __future__ import annotations

import math

import numpy as np

from .robust import EmpiricalUCB


class GoodputAckTS(EmpiricalUCB):
    def __init__(self, K, rates, rng=None):
        super().__init__(K, "ack")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator (explicit seeding)")
        self.rates, self.rng = np.asarray(rates, float), rng

    def select(self, now_s):
        self.t += 1
        return int(np.argmax(self.rates * self.rng.beta(1 + self.s, 1 + self.n - self.s)))


class GoodputAckUCB1(EmpiricalUCB):
    def __init__(self, K, rates):
        super().__init__(K, "goodput", sigma=1.0)
        self.rmax = float(np.max(rates))

    def observe(self, arm, tx_time_s, reward):
        super().observe(arm, tx_time_s, {"goodput": reward["goodput"] / self.rmax})


class StructuredGoodputUCB:
    GH_X, GH_W = np.polynomial.hermite_e.hermegauss(9)

    def __init__(self, J, M, rates, table, calibration="noise_stream", width="gauss", detector_params=None,
                 inflate=2.0, inflate_window=100, min_noise_samples=1000, fading_aware=False):
        from ..estimation.noise_state import LogMomentTracker, NoiseState
        from ..estimation.regime import RegimeDetector
        if width not in ("gauss", "bernstein"):
            raise ValueError("width must be 'gauss' or 'bernstein'")
        self.J, self.M, self.K = int(J), int(M), int(J) * int(M)
        self.rates = np.asarray(rates, float)
        self.table, self.width = table, width
        self.t = 0
        self.nj = np.zeros(J, dtype=int)
        self.sj = np.zeros(J)
        self.qj = np.zeros(J)                             # sum of gamma-hat^2 (fading-aware)
        self.fading_aware = bool(fading_aware)
        self._pending = np.zeros(J, dtype=int)
        self.fixed_alpha = None if calibration == "noise_stream" else float(calibration)
        self.alpha = self.fixed_alpha
        if self.fixed_alpha is None:
            dp = RegimeDetector.REAL_NOISE if detector_params is None else detector_params
            self.ns = NoiseState("log_moment", detector=RegimeDetector(**dp) if dp is not False else None)
            self._new_tracker = lambda: LogMomentTracker(1.0)
            self._since_reset = 0
            self.min_noise = int(min_noise_samples)
        self.inflate, self.inflate_window = float(inflate), int(inflate_window)
        self._inflate_until = -1

    def observe_noise(self, now_s, samples):
        if self.fixed_alpha is not None:
            return
        out = self.ns.update(samples)
        self._since_reset += np.size(samples)
        if out.get("regime_change"):
            self._inflate_until = self.t + self.inflate_window
            self.ns.tracker = self._new_tracker()
            self._since_reset = 0
            return
        if self._since_reset >= self.min_noise and np.isfinite(out["alpha"]):
            self.alpha = float(out["alpha"])

    def observe(self, arm, tx_time_s, reward):
        j = arm // self.M
        self.nj[j] += 1
        self._pending[j] = max(self._pending[j] - 1, 0)
        self.sj[j] += reward["gamma_db"]
        self.qj[j] += reward["gamma_db"] ** 2

    def sub_band_index(self) -> np.ndarray:
        """Optimistic location S_j + w_j per sub-band."""
        xmean, sigma, nu2, b = self.table.law(self.alpha)
        L = math.log(max(self.t, 2))
        n = self.nj.astype(float)
        sf2 = self.fading_var(sigma) if self.fading_aware else 0.0
        if self.width == "gauss":
            w = math.sqrt(sigma ** 2 + sf2) * np.sqrt(2.0 * L / n)
        else:
            w = np.min(np.maximum(np.sqrt(2.0 * np.outer(1.0 / n, nu2 + sf2) * L), 2.0 * np.outer(1.0 / n, b) * L), axis=1)
        w = w * (self.inflate if self.t <= self._inflate_until else 1.0)
        return self.sj / n - xmean + w

    def fading_var(self, sigma_x) -> float:
        """Pooled within-sub-band variance of gamma-hat minus the noise-law variance (>= 0)."""
        n = self.nj.astype(float)
        ok = n >= 2
        if not np.any(ok):
            return 0.0
        ss = np.sum(self.qj[ok] - self.sj[ok] ** 2 / n[ok])
        return max(0.0, ss / np.sum(n[ok] - 1) - sigma_x ** 2)

    def expected_success(self, S) -> np.ndarray:
        """P_m(S) (J, M), averaged over the estimated fading if fading-aware."""
        if not self.fading_aware:
            return self.table.psucc_at(self.alpha, S)
        sf = math.sqrt(self.fading_var(self.table.law(self.alpha)[1]))
        if sf == 0.0:
            return self.table.psucc_at(self.alpha, S)
        acc = 0.0
        for x, wgt in zip(self.GH_X, self.GH_W):
            acc = acc + wgt * self.table.psucc_at(self.alpha, np.asarray(S) + sf * x)
        return acc / math.sqrt(2.0 * math.pi)

    def select(self, now_s):
        self.t += 1
        z = np.flatnonzero((self.nj == 0) & (self._pending == 0))
        if z.size:                                        # delay-safe first probe of each sub-band, lowest MCS
            self._pending[z[0]] += 1
            return int(z[0]) * self.M
        if np.any(self.nj == 0) or self.alpha is None:
            j = int(np.argmin(self.nj + self._pending)) if np.any(self.nj == 0) else (self.t - 1) % self.J
            return j * self.M
        g = self.rates[None, :] * self.expected_success(self.sub_band_index())             # (J, M)
        return int(np.argmax(g))                                                           # = j * M + m


class ThresholdAMC:
    def __init__(self, J, M, rates, table, alpha_cal=2.0, target=0.9, window=16):
        self.J, self.M, self.K = int(J), int(M), int(J) * int(M)
        self.rates = np.asarray(rates, float)
        P = table.psucc_at(alpha_cal, table.s_grid)                     # (n_S, M)
        xmean = table.law(alpha_cal)[0]
        self.thr = np.array([table.s_grid[np.argmax(P[:, m] >= target)] if np.any(P[:, m] >= target) else np.inf
                             for m in range(self.M)]) + xmean           # thresholds on the gamma-hat scale
        self.window = int(window)
        self.hist = [[] for _ in range(J)]
        self.t = 0
        self._probed = np.zeros(J, dtype=bool)

    def observe_noise(self, now_s, samples):
        return

    def observe(self, arm, tx_time_s, reward):
        h = self.hist[arm // self.M]
        h.append(float(reward["gamma_db"]))
        if len(h) > self.window:
            del h[0]

    def select(self, now_s):
        self.t += 1
        z = np.flatnonzero(~self._probed)
        if z.size:
            self._probed[z[0]] = True
            return int(z[0]) * self.M
        best, arm = -1.0, 0
        for j in range(self.J):
            if not self.hist[j]:
                continue
            med = float(np.median(self.hist[j]))
            ok = np.flatnonzero(self.thr <= med)
            m = int(ok[-1]) if ok.size else 0
            if self.rates[m] > best:
                best, arm = self.rates[m], j * self.M + m
        return arm
