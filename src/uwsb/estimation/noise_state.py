"""Recursive noise-state estimation from the dense noise stream (THEORY-B §2.1, T4; D-B3).

Two candidate tail/scale trackers (D-B3 is decided by their tested convergence):
- LogMomentTracker: exponentially weighted first two moments of u = log|x|; alpha and c from
  Var u = (pi^2/6)(1/alpha^2 + 1/2) and E u = gamma(1/alpha - 1) + log c (as the batch
  log_moment_fit). Memory O(1); per-sample cost O(1); block updates are exact
  (weights lambda^(n-i) summed in closed form). forget = 1 gives the batch estimator exactly.
- QuantileTracker: exponentially weighted block quantiles (q05, q25, q75, q95) of x, then the
  quantile-ratio table of tail_index.stable_quantile_fit. Cost O(log B) per sample for blocks
  of B samples (a sort per block), memory O(1).
ExceedanceCounter counts |x| > k * c_hat with the scale estimate available BEFORE each block
(no look-ahead), for the warm-up analysis (T4). Zeros in x are skipped for log|x| and counted.
"""

from __future__ import annotations

import numpy as np

from .regime import RegimeDetector
from .tail_index import EULER_GAMMA, _ALPHA_GRID, _quantile_table


def _alpha_from_logvar(v: float) -> float:
    inv_a2 = 6.0 * v / np.pi ** 2 - 0.5
    return 2.0 if inv_a2 <= 0.25 else float(min(2.0, inv_a2 ** -0.5))


class LogMomentTracker:
    def __init__(self, forget: float = 1.0):
        if not 0.0 < forget <= 1.0:
            raise ValueError("forget must be in (0, 1]")
        self.forget = float(forget)
        self.w = self.s1 = self.s2 = 0.0          # weighted count, sum u, sum u^2
        self.n_zero = 0

    def update(self, x) -> None:
        x = np.asarray(x, dtype=float).ravel()
        nz = x != 0
        self.n_zero += int(np.sum(~nz))
        u = np.log(np.abs(x[nz]))
        n = u.size
        if n == 0:
            return
        lam = self.forget
        if lam == 1.0:
            wts = np.ones(n)
        else:
            wts = lam ** np.arange(n - 1, -1, -1, dtype=float)
        decay = lam ** n
        self.w = decay * self.w + float(wts.sum())
        self.s1 = decay * self.s1 + float(wts @ u)
        self.s2 = decay * self.s2 + float(wts @ (u * u))

    @property
    def n_eff(self) -> float:
        return self.w

    def estimate(self) -> tuple[float, float]:
        if self.w < 2:
            return float("nan"), float("nan")
        m = self.s1 / self.w
        v = max(self.s2 / self.w - m * m, 0.0)
        alpha = _alpha_from_logvar(v)
        return alpha, float(np.exp(m - EULER_GAMMA * (1.0 / alpha - 1.0)))


class QuantileTracker:
    def __init__(self, forget_blocks: float = 1.0, min_block: int = 200):
        if not 0.0 < forget_blocks <= 1.0:
            raise ValueError("forget_blocks must be in (0, 1]")
        self.forget = float(forget_blocks)
        self.min_block = int(min_block)
        self.w = 0.0
        self.q = np.zeros(4)                        # weighted mean of block quantiles

    def update(self, x) -> None:
        x = np.asarray(x, dtype=float).ravel()
        if x.size < self.min_block:
            raise ValueError(f"blocks must have at least {self.min_block} samples")
        qb = np.quantile(x, [0.05, 0.25, 0.75, 0.95])
        self.q = (self.forget * self.w * self.q + qb) / (self.forget * self.w + 1.0)
        self.w = self.forget * self.w + 1.0

    def estimate(self) -> tuple[float, float]:
        if self.w == 0:
            return float("nan"), float("nan")
        q05, q25, q75, q95 = self.q
        nu, iqr = _quantile_table()
        alpha = float(np.interp((q95 - q05) / (q75 - q25), nu[::-1], _ALPHA_GRID[::-1]))
        return alpha, float((q75 - q25) / np.interp(alpha, _ALPHA_GRID, iqr))


class ExceedanceCounter:
    def __init__(self, k_sigma: float = 5.0):
        self.k = float(k_sigma)
        self.n = 0
        self.n_exceed = 0

    def update(self, x, c_hat: float) -> None:
        x = np.asarray(x, dtype=float).ravel()
        self.n += x.size
        if np.isfinite(c_hat) and c_hat > 0:
            self.n_exceed += int(np.sum(np.abs(x) > self.k * c_hat))

    @property
    def rate(self) -> float:
        return self.n_exceed / self.n if self.n else float("nan")


class NoiseState:
    """Composite tracker fed with successive noise-only blocks (e.g. per slot)."""

    def __init__(self, method: str = "log_moment", forget: float = 1.0, k_sigma: float = 5.0,
                 detector: RegimeDetector | None = None):
        if method == "log_moment":
            self.tracker = LogMomentTracker(forget)
        elif method == "quantile":
            self.tracker = QuantileTracker(forget)
        else:
            raise ValueError(f"method must be 'log_moment' or 'quantile', got {method!r}")
        self.method = method
        self.exceed = ExceedanceCounter(k_sigma)
        self.detector = detector

    def update(self, x) -> dict:
        _, c_before = self.tracker.estimate()
        self.exceed.update(x, c_before)             # uses only past information
        self.tracker.update(x)
        alpha, c = self.tracker.estimate()
        out = {"alpha": alpha, "c": c, "n_exceed": self.exceed.n_exceed,
               "exceed_rate": self.exceed.rate, "regime_change": False, "regime_id": 0}
        if self.detector is not None:
            d = self.detector.update(x)
            out["regime_change"], out["regime_id"] = d["alarm"], d["regime_id"]
        return out
