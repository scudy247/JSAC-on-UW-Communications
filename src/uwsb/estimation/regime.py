"""Regime-change detector on the dense noise stream (proposal §3; D-B5 default: widths only).

Per noise block two statistics are formed:
  level  = log(robust sigma)                 (insensitive to sparse impulses)
  impuls = Var(log|x|) = (pi^2/6)(1/alpha^2 + 1/2) for SaS  (responds to the tail index)
Each is standardised against a baseline (mean and sd over the first `warmup` blocks of the
current regime) and fed to a two-sided CUSUM with drift k and threshold h (in baseline-sd
units). An alarm on either statistic declares a regime change: regime_id increments and the
baseline is re-learned over the next `warmup` blocks (no detection while learning). NIR-UCB
uses the alarm to inflate its widths without restarting. Cost: O(1) state per block.
"""

from __future__ import annotations

import numpy as np

from ..noise.impulsiveness import robust_sigma


def block_statistics(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float).ravel()
    nz = x[x != 0]
    if nz.size < 10:
        raise ValueError("block needs at least 10 non-zero samples")
    return float(np.log(robust_sigma(nz))), float(np.var(np.log(np.abs(nz))))


class RegimeDetector:
    def __init__(self, warmup: int = 30, k: float = 0.5, h: float = 8.0):
        if warmup < 5 or k < 0 or h <= 0:
            raise ValueError("need warmup >= 5, k >= 0, h > 0")
        self.warmup, self.k, self.h = int(warmup), float(k), float(h)
        self.regime_id = 0
        self.n_blocks = 0
        self.alarms: list[int] = []
        self._reset()

    def _reset(self):
        self._buf = []                       # baseline statistics while learning (<= warmup)
        self._mu = self._sd = None
        self._g = np.zeros((2, 2))           # [statistic, (up, down)]

    def update(self, x) -> dict:
        s = np.array(block_statistics(x))
        b = self.n_blocks
        self.n_blocks += 1
        if self._mu is None:
            self._buf.append(s)
            if len(self._buf) == self.warmup:
                arr = np.array(self._buf)
                self._mu = arr.mean(axis=0)
                self._sd = np.maximum(arr.std(axis=0, ddof=1), 1e-12)
                self._buf = []
            return {"alarm": False, "regime_id": self.regime_id, "learning": True}
        z = (s - self._mu) / self._sd
        self._g[:, 0] = np.maximum(0.0, self._g[:, 0] + z - self.k)
        self._g[:, 1] = np.maximum(0.0, self._g[:, 1] - z - self.k)
        if np.any(self._g > self.h):
            self.regime_id += 1
            self.alarms.append(b)
            self._reset()
            return {"alarm": True, "regime_id": self.regime_id, "learning": False}
        return {"alarm": False, "regime_id": self.regime_id, "learning": False}
