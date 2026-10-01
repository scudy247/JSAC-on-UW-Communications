"""Regime-change detector on the dense noise stream (proposal §3; D-B5 default: widths only).

Per noise block two statistics are formed:
  level  = log(robust sigma)                 (insensitive to sparse impulses)
  impuls = Var(log|x|) = (pi^2/6)(1/alpha^2 + 1/2) for SaS  (responds to the tail index)
Each is standardised against a baseline (mean and sd over the first `warmup` blocks of the
current regime) and fed to a two-sided CUSUM with drift k and threshold h (in baseline-sd
units). An alarm on either statistic declares a regime change: regime_id increments and the
baseline is re-learned over the next `warmup` blocks (no detection while learning). NIR-UCB
uses the alarm to inflate its widths without restarting. Cost: O(1) state per block.

Recalibration for real noise (2026-10-01; reports/logs/regime_real_20260930.log: 27-40 alarms/h on
FK01/HI01 with sub-dB level changes). Two causes in real recordings that i.i.d. synthetic tests miss:
block statistics are positively correlated from block to block, and the level wanders slowly.
Opt-in options (defaults reproduce the original detector exactly):
  long_run=True     scale each statistic by its LONG-RUN sd, sd sqrt((1 + r) / (1 - r)), with r the
                    lag-1 autocorrelation over the baseline (AR(1) approximation; r clipped to
                    [0, 0.95]): the CUSUM sums correlated increments, whose variance the plain sd
                    understates by that factor;
  min_shift_db, min_shift_impuls
                    smallest change worth an alarm, in dB of noise level and in units of
                    Var log|x|; the CUSUM drift becomes max(k * scale, min_shift / 2) (the
                    classical choice: drift = half the shift to detect), so slow wander and
                    sub-threshold changes accumulate nothing.
REAL_NOISE holds the settings evaluated on real recordings
(experiments/analysis/regime_real_recal_20261001.py).
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
    # dB of noise amplitude -> log(robust sigma): 20 log10(sigma) = dB  =>  log sigma = dB ln(10) / 20
    DB_TO_LOG = float(np.log(10.0) / 20.0)
    REAL_NOISE = dict(warmup=120, k=0.5, h=8.0, long_run=True, min_shift_db=2.0, min_shift_impuls=0.2)

    def __init__(self, warmup: int = 30, k: float = 0.5, h: float = 8.0, long_run: bool = False,
                 min_shift_db: float = 0.0, min_shift_impuls: float = 0.0):
        if warmup < 5 or k < 0 or h <= 0 or min_shift_db < 0 or min_shift_impuls < 0:
            raise ValueError("need warmup >= 5, k >= 0, h > 0, min shifts >= 0")
        self.warmup, self.k, self.h = int(warmup), float(k), float(h)
        self.long_run = bool(long_run)
        self.min_shift = np.array([min_shift_db * self.DB_TO_LOG, float(min_shift_impuls)])
        self.regime_id = 0
        self.n_blocks = 0
        self.alarms: list[int] = []
        self.alarm_stat: list[str] = []     # which statistic fired: 'level' and/or 'impuls'
        self._reset()

    @classmethod
    def for_real_noise(cls, **overrides):
        return cls(**{**cls.REAL_NOISE, **overrides})

    def _reset(self):
        self._buf = []                       # baseline statistics while learning (<= warmup)
        self._mu = self._sd = None
        self._g = np.zeros((2, 2))           # [statistic, (up, down)], in scale units

    def _learn(self):
        arr = np.array(self._buf)
        self._mu = arr.mean(axis=0)
        sd = np.maximum(arr.std(axis=0, ddof=1), 1e-12)
        if self.long_run:
            d = arr - self._mu
            r = np.sum(d[1:] * d[:-1], axis=0) / np.maximum(np.sum(d * d, axis=0), 1e-300)
            r = np.clip(r, 0.0, 0.95)
            sd = sd * np.sqrt((1 + r) / (1 - r))
        self._sd = sd
        self._drift = np.maximum(self.k, 0.5 * self.min_shift / sd)     # in scale units
        self._buf = []

    def update(self, x) -> dict:
        return self.update_stats(block_statistics(x))

    def update_stats(self, s) -> dict:
        """Same as update() for precomputed block_statistics(x)."""
        s = np.asarray(s, dtype=float)
        b = self.n_blocks
        self.n_blocks += 1
        if self._mu is None:
            self._buf.append(s)
            if len(self._buf) == self.warmup:
                self._learn()
            return {"alarm": False, "regime_id": self.regime_id, "learning": True}
        z = (s - self._mu) / self._sd
        self._g[:, 0] = np.maximum(0.0, self._g[:, 0] + z - self._drift)
        self._g[:, 1] = np.maximum(0.0, self._g[:, 1] - z - self._drift)
        if np.any(self._g > self.h):
            self.regime_id += 1
            self.alarms.append(b)
            fired = np.any(self._g > self.h, axis=1)
            self.alarm_stat.append("+".join(n for n, f in zip(("level", "impuls"), fired) if f))
            self._reset()
            return {"alarm": True, "regime_id": self.regime_id, "learning": False}
        return {"alarm": False, "regime_id": self.regime_id, "learning": False}
