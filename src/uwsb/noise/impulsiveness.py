"""Model-free impulsiveness statistics for ambient-noise recordings.

Used to screen real recordings before the gate (reports/noise_census.md, checkpoint 2).
Deliberately model-free: no tail-index estimate here. alpha-hat estimators (Hill,
McCulloch, log-moment) belong in estimation/tail_index.py (THEORY-B D-B3), built later.

Statistics per block of band-limited samples x (mean removed):
  robust_sigma   = 1.4826 * median(|x - median(x)|)   (= sigma for Gaussian x)
  excess_kurtosis = m4 / m2^2 - 3                     (0 for Gaussian, 3 for Laplace)
  exceed_frac    = fraction of |x - median(x)| > k * robust_sigma
  exceed_ratio   = exceed_frac / P(|Z| > k), Z ~ N(0,1)  (1 for Gaussian)

Band-limiting is causal (sosfilt) with filter state carried across blocks, so a
multi-hour file can be streamed block by block and gives the same output as one call.
Units in names (project.MD §7.6): fs_hz, band_hz.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

MAD_TO_SIGMA = 1.4826  # 1 / Phi^{-1}(3/4): MAD of N(0, s^2) is s / 1.4826


def gaussian_exceedance(k_sigma: float) -> float:
    """P(|Z| > k) for Z ~ N(0, 1)."""
    return math.erfc(k_sigma / math.sqrt(2.0))


def robust_sigma(x) -> float:
    """Scale from the median absolute deviation; equals sigma for Gaussian data."""
    x = np.asarray(x, dtype=float)
    return MAD_TO_SIGMA * float(np.median(np.abs(x - np.median(x))))


def block_stats(x, k_sigma: float = 5.0) -> dict:
    """Impulsiveness statistics of one block (see module docstring)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a 1-D array with at least 2 samples")
    c = x - x.mean()
    m2 = float(np.mean(c ** 2))
    if not m2 > 0.0:
        raise ValueError("block has zero variance")
    m4 = float(np.mean(c ** 4))
    s_rob = robust_sigma(x)
    if not s_rob > 0.0:
        raise ValueError("block has zero robust scale (more than half the samples equal)")
    exceed_frac = float(np.mean(np.abs(x - np.median(x)) > k_sigma * s_rob))
    return {
        "n": int(x.size),
        "std": math.sqrt(m2),
        "robust_sigma": s_rob,
        "excess_kurtosis": m4 / m2 ** 2 - 3.0,
        "k_sigma": float(k_sigma),
        "exceed_frac": exceed_frac,
        "exceed_ratio": exceed_frac / gaussian_exceedance(k_sigma),
    }


class BandpassStream:
    """Causal Butterworth band-pass whose state persists across calls."""

    def __init__(self, fs_hz: float, band_hz: tuple[float, float], order: int = 8):
        lo_hz, hi_hz = band_hz
        if not 0.0 < lo_hz < hi_hz < fs_hz / 2.0:
            raise ValueError(
                f"band {band_hz} Hz must satisfy 0 < lo < hi < fs/2 = {fs_hz / 2.0} Hz")
        self.fs_hz = float(fs_hz)
        self.band_hz = (float(lo_hz), float(hi_hz))
        self.sos = signal.butter(order, self.band_hz, btype="bandpass",
                                 fs=self.fs_hz, output="sos")
        self._zi = np.zeros((self.sos.shape[0], 2))

    def __call__(self, block):
        y, self._zi = signal.sosfilt(self.sos, np.asarray(block, dtype=float), zi=self._zi)
        return y
