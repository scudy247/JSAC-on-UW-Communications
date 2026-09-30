"""Symmetric alpha-stable noise generators (Proposal B, noise models).

Conventions (match the channel library's impulsive model, reports/library_notes.md §3):
- Real SaS with index alpha in (0, 2] and scale c: E[exp(i theta X)] = exp(-|c theta|^alpha).
  At alpha = 2 this is Gaussian with variance 2 c^2 (so c = 1/sqrt(2) gives unit variance,
  the library's choice).
- Complex isotropic (sub-Gaussian) SaS: Z = sqrt(A) (G1 + i G2), G1, G2 ~ N(0, 2c^2) i.i.d.,
  A positive (alpha/2)-stable with E[exp(-s A)] = exp(-s^(alpha/2)). Each of Re Z, Im Z is
  SaS(alpha, c) and the law is rotation invariant: E[exp(i Re(conj(w) Z))] = exp(-(c|w|)^alpha).
- Pseudo-power 2c^2 per real component (the library's SNR convention, [P] §VI-A).

Every function takes an explicit numpy Generator: no global random state (project.MD §7.4).
"""

from __future__ import annotations

import numpy as np


def _check_rng(rng):
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator (explicit seeding)")


def sas_real(alpha: float, scale: float, size, rng: np.random.Generator) -> np.ndarray:
    """Real symmetric alpha-stable variates (Chambers-Mallows-Stuck, beta = 0)."""
    _check_rng(rng)
    if not 0.0 < alpha <= 2.0:
        raise ValueError(f"alpha must be in (0, 2], got {alpha}")
    if not scale > 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    v = rng.uniform(-np.pi / 2, np.pi / 2, size)
    w = rng.exponential(1.0, size)
    if alpha == 1.0:
        x = np.tan(v)
    else:
        x = (np.sin(alpha * v) / np.cos(v) ** (1.0 / alpha)
             * (np.cos((1.0 - alpha) * v) / w) ** ((1.0 - alpha) / alpha))
    return scale * x


def positive_stable(a: float, size, rng: np.random.Generator) -> np.ndarray:
    """Totally skewed positive a-stable, 0 < a < 1, with E[exp(-s A)] = exp(-s^a) (Kanter 1975)."""
    _check_rng(rng)
    if not 0.0 < a < 1.0:
        raise ValueError(f"a must be in (0, 1), got {a}")
    u = rng.uniform(0.0, np.pi, size)
    w = rng.exponential(1.0, size)
    return (np.sin(a * u) / np.sin(u) ** (1.0 / a)
            * (np.sin((1.0 - a) * u) / w) ** ((1.0 - a) / a))


def sas_complex_isotropic(alpha: float, scale: float, size,
                          rng: np.random.Generator) -> np.ndarray:
    """Isotropic complex SaS (sub-Gaussian construction); alpha = 2 gives CN(0, 4c^2)."""
    _check_rng(rng)
    if not 0.0 < alpha <= 2.0:
        raise ValueError(f"alpha must be in (0, 2], got {alpha}")
    if not scale > 0.0:
        raise ValueError(f"scale must be > 0, got {scale}")
    sigma = np.sqrt(2.0) * scale
    g = rng.normal(0.0, sigma, size) + 1j * rng.normal(0.0, sigma, size)
    if alpha == 2.0:
        return g
    return np.sqrt(positive_stable(alpha / 2.0, size, rng)) * g
