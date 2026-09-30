"""Middleton Class A impulsive noise (complex baseband, sample-wise model).

Per sample: the number of active impulsive sources m ~ Poisson(A); conditionally on m the
noise is CN(0, s2_m) with
    s2_m = power * (m / A + Gamma) / (1 + Gamma),
so E|n|^2 = power for any (A, Gamma). A = impulsive index (small A = rare, strong impulses),
Gamma = Gaussian-to-impulsive power ratio.
Known properties (tested): P(m >= 1) = 1 - exp(-A); each real component has excess kurtosis
3 / (A (1 + Gamma)^2); A -> infinity gives Gaussian noise (THEORY-B §2.3, branch A).
Every function takes an explicit numpy Generator (project.MD §7.4).
"""

from __future__ import annotations

import numpy as np


def class_a_complex(A: float, Gamma: float, size, rng: np.random.Generator,
                    power: float = 1.0, return_states: bool = False):
    """Complex Class A samples; optionally also the Poisson states m (for tests/metrics only)."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator (explicit seeding)")
    if not A > 0.0:
        raise ValueError(f"A must be > 0, got {A}")
    if not Gamma >= 0.0:
        raise ValueError(f"Gamma must be >= 0, got {Gamma}")
    if not power > 0.0:
        raise ValueError(f"power must be > 0, got {power}")
    m = rng.poisson(A, size)
    s2 = power * (m / A + Gamma) / (1.0 + Gamma)
    n = np.sqrt(s2 / 2.0) * (rng.standard_normal(size) + 1j * rng.standard_normal(size))
    return (n, m) if return_states else n


def excess_kurtosis_per_component(A: float, Gamma: float) -> float:
    """Analytic excess kurtosis of Re(n) (or Im(n)) for Class A: 3 / (A (1 + Gamma)^2)."""
    return 3.0 / (A * (1.0 + Gamma) ** 2)
