"""Regime-switching impulsive noise (Layer 1; THEORY-B §2.1, regime Z_t).

Blocks of `block` samples; the regime Z follows a Markov chain across blocks with transition
matrix P; within regime z the samples are i.i.d. SaS(alpha_z, c_z) (alpha_z = 2: Gaussian with
variance 2 c_z^2). Returns the samples and the true regime of every block (for metrics only:
never pass it to an agent, project.MD §7.2).
"""

from __future__ import annotations

import numpy as np

from .stable import sas_real


def regime_switching_noise(regimes, P, n_blocks: int, block: int, rng: np.random.Generator,
                           z0: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """regimes: list of (alpha, c); P: (R, R) row-stochastic. Returns (x[n_blocks*block], z[n_blocks])."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator (explicit seeding)")
    P = np.asarray(P, dtype=float)
    R = len(regimes)
    if P.shape != (R, R) or np.any(P < 0) or not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("P must be a row-stochastic (R, R) matrix")
    z = np.empty(n_blocks, dtype=int)
    z[0] = z0
    for b in range(1, n_blocks):
        z[b] = rng.choice(R, p=P[z[b - 1]])
    x = np.empty(n_blocks * block)
    for b in range(n_blocks):
        alpha, c = regimes[z[b]]
        x[b * block:(b + 1) * block] = sas_real(alpha, c, block, rng)
    return x, z
