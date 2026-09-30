"""Seedable re-implementation of the channel library's noise model (THEORY-B T-4 (b)).

The library's `uwa_channels.noisegen` (v0.7.1, MIT) draws its drivers from NumPy's global
RNG, so it cannot be seeded per call (reports/library_notes.md §4). This module applies the
same arithmetic with an explicit numpy Generator:

  drivers z[n, j] i.i.d.: N(0, 1) if alpha == 2, else SaS(alpha, scale 1/sqrt(2))
  mixing  w[n, i] = sum_j sum_k beta[i, j, k] z[n + k, j]      at the model rate Fs
  then    polyphase resampling Fs -> fs and truncation to the requested length.

`mix_drivers` is the deterministic part; the tests feed the same drivers to it and to the
library's private `_noise_mixing` and require identical output.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy import signal

from .stable import sas_real


def _beta_mmk(noise) -> np.ndarray:
    """Mixing coefficients as (M, M, K), undoing h5py's axis reversal (as noisegen does)."""
    beta = np.asarray(noise["beta"])
    if beta.ndim == 3 and beta.shape[0] != beta.shape[1]:
        beta = np.transpose(beta, (2, 1, 0))
    return beta


def model_params(noise) -> tuple[float, float, np.ndarray]:
    """(alpha, Fs_hz, beta[M, M, K]) of a library noise file loaded with load_noise."""
    return float(noise["alpha"][0, 0]), float(noise["Fs"][0, 0]), _beta_mmk(noise)


def n_driver_samples(n_samples: int, fs_hz: float, noise) -> tuple[int, int]:
    """(K, K_mix): output length at the model rate and mixing length, as in noisegen."""
    _, Fs_hz, beta = model_params(noise)
    return int(np.ceil(n_samples / fs_hz * Fs_hz)), beta.shape[2]


def mix_drivers(z: np.ndarray, n_samples: int, fs_hz: float, array_index, noise) -> np.ndarray:
    """Deterministic mixing + resampling of given drivers z with shape (K + K_mix, M)."""
    _, Fs_hz, beta = model_params(noise)
    K, K_mix = n_driver_samples(n_samples, fs_hz, noise)
    beta_sub = beta[list(array_index), :, :]
    if z.shape != (K + K_mix, beta.shape[0]):
        raise ValueError(f"z must have shape {(K + K_mix, beta.shape[0])}, got {z.shape}")
    w = np.zeros((K, len(beta_sub)))
    for k in range(K_mix):
        w += z[k:k + K, :] @ beta_sub[:, :, k].T
    frac = Fraction(fs_hz / Fs_hz).limit_denominator()
    w = signal.resample_poly(w, frac.numerator, frac.denominator, axis=0)
    return w[:n_samples, :]


def library_noise(n_samples: int, fs_hz: float, array_index, noise,
                  rng: np.random.Generator) -> np.ndarray:
    """Real passband noise (n_samples x len(array_index)) at fs_hz from a library noise file."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator (explicit seeding)")
    alpha, _, beta = model_params(noise)
    K, K_mix = n_driver_samples(n_samples, fs_hz, noise)
    shape = (K + K_mix, beta.shape[0])
    z = rng.standard_normal(shape) if alpha == 2.0 else sas_real(alpha, 1.0 / np.sqrt(2.0), shape, rng)
    return mix_drivers(z, n_samples, fs_hz, array_index, noise)
