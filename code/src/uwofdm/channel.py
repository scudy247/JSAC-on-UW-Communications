"""
Time-varying multipath channel with bulk and path-specific Doppler, in complex baseband (PLAN: Phase 1).

Each path p has a delay d_p [s], a complex gain g_p and its own Doppler-scale trajectory
eps_p(t) = eps_bulk(t) + delta_p(t). The bulk part is common to all paths (the link Doppler of motion.link_doppler);
delta_p is what surface reflections and different arrival angles add. Only the bulk part can be removed by one
resampling; delta_p remains after compensation and is counted as residual (README: Scope).

Paths are given by hand or drawn at random (exponential_paths). Planned: paths from a Bellhop arrivals file (delays,
amplitudes, angles), with delta_p = eps_bulk * (cos of the path's angle relative to the direct path - 1) as a
first-order geometric model.
"""
from dataclasses import dataclass

import numpy as np

from .waveform import warp, warp_delayed


@dataclass
class Path:
    delay: float  # [s], relative to the first arrival
    gain: object  # complex, or one value per output sample (fading within the packet)
    delta_eps: object = 0.0  # path-specific Doppler offset: constant, or one value per output sample


def apply(s, fs, fc, eps_bulk, paths, extra=0):
    """Received baseband: sum over paths of warp(delayed s, eps_bulk + delta_p) * gain, delays fractional (before
    12 Oct 2026 they were rounded to whole samples, which flattered per-path cancellation). The output has
    len(s) + extra samples (room for the longest delay and compression); eps_bulk per output sample, or scalar."""
    n = len(s) + extra
    eps_bulk = np.broadcast_to(np.asarray(eps_bulk, float), (n,)) if np.ndim(eps_bulk) == 0 else \
        np.pad(np.asarray(eps_bulk, float), (0, max(0, n - len(eps_bulk))), mode="edge")[:n]
    s_pad = np.pad(s, (0, extra))
    y = np.zeros(n, complex)
    for p in paths:
        g = p.gain if np.ndim(p.gain) == 0 else np.pad(np.asarray(p.gain, complex), (0, max(0, n - len(p.gain))),
                                                         mode="edge")[:n]
        y += g * warp_delayed(s_pad, fs, eps_bulk + p.delta_eps, fc, p.delay)  # fractional delays (12 Oct 2026)
    return y


def awgn(y, snr_db, rng, ref_power=None):
    """Add complex white noise at snr_db relative to ref_power (default: mean power of y)."""
    p = np.mean(np.abs(y) ** 2) if ref_power is None else ref_power
    sigma = np.sqrt(p / 10 ** (snr_db / 10) / 2)
    return y + sigma * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))


def exponential_paths(n_paths, spread, rng, decay_db=6.0, delta_eps_std=0.0):
    """Random sparse channel: n_paths delays uniform over `spread` [s] (first at 0), power decaying by decay_db over
    the spread, Rayleigh amplitudes, unit total power; optional path-specific Doppler offsets (std delta_eps_std)."""
    d = np.sort(np.concatenate([[0.0], rng.uniform(0, spread, n_paths - 1)]))
    pw = 10 ** (-decay_db * d / spread / 10)
    g = np.sqrt(pw / 2) * (rng.standard_normal(n_paths) + 1j * rng.standard_normal(n_paths))
    g /= np.sqrt(np.sum(np.abs(g) ** 2))
    de = delta_eps_std * rng.standard_normal(n_paths)
    de[0] = 0.0
    return [Path(float(a), complex(b), float(c)) for a, b, c in zip(d, g, de)]
