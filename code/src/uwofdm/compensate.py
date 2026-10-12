"""
C1: Doppler compensation inside the packet, at the receiver (README: C1; PLAN: Phase 2).

Estimation is an input (README ground rule): the receiver has M estimates of the Doppler scale at known instants
inside the packet (e.g. from preamble, pilots or decision feedback), with a stated error. It fits a trajectory model
to them and resamples along it (waveform.unwarp). Models:
    order 0   one scale for the packet
    order 1   linear in time (constant acceleration, as Cho-Kim 2026)
    order 2   quadratic (adds jerk)
    'piece'   piecewise constant, one scale per estimate (per symbol when estimates are per symbol)
"""
import numpy as np

from .waveform import unwarp


def fit_trajectory(t_est, eps_est, t_grid, model):
    """Trajectory eps_hat on t_grid from estimates eps_est at t_est, for model 0, 1, 2 or 'piece'."""
    t_est, eps_est = np.asarray(t_est, float), np.asarray(eps_est, float)
    if model == "piece":
        edges = 0.5 * (t_est[1:] + t_est[:-1])
        return eps_est[np.searchsorted(edges, t_grid)]
    order = int(model)
    tc = t_est.mean()
    coef = np.polyfit(t_est - tc, eps_est, order)
    return np.polyval(coef, np.asarray(t_grid, float) - tc)


def compensate(y, fs, fc, eps_hat):
    """Receiver compensation: resample y along the estimated trajectory eps_hat (one value per sample)."""
    return unwarp(y, fs, eps_hat, fc)
