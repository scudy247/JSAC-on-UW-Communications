"""
E2 fusion (PLAN: Phase 5): Kalman filter + RTS smoother for what the sensor prior missed, over one packet.

After resampling along the prior (side information + own IMU), the residual Doppler r(t) is small and smooth (IMU
drift, latency error, quantisation of the side information). State x = [tau, r, r'] with tau = int r dt (timing
drift, s), r' driven by white jerk noise of spectral density q (integrated random walk on r). Measurements:
  'tau' : arrival time of a chirp relative to its nominal position (postamble; the preamble fixes tau(0) = 0)
  'r'   : a residual Doppler scale, e.g. a per-symbol frequency offset / fc, with its variance
The smoother returns r(t) on any grid, used for one resampling (no per-symbol jumps). Prior uncertainty
(sigma_r0, sigma_dr0) and q describe how far the prior may be off: wide -> trust the signal, narrow -> trust the prior.
"""
import numpy as np


def _F(dt):
    return np.array([[1.0, dt, dt * dt / 2], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])


def _Q(dt, q):
    return q * np.array([[dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                         [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                         [dt ** 3 / 6, dt ** 2 / 2, dt]])


H = {"tau": np.array([1.0, 0.0, 0.0]), "r": np.array([0.0, 1.0, 0.0])}


def smooth(events, t_out, sigma_r0, sigma_dr0, q):
    """events: list of (t, kind, value, variance) with kind in H; t_out: times at which r is returned.
    Returns (r_hat, r_std) at t_out. tau(0) = 0 exactly (the preamble is the time reference)."""
    times = np.unique(np.concatenate([[0.0], [e[0] for e in events], np.asarray(t_out, float)]))
    meas = {}
    for t, kind, v, var in events:
        meas.setdefault(t, []).append((kind, v, var))
    n = len(times)
    xp, Pp, xf, Pf = np.zeros((n, 3)), np.zeros((n, 3, 3)), np.zeros((n, 3)), np.zeros((n, 3, 3))
    x = np.zeros(3)
    P = np.diag([0.0, sigma_r0 ** 2, sigma_dr0 ** 2])
    for i, t in enumerate(times):
        if i > 0:
            dt = t - times[i - 1]
            Fm = _F(dt)
            x, P = Fm @ x, Fm @ P @ Fm.T + _Q(dt, q)
        xp[i], Pp[i] = x, P
        for kind, v, var in meas.get(t, []):
            h = H[kind]
            s = h @ P @ h + var
            k = P @ h / s
            x = x + k * (v - h @ x)
            P = P - np.outer(k, h @ P)
        xf[i], Pf[i] = x, P
    xs, Ps = xf.copy(), Pf.copy()
    for i in range(n - 2, -1, -1):
        dt = times[i + 1] - times[i]
        Fm = _F(dt)
        C = Pf[i] @ Fm.T @ np.linalg.pinv(Pp[i + 1])
        xs[i] = xf[i] + C @ (xs[i + 1] - xp[i + 1])
        Ps[i] = Pf[i] + C @ (Ps[i + 1] - Pp[i + 1]) @ C.T
    idx = np.searchsorted(times, t_out)
    return xs[idx, 1], np.sqrt(np.maximum(Ps[idx, 1, 1], 0.0))
