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


# --------------------------------------------------------------------------- wave-state model (PLAN: fusion.WaveState)
def wave_model(f_p, zeta, sigma_w, sigma_b0=1e-5, q_b=(1e-6) ** 2):
    """Continuous-time model of the residual Doppler as wave motion: state x = [tau, b, w, w'] with r = b + w,
    tau' = r, b a random walk (slow drift, IMU bias; spectral density q_b, initial std sigma_b0) and w a damped
    oscillator w'' + 2 zeta w0 w' + w0^2 w = noise (w0 = 2 pi f_p) whose stationary std is sigma_w (the wave band of
    the motion spectrum: peak f_p, damping zeta ~ half-power bandwidth / (2 f_p)). Returns (A, Qc, P0) for
    smooth_lti."""
    w0 = 2 * np.pi * f_p
    A = np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -w0 ** 2, -2 * zeta * w0]])
    Qc = np.diag([0.0, q_b, 0.0, 4 * zeta * w0 ** 3 * sigma_w ** 2])
    P0 = np.diag([0.0, sigma_b0 ** 2, sigma_w ** 2, (w0 * sigma_w) ** 2])
    return A, Qc, P0


H_WAVE = {"tau": np.array([1.0, 0.0, 0.0, 0.0]), "r": np.array([0.0, 1.0, 1.0, 0.0])}


def _discretise(A, Qc, dt):
    """Exact discretisation (Van Loan): transition F and process-noise covariance Q over dt."""
    from scipy.linalg import expm
    n = len(A)
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n], M[:n, n:], M[n:, n:] = -A, Qc, A.T
    E = expm(M * dt)
    F = E[n:, n:].T
    return F, F @ E[:n, n:]


def smooth_lti(events, t_out, A, Qc, P0, H=H_WAVE):
    """Kalman filter + RTS smoother for a linear time-invariant SDE (e.g. wave_model); events as in smooth. Returns
    (r_hat, r_std) at t_out, r = H['r'] x. tau(0) = 0 exactly (P0[0, 0] = 0)."""
    times = np.unique(np.concatenate([[0.0], [e[0] for e in events], np.asarray(t_out, float)]))
    meas = {}
    for t, kind, v, var in events:
        meas.setdefault(t, []).append((kind, v, var))
    n, m = len(times), len(A)
    cache = {}
    xp, Pp, xf, Pf = np.zeros((n, m)), np.zeros((n, m, m)), np.zeros((n, m)), np.zeros((n, m, m))
    x, P = np.zeros(m), np.array(P0, float)
    for i, t in enumerate(times):
        if i > 0:
            dt = round(t - times[i - 1], 12)
            if dt not in cache:
                cache[dt] = _discretise(A, Qc, dt)
            Fm, Q = cache[dt]
            x, P = Fm @ x, Fm @ P @ Fm.T + Q
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
        Fm, _ = cache[round(times[i + 1] - times[i], 12)]
        C = Pf[i] @ Fm.T @ np.linalg.pinv(Pp[i + 1])
        xs[i] = xf[i] + C @ (xs[i + 1] - xp[i + 1])
        Ps[i] = Pf[i] + C @ (Ps[i + 1] - Pp[i + 1]) @ C.T
    idx = np.searchsorted(times, t_out)
    h = H["r"]
    return xs[idx] @ h, np.sqrt(np.maximum(np.einsum("i,nij,j->n", h, Ps[idx], h), 0.0))


def fit_wave(x, fs, f_min=0.03):
    """Wave-model parameters (f_p, zeta, sigma_w) from a Doppler (or residual) series: spectral peak above f_min,
    half-power bandwidth, standard deviation."""
    from .theory import spectrum
    f, S = spectrum(x, fs)
    k = f > f_min
    f, S = f[k], S[k]
    i = int(np.argmax(S))
    half = S >= S[i] / 2
    lo, hi = i, i
    while lo > 0 and half[lo - 1]:
        lo -= 1
    while hi < len(S) - 1 and half[hi + 1]:
        hi += 1
    bw = max(f[hi] - f[lo], f[1] - f[0])
    return float(f[i]), float(np.clip(bw / (2 * f[i]), 0.05, 1.0)), float(np.std(x))
