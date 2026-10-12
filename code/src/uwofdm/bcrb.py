"""
T3: Bayesian Cramer-Rao bound on the Doppler trajectory over a packet, with and without the sensor prior (PLAN:
Phase 5, theory).

Unknown: the residual Doppler r(t) over the frame on a grid t (default 10 ms). Prior: r ~ N(0, K), K estimated from
real motion: signal-only, K = covariance of the link Doppler itself (nothing is known inside the packet); sensor-aided,
K = covariance of what the sensor prior (side information + own IMU) misses. Measurements z = H r + n, n ~ N(0, R),
all linear in r:
  'cfo'     per FFT window, the mean of r over the window (null-subcarrier offset): var = cfo_std_0db^2 / snr / fc^2
  'pilot'   per pair of consecutive symbols, the mean of r between their centres (carrier-phase rotation of the
            pilots): var = 2 / (n_pilots snr) / (2 pi fc T_sym)^2
  'timing'  postamble arrival relative to the preamble: integral of r from 0 to the postamble (chirp delay):
            var = 1 / (8 pi^2 B_rms^2 E/N0), B_rms = sweep B / sqrt(12), E/N0 = n_chirp samples x snr / oversampling
(snr = per-subcarrier SNR, linear). For this linear-Gaussian model the bound is the error covariance of the MMSE
smoother, P = (K^-1 + H^T R^-1 H)^-1, reached by the Kalman / RTS smoother with the right model. Interference: the
residual phase theta = 2 pi fc int (r - r_hat) dt, variance within each data FFT window (window mean removed, as in
sideinfo.interference), averaged over windows; SIR = 1 / that. It bounds estimation from Doppler measurements (not
from the raw waveform) and describes the mean, not the tail of bad packets.
"""
import numpy as np

CFO_STD_0DB = 0.5  # [Hz] null-subcarrier offset error at 0 dB per-subcarrier SNR (measured, chain.FUSION)


def grid(F, dt=0.01):
    return np.arange(0.0, F.length / F.o.fs + dt / 2, dt)


def _avg_row(t, a, b):
    """Row of H giving the mean of r over [a, b] on the grid (trapezoidal weights)."""
    w = np.zeros(len(t))
    for i in range(len(t) - 1):
        lo, hi = max(a, t[i]), min(b, t[i + 1])
        if hi > lo:
            f0 = (lo - t[i]) / (t[i + 1] - t[i])
            f1 = (hi - t[i]) / (t[i + 1] - t[i])
            w[i] += (hi - lo) * (1 - 0.5 * (f0 + f1))
            w[i + 1] += (hi - lo) * 0.5 * (f0 + f1)
    return w / max(b - a, 1e-12)


def cumulative(t):
    """Matrix C with (C r)_i = int_0^{t_i} r dt (trapezoidal)."""
    n = len(t)
    C = np.zeros((n, n))
    for i in range(1, n):
        C[i] = C[i - 1]
        d = t[i] - t[i - 1]
        C[i, i - 1] += d / 2
        C[i, i] += d / 2
    return C


def windows(F, t):
    """Grid indices of each data FFT window."""
    o = F.o
    out = []
    for k in range(F.n_data):
        a = (F.data_start + k * o.n_symbol + o.n_cp) / o.fs
        out.append(np.where((t >= a) & (t <= a + o.n_fft / o.fs))[0])
    return out


def measurements(F, t, snr_db, kinds=("cfo", "timing"), n_pilots=None):
    """(H, R) for the requested measurement kinds at per-subcarrier SNR snr_db."""
    o = F.o
    snr = 10 ** (snr_db / 10)
    t_sym = o.n_symbol / o.fs
    rows, var = [], []
    starts = [F.header_start] + [F.data_start + k * o.n_symbol for k in range(F.n_data)]
    if "cfo" in kinds:
        for s in starts:
            a = (s + o.n_cp) / o.fs
            rows.append(_avg_row(t, a, a + o.n_fft / o.fs))
            var.append(CFO_STD_0DB ** 2 / snr / o.fc ** 2)
    if "pilot" in kinds:
        n_p = n_pilots or int(np.sum(o.pilots() & o.active))
        centres = [(F.data_start + k * o.n_symbol + o.n_cp) / o.fs + o.n_fft / o.fs / 2 for k in range(F.n_data)]
        for c0, c1 in zip(centres[:-1], centres[1:]):
            rows.append(_avg_row(t, c0, c1))
            var.append(2 / (n_p * snr) / (2 * np.pi * o.fc * t_sym) ** 2)
    if "timing" in kinds:
        tp = F.post_start / o.fs
        rows.append(_avg_row(t, 0.0, tp) * tp)
        b_rms = F.sweep * o.bandwidth / np.sqrt(12)
        en0 = F.n_chirp * snr / o.oversampling
        var.append(1 / (8 * np.pi ** 2 * b_rms ** 2 * en0))
    return np.array(rows), np.diag(var)


def bound(K, H, R):
    """Bayesian CRB / MMSE error covariance of r given the prior covariance K and z = H r + n:
    (K^-1 + H^T R^-1 H)^-1 = K - K H^T (H K H^T + R)^-1 H K (this form needs no inverse of K, which is nearly singular
    for an empirical covariance of smooth trajectories)."""
    KH = K @ H.T
    P = K - KH @ np.linalg.solve(H @ KH + R, KH.T)
    return 0.5 * (P + P.T)


def interference(P, F, t):
    """Mean over data FFT windows of the within-window variance of theta = 2 pi fc int e dt, e ~ N(0, P)."""
    C = cumulative(t)
    S = (2 * np.pi * F.o.fc) ** 2 * C @ P @ C.T
    vals = []
    for idx in windows(F, t):
        n = len(idx)
        M = np.eye(n) - np.ones((n, n)) / n
        vals.append(np.trace(M @ S[np.ix_(idx, idx)] @ M) / n)
    return float(np.mean(vals))


def sir_db(P, F, t):
    return float(-10 * np.log10(max(interference(P, F, t), 1e-30)))
