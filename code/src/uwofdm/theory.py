"""
Closed forms shared by R1 (compensation inside the packet) and R2 (staleness and placement). Proposal of 9 Oct 2026,
to be checked against the literature in literature/g1_intrapacket_compensation_check.md.

Both results ask how well limited information represents the velocity v(t), a stationary process with one-sided
spectrum S_v(f):
  R2, across time:   an estimate H seconds old         E[(v(t+H) - v(t))^2] = int S_v(f) 2 (1 - cos 2 pi f H) df
  R1, within packet: the best order-N polynomial over a packet of length T leaves, per frequency, the fraction
                     K_N(f, T) = 1 - sum_{k=0..N} (2k+1) j_k(pi f T)^2
                     (j_k: spherical Bessel functions; projection of exp(j 2 pi f t) on the Legendre polynomials of the
                     interval; N = 0 gives 1 - sinc^2).
Estimate noise: fitting order N to M equally spaced estimates of eps with noise sigma adds on average sigma^2 (N+1)/M
over the packet.
Interference: a residual Doppler scale e, roughly constant over one OFDM symbol of length T_s, is a frequency offset
fc * e; for small offsets the interference between subcarriers has power (pi fc e T_s)^2 / 3 relative to the signal.
"""
import numpy as np
from scipy.special import spherical_jn

C = 1500.0


def kernel_poly(f, T, order):
    """Fraction of a velocity component at frequency f left after the best order-`order` polynomial fit over T."""
    x = np.pi * np.asarray(f, float) * T
    kept = sum((2 * k + 1) * spherical_jn(k, x) ** 2 for k in range(order + 1))
    return np.clip(1.0 - kept, 0.0, 1.0)


def kernel_stale(f, H):
    """Error of using a value H seconds old, per unit spectral density at frequency f."""
    return 2.0 * (1.0 - np.cos(2 * np.pi * np.asarray(f, float) * H))


def kernel_extrapolate(f, L):
    """Error of extrapolating across a latency L with value and slope: v(t) - [v(t - L) + L v'(t - L)], per unit
    spectral density: |exp(j w L) - 1 - j w L|^2."""
    w = 2 * np.pi * np.asarray(f, float) * L
    return np.abs(np.exp(1j * w) - 1 - 1j * w) ** 2


def kernel_stale_packet(f, H, T, n=64):
    """Staleness kernel averaged over a packet: a value taken H seconds before the packet starts, used over [0, T]."""
    h = H + (np.arange(n) + 0.5) * T / n
    return np.mean([kernel_stale(f, x) for x in h], axis=0)


def gauss_markov_spectrum(f, sigma, tau):
    """One-sided spectral density of a first-order Gauss-Markov process (std sigma, correlation time tau)."""
    return 4 * sigma ** 2 * tau / (1 + (2 * np.pi * np.asarray(f, float) * tau) ** 2)


def spectrum(v, fs, nseg=None):
    """One-sided spectral density of a velocity record [ (m/s)^2 / Hz ] (Welch, Hann, 50% overlap); returns f, S
    with sum(S) * df = var(v)."""
    from scipy.signal import welch
    n = len(v)
    nper = nseg or min(n, int(2 ** np.floor(np.log2(max(256, n // 8)))))
    f, S = welch(np.asarray(v, float) - np.mean(v), fs=fs, nperseg=nper)
    return f, S


def residual_eps_var(f, S, kernel, c=C):
    """Residual Doppler-scale variance from velocity spectrum S(f) and a kernel evaluated on f."""
    df = f[1] - f[0]
    return float(np.sum(S * kernel) * df) / c ** 2


def noise_term(sigma_eps, order, m):
    """Average variance over the packet of an order-N least-squares fit to m equally spaced noisy estimates."""
    return sigma_eps ** 2 * (order + 1) / m


def ici_sir_db(e_var, fc, t_sym):
    """Signal-to-interference ratio between subcarriers for a residual Doppler scale of variance e_var [dB]."""
    return float(10 * np.log10(3.0 / ((np.pi * fc * t_sym) ** 2 * e_var)))


def _sample_fit(t_samples, T, order, n):
    """Weights mapping the samples to the order-N least-squares fit, evaluated on n points across [0, T]."""
    g = (np.arange(n) + 0.5) * T / n
    s = np.asarray(t_samples, float)
    scale = max(T, 1e-12)
    Bs, Bg = np.vander(s / scale, order + 1), np.vander(g / scale, order + 1)
    return g, s, Bg @ np.linalg.pinv(Bs)


def kernel_samples(f, t_samples, T, order, n=128):
    """Like kernel_poly, for the fit the receiver actually makes: order-N least squares through estimates taken at
    t_samples (e.g. the first and last symbol), error averaged over the packet [0, T]. With two samples and order 1
    this is the line through two points, which leaves up to 6x (7.8 dB) more curvature error than the best line."""
    f = np.atleast_1d(np.asarray(f, float))
    g, s, W = _sample_fit(t_samples, T, order, n)
    w = 2 * np.pi * f[:, None]
    R = np.exp(1j * w * g[None, :]) - np.exp(1j * w * s[None, :]) @ W.T
    return np.mean(np.abs(R) ** 2, axis=1)


def noise_samples(sigma_eps, t_samples, T, order, n=128):
    """Average variance over the packet of the order-N fit to independent estimates (std sigma_eps) at t_samples."""
    _, _, W = _sample_fit(t_samples, T, order, n)
    return float(sigma_eps ** 2 * np.mean(np.sum(W ** 2, axis=1)))


def kernel_ici(f, t_grid, t_samples, W, windows, fc, chunk=256):
    """Interference between subcarriers per unit spectral density of eps at frequency f, for any linear estimator.

    The residual of a Doppler component exp(j w t) is r(t) = exp(j w t) - sum_s W[t, s] exp(j w t_s) (W: weights from
    the estimates at t_samples to the compensation trajectory on t_grid; W = 0 if nothing is compensated). Its phase
    Theta(t) = 2 pi fc int r dt rotates the subcarriers; the common phase of each FFT window is removed (channel
    estimate), the rest is interference: kernel = mean over windows of the variance of Theta within the window.
    A constant offset gives (pi fc e T)^2 / 3 (ici_sir_db). After per-symbol compensation the residual inside a window
    is a zero-mean ramp, whose quadratic phase gives 5x (7 dB) less than an offset of the same variance: treating every
    residual as an offset made the per-symbol prediction 5-7 dB pessimistic.
    windows: list of (start index, stop index) into t_grid."""
    f = np.atleast_1d(np.asarray(f, float))
    t = np.asarray(t_grid, float)
    dt = np.diff(t)
    out = np.empty(len(f))
    for a in range(0, len(f), chunk):
        w = 2 * np.pi * f[a:a + chunk, None]
        r = np.exp(1j * w * t[None, :])
        if W is not None:
            r = r - np.exp(1j * w * np.asarray(t_samples, float)[None, :]) @ W.T
        th = np.concatenate([np.zeros((len(w), 1)), np.cumsum(0.5 * (r[:, 1:] + r[:, :-1]) * dt, axis=1)], axis=1)
        v = [np.mean(np.abs(th[:, i:j] - th[:, i:j].mean(axis=1, keepdims=True)) ** 2, axis=1) for i, j in windows]
        out[a:a + chunk] = np.mean(v, axis=0)
    return (2 * np.pi * fc) ** 2 * out
