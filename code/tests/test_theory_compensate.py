"""Closed-form kernels against direct computation; trajectory fitting."""
import numpy as np

from uwofdm import compensate, theory


def _projection_residual(f, T, order, n=4001):
    """Direct numerical residual of the best order-N polynomial fit (least squares over T) to exp(j 2 pi f t)."""
    t = np.linspace(-T / 2, T / 2, n)
    z = np.exp(2j * np.pi * f * t)
    A = np.vander(t / T, order + 1)
    coef = np.linalg.lstsq(A, z, rcond=None)[0]
    return np.mean(np.abs(z - A @ coef) ** 2)


def test_kernel_poly_matches_projection():
    for order in (0, 1, 2, 3):
        for f in (0.01, 0.1, 0.3, 1.0):
            assert abs(theory.kernel_poly(f, 2.0, order) - _projection_residual(f, 2.0, order)) < 2e-3
    assert np.isclose(theory.kernel_poly(0.2, 1.5, 0), 1 - np.sinc(0.2 * 1.5) ** 2)


def test_stale_kernel_and_variance():
    rng = np.random.default_rng(0)
    fs, n = 10.0, 200000
    v = np.convolve(rng.standard_normal(n), np.ones(20) / 20, "same")  # low-pass noise
    f, S = theory.spectrum(v, fs)
    assert abs(np.sum(S) * (f[1] - f[0]) / v.var() - 1) < 0.05
    H = 7  # samples
    emp = np.mean((v[H:] - v[:-H]) ** 2)
    pred = np.sum(S * theory.kernel_stale(f, H / fs)) * (f[1] - f[0])
    assert abs(pred / emp - 1) < 0.05


def test_fit_trajectory():
    t = np.linspace(0, 2, 9)
    eps = 1e-4 + 2e-5 * t - 3e-6 * t ** 2
    g = np.linspace(0, 2, 100)
    assert np.allclose(compensate.fit_trajectory(t, eps, g, 2), 1e-4 + 2e-5 * g - 3e-6 * g ** 2)
    assert np.allclose(compensate.fit_trajectory(t, eps, g, 0), eps.mean())
    pc = compensate.fit_trajectory(t, eps, g, "piece")
    assert np.allclose(pc[0], eps[0]) and np.allclose(pc[-1], eps[-1])


def test_gauss_markov_spectrum_integrates_to_variance():
    f = np.linspace(0, 50, 500001)
    S = theory.gauss_markov_spectrum(f, 0.03, 4.0)
    assert abs(np.sum(S) * (f[1] - f[0]) / 0.03 ** 2 - 1) < 0.01


def test_extrapolation_kernel():
    rng = np.random.default_rng(5)
    fs, n = 50.0, 400000
    v = np.convolve(rng.standard_normal(n), np.hanning(101), "same")
    a = np.gradient(v, 1 / fs)
    L = 3  # samples
    emp = np.mean((v[L:] - v[:-L] - (L / fs) * a[:-L]) ** 2)
    f, S = theory.spectrum(v, fs)
    pred = np.sum(S * theory.kernel_extrapolate(f, L / fs)) * (f[1] - f[0])
    assert abs(pred / emp - 1) < 0.1


def test_kernel_samples():
    """Many samples -> the projection kernel; two samples -> the line through two points (direct computation)."""
    T = 2.0
    many = (np.arange(400) + 0.5) * T / 400
    for f in (0.05, 0.3):
        assert abs(theory.kernel_samples(f, many, T, 1)[0] / theory.kernel_poly(f, T, 1) - 1) < 0.02
    f, s = 0.2, np.array([0.1, 1.9])
    g = (np.arange(128) + 0.5) * T / 128
    z = np.exp(2j * np.pi * f * s)
    line = z[0] + (z[1] - z[0]) * (g - s[0]) / (s[1] - s[0])
    assert np.isclose(theory.kernel_samples(f, s, T, 1)[0], np.mean(np.abs(np.exp(2j * np.pi * f * g) - line) ** 2))
    assert np.isclose(theory.noise_samples(1.0, s, T, 0), 0.5)
