"""T3 bound (bcrb.py): the MMSE smoother reaches it on the linear-Gaussian model; measurements only reduce it."""
import numpy as np

from uwofdm import bcrb, frame


def _K(t, sigma=3e-4, f=0.12):
    d = t[:, None] - t[None, :]
    return sigma ** 2 * np.cos(2 * np.pi * f * d) * np.exp(-(d / 8.0) ** 2)  # narrow-band wave-like covariance


def test_mmse_smoother_reaches_the_bound():
    F = frame.Frame()
    t = bcrb.grid(F, 0.02)
    K = _K(t)
    H, R = bcrb.measurements(F, t, 10.0, ("cfo", "pilot", "timing"))
    P = bcrb.bound(K, H, R)
    rng = np.random.default_rng(0)
    L = np.linalg.cholesky(K + 1e-18 * np.eye(len(t)))
    G = K @ H.T @ np.linalg.inv(H @ K @ H.T + R)
    E = []
    for _ in range(4000):
        r = L @ rng.standard_normal(len(t))
        z = H @ r + np.sqrt(np.diag(R)) * rng.standard_normal(len(R))
        E.append(r - G @ z)
    emp = np.cov(np.array(E).T)
    assert abs(np.trace(emp) / np.trace(P) - 1) < 0.08


def test_more_information_lowers_the_bound():
    F = frame.Frame()
    t = bcrb.grid(F, 0.02)
    K = _K(t)
    sir = lambda kinds, snr, Kp=K: bcrb.sir_db(bcrb.bound(Kp, *bcrb.measurements(F, t, snr, kinds)), F, t)
    assert sir(("cfo", "timing", "pilot"), 20) >= sir(("cfo", "timing"), 20) - 1e-9
    assert sir(("cfo", "timing"), 30) > sir(("cfo", "timing"), 10)
    assert sir(("cfo", "timing"), 20, K / 100) > sir(("cfo", "timing"), 20)  # tighter prior (sensors)
