"""Long-integration waveform receiver (longint.py): exact compensation gives full efficiency; at high CNR the
signal-only receiver recovers a sinusoidal Doppler; without compensation the efficiency collapses; the frequency search
stays inside its window."""
import numpy as np

from uwofdm import chain, longint


def test_long_integration_receiver():
    fs, chip, fc, T = 4000.0, 1000.0, 16e3, 8.0
    rng = np.random.default_rng(0)
    n = int(T * fs)
    t = np.arange(n) / fs
    eps = 2e-4 + 3e-4 * np.sin(2 * np.pi * 0.12 * t)
    c = longint.dsss(int(T * chip), chip, fs, rng)
    y0, y = longint.received(c, fs, fc, eps, 45.0, rng)
    from uwofdm.waveform import unwarp
    assert longint.efficiency(unwarp(y0, fs, eps, fc), c, fs, 8.0) > 0.98
    assert longint.efficiency(y0, c, fs, 8.0) < 0.1
    eff, traj = longint.receive(y, y0, c, fs, fc, np.full(n, np.mean(eps)), 45.0, 0.5, dict(chain.SIGNAL_FUSION))
    assert eff(8.0) > 0.8


def test_search_window():
    fs, t_seg = 4000.0, 0.5
    rng = np.random.default_rng(1)
    n = int(4 * fs)
    c = longint.dsss(4000, 1000.0, fs, rng)
    t = np.arange(n) / fs
    z = c * np.exp(2j * np.pi * 7.3 * t)  # a clean tone at 7.3 Hz
    assert all(abs(f - 7.3) < 0.1 for _, f, _ in longint.segment_freqs(z, c, fs, t_seg, band_hz=20.0))
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 10
    assert all(abs(f) <= 20.0 + 1e-9 for _, f, _ in longint.segment_freqs(noise, c, fs, t_seg, band_hz=20.0))
