"""Partial-FFT demodulation (pfft.py, chain.receive_pfft): the signal-only answer to Doppler varying within a symbol."""
import numpy as np

from uwofdm import chain, frame, pfft, waveform
from uwofdm.experiment import genie_powers


def test_segments_sum_to_the_ordinary_fft():
    F = frame.Frame()
    o = F.o
    seg = np.random.default_rng(0).standard_normal(o.n_fft) + 0j
    Yp = pfft.segment_ffts(seg, o, 4)
    assert np.allclose(Yp.sum(0), np.fft.fft(seg)[o.bins()] * np.sqrt(o.n_sc) / o.n_fft)


def test_pfft_after_signal_kalman_follows_fast_doppler():
    """Slow 6e-4 wave plus a fast 3e-4 component at 0.8 Hz (strong variation within the 128 ms symbols), 40 dB:
    partial FFT (8 segments) after the signal-only Kalman resampling gains several dB; with no fast part it does not
    lose."""
    F = frame.Frame()
    o = F.o
    sir = lambda Y, X: 10 * np.log10(np.divide(*genie_powers(Y, X, F.used)))
    for amp, gain in ((3e-4, 4.0), (0.0, -0.5)):
        rng = np.random.default_rng(2)
        X = o.random_symbols(F.n_data, rng)
        X[:, F.nulls] = 0
        s = F.build(o.random_symbols(F.n_header, rng), X)
        lead = 1500
        x = np.concatenate([np.zeros(lead, complex), s, np.zeros(8000, complex)])
        t = np.arange(len(x)) / o.fs
        y = waveform.warp(x, o.fs, 6e-4 * np.sin(2 * np.pi * 0.15 * t) + amp * np.sin(2 * np.pi * 0.8 * t + 1), o.fc)
        y = y + np.sqrt(4e-4 / 2) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
        se = (0, lead + 1500)
        base = sir(chain.receive_signal_fused(y, F, se)[0], X)
        assert sir(chain.receive_pfft(y, F, X, se, front="kalman", n_seg=8)[0], X) > base + gain
