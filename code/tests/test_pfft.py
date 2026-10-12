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


def test_ici_aware_equaliser_removes_most_intra_symbol_ici():
    """Fast 3e-4 Doppler component at 40 dB: the ICI-aware equaliser (ici.py, 8 segments) gains several dB over the
    signal-only Kalman alone (it lands about 1 dB below partial FFT on these synthetic cases, see RESULTS.md)."""
    F = frame.Frame()
    o = F.o
    rng = np.random.default_rng(2)
    X = o.random_symbols(F.n_data, rng)
    X[:, F.nulls] = 0
    s = F.build(o.random_symbols(F.n_header, rng), X)
    lead = 1500
    x = np.concatenate([np.zeros(lead, complex), s, np.zeros(8000, complex)])
    t = np.arange(len(x)) / o.fs
    y = waveform.warp(x, o.fs, 6e-4 * np.sin(2 * np.pi * 0.15 * t) + 3e-4 * np.sin(2 * np.pi * 0.8 * t + 1), o.fc)
    y = y + np.sqrt(4e-4 / 2) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
    sir = lambda Y: 10 * np.log10(np.divide(*genie_powers(Y, X, F.used)))
    Ys, inf = chain.receive_signal_fused(y, F, (0, lead + 1500))
    assert sir(chain._demod_ici(inf["z"], F, X, 8, 2)[0]) > sir(Ys) + 5


def test_pilot_tracking_closed_loop_on_long_packet():
    """4 s packet, slow 6e-4 wave + 6e-4 at 0.3 Hz, 30 dB: closed-loop pilot tracking (chain.receive_pilot_track)
    is not worse than the signal-only Kalman (an open-loop version collapsed to ~15 dB: inter-symbol timing ramp and
    ICI-blurred pilots)."""
    F = frame.Frame(n_data=27)
    o = F.o
    rng = np.random.default_rng(2)
    X = o.random_symbols(F.n_data, rng)
    X[:, F.nulls] = 0
    s = F.build(o.random_symbols(F.n_header, rng), X)
    lead = 1500
    x = np.concatenate([np.zeros(lead, complex), s, np.zeros(8000, complex)])
    t = np.arange(len(x)) / o.fs
    y = waveform.warp(x, o.fs, 6e-4 * np.sin(2 * np.pi * 0.15 * t) + 6e-4 * np.sin(2 * np.pi * 0.3 * t + 1), o.fc)
    y = y + np.sqrt(4e-3 / 2) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
    sir = lambda Y: 10 * np.log10(np.divide(*genie_powers(Y, X, F.used)))
    se = (0, lead + 1500)
    pt = sir(chain.receive_pilot_track(y, F, X, se)[0])
    assert pt > 28.0 and pt > sir(chain.receive_signal_fused(y, F, se)[0]) - 0.5
