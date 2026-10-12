"""Cho-Kim baseline: constant acceleration is recovered; a swell over a long frame is not."""
import numpy as np

from uwofdm import chain, frame, waveform
from uwofdm.experiment import genie_powers


def _setup(n_data, rng):
    F = frame.Frame(n_data=n_data)
    o = F.o
    X = o.random_symbols(F.n_data, rng)
    X[:, F.nulls] = 0
    s = F.build(o.random_symbols(F.n_header, rng), X)
    lead = 1500
    x = np.concatenate([np.zeros(lead, complex), s, np.zeros(9000, complex)])
    return F, o, X, x, lead, np.arange(len(x)) / o.fs


def test_constant_acceleration_recovered():
    """0.4 m/s^2 (as in their evaluation), 1.3 s frame: acceleration estimate within 10%, high SIR."""
    rng = np.random.default_rng(0)
    F, o, X, x, lead, t = _setup(7, rng)
    acc = 0.4
    eps = (1.0 + acc * (t - t[lead])) / 1500.0
    y = waveform.warp(x, o.fs, eps, o.fc)
    Y, info = chain.receive_cho_kim(y, F, (0, lead + 1500))
    assert abs(info["accel_m_s2"] - acc) < 0.04
    assert 10 * np.log10(np.divide(*genie_powers(Y, X, F.used))) > 30


def test_swell_over_long_frame_is_not_a_line():
    """0.5 m/s swell, 8 s period, 4 s frame: constant acceleration leaves a large residual (our 46%-of-a-subcarrier
    estimate in literature/baseline_cho_kim_2026.md); the per-symbol CFO front end does better."""
    rng = np.random.default_rng(1)
    F, o, X, x, lead, t = _setup(27, rng)
    eps = 0.5 * np.sin(2 * np.pi * t / 8 + 0.3) / 1500.0
    y = waveform.warp(x, o.fs, eps, o.fc)
    sir = lambda Y: 10 * np.log10(np.divide(*genie_powers(Y, X, F.used)))
    s_ck = sir(chain.receive_cho_kim(y, F, (0, lead + 1500))[0])
    s_li = sir(chain.receive_signal_only(y, F, (0, lead + 1500))[0])
    assert s_ck < 20 and s_li > s_ck + 5
