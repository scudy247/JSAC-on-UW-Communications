"""Kalman fusion: smoother on synthetic measurements; fused receiver vs per-symbol corrections at low and high SNR."""
import numpy as np

from uwofdm import chain, frame, fusion, waveform


def test_smoother_tracks_line_and_weights_measurements():
    rng = np.random.default_rng(0)
    t = np.linspace(0.1, 1.2, 9)
    r_true = 1e-4 + 5e-5 * t
    ev = [(ti, "r", ri + 2e-5 * rng.standard_normal(), (2e-5) ** 2) for ti, ri in zip(t, r_true)]
    ev.append((1.3, "tau", 1e-4 * 1.3 + 2.5e-5 * 1.3 ** 2, (1e-6) ** 2))
    r_hat, r_std = fusion.smooth(ev, t, 3e-4, 3e-4, (1e-5) ** 2)
    assert np.sqrt(np.mean((r_hat - r_true) ** 2)) < 1e-5  # better than a single measurement (2e-5)
    # no measurements: stays at the prior (0) with the prior's uncertainty
    r0, s0 = fusion.smooth([], np.array([0.5]), 3e-4, 0.0, 0.0)
    assert r0[0] == 0 and abs(s0[0] - 3e-4) < 1e-12


def _sir(Y, X, F):
    from uwofdm.experiment import genie_powers
    return 10 * np.log10(np.divide(*genie_powers(Y, X, F.used)))


def test_fused_receiver_follows_within_packet_residual():
    """Prior off by a residual that varies within the packet (1.5e-4 + 2e-4 sin(0.6 Hz)): at 35 dB per-subcarrier SNR
    the fused trajectory beats constant scale + independent per-symbol CFO by several dB (it also removes the
    intra-symbol ramp); at 10 dB both are noise-limited and fusion is not worse."""
    rng = np.random.default_rng(1)
    F = frame.Frame()
    o = F.o
    X = o.random_symbols(F.n_data, rng)
    X[:, F.nulls] = 0
    s = F.build(o.random_symbols(F.n_header, rng), X)
    lead = 1500
    x = np.concatenate([np.zeros(lead, complex), s, np.zeros(8000, complex)])
    t = np.arange(len(x)) / o.fs
    prior_part = 4e-4 * np.sin(2 * np.pi * 0.2 * t)
    resid = 1.5e-4 + 2e-4 * np.sin(2 * np.pi * 0.6 * t + 1)
    y0 = waveform.warp(x, o.fs, prior_part + resid, o.fc)
    prior = prior_part[lead:]
    out = {}
    for snr in (10, 35):
        sig = np.sqrt(4.0 / 10 ** (snr / 10) / 2)
        y = y0 + sig * (rng.standard_normal(len(y0)) + 1j * rng.standard_normal(len(y0)))
        out[snr] = (_sir(chain.receive_aided(y, F, prior, (0, lead + 1500))[0], X, F),
                    _sir(chain.receive_fused(y, F, prior, (0, lead + 1500))[0], X, F))
    assert out[35][1] > out[35][0] + 5
    assert out[10][1] > out[10][0] - 0.5


def test_sic_cancels_second_path_with_different_doppler():
    """Two equal paths 10 ms apart, the second with an extra Doppler of 1.5e-4 (heave on a bottom bounce): one
    resampling along path 0 leaves the second path's inter-carrier interference (SINR ~7.5 dB, which the rate-1/2 code
    still decodes since the sparse channel estimate of 12 Oct 2026; with the old LS estimate, 5.5 dB, it did not);
    decoder-aided cancellation with the known per-path Doppler removes it (~28 dB) and decodes."""
    from uwofdm import channel, receiver
    rng = np.random.default_rng(7)
    F = frame.Frame()
    o = F.o
    info = rng.integers(0, 2, receiver.n_info_bits(F))
    X = receiver.build_data(info, F)
    s = F.build(o.random_symbols(F.n_header, rng), X)
    lead, extra = 1500, int(0.09 * o.fs)
    x = np.concatenate([np.zeros(lead, complex), s, np.zeros(9000, complex)])
    n = len(x) + extra
    t = np.arange(n) / o.fs
    bulk = 3e-4 * np.sin(2 * np.pi * 0.1 * t)
    d2 = 1.5e-4 + 3e-5 * np.sin(2 * np.pi * 0.2 * t)
    paths = [channel.Path(0.0, 0.75, 0.0), channel.Path(0.010, 0.66j, d2)]
    y = channel.apply(x, o.fs, o.fc, bulk, paths, extra=extra)
    y = y + np.sqrt(4e-3 / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    trs = [bulk[lead:], bulk[lead:] + d2[lead:]]
    Y1 = chain.receive_branches(y, F, [trs[0]], (0, lead + 1500))[0]
    s1 = receiver.sinr(Y1, X, F)
    branches, _ = chain.receive_sic(y, F, trs, [0.0, 0.010], (0, lead + 1500), iters=4)
    s2 = receiver.sinr_combined(branches, X, F)
    assert 10 * np.log10(s2[0] / s2[1]) > 10 * np.log10(s1[0] / s1[1]) + 15
    assert np.array_equal(receiver.decode_combined(branches, F), info)
