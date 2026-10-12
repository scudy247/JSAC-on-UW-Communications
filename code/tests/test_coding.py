"""Convolutional code, Viterbi, QPSK LLRs."""
import numpy as np

from uwofdm import coding


def test_encode_decode_noiseless_and_noisy():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 500)
    c = coding.encode(bits)
    assert len(c) == 2 * (500 + coding.K - 1)
    llr = 4.0 * (1 - 2 * c)
    assert np.array_equal(coding.viterbi(llr, 500), bits)
    # BPSK at Eb/N0 = 4 dB (Es/N0 = 1 dB): rate-1/2 K=7 gives BER ~ 1e-4; allow a few errors
    esn0 = 10 ** 0.1
    y = (1 - 2 * c) + rng.standard_normal(len(c)) / np.sqrt(2 * esn0)
    dec = coding.viterbi(4 * esn0 * y, 500)
    assert np.sum(dec != bits) <= 3


def test_qpsk_llr_signs_and_interleaver():
    rng = np.random.default_rng(1)
    b = rng.integers(0, 2, 200)
    x = coding.qpsk(b)
    h = np.exp(1j * rng.uniform(0, 2 * np.pi, len(x)))
    llr = coding.qpsk_llr(h * x, h, 0.1)
    assert np.array_equal((llr < 0).astype(int), b)
    p = coding.interleaver(100)
    assert sorted(p) == list(range(100)) and np.array_equal(p, coding.interleaver(100))


def test_coded_packet_through_chain():
    """Clean packet through the signal-only chain decodes without error; at -2 dB per-subcarrier SNR it fails."""
    from uwofdm import chain, frame, receiver, waveform
    rng = np.random.default_rng(2)
    F = frame.Frame()
    o = F.o
    info = rng.integers(0, 2, receiver.n_info_bits(F))
    s = F.build(o.random_symbols(F.n_header, rng), receiver.build_data(info, F))
    lead = 1200
    x = np.concatenate([np.zeros(lead, complex), s, np.zeros(6000, complex)])
    y0 = waveform.warp(x, o.fs, np.full(len(x), 4e-4), o.fc)
    for snr, ok in ((15, True), (-2, False)):
        sig = np.sqrt(4.0 / 10 ** (snr / 10) / 2)
        y = y0 + sig * (rng.standard_normal(len(y0)) + 1j * rng.standard_normal(len(y0)))
        Y, _ = chain.receive_signal_only(y, F, search=(0, lead + 1500))
        assert np.array_equal(receiver.decode(Y, F), info) == ok


def test_qam16_mapping_and_llrs():
    rng = np.random.default_rng(3)
    b = rng.integers(0, 2, 400)
    x = coding.qam16(b)
    assert abs(np.mean(np.abs(x) ** 2) - 1) < 0.1
    h = 0.8 * np.exp(1j * rng.uniform(0, 2 * np.pi, len(x)))
    llr = coding.qam16_llr(h * x, h, 1e-3)
    assert np.array_equal((llr < 0).astype(int), b)


def test_coded_16qam_packet_through_chain():
    from uwofdm import chain, frame, receiver, waveform
    rng = np.random.default_rng(4)
    F = frame.Frame(n_data=7)
    o = F.o
    info = rng.integers(0, 2, receiver.n_info_bits(F, "16qam"))
    s = F.build(o.random_symbols(F.n_header, rng), receiver.build_data(info, F, "16qam"))
    lead = 900
    x = np.concatenate([np.zeros(lead, complex), s, np.zeros(6000, complex)])
    y = waveform.warp(x, o.fs, np.full(len(x), -3e-4), o.fc)
    y = y + np.sqrt(4.0 / 10 ** 2.2 / 2) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))  # 22 dB
    Y, _ = chain.receive_signal_only(y, F, search=(0, lead + 1500))
    assert np.array_equal(receiver.decode(Y, F, "16qam"), info)


def test_sparse_channel_estimate_close_to_genie():
    """SOMP channel estimate (receiver.channel_estimates): on a static 5-path channel (15 ms spread) its error is a
    small fraction of the noise, and rate-1/2 QPSK decodes at 6 dB per-subcarrier SNR (the old per-symbol LS
    estimate had an error equal to the noise: about 4 dB lost)."""
    from uwofdm import channel, frame, receiver
    rng = np.random.default_rng(5)
    F = frame.Frame()
    o = F.o
    data, _ = receiver.masks(F)
    f = (np.arange(o.n_sc) - o.n_sc // 2) * o.df
    nv = 10 ** (-6 / 10)
    ok = 0
    for _ in range(4):
        paths = channel.exponential_paths(5, 0.015, rng)
        Ht = sum(p.gain * np.exp(-2j * np.pi * f * p.delay) for p in paths)
        info = rng.integers(0, 2, receiver.n_info_bits(F))
        X = receiver.build_data(info, F)
        Y = X * Ht + np.sqrt(nv / 2) * (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape))
        H, _ = receiver.channel_estimates(Y, F, method="somp")
        assert np.mean(np.abs(H[:, data] - Ht[data]) ** 2) < 0.2 * nv
        H_ls, _ = receiver.channel_estimates(Y, F, method="ls")
        assert np.mean(np.abs(H_ls[:, data] - Ht[data]) ** 2) > 0.5 * nv
        ok += np.array_equal(receiver.decode(Y, F), info)
    assert ok >= 3
