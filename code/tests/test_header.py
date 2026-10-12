"""Side-information header: bit packing, CRC, DBPSK round trip, decoding through the full chain under Doppler."""
import numpy as np

from uwofdm import chain, frame, header, sideinfo, waveform


def test_pack_unpack_crc():
    alloc = np.array([0, 4, 6, 3, 0])
    ranges = np.array([1.0, 1.0, 0.5, 0.2, 0.1])
    c = np.array([0.3, -0.42, 0.11, 0.05, 0.02])
    bits = header.pack(c, alloc, ranges)
    assert len(bits) == alloc.sum() + 8
    payload, ok = header.check(bits)
    assert ok
    q = header.unpack(payload, alloc, ranges)
    ref = sideinfo.quantise_alloc(c, alloc, ranges)
    assert np.allclose(q, ref)
    bad = bits.copy()
    bad[3] ^= 1
    assert not header.check(bad)[1]


def test_dbpsk_round_trip_with_noise():
    rng = np.random.default_rng(0)
    o = waveform.OFDM()
    bits = rng.integers(0, 2, 40)
    X = header.modulate(bits, o)
    H = np.exp(1j * (0.7 + 0.002 * np.arange(o.n_sc)))  # smooth channel phase
    Y = X[0] * H + (rng.standard_normal(o.n_sc) + 1j * rng.standard_normal(o.n_sc)) * np.sqrt(0.5 / 10 ** 0.0)
    got, soft = header.demodulate(Y, len(bits), o)
    assert np.array_equal(got, bits)  # 0 dB per subcarrier, ~24 repetitions per bit


def test_header_through_chain_under_wave_doppler():
    """Transmitter share 8e-4 sin(0.3 Hz) (rough sea), receiver share known by its IMU: the header is read and the aided
    receiver beats the signal-only front end."""
    from uwofdm.experiment import genie_powers
    rng = np.random.default_rng(1)
    F = frame.Frame()
    o = F.o
    T = F.length / o.fs
    lead, tail = 1800, 8000
    m = np.arange(F.length + tail // 2)
    t = m / o.fs
    tx_share = 8e-4 * np.sin(2 * np.pi * 0.3 * t + 0.4)
    rx_share = 6e-4 * np.sin(2 * np.pi * 0.2 * t + 2.0)
    K = 12
    c = sideinfo.project(tx_share[:F.length], t[:F.length], T, K)
    ranges = 1.5 * np.abs(c) + 1e-7
    alloc = np.array([6, 6, 6, 5, 4, 3, 2, 0, 0, 0, 0, 0])
    X = o.random_symbols(F.n_data, rng)
    s = F.build(header.modulate(header.pack(c, alloc, ranges), o, F.used), X)
    x = np.concatenate([np.zeros(lead, complex), s, np.zeros(tail, complex)])
    eps = np.concatenate([np.full(lead, tx_share[0] + rx_share[0]), (tx_share + rx_share)[:len(x) - lead]])
    eps = np.pad(eps, (0, len(x) - len(eps)), mode="edge")
    y = waveform.warp(x, o.fs, eps, o.fc)
    y += 0.005 * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
    basis_fn = lambda cc, mm: sideinfo.reconstruct(cc, np.minimum(mm / o.fs, T), T)
    Y, info = chain.receive_with_header(y, F, rx_share, alloc, ranges, basis_fn, search=(0, lead + 1500))
    assert info["header_ok"] and not info["fallback"]
    Y0, _ = chain.receive_signal_only(y, F, search=(0, lead + 1500))
    sir = lambda YY: 10 * np.log10(np.divide(*genie_powers(YY, X, F.used)))
    assert sir(Y) > sir(Y0) + 5
