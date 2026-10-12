"""OFDM, time warp and channel: round trips, analytic Doppler of a tone, inverse warps, residual interference."""
import numpy as np

from uwofdm import channel, waveform
from uwofdm.waveform import OFDM, unwarp, warp


def db(x):
    return 10 * np.log10(x)


def test_ofdm_round_trip():
    o = OFDM()
    rng = np.random.default_rng(0)
    X = o.random_symbols(3, rng, bits=4)
    assert np.allclose(o.demodulate(o.modulate(X), 3), X)


def test_warp_identity_and_tone_doppler():
    fs, fc, f0, eps = 32000.0, 16000.0, 1000.0, 3e-4
    n = np.arange(32000)
    s = np.exp(2j * np.pi * f0 * n / fs)
    assert np.allclose(warp(s, fs, np.zeros(len(n)), fc)[100:-100], s[100:-100])
    y = warp(s, fs, np.full(len(n), eps), fc)[200:-200]
    f_meas = np.mean(np.diff(np.unwrap(np.angle(y)))) * fs / (2 * np.pi)
    assert abs(f_meas - (f0 + eps * (f0 + fc))) < 0.01  # passband frequency scaled by (1 + eps)


def _err_db(a, b, m=400):
    return db(np.mean(np.abs(a[m:-m] - b[m:-m]) ** 2) / np.mean(np.abs(b[m:-m]) ** 2))


def test_inverse_warps():
    """Receiver compensation unwarp(warp(s)) and transmit-side pre-compensation warp(unwarp(s)) both return s."""
    o = OFDM()
    rng = np.random.default_rng(1)
    s = o.modulate(o.random_symbols(4, rng))
    t = np.arange(len(s)) / o.fs
    eps = 3e-4 * np.sin(2 * np.pi * t / 8.0 + 0.3) + 1e-4  # swell-like trajectory
    assert _err_db(unwarp(warp(s, o.fs, eps, o.fc), o.fs, eps, o.fc), s) < -50
    assert _err_db(warp(unwarp(s, o.fs, eps, o.fc), o.fs, eps, o.fc), s) < -50


def test_residual_interference():
    """Uncompensated Doppler destroys the subcarriers; exact compensation restores them; a residual error in between."""
    o = OFDM()
    rng = np.random.default_rng(2)
    X = o.random_symbols(4, rng)
    s = o.modulate(X)
    eps = np.full(len(s), 2e-4)  # 0.3 m/s
    y = warp(s, o.fs, eps, o.fc)
    sinr = lambda Y: db(np.mean(np.abs(X[:, o.active]) ** 2) / np.mean(np.abs(Y[:, o.active] - X[:, o.active]) ** 2))
    raw = sinr(o.demodulate(y, 4))
    exact = sinr(o.demodulate(unwarp(y, o.fs, eps, o.fc), 4))
    resid = sinr(o.demodulate(unwarp(y, o.fs, eps - 2e-6, o.fc), 4))
    assert raw < 0 and exact > 40 and raw < resid < exact


def test_channel_single_path_is_warp():
    o = OFDM()
    rng = np.random.default_rng(3)
    s = o.modulate(o.random_symbols(2, rng))
    eps = np.full(len(s), 1e-4)
    y = channel.apply(s, o.fs, o.fc, eps, [channel.Path(0.0, 1.0)])
    assert np.allclose(y, warp(s, o.fs, eps, o.fc))
    paths = channel.exponential_paths(8, 0.015, rng, delta_eps_std=1e-5)
    assert np.isclose(sum(abs(p.gain) ** 2 for p in paths), 1) and paths[0].delay == 0 and paths[0].delta_eps == 0


def test_precompensation_needs_room_at_the_end():
    """Transmit-side pre-compensation stretches the packet: with zero padding the whole packet survives."""
    o = OFDM()
    rng = np.random.default_rng(7)
    X = o.random_symbols(8, rng)
    s = np.pad(o.modulate(X), (0, 4096))
    for e0 in (1e-3, -1e-3):
        eps = np.full(len(s), e0)
        y = warp(unwarp(s, o.fs, eps, o.fc), o.fs, eps, o.fc)
        Y = o.demodulate(y, 8)
        err = np.mean(np.abs(Y[:, o.active] - X[:, o.active]) ** 2) / np.mean(np.abs(X[:, o.active]) ** 2)
        assert db(err) < -45


def test_pilot_metric_and_multipath():
    """Pilot-based channel estimation equalises a static multipath channel; per-path Doppler degrades it."""
    from uwofdm import experiment
    o = OFDM()
    rng = np.random.default_rng(8)
    X = o.random_symbols(4, rng)
    s = np.pad(o.modulate(X), (0, 4096))
    paths = [channel.Path(d, g) for d, g in ((0.0, 0.8), (0.004, 0.5j), (0.011, -0.3))]
    y = channel.apply(s, o.fs, o.fc, 0.0, paths)
    sig, err = experiment.pilot_powers(o.demodulate(y, 4), X, o)
    assert db(err / sig) < -40  # regularised delay-domain estimator: about -42 dB on a static channel
    paths[1].delta_eps = 5e-5
    sig2, err2 = experiment.pilot_powers(o.demodulate(channel.apply(s, o.fs, o.fc, 0.0, paths), 4), X, o)
    assert err2 / sig2 > 3 * err / sig


def test_pilot_metric_tolerates_early_arrival():
    """A small negative timing offset (signal early, from residual Doppler drift) must not break the estimator."""
    from uwofdm import experiment
    o = OFDM()
    rng = np.random.default_rng(9)
    X = o.random_symbols(4, rng)
    s = np.pad(o.modulate(X), (0, 4096))
    y = np.concatenate([s[30:], np.zeros(30, complex)])  # 30 samples (0.9 ms) early
    sig, err = experiment.pilot_powers(o.demodulate(y, 4, backoff=int(experiment.BACKOFF * o.fs)), X, o)
    assert db(err / sig) < -35  # back-off keeps the next symbol out; the estimator ceiling is about -40 dB


def test_pilot_estimator_does_not_amplify_noise():
    from uwofdm import experiment
    _, _, M = experiment._estimator(OFDM())
    assert db(np.mean(np.sum(np.abs(M) ** 2, axis=1))) < 1.0


def test_genie_matches_ici_closed_form():
    """The genie measurement must follow (pi fc e T)^2 / 3 also at strong interference (0.2-subcarrier offset): the
    adjacent-product phase-ramp estimate was biased there by up to 20 dB."""
    from uwofdm import theory, waveform
    from uwofdm.experiment import PAD, genie_powers
    o = waveform.OFDM()
    X = o.random_symbols(4, np.random.default_rng(1))
    s = np.pad(o.modulate(X), (0, PAD))
    for e in (5e-6, 3.6e-5, 1e-4):
        sig, err = genie_powers(o.demodulate(waveform.warp(s, o.fs, np.full(len(s), e), o.fc), 4), X, o.active)
        assert abs(10 * np.log10(sig / err) - theory.ici_sir_db(e ** 2, o.fc, o.n_fft / o.fs)) < 1.0
    sig, err = genie_powers(o.demodulate(np.roll(s, 7), 4), X, o.active)  # a timing offset is removed
    assert 10 * np.log10(sig / err) > 60


def test_warp_delayed_matches_integer_shift_and_interpolates():
    """waveform.warp_delayed: equals warp of a zero-padded signal for an integer delay; a half-sample delay lands
    between the two integer neighbours (band-limited)."""
    from uwofdm import waveform
    rng = np.random.default_rng(0)
    o = OFDM()
    s = o.modulate(o.random_symbols(2, rng))
    eps = np.full(len(s), 2e-4)
    a = waveform.warp_delayed(s, o.fs, eps, o.fc, 7 / o.fs)
    b = waveform.warp(np.concatenate([np.zeros(7, complex), s])[:len(s)], o.fs, eps, o.fc)
    assert np.max(np.abs(a[64:-64] - b[64:-64])) < 1e-6
    h = waveform.warp_delayed(s, o.fs, eps, o.fc, 7.5 / o.fs)
    c = waveform.warp_delayed(s, o.fs, eps, o.fc, 8 / o.fs)
    assert np.mean(np.abs(h - 0.5 * (a + c)) ** 2) < 0.05 * np.mean(np.abs(a) ** 2)
