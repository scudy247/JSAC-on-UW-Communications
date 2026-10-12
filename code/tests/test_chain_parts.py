"""End-to-end chain, step 1: frame layout, preamble/postamble scale estimate, residual CFO from null subcarriers."""
import numpy as np

from uwofdm import estimators, frame, waveform


def _frame_signal(rng, n_data=7):
    F = frame.Frame(n_data=n_data)
    o = F.o
    s = F.build(o.random_symbols(F.n_header, rng), o.random_symbols(n_data, rng))
    return F, s


def test_frame_layout():
    F, s = _frame_signal(np.random.default_rng(0))
    assert len(s) == F.length and F.gap == F.post_start
    assert np.allclose(s[F.post_start:], F.chirp())
    assert np.allclose(s[F.n_chirp:F.header_start], 0)


def test_scale_from_preamble_and_postamble():
    rng = np.random.default_rng(1)
    F, s = _frame_signal(rng)
    lead = 3000
    for eps in (-1e-3, 2e-4, 8e-4):
        x = np.concatenate([np.zeros(lead, complex), s, np.zeros(4096, complex)])
        y = waveform.warp(x, F.o.fs, np.full(len(x), eps), F.o.fc)
        y += 0.05 * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
        n_pre, _ = estimators.detect_chirp(y, F, 0, lead + 2000)
        n_post, _ = estimators.detect_chirp(y, F, int(n_pre + 0.9 * F.gap), int(n_pre + 1.1 * F.gap))
        assert abs(estimators.scale_pre_post(n_pre, n_post, F.gap) - eps) < 1e-5


def test_residual_cfo():
    rng = np.random.default_rng(2)
    o = waveform.OFDM()
    seg = o.modulate(o.random_symbols(1, rng))[o.n_cp:]
    for d in (-2.0, 0.7, 3.1):  # Hz (subcarrier spacing 7.8 Hz)
        y = seg * np.exp(2j * np.pi * d * np.arange(len(seg)) / o.fs)
        assert abs(estimators.residual_cfo(y, o) - d) < 0.1


def _sir(Y, X, o, used=None):
    from uwofdm.experiment import genie_powers
    s, e = genie_powers(Y, X, o.active if used is None else used)
    return 10 * np.log10(s / e)


def test_receivers_on_wave_doppler():
    """Constant scale: both receivers are clean. Wave-like time-varying scale (0.3 m/s at 0.12 Hz on a 1.3 s frame):
    the one-scale-per-packet front end with per-symbol CFO loses several dB, the receiver with an exact prior does not."""
    from uwofdm import chain
    rng = np.random.default_rng(3)
    F = frame.Frame()
    o = F.o
    X = o.random_symbols(F.n_data, rng)
    s = F.build(o.random_symbols(F.n_header, rng), X)
    lead, tail = 2500, 6000
    x = np.concatenate([np.zeros(lead, complex), s, np.zeros(tail, complex)])
    t = np.arange(len(x)) / o.fs
    for eps in (np.full(len(x), 6e-4), 6e-4 + 2e-4 * np.sin(2 * np.pi * 0.12 * t + 1.0)):
        y = waveform.warp(x, o.fs, eps, o.fc)
        Y0, info0 = chain.receive_signal_only(y, F, search=(0, lead + 1000))
        prior = eps[lead:lead + F.length + tail // 2]
        Y1, info1 = chain.receive_aided(y, F, prior, search=(0, lead + 1000))
        s0, s1 = _sir(Y0, X, o, F.used), _sir(Y1, X, o, F.used)
        assert s1 > 40
        if np.ptp(eps) == 0:
            assert s0 > 40 and abs(info0["eps"] - 6e-4) < 1e-5
        else:
            assert s0 < s1 - 5


def test_residual_cfo_low_snr_with_inband_nulls():
    """With in-band nulls the per-symbol offset is found at 0 dB per-subcarrier SNR; with all out-of-band bins it is
    not (the failure that broke the first low-SNR runs)."""
    rng = np.random.default_rng(4)
    F = frame.Frame()
    o = F.o
    X = o.random_symbols(1, rng)
    X[:, F.nulls] = 0
    seg = o.modulate(X)[o.n_cp:]
    d = 2.3
    errs_n, errs_a = [], []
    for _ in range(20):
        y = seg * np.exp(2j * np.pi * d * np.arange(len(seg)) / o.fs)
        y = y + np.sqrt(2.0) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))  # 0 dB per subcarrier
        errs_n.append(estimators.residual_cfo(y, o, span=2, n_grid=81, null_bins=F.null_bins()) - d)
        errs_a.append(estimators.residual_cfo(y, o, span=2, n_grid=81) - d)
    assert np.sqrt(np.mean(np.square(errs_n))) < 0.8  # < 0.1 subcarrier spacing
    assert np.sqrt(np.mean(np.square(errs_a))) > 3.0


def test_mean_scale_unbiased_under_acceleration():
    """With up- and down-sweeps detected separately and averaged, the preamble / postamble scale is the true mean under
    0.4 m/s^2 within 1e-5 (a single up-chirp was
    biased by 3.6e-5 through range-Doppler coupling)."""
    rng = np.random.default_rng(5)
    F, s = _frame_signal(rng)
    lead = 1500
    x = np.concatenate([np.zeros(lead, complex), s, np.zeros(9000, complex)])
    t = np.arange(len(x)) / F.o.fs
    eps = (1.0 + 0.4 * (t - t[lead])) / 1500.0
    y = waveform.warp(x, F.o.fs, eps, F.o.fc)
    n_pre, _ = estimators.detect_chirp(y, F, 0, lead + 1500)
    n_post, _ = estimators.detect_chirp(y, F, int(n_pre + 0.99 * F.gap), int(n_pre + 1.01 * F.gap))
    true_mean = (1.0 + 0.4 * F.gap / 2 / F.o.fs) / 1500.0
    assert abs(estimators.scale_pre_post(n_pre, n_post, F.gap) - true_mean) < 1e-5  # single up-chirp: 3.6e-5


def test_detect_refinement_stays_within_half_a_sample():
    """estimators.detect: the parabolic refinement is clamped to +-0.5 sample (a near-flat peak at low SNR once moved
    the preamble position past the end of the signal)."""
    from uwofdm import estimators
    ref = np.ones(8, complex)
    y = np.zeros(64, complex)
    y[20:28] = 1.0
    y[19] = y[28] = 0.999999  # nearly flat top: tiny parabola curvature
    pos, _ = estimators.detect(y, ref, first_path=None)
    assert 18.5 <= pos <= 21.5
