import math

import numpy as np
import pytest
from scipy.stats import levy_stable

from uwsb.noise.impulsiveness import (
    BandpassStream, block_stats, gaussian_exceedance, robust_sigma,
)


def test_gaussian_exceedance_known_values():
    assert gaussian_exceedance(0.0) == pytest.approx(1.0)
    assert gaussian_exceedance(1.0) == pytest.approx(0.31731, rel=1e-4)
    assert gaussian_exceedance(3.0) == pytest.approx(2.6998e-3, rel=1e-4)
    assert gaussian_exceedance(5.0) == pytest.approx(5.7330e-7, rel=1e-4)


def test_robust_sigma_gaussian_and_laplace():
    rng = np.random.default_rng(1)
    assert robust_sigma(2.0 * rng.standard_normal(1_000_000)) == pytest.approx(2.0, rel=0.01)
    # Laplace(b): MAD = b ln 2, so robust_sigma = 1.4826 * b ln 2
    b = 1.5
    x = rng.laplace(scale=b, size=1_000_000)
    assert robust_sigma(x) == pytest.approx(1.4826 * b * math.log(2.0), rel=0.01)


def test_block_stats_gaussian_is_null_case():
    rng = np.random.default_rng(2)
    s = block_stats(3.0 * rng.standard_normal(1_000_000), k_sigma=3.0)
    assert s["excess_kurtosis"] == pytest.approx(0.0, abs=0.03)   # SE ~ sqrt(24/n) = 0.005
    assert s["exceed_ratio"] == pytest.approx(1.0, abs=0.1)       # ~2700 exceedances
    assert s["std"] == pytest.approx(3.0, rel=0.01)


def test_block_stats_laplace_kurtosis_is_three():
    rng = np.random.default_rng(3)
    s = block_stats(rng.laplace(scale=1.0, size=2_000_000))
    assert s["excess_kurtosis"] == pytest.approx(3.0, abs=0.15)
    assert s["exceed_ratio"] > 1.0                                  # heavier than Gaussian


def test_block_stats_flags_symmetric_alpha_stable():
    x = levy_stable.rvs(1.5, 0.0, size=200_000, random_state=np.random.default_rng(4))
    s = block_stats(x, k_sigma=5.0)
    assert s["exceed_ratio"] > 100.0     # power-law tail vs Gaussian 5.7e-7
    assert s["excess_kurtosis"] > 10.0


def test_block_stats_rejects_degenerate_input():
    with pytest.raises(ValueError):
        block_stats(np.ones(100))
    with pytest.raises(ValueError):
        block_stats(np.array([1.0]))


def test_bandpass_passes_in_band_and_rejects_out_of_band():
    fs_hz = 48_000.0
    t_s = np.arange(int(fs_hz)) / fs_hz                  # 1 s
    bp_in = BandpassStream(fs_hz, (10_500.0, 15_500.0))
    bp_out = BandpassStream(fs_hz, (10_500.0, 15_500.0))
    y_in = bp_in(np.sin(2 * np.pi * 13_000.0 * t_s))[4800:]    # drop 0.1 s transient
    y_out = bp_out(np.sin(2 * np.pi * 3_000.0 * t_s))[4800:]
    assert np.mean(y_in ** 2) == pytest.approx(0.5, rel=0.05)  # unit tone: power 1/2
    assert 10 * np.log10(np.mean(y_out ** 2) / 0.5) < -60.0


def test_bandpass_streaming_equals_single_call():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(30_000)
    one = BandpassStream(96_000.0, (20_000.0, 30_000.0))(x)
    bp = BandpassStream(96_000.0, (20_000.0, 30_000.0))
    blocks = np.concatenate([bp(x[i:i + 7_000]) for i in range(0, x.size, 7_000)])
    np.testing.assert_allclose(blocks, one, rtol=0, atol=1e-12)


def test_bandpass_rejects_band_above_nyquist():
    with pytest.raises(ValueError):
        BandpassStream(48_000.0, (20_000.0, 30_000.0))   # Red band needs fs >= 60 kHz
