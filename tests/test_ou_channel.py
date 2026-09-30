import numpy as np
import pytest

from uwsb.channels.ou import OUChannel, OUChannelConfig


def test_reproducible_given_seed():
    cfg = OUChannelConfig(m=[0.5, 0.4], sigma_idio=0.3, Tc_s=20.0, T_slot_s=1.0)
    a = OUChannel(cfg).simulate(1000, np.random.default_rng(7))
    b = OUChannel(cfg).simulate(1000, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_stationary_variance_matches_sigma_dev():
    cfg = OUChannelConfig(m=[0.0], sigma_idio=0.5, Tc_s=15.0, T_slot_s=1.0)
    mu = OUChannel(cfg).simulate(200_000, np.random.default_rng(1))
    assert mu[:, 0].std() == pytest.approx(0.5, rel=0.05)   # sigma_dev = sigma_idio


def test_autocorrelation_is_exponential():
    Tc, Tslot = 30.0, 1.0
    cfg = OUChannelConfig(m=[0.0], sigma_idio=1.0, Tc_s=Tc, T_slot_s=Tslot)
    x = OUChannel(cfg).simulate(400_000, np.random.default_rng(2))[:, 0]
    x = x - x.mean()
    var = np.dot(x, x) / len(x)
    for lag in (10, 30, 60):
        ac = np.dot(x[:-lag], x[lag:]) / (len(x) - lag) / var
        assert ac == pytest.approx(np.exp(-lag * Tslot / Tc), abs=0.03)


def test_shared_factor_couples_arms():
    # b > 0 => positive cross-arm correlation; b = 0 => ~0.
    for b, expect_corr in [(1.0, True), (0.0, False)]:
        cfg = OUChannelConfig(m=[0.0, 0.0], sigma_idio=0.5, b=b, Tc_s=20.0,
                              T_slot_s=1.0)
        mu = OUChannel(cfg).simulate(200_000, np.random.default_rng(3))
        c = np.corrcoef(mu[:, 0], mu[:, 1])[0, 1]
        if expect_corr:
            assert c > 0.5
        else:
            assert abs(c) < 0.05


def test_sigma_dev_property():
    cfg = OUChannelConfig(m=[0.0, 0.0], sigma_idio=0.3, b=0.4, Tc_s=10.0,
                          T_slot_s=1.0)
    assert np.allclose(cfg.sigma_dev, np.sqrt(0.3 ** 2 + 0.4 ** 2))


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        OUChannelConfig(m=[0.1, 0.2], sigma_idio=[0.1, 0.2, 0.3], Tc_s=5.0,
                        T_slot_s=1.0)
