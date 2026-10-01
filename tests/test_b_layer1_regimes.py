"""Regime-switching Layer-1 tables (experiments/run_layer1_b.py, THEORY-B G-0)."""
import numpy as np
import pytest

from experiments import run_layer1_b as r


def _cfg(**kw):
    cfg = {"reward_model": "sas", "T": 4000, "nu": 50, "noise": {"alpha": 1.9, "c": 1.0},
           "regimes": {"laws": [{"alpha": 1.9, "c": 1.0}, {"alpha": 1.9, "c": 5.0}],
                       "P": [[0.99, 0.01], [0.02, 0.98]]},
           "arms": {"means": [0.0, 1.0]}, "ack_threshold": 0.5}
    cfg.update(kw)
    return cfg


def test_regime_path_follows_the_chain():
    z = r.regime_path([[0.99, 0.01], [0.02, 0.98]], 200_000, np.random.default_rng(1))
    assert z[0] == 0
    assert np.mean(z[1:][z[:-1] == 0] == 0) == pytest.approx(0.99, abs=0.002)
    assert np.mean(z[1:][z[:-1] == 1] == 1) == pytest.approx(0.98, abs=0.003)
    with pytest.raises(ValueError):
        r.regime_path([[0.5, 0.4], [0.1, 0.9]], 10, np.random.default_rng(0))


def test_regime_table_slots_follow_their_law():
    tab = r.make_table(_cfg(), 3)
    z = tab.meta["regime"]
    assert set(np.unique(z)) == {0, 1}
    assert np.array_equal(tab.truth_mean, np.tile([0.0, 1.0], (4000, 1)))     # S: means do not move
    s0 = np.median(np.abs(tab.noise[z == 0]))
    s1 = np.median(np.abs(tab.noise[z == 1]))
    assert s1 / s0 == pytest.approx(5.0, rel=0.1)                              # stream scale follows c
    d0 = np.median(np.abs(tab.fields["gamma_db"][z == 0] - [0.0, 1.0]))
    d1 = np.median(np.abs(tab.fields["gamma_db"][z == 1] - [0.0, 1.0]))
    assert d1 / d0 == pytest.approx(5.0, rel=0.1)                              # rewards too
    assert np.array_equal(r.make_table(_cfg(), 3).fields["gamma_db"], tab.fields["gamma_db"])   # seeded


def test_regime_config_validation():
    bad = _cfg(noise={"alpha": 1.5, "c": 1.0})
    with pytest.raises(ValueError):
        r.make_table(bad, 0)
    with pytest.raises(ValueError):
        r.make_agent({"type": "empirical_ucb", "params": {"sigma": "oracle"}}, _cfg(), 2, 0)


def test_stationary_tables_unchanged_by_the_regime_code():
    cfg = _cfg()
    del cfg["regimes"]
    a = r.make_table(cfg, 5)
    rng = np.random.default_rng(5)                                             # the original construction
    g = np.array([0.0, 1.0]) + r.sas_real(1.9, 1.0, (4000, 2), rng)
    assert np.array_equal(a.fields["gamma_db"], g)
