"""Layer-1 goodput model (envs/goodput.py) and its agents (bandits/goodput.py), THEORY-B G-1/G-2."""
import math

import numpy as np
import pytest
from scipy.special import digamma, gammainc

from uwsb.bandits.goodput import GoodputAckTS, GoodputAckUCB1, StructuredGoodputUCB, ThresholdAMC
from uwsb.envs.goodput import (GoodputModel, SuccessTable, achievable_rate, cached_success_table,
                               expected_goodput, make_goodput_table)
from uwsb.envs.table_env import TableEnv

S_GRID = np.arange(-5.0, 20.5, 0.5)


def test_achievable_rate_formulas():
    p2 = np.array([[0.5, 1.0, 1.5, 100.0]])
    s = np.array([3.0])
    mp = achievable_rate(GoodputModel(success_model="mean_power"), s, p2)
    assert mp[0] == pytest.approx(math.log2(1 + 3.0 / np.mean(p2)))
    mi = achievable_rate(GoodputModel(success_model="miesm"), s, p2)
    assert mi[0] == pytest.approx(np.mean(np.log2(1 + 3.0 / p2)))
    bl = achievable_rate(GoodputModel(success_model="blanking", blank_db=10.0), s, p2)   # 100 > 10: blanked
    assert bl[0] == pytest.approx(0.75 * math.log2(1 + 3.0 / 1.0))
    with pytest.raises(ValueError):
        achievable_rate(GoodputModel(success_model="nope"), s, p2)


def test_table_structure_and_seeding():
    m = GoodputModel(alpha=1.6)
    tab = make_goodput_table(m, 500, 20, seed=3)
    g, ack, gp = tab.fields["gamma_db"], tab.fields["ack"], tab.fields["goodput"]
    assert g.shape == (500, m.J * m.M)
    for j in range(m.J):                                         # gamma-hat shared by every MCS on a sub-band
        assert np.all(g[:, j * m.M:(j + 1) * m.M] == g[:, [j * m.M]])
        a = ack[:, j * m.M:(j + 1) * m.M]
        assert np.all(np.diff(a, axis=1) <= 0)                    # success at rate m => success at lower rates
    assert np.array_equal(gp, ack * m.arm_rates)
    assert np.array_equal(make_goodput_table(m, 500, 20, seed=3).fields["ack"], ack)


def test_expected_goodput_matches_long_table():
    m = GoodputModel(alpha=1.6)
    eg = expected_goodput(m, n_mc=40_000)
    tab = make_goodput_table(m, 20_000, 1, seed=4, truth=eg)
    assert np.allclose(tab.fields["goodput"].mean(axis=0), eg, atol=0.03)


def test_gaussian_mean_power_success_is_exact():
    # alpha = 2, c = 0.5: |z|^2 ~ Exp(1), mean over n ~ Gamma(n, 1)/n; success iff mean <= s / (2^R - 1)
    m = GoodputModel(success_model="mean_power", n_pilot=16, n_data=64)
    tab = SuccessTable(m, [1.8, 2.0], S_GRID, 20_000, 1)
    s = 10 ** ((S_GRID - m.gap_db) / 10)
    for k, R in enumerate(m.rates):
        exact = gammainc(64, 64 * s / (2 ** R - 1))
        assert np.max(np.abs(tab.psucc_at(2.0, S_GRID)[:, k] - exact)) < 0.02
    # gamma-hat noise X = -10 log10(Gamma(16, 1)/16): E[X] = -(10 / ln 10)(digamma(16) - ln 16)
    assert tab.law(2.0)[0] == pytest.approx(-(10 / math.log(10)) * (digamma(16) - math.log(16)), abs=0.02)


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    m = GoodputModel()
    return cached_success_table(m, [1.2, 1.6, 2.0], S_GRID, 5000, 2, cache_dir=tmp_path_factory.mktemp("c"))


def test_ack_agents_index():
    rates = [0.5, 3.0]
    ts = GoodputAckTS(2, rates, rng=np.random.default_rng(0))
    for _ in range(50):
        ts.observe(0, 0, {"ack": 1.0})
        ts.observe(1, 0, {"ack": 0.0})
    assert ts.select(0) == 0                                      # 0.5 * ~1 beats 3 * ~0
    u = GoodputAckUCB1(2, rates)
    u.observe(1, 0, {"goodput": 3.0, "ack": 1.0})
    assert u.s[1] == 1.0


def test_structured_learner_finds_the_best_arm(table):
    m = GoodputModel(alpha=1.6)
    eg = expected_goodput(m, n_mc=20_000)
    tab = make_goodput_table(m, 3000, 200, seed=5, truth=eg)
    ag = StructuredGoodputUCB(m.J, m.M, m.rates, table, calibration="noise_stream")
    res = TableEnv(tab, 3).run(ag)
    assert np.mean(res.arms[-750:] == int(np.argmax(eg))) > 0.8
    assert abs(ag.alpha - 1.6) < 0.1


def test_awgn_calibration_is_biased_under_impulsive_noise(table):
    # AWGN-calibrated structured learner: gamma-hat bias E[X] assumed ~0 while it is ~-5 dB at alpha 1.6 ->
    # it underestimates the sub-band SNR and settles on a lower-goodput arm than the calibrated one
    m = GoodputModel(alpha=1.6)
    eg = expected_goodput(m, n_mc=20_000)
    tab = make_goodput_table(m, 3000, 200, seed=6, truth=eg)
    fixed = TableEnv(tab, 3).run(StructuredGoodputUCB(m.J, m.M, m.rates, table, calibration=2.0))
    calib = TableEnv(tab, 3).run(StructuredGoodputUCB(m.J, m.M, m.rates, table, calibration="noise_stream"))
    assert calib.regret_inst.sum() < 0.5 * fixed.regret_inst.sum()


def test_threshold_amc_thresholds_and_choice(table):
    m = GoodputModel()
    amc = ThresholdAMC(m.J, m.M, m.rates, table, alpha_cal=2.0, target=0.9)
    assert np.all(np.diff(amc.thr[np.isfinite(amc.thr)]) > 0)    # higher rate needs higher gamma-hat
    assert [amc.select(0) for _ in range(3)] == [0, m.M, 2 * m.M]  # one probe per sub-band, lowest MCS
    amc.observe(0, 0, {"gamma_db": 50.0})
    assert amc.select(0) == m.M - 1                               # very high gamma-hat: top MCS on sub-band 0


def test_layer1_runner_goodput_smoke(tmp_path, monkeypatch):
    import json

    import yaml

    from experiments import run_layer1_b
    from uwsb import runtime
    monkeypatch.setenv("UWSB_RESULTS_DIR", str(tmp_path))
    cfg_path = "experiments/configs/b_layer1_smoke_goodput.yaml"
    assert run_layer1_b.main(["--config", cfg_path, "--smoke", "--n-workers", "2"]) == 0
    cfg = yaml.safe_load(open(cfg_path))
    summary = json.loads((runtime.result_dir(cfg) / "summary.json").read_text())
    assert set(summary["agents"]) == {a["name"] for a in cfg["agents"]}
    assert len(summary["true_means"]) == 15 and "ack_threshold" not in summary


def test_fading_tables_and_fading_aware_learner(table):
    m = GoodputModel(alpha=1.8, fading_db=3.0, fading_corr=0.9)
    tab = make_goodput_table(m, 3000, 200, seed=7)
    f = tab.meta["fading"]
    assert f.std() == pytest.approx(3.0, rel=0.15) and np.corrcoef(f[1:, 0], f[:-1, 0])[0, 1] == pytest.approx(0.9, abs=0.05)
    g = tab.fields["gamma_db"][:, 0]
    assert np.corrcoef(g, f[:, 0])[0, 1] > 0.5                       # gamma-hat follows the channel
    assert not np.allclose(tab.truth_mean[0], tab.truth_mean[1000])  # per-slot oracle
    nofade = make_goodput_table(GoodputModel(alpha=1.8), 300, 20, seed=7)
    assert np.array_equal(nofade.fields["ack"], make_goodput_table(GoodputModel(alpha=1.8), 300, 20, seed=7).fields["ack"])
    ag = StructuredGoodputUCB(m.J, m.M, m.rates, table, calibration="noise_stream", fading_aware=True)
    TableEnv(tab, 3).run(ag)
    assert ag.fading_var(table.law(ag.alpha)[1]) == pytest.approx(9.0, rel=0.35)   # fading variance recovered
