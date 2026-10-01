"""NIR-UCB v1 and the noise -> reward-noise law (THEORY-B G-0/G-5)."""
import math

import numpy as np
import pytest
from scipy.special import polygamma

from uwsb.bandits.nir import NIRUCBv1
from uwsb.envs.table_env import TableEnv
from uwsb.estimation.reward_law import EvmDbLaw, SasLaw, cached_evm_db_law, evm_db_noise, subexp_pairs


@pytest.fixture(scope="module")
def law():
    return EvmDbLaw(64, [1.4, 1.6, 1.8, 2.0], 20_000, 3)


def test_gaussian_limit_matches_trigamma(law):
    # alpha = 2: mean |z|^2 over n symbols = Gamma(n, .)/n, so sd of 10 log10 = (10/ln 10) sqrt(psi_1(n))
    assert law(2.0)[0] == pytest.approx(10 / math.log(10) * math.sqrt(polygamma(1, 64)), rel=0.02)
    assert law(1.6)[0] > law(1.8)[0] > law(2.0)[0]                    # heavier tails, wider reward noise


def test_subexponential_pairs_hold_on_fresh_samples():
    x = evm_db_noise(1.6, 64, 50_000, np.random.default_rng(4))
    sigma, nu2, b = subexp_pairs(x)
    y = evm_db_noise(1.6, 64, 50_000, np.random.default_rng(5))
    d = y - y.mean()
    for j in (0, 4, 9):
        lam = 1.0 / b[j]
        for side in (1, -1):
            psi = math.log(np.mean(np.exp(side * lam * d)))
            assert psi <= 0.5 * lam * lam * nu2[j] * 1.1                # 10 % Monte Carlo slack
    assert np.all(np.diff(b) <= 1e-12)                                 # larger nu2 -> smaller (or equal) b


def test_law_interpolates_and_caches(law, tmp_path):
    s16, s18 = law(1.6)[0], law(1.8)[0]
    assert law(1.7)[0] == pytest.approx(0.5 * (s16 + s18))
    assert law(1.0)[0] == law(1.4)[0]                                   # clipped to the grid
    a = cached_evm_db_law(16, [1.5, 2.0], 500, 1, cache_dir=tmp_path)
    b = cached_evm_db_law(16, [1.5, 2.0], 500, 1, cache_dir=tmp_path)   # loaded from disk
    assert len(list(tmp_path.iterdir())) == 1 and np.array_equal(a.b, b.b) and np.array_equal(a.sigma, b.sigma)
    assert SasLaw()(1.5, 2.0) == (math.sqrt(2) * 2.0, None, None)


def test_widths_are_the_stated_formulas(law):
    ag = NIRUCBv1(3, law=law, width="bernstein")
    ag.params = law(1.6)
    ag.t, ag.n = 100, np.array([5, 50, 500])
    sigma, nu2, b = ag.params
    L = math.log(100)
    want = [min(max(math.sqrt(2 * v * L / n), 2 * bb * L / n) for v, bb in zip(nu2, b)) for n in (5, 50, 500)]
    assert np.allclose(ag.widths(), want)
    ag.width = "gauss"
    assert np.allclose(ag.widths(), sigma * np.sqrt(2 * L / np.array([5, 50, 500])))
    ag._inflate_until = 100
    assert np.allclose(ag.widths(), 2 * sigma * np.sqrt(2 * L / np.array([5, 50, 500])))
    with pytest.raises(ValueError):
        NIRUCBv1(2, law=None)


def test_v1_learns_on_branch_e_tables(law):
    from experiments import run_layer1_b as r
    cfg = {"reward_model": "evm_db", "T": 3000, "nu": 200, "noise": {"alpha": 1.6, "c": 0.1},
           "arms": {"snr_db": [0.0, 2.0]}, "evm": {"n_sym": 64, "mc_packets": 2000, "mc_seed": 7},
           "ack_threshold": 0.0}
    tab = r.make_table(cfg, 9)
    for width in ("gauss", "bernstein"):
        res = TableEnv(tab, 3).run(NIRUCBv1(2, law=law, width=width))
        first, last = (res.arms[:750] == 1).mean(), (res.arms[-750:] == 1).mean()
        assert last > first and last > 0.8


def test_regime_alarm_restarts_tracker_and_keeps_params(law):
    from uwsb.noise.stable import sas_real
    ag = NIRUCBv1(2, law=law, min_noise_samples=1000)
    rng = np.random.default_rng(10)
    for _ in range(200):
        ag.observe_noise(0.0, sas_real(1.8, 1.0, 200, rng))
    before = ag.params
    assert before is not None and ag.alarms == 0
    for _ in range(10):
        ag.observe_noise(0.0, sas_real(1.8, 10.0, 200, rng))           # +20 dB level step
    assert ag.alarms == 1 and ag.ns.tracker.n_eff < 10 * 200
    assert ag._inflate_until > ag.t


def test_law_bias_gaussian_limit_and_scale(law):
    from scipy.special import digamma
    # alpha = 2, c = 1: |z1|^2 / 4 ~ Exp(1); X1 = -10 log10(4 Gamma(n, 1) / n)
    exact = -10 * math.log10(4) - (10 / math.log(10)) * (digamma(64) - math.log(64))
    assert law.bias(2.0, 1.0) == pytest.approx(exact, abs=0.03)
    assert law.bias(1.6, 0.1) == pytest.approx(law.bias(1.6, 1.0) + 20.0, abs=1e-9)
    assert law.bias(1.6, 1.0) < law.bias(2.0, 1.0) - 2.0                 # impulsive noise lowers gamma-hat
    assert SasLaw().bias(1.5, 2.0) == 0.0


def test_debiased_means_recover_arm_snr_across_regimes(law):
    from experiments import run_layer1_b as r
    cfg = {"reward_model": "evm_db", "T": 4000, "nu": 200, "noise": {"alpha": 1.9, "c": 0.1},
           "regimes": {"laws": [{"alpha": 1.9, "c": 0.1}, {"alpha": 1.5, "c": 0.1}],
                       "P": [[0.998, 0.002], [0.002, 0.998]]},
           "arms": {"snr_db": [0.0, 2.0]}, "evm": {"n_sym": 64, "mc_packets": 2000, "mc_seed": 7},
           "ack_threshold": 0.0}
    tab = r.make_table(cfg, 11)
    assert len(set(tab.meta["regime"])) == 2
    ag = NIRUCBv1(2, law=law, debias=True, detector_params={})
    TableEnv(tab, 3).run(ag)
    est = ag.s / ag.n
    assert est[1] == pytest.approx(2.0, abs=0.5)                          # best arm: many samples
    raw = NIRUCBv1(2, law=law)                                            # no debias: the average carries E[X]
    TableEnv(tab, 3).run(raw)                                             # (+20 dB from c = 0.1, minus impulses)
    assert abs((raw.s / raw.n)[1] - 2.0) > 3.0


def test_debias_retroactively_corrects_early_observations(law):
    ag = NIRUCBv1(2, law=law, debias=True, min_noise_samples=10)
    ag.observe(0, 0.0, {"gamma_db": 5.0})                                 # before any calibration: raw
    assert ag.s[0] == 5.0 and ag._early
    from uwsb.noise.stable import sas_real
    ag.observe_noise(0.0, sas_real(1.8, 1.0, 200, np.random.default_rng(0)))
    assert not ag._early and ag.n[0] == 1
    assert ag.s[0] == pytest.approx(5.0 - law.bias(ag._alpha, ag._c))
    assert ag.v[0] == pytest.approx(ag.params[0] ** 2)


def test_redebias_after_alarm_uses_the_new_calibration(law):
    ag = NIRUCBv1(2, law=law, debias=True, redebias_lookback=5, min_noise_samples=10)
    ag.params, ag._alpha, ag._c = law(1.9), 1.9, 1.0
    for t in range(1, 11):
        ag.t = t
        ag.observe(0, 0.0, {"gamma_db": 1.0})
    stale = law.bias(1.9, 1.0)
    assert ag.s[0] == pytest.approx(10 * (1.0 - stale))
    ag._redebias_from = 10 - 5                                        # as set by an alarm at decision 10
    ag._alpha, ag._c, ag.params = 1.3, 1.0, law(1.3)
    ag._redebias(ag._redebias_from)
    new = law.bias(1.3, 1.0)
    assert ag.s[0] == pytest.approx(4 * (1.0 - stale) + 6 * (1.0 - new))  # decisions 5..10 re-debiased
    assert ag.v[0] == pytest.approx(4 * law(1.9)[0] ** 2 + 6 * law(1.3)[0] ** 2)


def test_inverse_variance_weighting(law):
    ag = NIRUCBv1(2, law=law, debias=True, weighting="inverse_variance", redebias_lookback=5)
    ag._alpha, ag._c = 2.0, 1.0
    ag.params = law(2.0)                                                  # calm: small sigma
    ag.t = 1
    ag.observe(0, 0.0, {"gamma_db": 1.0})
    ag.params, ag._alpha = law(1.4), 1.4                                  # impulsive: large sigma
    ag.t = 2
    ag.observe(0, 0.0, {"gamma_db": 9.0})
    s1, s2 = law(2.0)[0] ** 2, law(1.4)[0] ** 2
    y1, y2 = 1.0 - law.bias(2.0, 1.0), 9.0 - law.bias(1.4, 1.0)
    assert ag.sw[0] / ag.W[0] == pytest.approx((y1 / s1 + y2 / s2) / (1 / s1 + 1 / s2))
    ag.t = 10
    assert ag.widths()[0] == pytest.approx(math.sqrt(2 * math.log(10) / (1 / s1 + 1 / s2)))
    with pytest.raises(ValueError):
        NIRUCBv1(2, law=law, weighting="inverse_variance")               # needs debias
