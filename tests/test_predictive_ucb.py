"""Analytic unit tests for the 2-state Kalman PUCB (THEORY.md §5, project.MD §7.1).

Every estimator is checked against a case with a known closed form BEFORE it is
trusted in any experiment.
"""
import math

import numpy as np
import pytest

from uwsb.bandits.predictive_ucb import PredictiveUCB, PredictiveThompson, beta_theory


def R(a, Tc):
    return math.exp(-a / Tc)


def test_single_sample_matches_brief_formula():
    """THEORY.md §5 test 1: known mean (prior_m_var=0), noiseless obs.

    After one sample r of age a, the predicted reward-mean posterior must be
    mean = m + R(a)(r - m),  var = sigma^2 (1 - R(a)^2).
    """
    Tc, sigma2, m = 10.0, 0.4, 0.5
    agent = PredictiveUCB(K=1, Tc_hat_s=Tc, sigma2=sigma2, sigma_n2=0.0,
                          prior_m_mean=m, prior_m_var=0.0)
    r = 0.9
    agent.observe(arm=0, tx_time_s=0.0, reward=r)
    for a in (1.0, 5.0, 25.0):
        mean, std = agent.posterior(now_s=a)
        assert mean[0] == pytest.approx(m + R(a, Tc) * (r - m), rel=1e-9)
        assert std[0] ** 2 == pytest.approx(sigma2 * (1 - R(a, Tc) ** 2), rel=1e-9)


def test_noiseless_most_recent_sample_pins_state():
    """With known mean and noiseless obs, a redundant older sample must not
    change the posterior: the freshest sample pins d exactly."""
    Tc, sigma2, m = 8.0, 0.3, 0.2
    a1 = PredictiveUCB(K=1, Tc_hat_s=Tc, sigma2=sigma2, sigma_n2=0.0,
                       prior_m_mean=m, prior_m_var=0.0)
    a2 = PredictiveUCB(K=1, Tc_hat_s=Tc, sigma2=sigma2, sigma_n2=0.0,
                       prior_m_mean=m, prior_m_var=0.0)
    a1.observe(0, 0.0, 0.7)
    a1.observe(0, 3.0, 0.6)           # older + newer
    a2.observe(0, 3.0, 0.6)           # newer only
    m1, s1 = a1.posterior(10.0)
    m2, s2 = a2.posterior(10.0)
    assert m1[0] == pytest.approx(m2[0], rel=1e-9)
    assert s1[0] == pytest.approx(s2[0], rel=1e-9)


def test_graceful_degradation_to_mean_at_infinite_staleness():
    """THEORY.md §5 test 2: as the prediction gap -> inf, the deviation is
    forgotten and the index reduces to mean = m_hat, var = Pmm + sigma^2."""
    Tc, sigma2 = 5.0, 0.25
    agent = PredictiveUCB(K=1, Tc_hat_s=Tc, sigma2=sigma2, sigma_n2=0.05,
                          prior_m_mean=0.0, prior_m_var=1.0)
    for tx, r in [(0.0, 0.8), (2.0, 0.9), (4.0, 0.7)]:
        agent.observe(0, tx, r)
    mean_far, std_far = agent.posterior(now_s=1e6)
    # deviation contribution gone: var -> Pmm + sigma^2, mean -> m_hat
    assert std_far[0] ** 2 == pytest.approx(agent.Pmm[0] + sigma2, rel=1e-6)
    assert mean_far[0] == pytest.approx(agent.m_hat[0], rel=1e-9)


def test_static_channel_learns_the_mean():
    """Feed a constant reward (true deviation 0): m_hat must converge to it."""
    agent = PredictiveUCB(K=1, Tc_hat_s=10.0, sigma2=0.2, sigma_n2=0.01,
                          prior_m_mean=0.0, prior_m_var=1.0)
    c = 0.73
    for i in range(2000):
        agent.observe(0, float(i), c)
    mean, _ = agent.posterior(now_s=2000.0)
    assert mean[0] == pytest.approx(c, abs=0.02)


def test_variance_stays_nonnegative_under_random_stream():
    rng = np.random.default_rng(0)
    agent = PredictiveUCB(K=3, Tc_hat_s=6.0, sigma2=0.5, sigma_n2=0.1,
                          prior_m_var=1.0, q_m=1e-4)
    t = 0.0
    for _ in range(5000):
        k = int(rng.integers(3))
        agent.observe(k, t, float(rng.normal()))
        t += 1.0
        _ = agent.select(t, rng)      # exercises _predict_forward + clip asserts
    assert np.all(agent.Pmm >= 0) and np.all(agent.Pdd >= 0)


def test_out_of_order_feedback_rejected():
    agent = PredictiveUCB(K=1, Tc_hat_s=5.0, sigma2=0.2, sigma_n2=0.0)
    agent.observe(0, 10.0, 0.5)
    with pytest.raises(ValueError):
        agent.observe(0, 3.0, 0.5)    # earlier tx after a later one


def test_unseen_arm_is_explored_over_modest_incumbent():
    """A single in-range observation must not starve exploration of an unseen
    arm. (Note: PUCB uses a proper Bayesian prior, not infinite optimism, so a
    genuinely huge observed mean CAN outrank an unseen arm -- that is correct;
    here the incumbent's reward is on-scale so the unseen arm's optimism wins.)"""
    agent = PredictiveUCB(K=2, Tc_hat_s=5.0, sigma2=0.2, sigma_n2=0.1,
                          prior_m_mean=0.5, prior_m_var=1.0, beta=1.5)
    agent.observe(0, 0.0, 0.55)       # arm 0: a normal, slightly-good reward
    assert agent.select(now_s=1.0) == 1   # unseen arm 1 still explored


def test_thompson_runs_and_requires_rng():
    agent = PredictiveThompson(K=2, Tc_hat_s=5.0, sigma2=0.2, sigma_n2=0.1)
    with pytest.raises(ValueError):
        agent.select(now_s=1.0, rng=None)
    rng = np.random.default_rng(1)
    a = agent.select(now_s=1.0, rng=rng)
    assert a in (0, 1)


def test_beta_theory_schedule_grows():
    b = beta_theory()
    assert b(10) < b(1000)
    assert b(2) == pytest.approx(math.sqrt(2 * math.log(2)))
