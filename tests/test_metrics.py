import numpy as np
import pytest

from uwsb.channels.ou import OUChannel, OUChannelConfig
from uwsb.envs.link_adaptation import LinkAdaptationEnv, GaussianReward
from uwsb.metrics import (
    staleness_number, staleness_number_slots, regret_report,
    dynamic_oracle_reward, earned_expected_reward,
)
from uwsb.bandits.predictive_ucb import PredictiveUCB


def test_staleness_number():
    assert staleness_number(2.0, 4.0) == pytest.approx(0.5)
    assert staleness_number_slots(30, 15.0, 1.0) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        staleness_number(1.0, 0.0)


def _run(agent, ch, tau_rt=20, tau_ow=10, T=1500, seed=0):
    env = LinkAdaptationEnv(ch, GaussianReward(0.1), tau_rt_slots=tau_rt,
                            tau_ow_slots=tau_ow, T_slot_s=1.0)
    return env.run(agent, T=T, rng=np.random.default_rng(seed))


def test_dynamic_oracle_dominates_earned():
    cfg = OUChannelConfig(m=[0.6, 0.5, 0.4], sigma_idio=0.2, Tc_s=40.0, T_slot_s=1.0)
    ch = OUChannel(cfg); ch.simulate(3000, np.random.default_rng(0))
    agent = PredictiveUCB(K=3, Tc_hat_s=40.0, sigma2=0.04, sigma_n2=0.01, beta=1.5)
    res = _run(agent, ch)
    dyn = dynamic_oracle_reward(ch, res, GaussianReward(0.1))
    earned = earned_expected_reward(ch, res, GaussianReward(0.1))
    assert np.all(dyn + 1e-9 >= earned)                    # oracle never worse
    rep = regret_report(ch, res, GaussianReward(0.1), Tc_true_s=40.0)
    assert np.all(np.diff(rep["regret_vs_dynamic"]) >= -1e-9)   # non-decreasing


def test_constant_channel_zero_regret_for_good_agent():
    # No fluctuation (sigma=0): dynamic == stationary, PUCB should match after
    # it has identified the best arm.
    cfg = OUChannelConfig(m=[0.9, 0.5, 0.3], sigma_idio=0.0, Tc_s=10.0, T_slot_s=1.0)
    ch = OUChannel(cfg); ch.simulate(3000, np.random.default_rng(0))
    agent = PredictiveUCB(K=3, Tc_hat_s=10.0, sigma2=1e-6, sigma_n2=0.01, beta=1.0)
    res = _run(agent, ch, T=2500)
    rep = regret_report(ch, res, GaussianReward(0.1), Tc_true_s=10.0)
    # tail per-step regret vs dynamic ~ 0
    tail = np.diff(rep["regret_vs_dynamic"])[-500:]
    assert tail.mean() < 1e-6


def test_causal_between_dynamic_and_stationary():
    cfg = OUChannelConfig(m=[0.55, 0.45], sigma_idio=0.4, Tc_s=50.0, T_slot_s=1.0)
    ch = OUChannel(cfg); ch.simulate(4000, np.random.default_rng(5))
    agent = PredictiveUCB(K=2, Tc_hat_s=50.0, sigma2=0.16, sigma_n2=0.01, beta=1.5)
    res = _run(agent, ch, tau_rt=15, tau_ow=8, T=3000)
    rm = GaussianReward(0.1)
    rep = regret_report(ch, res, rm, Tc_true_s=50.0)
    dyn = rep["reward_dynamic"].mean()
    caus = rep["reward_causal"].mean()
    stat = rep["reward_stationary"].mean()
    # dynamic >= causal >= stationary  (the oracle ladder, THEORY.md §2)
    assert dyn + 1e-9 >= caus >= stat - 1e-9
