"""The delayed-feedback guarantees (project.MD §7.2, §7.3).

These are the tests that must FAIL if any sample younger than tau_rt ever reaches
an agent, or if the agent is handed anything but (arm, tx_time, reward)/(now).
"""
import numpy as np
import pytest

from uwsb.channels.ou import OUChannel, OUChannelConfig
from uwsb.envs.link_adaptation import LinkAdaptationEnv, GaussianReward


def make_channel(K=3, Tc_s=40.0, T_slot_s=1.0, n=4000, seed=0):
    cfg = OUChannelConfig(m=np.linspace(0.6, 0.4, K), sigma_idio=0.2,
                          Tc_s=Tc_s, T_slot_s=T_slot_s)
    ch = OUChannel(cfg)
    ch.simulate(n, np.random.default_rng(seed))
    return ch


class AgeSpyAgent:
    """Counts delivered samples. (Ages are checked from the env's own true-time
    measurement `res.min_delivered_age_slots`; a spy cannot time the age itself
    because feedback is delivered before `select` runs each slot.)"""

    def __init__(self, K):
        self.K = K
        self.n_observed = 0

    def observe(self, arm, tx_time_s, reward):
        self.n_observed += 1

    def select(self, now_s, rng=None):
        return int(now_s) % self.K       # deterministic round-robin-ish


def test_no_sample_younger_than_tau_rt():
    tau_rt, tau_ow = 30, 15
    ch = make_channel()
    env = LinkAdaptationEnv(ch, GaussianReward(0.1), tau_rt_slots=tau_rt,
                            tau_ow_slots=tau_ow, T_slot_s=1.0)
    agent = AgeSpyAgent(ch.cfg.K)
    res = env.run(agent, T=1000, rng=np.random.default_rng(1))
    assert agent.n_observed > 0, "no feedback was ever delivered"
    # Env measures age with the true current slot; guard from project.MD §7.3.
    assert res.min_delivered_age_slots >= tau_rt
    # The env raises internally if an early delivery ever occurs; reaching here
    # without exception is itself part of the guarantee.


def test_one_way_exceeding_round_trip_rejected():
    """Construction guard: tau_ow <= tau_rt (D2)."""
    ch = make_channel()
    with pytest.raises(ValueError):
        LinkAdaptationEnv(ch, GaussianReward(0.0), tau_rt_slots=5, tau_ow_slots=10)


def test_last_tau_rt_packets_not_delivered_within_horizon():
    tau_rt = 25
    ch = make_channel()
    env = LinkAdaptationEnv(ch, GaussianReward(0.05), tau_rt_slots=tau_rt,
                            tau_ow_slots=10, T_slot_s=1.0)
    agent = AgeSpyAgent(ch.cfg.K)
    T = 500
    env.run(agent, T=T, rng=np.random.default_rng(2))
    # Exactly the packets sent in the last tau_rt slots cannot have matured.
    assert agent.n_observed == T - tau_rt


def test_agent_receives_only_declared_interface():
    """Oracle non-leakage (project.MD §7.2): the env must call select(now, rng)
    and observe(arm, tx, reward) with primitive types only -- never the channel
    or true mu."""
    tau_rt = 10
    ch = make_channel()
    env = LinkAdaptationEnv(ch, GaussianReward(0.1), tau_rt_slots=tau_rt,
                            tau_ow_slots=5, T_slot_s=1.0)

    calls = {"observe": [], "select": []}

    class Recorder:
        def observe(self, arm, tx_time_s, reward):
            calls["observe"].append((arm, tx_time_s, reward))
        def select(self, now_s, rng=None):
            calls["select"].append(now_s)
            return 0

    env.run(Recorder(), T=200, rng=np.random.default_rng(3))
    # every observe arg is a primitive scalar, no arrays / channel objects
    for arm, tx, r in calls["observe"]:
        assert isinstance(arm, (int, np.integer))
        assert np.isscalar(tx) and np.isscalar(r)
    assert all(np.isscalar(x) for x in calls["select"])


def test_channel_too_short_raises():
    ch = make_channel(n=100)
    env = LinkAdaptationEnv(ch, GaussianReward(0.0), tau_rt_slots=10,
                            tau_ow_slots=5)
    with pytest.raises(ValueError):
        env.run(AgeSpyAgent(ch.cfg.K), T=100, rng=np.random.default_rng(0))
