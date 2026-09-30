import numpy as np
import pytest

from uwsb.envs.table_env import OutcomeTable, TableEnv

T_SLOTS, K = 60, 3


def _table(noise_nu=5, seed=0):
    rng = np.random.default_rng(seed)
    mean = np.tile([0.2, 0.5, 0.4], (T_SLOTS, 1))
    g = mean + 0.1 * rng.standard_normal((T_SLOTS, K))
    ack = (rng.random((T_SLOTS, K)) < mean).astype(float)
    noise = np.arange(T_SLOTS * noise_nu, dtype=float).reshape(T_SLOTS, noise_nu)
    return OutcomeTable(fields={"gamma_db": g, "ack": ack}, truth_mean=mean, noise=noise, T_slot_s=0.5)


class Spy:
    """Agent that records everything it is given; picks arms from a fixed script."""

    def __init__(self, script):
        self.script, self.now = script, None
        self.got, self.noise = [], []

    def select(self, now_s):
        self.now = now_s
        return self.script(int(round(now_s / 0.5)))

    def observe(self, arm, tx_time_s, reward):
        self.got.append((self.now, arm, tx_time_s, dict(reward)))

    def observe_noise(self, now_s, samples):
        self.noise.append((now_s, samples))


@pytest.mark.parametrize("tau", [0, 1, 7])
def test_rewards_arrive_exactly_after_the_round_trip(tau):
    tab = _table()
    spy = Spy(lambda t: t % K)
    res = TableEnv(tab, tau).run(spy)
    # a reward is usable at the earliest by the next decision (same convention as Proposal 0)
    assert res.n_delivered == T_SLOTS - max(tau, 1)
    assert res.min_delivered_age_slots == max(tau, 1)
    for now_prev, arm, tx_s, r in spy.got:
        # delivered at the start of slot tx + tau, i.e. before the select of that slot
        tx = int(round(tx_s / 0.5))
        assert arm == tx % K
        assert r == {"gamma_db": tab.fields["gamma_db"][tx, arm], "ack": tab.fields["ack"][tx, arm]}
    delivered_slots = sorted(int(round(tx / 0.5)) for _, _, tx, _ in spy.got)
    assert delivered_slots == list(range(T_SLOTS - max(tau, 1)))


def test_no_leakage_only_pulled_arm_fields_no_truth():
    tab = _table()
    spy = Spy(lambda t: 1)
    TableEnv(tab, 2).run(spy)
    for _, arm, _, r in spy.got:
        assert arm == 1 and set(r) == {"gamma_db", "ack"}          # no truth, no other arms


@pytest.mark.parametrize("ndelay", [0, 3])
def test_noise_stream_slices_and_delay(ndelay):
    tab = _table(noise_nu=4)
    spy = Spy(lambda t: 0)
    res = TableEnv(tab, 1, noise_delay_slots=ndelay).run(spy)
    assert res.n_noise_delivered == T_SLOTS - 1 - ndelay
    for now_s, samples in spy.noise:
        t = int(round(now_s / 0.5))
        np.testing.assert_array_equal(samples, tab.noise[t - 1 - ndelay])
    spy.noise[0][1][:] = -1.0                                        # agent gets a copy
    assert tab.noise[0, 0] == 0.0


def test_regret_oracle_zero_and_fixed_arm_linear():
    tab = _table()
    assert TableEnv(tab, 3).run(Spy(lambda t: 1)).regret == 0.0              # arm 1 is best
    r = TableEnv(tab, 3).run(Spy(lambda t: 0)).regret
    assert r == pytest.approx(0.3 * T_SLOTS)                                   # gap 0.3 each slot


def test_deterministic_and_validated():
    tab = _table()
    a = TableEnv(tab, 2).run(Spy(lambda t: t % 2))
    b = TableEnv(tab, 2).run(Spy(lambda t: t % 2))
    assert np.array_equal(a.arms, b.arms) and a.regret == b.regret
    with pytest.raises(ValueError):
        TableEnv(tab, -1)
    with pytest.raises(ValueError):
        TableEnv(tab, 1).run(Spy(lambda t: K))                           # invalid arm
    with pytest.raises(ValueError):
        OutcomeTable(fields={"g": np.zeros((5, 2))}, truth_mean=np.zeros((5, 3)))
    with pytest.raises(ValueError):
        TableEnv(tab, 1).run(Spy(lambda t: 0), T=T_SLOTS + 1)
