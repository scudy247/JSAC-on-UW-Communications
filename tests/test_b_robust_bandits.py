import math

import numpy as np
import pytest
from scipy.special import gamma as G

from uwsb.bandits.robust import (
    AckUCB1, BernoulliTS, EmpiricalUCB, GaussianTS, TruncatedMeanUCB,
    bcl_width, sas_abs_moment, truncated_mean,
)
from uwsb.envs.table_env import OutcomeTable, TableEnv
from uwsb.noise.stable import sas_real


def test_sas_abs_moment_closed_forms_and_monte_carlo():
    c = 0.7
    gauss = (math.sqrt(2) * c) ** 1.5 * 2 ** 0.75 * G(1.25) / math.sqrt(math.pi)   # N(0, 2c^2)
    assert sas_abs_moment(1.5, 2.0, c) == pytest.approx(gauss, rel=1e-12)
    assert sas_abs_moment(0.5, 1.0, 1.0) == pytest.approx(1 / math.cos(math.pi / 4), rel=1e-12)
    x = sas_real(1.5, 1.3, 2_000_000, np.random.default_rng(1))
    assert np.mean(np.abs(x) ** 0.5) == pytest.approx(sas_abs_moment(0.5, 1.5, 1.3), rel=0.005)
    with pytest.raises(ValueError):
        sas_abs_moment(1.6, 1.5, 1.0)                                  # p >= alpha: infinite


def test_truncated_mean_respects_bcl_lemma1_coverage():
    alpha, c, mu = 1.5, 1.0, 0.3
    eps = 0.4                                                           # 1 + eps = 1.4 < alpha
    u = 2 ** eps * (abs(mu) ** (1 + eps) + sas_abs_moment(1 + eps, alpha, c))   # c_r bound, raw moment
    delta, n, reps = 0.05, 400, 2000
    L = math.log(1 / delta)
    w = bcl_width(u, eps, L, n)
    rng = np.random.default_rng(2)
    up = down = 0
    for _ in range(reps):
        x = mu + sas_real(alpha, c, n, rng)
        m = truncated_mean(x, u, eps, L)
        up += m > mu + w
        down += m < mu - w
    assert up / reps <= delta and down / reps <= delta                  # one-sided, each <= delta


def _bern_table(T, means, seed):
    rng = np.random.default_rng(seed)
    means = np.asarray(means)
    ack = (rng.random((T, means.size)) < means).astype(float)
    return OutcomeTable(fields={"ack": ack}, truth_mean=np.tile(means, (T, 1)))


def _sas_table(T, means, alpha, c, seed):
    rng = np.random.default_rng(seed)
    means = np.asarray(means)
    g = means + sas_real(alpha, c, (T, means.size), rng)
    return OutcomeTable(fields={"gamma_db": g}, truth_mean=np.tile(means, (T, 1)))


@pytest.mark.parametrize("make", [lambda: AckUCB1(2), lambda: BernoulliTS(2, rng=np.random.default_rng(3))])
def test_ack_agents_learn_the_best_arm_logarithmically(make):
    T, gap = 5000, 0.1
    tab = _bern_table(T, [0.5, 0.6], seed=4)
    res = TableEnv(tab, tau_rt_slots=3).run(make())
    bad = res.arms == 0
    # Auer, Cesa-Bianchi, Fischer (2002), Thm 1 (UCB1): E[T_bad] <= 8 ln T / gap^2 + 1 + pi^2/3
    assert bad.sum() <= 8 * math.log(T) / gap ** 2 + 1 + math.pi ** 2 / 3
    assert bad[-1000:].mean() < bad[:1000].mean()                       # learning: share decreases


def test_gaussian_agents_work_on_gaussian_rewards():
    tab = OutcomeTable(fields={"gamma_db": np.array([0.0, 0.5]) + np.random.default_rng(5).standard_normal((4000, 2))},
                       truth_mean=np.tile([0.0, 0.5], (4000, 1)))
    for agent in (EmpiricalUCB(2, sigma=1.0), GaussianTS(2, sigma=1.0, rng=np.random.default_rng(6))):
        res = TableEnv(tab, tau_rt_slots=5).run(agent)
        assert np.mean(res.arms[-1000:] == 1) > 0.9


def test_truncated_mean_ucb_finds_best_arm_under_heavy_tails():
    alpha, c, means = 1.5, 1.0, [0.0, 1.0]
    eps = 0.4
    u = 2 ** eps * (1.0 + sas_abs_moment(1 + eps, alpha, c))
    tab = _sas_table(6000, means, alpha, c, seed=7)
    agent = TruncatedMeanUCB(2, eps=eps, u=u)
    res = TableEnv(tab, tau_rt_slots=2).run(agent)
    best = res.arms == 1
    # BCL's worst-case width is conservative (6.5x the gap after 1000 pulls here): check
    # learning, not speed, and check the index is exactly Lemma 1 + Prop. 1
    assert best[-2000:].mean() > best[:2000].mean() and best.mean() > 0.5
    L = 2 * math.log(agent.t)
    manual = [truncated_mean(agent.x[a], u, eps, L) + bcl_width(u, eps, L, agent.n[a]) for a in (0, 1)]
    np.testing.assert_allclose(agent.index(), manual, rtol=1e-12)
    j = np.arange(1, 6)
    assert truncated_mean([1.0, -50.0, 2.0, 0.5, 100.0], 4.0, 1.0, 2.0) == pytest.approx(
        np.sum(np.where(np.abs([1.0, -50.0, 2.0, 0.5, 100.0]) <= np.sqrt(4.0 * j / 2.0),
                        [1.0, -50.0, 2.0, 0.5, 100.0], 0.0)) / 5)


def test_delay_safe_initialisation_and_validation():
    tab = _bern_table(50, [0.2, 0.8, 0.5], seed=8)
    agent = AckUCB1(3)
    res = TableEnv(tab, tau_rt_slots=10).run(agent)
    assert set(res.arms[:3]) == {0, 1, 2}                               # each arm once before feedback
    with pytest.raises(ValueError):
        TruncatedMeanUCB(2, eps=1.5)
    with pytest.raises(TypeError):
        GaussianTS(2, rng=np.random.RandomState(0))
    with pytest.raises(ValueError):
        AckUCB1(1)
