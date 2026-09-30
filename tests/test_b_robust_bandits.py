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


def test_median_of_means_matches_lemma2_and_its_coverage():
    from uwsb.bandits.robust import median_of_means, mom_width
    x = np.arange(1.0, 21.0)                                             # n = 20
    lt = math.log(math.exp(0.125) / 0.5)                                 # delta = 0.5 -> 8*lt = 6.54 -> k = 6
    k, N = 6, 3
    assert median_of_means(x, lt) == pytest.approx(np.median(x[: k * N].reshape(k, N).mean(axis=1)))
    alpha, c, eps, delta, n, reps = 1.5, 1.0, 0.4, 0.05, 400, 2000
    v = sas_abs_moment(1 + eps, alpha, c)                                # centred moment, no mean needed
    lt = math.log(math.exp(0.125) / delta)
    w = mom_width(v, eps, lt, n)
    rng = np.random.default_rng(9)
    up = sum(median_of_means(5.0 + sas_real(alpha, c, n, rng), lt) > 5.0 + w for _ in range(reps))
    assert up / reps <= delta                                            # holds even with a large mean


def test_median_of_means_ucb_learns():
    from uwsb.bandits.robust import MedianOfMeansUCB
    alpha, c, eps = 1.5, 1.0, 0.4
    tab = _sas_table(6000, [5.0, 6.0], alpha, c, seed=10)                # large means: MoM is centred
    agent = MedianOfMeansUCB(2, eps=eps, v=sas_abs_moment(1 + eps, alpha, c))
    res = TableEnv(tab, tau_rt_slots=2).run(agent)
    best = res.arms == 1
    assert best[-2000:].mean() > best[:2000].mean() and best.mean() > 0.5


# --- AdaR-UCB ---------------------------------------------------------------------------------
def test_adar_threshold_solves_eq13_and_follows_prop9():
    from uwsb.bandits.robust import ADAR_C, adar_threshold
    xp = sas_real(1.5, 1.0, 500, np.random.default_rng(11))
    L = math.log(1 / 0.05)
    M = adar_threshold(xp, L)
    assert np.sum(np.minimum(xp ** 2 / M ** 2, 1.0)) == pytest.approx(ADAR_C * L, rel=1e-9)
    few = np.zeros(100)
    few[:int(ADAR_C * L)] = 1.0                                   # nonzero count <= c log(1/delta)
    assert adar_threshold(few, L) is None


def test_adar_index_is_line9_and_arm_played_twice():
    from uwsb.bandits.robust import AdaRUCB, adar_threshold
    tab = _sas_table(400, [-1.0, -0.2], 1.5, 1.0, seed=12)
    agent = AdaRUCB(2)
    res = TableEnv(tab, tau_rt_slots=1).run(agent)
    assert np.all(res.arms[0::2] == res.arms[1::2])               # pairs: played twice per round
    L = 3 * math.log(agent.tau)
    B = agent.index()
    for i in range(2):
        x = np.asarray(agent.X[i]); M = adar_threshold(agent.Xp[i], L)
        if M is None:
            assert np.isinf(B[i]); continue
        y = np.where(np.abs(x) <= M, x, 0.0); N = x.size
        manual = y.mean() + math.sqrt(2 * np.var(y, ddof=1) * L / N) + 10 * M * L / N
        assert B[i] == pytest.approx(manual, rel=1e-12)


def test_adar_learns_under_its_assumption_1():
    from uwsb.bandits.robust import AdaRUCB
    # symmetric noise around NEGATIVE means: E[X 1{|X|>M}] <= 0 for the optimal arm (Assumption 1)
    tab = _sas_table(8000, [-1.5, -0.5], 1.5, 1.0, seed=13)
    res = TableEnv(tab, tau_rt_slots=1).run(AdaRUCB(2))
    best = res.arms == 1
    assert best[-2000:].mean() > best[:2000].mean() and best.mean() > 0.5


# --- NIR-UCB v0 --------------------------------------------------------------------------------
def test_nir_eps_rule_is_the_brute_force_minimiser():
    from uwsb.bandits.robust import NIRUCB
    agent = NIRUCB(2)
    L, n = 2 * math.log(1000), 300
    eps, u = agent.choose_eps(1.5, 1.0, n, L)
    grid = np.linspace(0.001, 0.499, 2000)
    w = [4 * sas_abs_moment(1 + e, 1.5, 1.0) ** (1 / (1 + e)) * (L / n) ** (e / (1 + e)) for e in grid]
    assert eps == pytest.approx(grid[int(np.argmin(w))], abs=0.03)
    assert 0 < eps < 0.5 and u == pytest.approx(sas_abs_moment(1 + eps, 1.5, 1.0))
    assert agent.choose_eps(2.0, 1.0, n, L) == (1.0, 2.0)          # Gaussian: eps = 1, u = 2c^2


def _nir_table(T, means, alpha, c, seed, nu=200):
    rng = np.random.default_rng(seed)
    g = np.asarray(means) + sas_real(alpha, c, (T, len(means)), rng)
    noise = sas_real(alpha, c, (T, nu), rng)                          # same law: Layer-1 assumption
    return OutcomeTable(fields={"gamma_db": g}, truth_mean=np.tile(means, (T, 1)), noise=noise)


def test_nir_round_robin_before_side_information_then_learns():
    from uwsb.bandits.robust import NIRUCB
    tab = _nir_table(6000, [5.0, 6.0], 1.5, 1.0, seed=14)            # large means: shift needed
    agent = NIRUCB(2, min_noise_samples=2000)
    res = TableEnv(tab, tau_rt_slots=3).run(agent)
    assert list(res.arms[:10]) == [0, 1] * 5                          # no side information yet
    assert agent.state["alpha"] == pytest.approx(1.5, abs=0.05)       # noise state learned
    best = res.arms == 1
    assert best[-2000:].mean() > best[:2000].mean() and best.mean() > 0.5


def test_nir_regime_alarm_inflates_widths():
    from uwsb.bandits.robust import NIRUCB
    agent = NIRUCB(2, inflate=3.0, inflate_window=50)
    rng = np.random.default_rng(15)
    for _ in range(40):
        agent.observe_noise(0.0, sas_real(1.8, 1.0, 1000, rng))
    for k in range(2):
        for v in np.random.default_rng(16 + k).standard_normal(50):
            agent.observe(k, 0.0, {"gamma_db": v})
    agent.t = 100
    base = agent.index()
    agent.observe_noise(0.0, sas_real(1.8, 10.0, 1000, rng))          # 20 dB level jump -> alarm
    assert agent.state["regime_change"]
    assert np.all(agent.index() > base)                                # inflated widths


def test_adar_exact_threshold_equals_bisection():
    from uwsb.bandits.robust import ADAR_C, _adar_threshold_bisect, adar_threshold
    rng = np.random.default_rng(17)
    for n in (30, 300, 3000):
        xp = sas_real(1.4, 2.0, n, rng)
        xp[: n // 10] = 0.0                                             # zeros are allowed
        L = 3 * math.log(50)
        exact = adar_threshold(xp, L)
        if np.count_nonzero(xp) <= ADAR_C * L:                          # Prop. 9: no root
            assert exact is None
            with pytest.raises(ValueError):
                _adar_threshold_bisect(np.sort(xp ** 2), ADAR_C * L)
            continue
        ref = _adar_threshold_bisect(np.sort(xp ** 2), ADAR_C * L)
        assert exact == pytest.approx(ref, rel=1e-9)
    assert exact is not None                                            # n = 3000 does have a root
