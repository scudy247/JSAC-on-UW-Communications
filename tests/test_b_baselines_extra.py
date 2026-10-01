"""Remaining plan §7 baselines: Catoni-UCB (BCL 2013 §2.3, Fig. 2), fixed clipping, median-filtered
greedy (threshold-AMC proxy), tuned-width naive UCB, best fixed arm / dynamic oracle."""
import math

import numpy as np
import pytest

from uwsb.bandits.robust import (
    CatoniUCB, ClippedUCB, MedianFilterGreedy, catoni_mean, catoni_psi,
)
from uwsb.envs.table_env import OutcomeTable, TableEnv, best_fixed_arm_regret


def _table(g, means):
    T = g.shape[0]
    return OutcomeTable(fields={"gamma_db": g}, truth_mean=np.tile(np.asarray(means, float), (T, 1)))


def test_catoni_psi_meets_bcl_bounds_and_is_increasing():
    x = np.linspace(-50, 50, 200_001)
    p = catoni_psi(x)
    assert np.all(-np.log(1 - x + x * x / 2) <= p + 1e-12)
    assert np.all(p <= np.log(1 + x + x * x / 2) + 1e-12)
    assert np.all(np.diff(p) > 0)


def test_catoni_mean_root_and_deviation_bound():
    # Student t with 3 dof: variance 3, infinite 4th moment; Catoni's bound mu_C <= mu + 2 sqrt(v L / n)
    rng = np.random.default_rng(11)
    v, n, delta, reps, mu = 3.0, 200, 0.05, 2000, 0.4
    L = math.log(1 / delta)
    assert n >= 4 * L
    w = 2 * math.sqrt(v * L / n)
    up = down = 0
    for _ in range(reps):
        x = mu + rng.standard_t(3, n)
        m = catoni_mean(x, v, L)
        up += m > mu + w
        down += m < mu - w
    assert up / reps <= delta and down / reps <= delta
    x = rng.standard_t(3, 50)
    a = math.sqrt(2 * L / (50 * (v + 2 * v * L / (50 - 2 * L))))
    assert abs(np.sum(catoni_psi(a * (x - catoni_mean(x, v, L))))) < 1e-8
    with pytest.raises(ValueError):
        catoni_mean(np.ones(5), v, L)                    # n <= 2 log(1/delta)


def test_catoni_ucb_index_is_fig2():
    ag = CatoniUCB(2, v=2.0)
    rng = np.random.default_rng(12)
    for _ in range(40):
        ag.observe(0, 0.0, {"gamma_db": float(rng.normal())})
    for _ in range(5):
        ag.observe(1, 0.0, {"gamma_db": float(rng.normal())})
    ag.t = 100                                           # 8 log 100 = 36.8: arm 0 eligible, arm 1 not
    B = ag.index()
    L = 2 * math.log(100)
    assert B[0] == pytest.approx(catoni_mean(ag.x[0], 2.0, L) + math.sqrt(4 * 2.0 * L / 40))
    assert B[1] == np.inf


def test_catoni_ucb_learns_with_finite_variance():
    rng = np.random.default_rng(13)
    T, means = 4000, [0.0, 1.0]
    g = np.asarray(means) + rng.standard_t(3, (T, 2))
    res = TableEnv(_table(g, means), 3).run(CatoniUCB(2, v=3.0))
    first, last = (res.arms[: T // 4] == 1).mean(), (res.arms[-T // 4:] == 1).mean()
    assert last > first and last > 0.9


def test_clipped_ucb_clips_and_uses_range_width():
    ag = ClippedUCB(2, lo=-1.0, hi=2.0)
    assert ag.sigma == 3.0
    ag.observe(0, 0.0, {"gamma_db": 50.0})
    ag.observe(1, 0.0, {"gamma_db": -50.0})
    assert ag.s.tolist() == [2.0, -1.0]
    with pytest.raises(ValueError):
        ClippedUCB(2, lo=1.0, hi=1.0)


def test_median_filter_greedy_ignores_isolated_outlier_and_slides():
    ag = MedianFilterGreedy(2, window=3)
    for r in (1.0, 1.0, 1.0):
        ag.observe(0, 0.0, {"gamma_db": r})
    for r in (0.0, 0.0, 99.0):                           # one huge outlier on arm 1
        ag.observe(1, 0.0, {"gamma_db": r})
    assert ag.select(0.0) == 0
    for r in (5.0, 5.0):                                 # window slides: arm 1's median becomes 5
        ag.observe(1, 0.0, {"gamma_db": r})
    assert ag.hist[1] == [99.0, 5.0, 5.0]
    assert ag.select(0.0) == 1


def test_best_fixed_arm_reference():
    T = 10
    stat = OutcomeTable(fields={"x": np.zeros((T, 2))}, truth_mean=np.tile([0.0, 1.0], (T, 1)))
    assert np.all(best_fixed_arm_regret(stat) == 0.0)
    tr = np.zeros((T, 2))
    tr[:7, 0], tr[7:, 1] = 1.0, 1.0                      # regime switch at slot 7: best fixed = arm 0
    sw = OutcomeTable(fields={"x": np.zeros((T, 2))}, truth_mean=tr)
    assert best_fixed_arm_regret(sw).tolist() == [0.0] * 7 + [1.0] * 3
