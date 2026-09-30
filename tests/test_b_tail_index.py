import numpy as np
import pytest

from uwsb.estimation.tail_index import (
    bootstrap_ci, hill, hill_plot, log_moment_fit, mean_excess, stable_quantile_fit,
)
from uwsb.noise.stable import sas_real

N = 200_000


@pytest.mark.parametrize("alpha", [1.2, 1.5, 1.8])
def test_quantile_and_log_moment_recover_alpha_and_scale(alpha):
    c = 2.5
    x = sas_real(alpha, c, N, np.random.default_rng(1))
    a_q, c_q = stable_quantile_fit(x)
    a_l, c_l = log_moment_fit(x)
    assert a_q == pytest.approx(alpha, abs=0.05) and c_q == pytest.approx(c, rel=0.05)
    assert a_l == pytest.approx(alpha, abs=0.05) and c_l == pytest.approx(c, rel=0.05)


def test_gaussian_and_cauchy_limits():
    g = np.random.default_rng(2).standard_normal(N) * np.sqrt(2.0)       # SaS(2, c=1)
    a_q, c_q = stable_quantile_fit(g)
    a_l, _ = log_moment_fit(g)
    assert a_q == pytest.approx(2.0, abs=0.05) and c_q == pytest.approx(1.0, rel=0.05)
    assert a_l == pytest.approx(2.0, abs=0.05)
    cau = np.random.default_rng(3).standard_cauchy(N)
    assert stable_quantile_fit(cau)[0] == pytest.approx(1.0, abs=0.05)
    assert log_moment_fit(cau)[0] == pytest.approx(1.0, abs=0.05)


def test_hill_exact_on_pareto_and_consistent_on_stable():
    a = 1.7
    p = (1.0 - np.random.default_rng(4).random(1_000_000)) ** (-1.0 / a)
    assert hill(p, 5_000) == pytest.approx(a, rel=0.05)
    x = sas_real(1.5, 1.0, 1_000_000, np.random.default_rng(5))
    assert hill(x, 2_000) == pytest.approx(1.5, abs=0.15)             # top 0.2 %
    hp = hill_plot(x, [500, 1_000, 2_000])
    assert hp[2] == pytest.approx(hill(x, 2_000))


def test_mean_excess_flags_exponential_where_stable_fit_says_alpha_below_two():
    """Documents the false-GO risk (evidence pack §1): a stable fit on Laplace data
    (exponential tails, all moments finite) returns alpha < 2, while mean excess is flat."""
    lap = np.random.default_rng(6).laplace(size=N)
    assert stable_quantile_fit(lap)[0] < 1.9
    e = [v for _, v in mean_excess(np.abs(lap))]
    assert max(e) / min(e) < 1.15                                        # flat -> exponential
    par = (1.0 - np.random.default_rng(7).random(N)) ** (-1.0 / 1.7)
    ep = [v for _, v in mean_excess(par, qs=(0.9, 0.99))]
    # power law: e(u) = u/(a-1), so e grows by the threshold ratio u99/u90 = 10^(1/a) (3.87 at a=1.7)
    assert ep[1] / ep[0] == pytest.approx(10 ** (1 / 1.7), rel=0.1)


def test_bootstrap_ci_covers_and_shrinks():
    x_small = sas_real(1.5, 1.0, 5_000, np.random.default_rng(8))
    x_big = sas_real(1.5, 1.0, 50_000, np.random.default_rng(8))
    f = lambda z: stable_quantile_fit(z)[0]
    lo_s, hi_s = bootstrap_ci(f, x_small, np.random.default_rng(9), n_boot=100)
    lo_b, hi_b = bootstrap_ci(f, x_big, np.random.default_rng(9), n_boot=100)
    assert lo_s <= 1.5 <= hi_s and lo_b <= 1.5 <= hi_b
    assert (hi_b - lo_b) < 0.5 * (hi_s - lo_s)


def test_validation():
    with pytest.raises(ValueError):
        stable_quantile_fit(np.ones(10))
    with pytest.raises(ValueError):
        hill(np.arange(1.0, 200.0), 0)
    with pytest.raises(ValueError):
        log_moment_fit(np.array([np.nan] * 200))
    with pytest.raises(TypeError):
        bootstrap_ci(np.mean, np.arange(200.0), np.random.RandomState(0))
