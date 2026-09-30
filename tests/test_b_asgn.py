import numpy as np
import pytest

from uwsb.noise.asgn import asgn_m, covariation_ratio

# Mahmood & Chitre (OCEANS 2015), Table I, dataset D1: alpha = 1.715, r_{1,1+k} / (2 delta^2)
ALPHA_D1 = 1.715
RHO_D1 = [1.0, 0.621, 0.237, 0.161, -0.054]


def test_m0_is_iid_sas():
    x = asgn_m(1.5, 0.8, [1.0], 200_000, np.random.default_rng(1))
    assert np.mean(np.cos(x)) == pytest.approx(np.exp(-0.8 ** 1.5), abs=0.01)


def test_marginal_is_sas_with_scale_delta():
    delta = 1.3
    x = asgn_m(ALPHA_D1, delta, RHO_D1, 150_000, np.random.default_rng(2))
    for theta in (0.3, 0.8):
        assert np.mean(np.cos(theta * x)) == pytest.approx(
            np.exp(-(delta * theta) ** ALPHA_D1), abs=0.012)


def test_recovers_memory_structure_with_papers_estimator():
    x = asgn_m(ALPHA_D1, 1.0, RHO_D1, 150_000, np.random.default_rng(3))
    for k in range(1, 5):
        assert covariation_ratio(x, k, p=1.2) == pytest.approx(RHO_D1[k], abs=0.04)
    # beyond the memory order the dependence need not vanish exactly, but the
    # estimator must at least return a finite value
    assert np.isfinite(covariation_ratio(x, 6, p=1.2))


def test_reproducible_seed_sensitive_and_validated():
    a = asgn_m(1.6, 1.0, [1.0, 0.5], 2000, np.random.default_rng(4))
    b = asgn_m(1.6, 1.0, [1.0, 0.5], 2000, np.random.default_rng(4))
    c = asgn_m(1.6, 1.0, [1.0, 0.5], 2000, np.random.default_rng(5))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    with pytest.raises(ValueError):
        asgn_m(2.0, 1.0, [1.0, 0.5], 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        asgn_m(1.5, 1.0, [1.0, 1.2], 10, np.random.default_rng(0))      # not positive definite
    with pytest.raises(TypeError):
        asgn_m(1.5, 1.0, [1.0, 0.5], 10, np.random.RandomState(0))


def test_regression_seed_1012_small_q_no_stall():
    """The prior-only sampler stalled here (q ~ 0); the tilted scheme must handle it."""
    x = asgn_m(ALPHA_D1, 1.0, RHO_D1, 100_000, np.random.default_rng(1012), max_tries=200_000)
    assert np.all(np.isfinite(x))


def _force(scheme, q, m, n, seed):
    from uwsb.noise.asgn import _Pool, _TiltedPool
    rng = np.random.default_rng(seed)
    a_half = ALPHA_D1 / 2
    pool, tilted = _Pool(a_half, rng), _TiltedPool(a_half, m, rng)
    out = np.empty(n)
    for i in range(n):
        if scheme == "prior":
            log_lmax = -0.5 * m * np.log(q / m) - 0.5 * m
            while True:
                a, u = pool.next()
                if np.log(u) <= -0.5 * m * np.log(a) - q / (2 * a) - log_lmax:
                    break
        else:
            while True:
                a, u = tilted.next()
                if u <= np.exp(-q / (2 * a)):
                    break
        out[i] = a
    return out


@pytest.mark.parametrize("q", [0.5, 2.0, 6.0])
def test_both_exact_schemes_sample_the_same_posterior(q):
    m = 4
    a_p = np.log(_force("prior", q, m, 20_000, 1))
    a_t = np.log(_force("tilted", q, m, 20_000, 2))
    for p in (0.1, 0.5, 0.9):
        assert np.quantile(a_p, p) == pytest.approx(np.quantile(a_t, p), abs=0.05)


def test_tilted_prior_matches_reweighted_prior_at_q0():
    from uwsb.noise.asgn import _TiltedPool
    from uwsb.noise.stable import positive_stable
    m, a_half = 4, ALPHA_D1 / 2
    tp = _TiltedPool(a_half, m, np.random.default_rng(3), size=200_000)
    lt = np.log(tp.draws)
    f = positive_stable(a_half, 2_000_000, np.random.default_rng(4))
    wts = f ** (-m / 2)
    assert np.mean(lt) == pytest.approx(np.sum(wts * np.log(f)) / np.sum(wts), abs=0.02)
