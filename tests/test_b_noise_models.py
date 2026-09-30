import numpy as np
import pytest

from uwsb.noise.class_a import class_a_complex, excess_kurtosis_per_component
from uwsb.noise.stable import positive_stable, sas_complex_isotropic, sas_real

N = 400_000


def _cf(x, theta):
    return float(np.mean(np.cos(theta * x)))        # symmetric law: CF is real


@pytest.mark.parametrize("alpha", [0.8, 1.0, 1.5, 1.7, 2.0])
def test_sas_real_characteristic_function(alpha):
    c = 0.7
    x = sas_real(alpha, c, N, np.random.default_rng(1))
    for theta in (0.5, 1.0, 2.0):
        assert _cf(x, theta) == pytest.approx(np.exp(-abs(c * theta) ** alpha), abs=0.01)


@pytest.mark.parametrize("alpha", [1.2, 1.5, 1.8])
def test_sas_real_tail_slope(alpha):
    x = np.abs(sas_real(alpha, 1.0, 2_000_000, np.random.default_rng(2)))
    q = np.quantile(x, 0.995)
    slope = np.log10(np.mean(x > q) / np.mean(x > 10 * q))   # P(|X| > x) ~ x^-alpha
    assert slope == pytest.approx(alpha, abs=0.12)


@pytest.mark.parametrize("a", [0.3, 0.6, 0.85])
def test_positive_stable_laplace_transform(a):
    s_ = positive_stable(a, N, np.random.default_rng(3))
    assert np.all(s_ > 0)
    for s in (0.5, 1.0, 2.0):
        assert np.mean(np.exp(-s * s_)) == pytest.approx(np.exp(-s ** a), abs=0.01)


@pytest.mark.parametrize("alpha", [1.2, 1.7])
def test_sas_complex_components_and_isotropy(alpha):
    c = 0.5
    z = sas_complex_isotropic(alpha, c, N, np.random.default_rng(4))
    target = np.exp(-(c * 1.0) ** alpha)
    for comp in (z.real, z.imag, (z * np.exp(1j * 0.7)).real):   # rotated projection too
        assert _cf(comp, 1.0) == pytest.approx(target, abs=0.01)


def test_sas_complex_alpha_two_is_gaussian_with_variance_2c2():
    c = 1.0 / np.sqrt(2.0)                         # library convention: unit variance
    z = sas_complex_isotropic(2.0, c, N, np.random.default_rng(5))
    assert np.var(z.real) == pytest.approx(1.0, rel=0.01)
    assert np.var(z.imag) == pytest.approx(1.0, rel=0.01)


def test_class_a_power_impulse_fraction_and_kurtosis():
    A, Gamma = 0.1, 0.01
    n, m = class_a_complex(A, Gamma, 2_000_000, np.random.default_rng(6), return_states=True)
    assert np.mean(np.abs(n) ** 2) == pytest.approx(1.0, rel=0.03)
    assert np.mean(m >= 1) == pytest.approx(1 - np.exp(-A), rel=0.02)
    x = n.real
    kurt = np.mean(x ** 4) / np.mean(x ** 2) ** 2 - 3
    assert kurt == pytest.approx(excess_kurtosis_per_component(A, Gamma), rel=0.1)


def test_class_a_large_index_is_gaussian():
    n = class_a_complex(50.0, 0.5, N, np.random.default_rng(7))
    x = n.real
    assert np.mean(x ** 4) / np.mean(x ** 2) ** 2 - 3 == pytest.approx(0.0, abs=0.05)


def test_generators_are_reproducible_and_seed_sensitive():
    for f in (lambda r: sas_real(1.5, 1.0, 1000, r),
              lambda r: sas_complex_isotropic(1.5, 1.0, 1000, r),
              lambda r: class_a_complex(0.1, 0.01, 1000, r)):
        assert np.array_equal(f(np.random.default_rng(9)), f(np.random.default_rng(9)))
        assert not np.array_equal(f(np.random.default_rng(9)), f(np.random.default_rng(10)))


def test_generators_validate_inputs():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sas_real(2.5, 1.0, 10, rng)
    with pytest.raises(ValueError):
        positive_stable(1.0, 10, rng)
    with pytest.raises(ValueError):
        class_a_complex(0.0, 0.1, 10, rng)
    with pytest.raises(TypeError):
        sas_real(1.5, 1.0, 10, np.random.RandomState(0))      # legacy RNG rejected
