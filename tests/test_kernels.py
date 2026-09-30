import math

import numpy as np
import pytest

from uwsb.estimation.kernels import (
    ExponentialKernel, TwoExponentialKernel, coherence_time_from_R,
)


def test_exponential_defines_Tc_at_1_over_e():
    k = ExponentialKernel(Tc_s=7.0)
    assert k(0.0) == pytest.approx(1.0)
    assert k(7.0) == pytest.approx(math.exp(-1.0))       # R(Tc) = 1/e  (D1)
    assert k.Tc_half_s == pytest.approx(7.0 * math.log(2.0))
    assert k(k.Tc_half_s) == pytest.approx(0.5)           # comms 1/2 convention


def test_rho_star_equals_R_for_gauss_markov():
    k = ExponentialKernel(Tc_s=3.0)
    for tau in (0.5, 3.0, 9.0):
        assert k.rho_star(tau) == pytest.approx(float(k(tau)))


def test_coherence_time_inversion_recovers_Tc():
    k = ExponentialKernel(Tc_s=12.5)
    assert coherence_time_from_R(k, level=math.exp(-1.0)) == pytest.approx(12.5, rel=1e-4)


def test_two_exp_normalized_and_monotone():
    k = TwoExponentialKernel(Tf_s=1.0, Ts_s=20.0, w=0.5)
    assert k(0.0) == pytest.approx(1.0)
    xs = np.linspace(0, 50, 100)
    r = k(xs)
    assert np.all(np.diff(r) <= 1e-12)                    # monotone decreasing
    assert r[-1] < 0.2


def test_kernel_even_extension():
    k = ExponentialKernel(Tc_s=4.0)
    assert k(-3.0) == pytest.approx(k(3.0))


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        ExponentialKernel(Tc_s=0.0)
    with pytest.raises(ValueError):
        TwoExponentialKernel(Tf_s=1.0, Ts_s=2.0, w=1.5)
