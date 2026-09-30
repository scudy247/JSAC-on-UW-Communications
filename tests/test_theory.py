"""Unit tests for the closed-form converse (THEORY.md Prop 2 sanity checks)."""
import numpy as np
import pytest

from uwsb.theory import (
    causal_surplus_over_stationary, relative_gain, relative_gain_uS,
    phase_boundary_u, phase_boundary_curve, compensated_gain, collapse_variable,
    psi,
)


def test_rho_one_gives_unit_gain():
    for u in (0.0, 0.5, 2.0, 5.0):
        assert relative_gain(u, 1.0, 1.0) == pytest.approx(1.0)


def test_symmetric_arms_gain_equals_rho():
    """Delta_bar = 0 => relative gain = rho (Prop 1, the delay tax)."""
    for rho in (0.9, 0.6, 0.3, 0.1):
        assert relative_gain(0.0, 1.0, rho) == pytest.approx(rho, rel=1e-9)


def test_infinite_staleness_gives_zero_gain():
    for u in (0.0, 1.0, 3.0):
        assert relative_gain(u, 1.0, 1e-9) == pytest.approx(0.0, abs=1e-6)


def test_dynamic_surplus_matches_halfnormal_at_delta_zero():
    """G(1) at Delta_bar=0 is sigma_D * phi(0) = sigma_D / sqrt(2 pi)."""
    sigma_D = 1.4
    g1 = causal_surplus_over_stationary(0.0, sigma_D, 1.0)
    assert g1 == pytest.approx(sigma_D / np.sqrt(2 * np.pi), rel=1e-9)


def test_relative_gain_monotone_decreasing_in_u():
    S = 0.7
    us = np.linspace(0.01, 8.0, 60)
    r = relative_gain_uS(us, S)
    assert np.all(np.diff(r) <= 1e-12)


def test_relative_gain_decreasing_in_S():
    u = 0.5
    Ss = np.linspace(0.05, 4.0, 50)
    r = relative_gain_uS(u, Ss)
    assert np.all(np.diff(r) <= 1e-12)


def test_field_collapses_onto_z():
    """The compensated gain r*e^S -> psi(z), z = u e^S, as S grows (THEORY.md
    Prop 3 collapse). Pairs sharing z converge to the same value."""
    for z in (0.5, 1.0, 2.0):
        vals = [float(compensated_gain(z * np.exp(-S), S)) for S in (3, 4, 5, 6)]
        # converged and equal to psi(z)
        assert vals[-1] == pytest.approx(float(psi(z)), abs=2e-3)
        assert max(vals[-2:]) - min(vals[-2:]) < 5e-3     # collapsed


def test_compensated_boundary_slope_is_minus_one():
    """THEORY.md Prop 3 / §8 prediction 2 (corrected): contours of constant
    compensated gain (equivalently constant z) are slope -1 in (ln u, S)."""
    S_grid = np.linspace(3.0, 8.0, 25)     # large-S limit where the collapse is tight
    for level in (0.5, 0.25, 0.1):
        u_star = phase_boundary_curve(level, S_grid)
        slope = np.polyfit(S_grid, np.log(u_star), 1)[0]
        assert slope == pytest.approx(-1.0, abs=0.02)


def test_boundary_root_satisfies_equation():
    for level in (0.3, 0.6):
        for S in (1.0, 2.0, 3.0):
            u = phase_boundary_u(level, S)
            assert float(compensated_gain(u, S)) == pytest.approx(level, abs=1e-6)


def test_collapse_variable_is_u_exp_S():
    assert collapse_variable(0.2, 1.5) == pytest.approx(0.2 * np.exp(1.5))
    assert float(psi(0.0)) == pytest.approx(1.0)         # z=0 -> full gain


def test_surplus_vectorizes():
    u = np.array([0.0, 0.5, 1.0])
    rho = np.array([1.0, 0.5, 0.2])
    out = causal_surplus_over_stationary(u, 1.0, rho)
    assert out.shape == (3,)
    assert np.all(out >= 0)
