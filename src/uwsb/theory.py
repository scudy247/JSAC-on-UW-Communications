"""Closed-form converse quantities (THEORY.md §3, Props 1-3).

Two-arm Gauss-Markov analysis. D(t) = mu_1(t) - mu_2(t) is OU with mean
Delta_bar = m_1 - m_2 >= 0 and stationary variance sigma_D^2. Writing
rho = rho(tau) = e^{-S} for the predictability coefficient (S = tau/Tc):

  Best causal per-step surplus over the best stationary policy (†):

     G(rho) = rho sigma_D phi(Delta_bar/(rho sigma_D))
              - Delta_bar Phi(-Delta_bar/(rho sigma_D))
            = E[(-M)^+],   M ~ N(Delta_bar, rho^2 sigma_D^2)

  It is the exact ceiling on how much ANY causal policy can beat stationary,
  and it is achieved by the delayed clairvoyant (Prop 2). The dynamic-oracle
  surplus is G(1). The relative gain G(rho)/G(1) is the phase-diagram field.

These are the curves the milestone-3 simulation must reproduce. Everything here
is deterministic; no randomness, no channel -- pure closed form for overlay and
for unit tests against the Prop-2 sanity checks.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

_SQRT2PI = np.sqrt(2.0 * np.pi)


def causal_surplus_over_stationary(delta_bar, sigma_D, rho):
    """G(rho) via (†). Vectorized over any broadcastable inputs.

    delta_bar >= 0, sigma_D > 0, rho in (0, 1]. Returns E[(-M)^+] with
    M ~ N(delta_bar, (rho sigma_D)^2).
    """
    delta_bar = np.asarray(delta_bar, dtype=float)
    sigma_D = np.asarray(sigma_D, dtype=float)
    rho = np.asarray(rho, dtype=float)
    s = rho * sigma_D                       # predictable std of the gap
    # Guard s -> 0 (rho -> 0 or sigma_D -> 0): surplus -> 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(s > 0, delta_bar / s, np.inf)
        g = s * norm.pdf(z) - delta_bar * norm.cdf(-z)
    return np.where(s > 0, np.maximum(g, 0.0), 0.0)


def relative_gain(delta_bar, sigma_D, rho):
    """G(rho)/G(1): fraction of the dynamic-oracle tracking surplus that a
    causal policy can still capture at staleness rho. In [0, 1].

    Closed-form limits (THEORY.md Prop 2, used as unit tests):
      rho = 1            -> 1
      delta_bar = 0      -> rho          (the delay tax, Prop 1)
      rho -> 0           -> 0
    """
    num = causal_surplus_over_stationary(delta_bar, sigma_D, rho)
    den = causal_surplus_over_stationary(delta_bar, sigma_D, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0, num / den, 0.0)
    return np.clip(r, 0.0, 1.0)


def relative_gain_uS(u, S):
    """Relative gain as a function of u = Delta_bar/sigma_D and S = tau/Tc.

    sigma_D cancels in the ratio, so the phase field depends only on (u, S).
    """
    u = np.asarray(u, dtype=float)
    rho = np.exp(-np.asarray(S, dtype=float))
    return relative_gain(u, 1.0, rho)


# --- the collapse variable and the exact phase boundary ---------------------
# The relative-gain field collapses onto the single dimensionless variable
#     z = Delta_bar / (rho sigma_D) = u * e^{S}
# i.e. the ratio of the mean gap to the *predictable* fluctuation scale rho*sigma_D
# (THEORY.md Prop 3: collapse once Delta_bar >~ rho sigma_D, i.e. z >~ 1). Writing
# psi(z) = h(z)/h(0) with h(x) = phi(x) - x Phi(-x), the exact large-S limit is
#     relative_gain(u, S) * e^{S}  ->  psi(z),           z = u e^{S},
# so the *staleness-compensated* gain depends on z alone. Contours of constant z
# are therefore  ln u = ln z - S : straight lines of slope -1 in the
# (ln(Delta_bar/sigma_D), S) plane -- EXACTLY, no O(1) drift. (Contours of the RAW
# gain drift, because they mix the e^{-S} prefactor with psi(z); that is the O(1)
# term in Prop 3. The compensated gain / z is the clean object.)

def collapse_variable(u, S):
    """z = Delta_bar/(rho sigma_D) = u * e^{S}. Field depends ~only on this."""
    return np.asarray(u, dtype=float) * np.exp(np.asarray(S, dtype=float))


def compensated_gain(u, S):
    """relative_gain(u, S) * e^{S}. -> psi(z) as S grows; in (0, 1]."""
    return relative_gain_uS(u, S) * np.exp(np.asarray(S, dtype=float))


def psi(z):
    """Limiting compensated-gain profile psi(z) = h(z)/h(0)."""
    z = np.asarray(z, dtype=float)
    h = norm.pdf(z) - z * norm.cdf(-z)
    return h / norm.pdf(0.0)


def phase_boundary_u(level: float, S: float,
                     u_lo: float = 1e-9, u_hi: float = 50.0) -> float:
    """Boundary u*(S): the gap at which the compensated gain equals `level`.

    compensated_gain is monotone decreasing in u, so the root is unique. As S
    grows, ln u*(S) -> ln z(level) - S (slope -1). `level` in (0, 1); level=1 is
    the degenerate z=0 boundary (symmetric arms).
    """
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0,1)")
    f = lambda u: float(compensated_gain(u, S)) - level
    if f(u_lo) < 0:
        return u_lo
    if f(u_hi) > 0:
        return u_hi
    return brentq(f, u_lo, u_hi, xtol=1e-12, rtol=1e-12)


def phase_boundary_curve(level: float, S_grid) -> np.ndarray:
    """u*(S) over a grid of S. Slope of ln(u*) vs S -> -1 (THEORY.md Prop 3)."""
    return np.array([phase_boundary_u(level, float(S)) for S in S_grid])
