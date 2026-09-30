"""Batch tail estimators (Proposal B; plan §3 layout, THEORY-B D-B3 and T4).

For i.i.d. symmetric alpha-stable data with CF exp(-|c theta|^alpha):
- hill(x, k): Hill estimate of the tail index of |x| from the k largest values;
- stable_quantile_fit(x): McCulloch-type quantile estimator of (alpha, c). The lookup table
  nu(alpha) = (q95 - q05) / (q75 - q25) and the scale factor q75 - q25 are COMPUTED from
  scipy.stats.levy_stable.ppf (beta = 0, unit scale) on an alpha grid, not copied from
  McCulloch (1986), which is not in literature/; nu is monotone in alpha and inverted by
  interpolation;
- log_moment_fit(x): alpha from Var(log|X|) = (pi^2/6)(1/alpha^2 + 1/2) (checked against the
  Gaussian, pi^2/8, and Cauchy, pi^2/4, closed forms), c from E[log|X|] = gamma(1/alpha - 1) + log c;
- mean_excess(x, qs): e(u) = E[X - u | X > u]: flat for exponential tails, growing ~ u/(a-1) for
  power laws (the tail-TYPE diagnostic; the alpha estimators above assume a stable law and
  return alpha < 2 on exponential-tailed data, see the tests);
- bootstrap_ci(estimator, x, rng, ...): percentile bootstrap interval.
alpha estimates are clipped to (0, 2].
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.stats import levy_stable

EULER_GAMMA = 0.5772156649015329
_ALPHA_GRID = np.round(np.arange(0.6, 2.0001, 0.02), 4)


@lru_cache(maxsize=1)
def _quantile_table():
    q = np.array([levy_stable.ppf([0.05, 0.25, 0.75, 0.95], a, 0.0) for a in _ALPHA_GRID])
    nu = (q[:, 3] - q[:, 0]) / (q[:, 2] - q[:, 1])
    iqr = q[:, 2] - q[:, 1]
    if not np.all(np.diff(nu) < 0):
        raise RuntimeError("quantile-ratio table is not monotone in alpha")
    return nu, iqr


def _check(x, n_min=100):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < n_min:
        raise ValueError(f"x must be 1-D with at least {n_min} samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("x contains non-finite values")
    return x


def hill(x, k: int) -> float:
    """Hill estimate of the tail index of |x| using the k largest order statistics."""
    a = np.sort(np.abs(_check(x)))[::-1]
    if not 1 <= k < a.size:
        raise ValueError(f"k must be in [1, {a.size - 1}]")
    if a[k] <= 0:
        raise ValueError("the (k+1)-th largest |x| is zero")
    return float(1.0 / np.mean(np.log(a[:k] / a[k])))


def hill_plot(x, ks) -> np.ndarray:
    """Hill estimates for each k in ks (for the Hill plot)."""
    a = np.sort(np.abs(_check(x)))[::-1]
    logs = np.log(a)
    csum = np.cumsum(logs)
    ks = np.asarray(ks, dtype=int)
    if ks.min() < 1 or ks.max() >= a.size:
        raise ValueError(f"ks must be in [1, {a.size - 1}]")
    return 1.0 / (csum[ks - 1] / ks - logs[ks])


def stable_quantile_fit(x) -> tuple[float, float]:
    """(alpha, c) for symmetric stable data from sample quantiles (McCulloch-type)."""
    x = _check(x)
    q05, q25, q75, q95 = np.quantile(x, [0.05, 0.25, 0.75, 0.95])
    nu_hat = (q95 - q05) / (q75 - q25)
    nu, iqr = _quantile_table()
    # nu decreases with alpha: interpolate on reversed arrays; outside the grid -> clip
    alpha = float(np.interp(nu_hat, nu[::-1], _ALPHA_GRID[::-1]))
    c = float((q75 - q25) / np.interp(alpha, _ALPHA_GRID, iqr))
    return alpha, c


def log_moment_fit(x) -> tuple[float, float]:
    """(alpha, c) for symmetric stable data from the first two moments of log|x|."""
    x = _check(x)
    x = x[x != 0]
    u = np.log(np.abs(x))
    v = float(np.var(u))
    inv_a2 = 6.0 * v / np.pi ** 2 - 0.5
    alpha = 2.0 if inv_a2 <= 0.25 else float(min(2.0, inv_a2 ** -0.5))
    c = float(np.exp(np.mean(u) - EULER_GAMMA * (1.0 / alpha - 1.0)))
    return alpha, c


def mean_excess(x, qs=(0.9, 0.99, 0.999)) -> list[tuple[float, float]]:
    """[(u, e(u))] at thresholds u = empirical quantiles qs of x."""
    x = _check(x, n_min=10)
    out = []
    for q in qs:
        u = float(np.quantile(x, q))
        tail = x[x > u]
        out.append((u, float(np.mean(tail - u)) if tail.size else float("nan")))
    return out


def bootstrap_ci(estimator, x, rng: np.random.Generator, n_boot: int = 200,
                 level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval of a scalar estimator(x)."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator (explicit seeding)")
    x = _check(x)
    vals = np.array([estimator(x[rng.integers(0, x.size, x.size)]) for _ in range(n_boot)])
    lo, hi = np.quantile(vals, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)
