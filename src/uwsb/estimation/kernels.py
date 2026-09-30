"""Temporal correlation kernels R(dt).

Locked decision D1 (THEORY.md): the exponential kernel R(dt)=exp(-|dt|/Tc)
defines T_c as its own time constant, i.e. R(T_c)=1/e. The comms 1/2-convention
value is T_c_half = Tc * ln 2; expose it but never use it internally.

All kernels are normalized: R(0)=1, R monotone in |dt|, R(inf)=0. The bandit and
the theory touch the kernel only through R(.) and through the *predictability
coefficient* rho_star(tau) (THEORY.md Prop 4); for Gauss-Markov kernels
rho_star(tau) = R(tau), which is why a single scalar drives everything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LN2 = math.log(2.0)


class Kernel:
    """Base class. Subclasses implement R(dt) for dt >= 0 (even extension)."""

    def R(self, dt):
        raise NotImplementedError

    def __call__(self, dt):
        dt = np.abs(np.asarray(dt, dtype=float))
        return self.R(dt)


@dataclass(frozen=True)
class ExponentialKernel(Kernel):
    """R(dt) = exp(-|dt|/Tc). Tc = time constant (R(Tc)=1/e), THEORY.md D1."""

    Tc_s: float

    def __post_init__(self):
        if not self.Tc_s > 0:
            raise ValueError(f"Tc_s must be > 0, got {self.Tc_s}")

    def R(self, dt):
        return np.exp(-np.asarray(dt, dtype=float) / self.Tc_s)

    @property
    def Tc_half_s(self) -> float:
        """Coherence time under the R=1/2 convention: Tc * ln 2."""
        return self.Tc_s * LN2

    def rho_star(self, tau_s: float) -> float:
        """Predictability coefficient corr(x(t), E[x(t)|past<=t-tau]).

        For a scalar Gauss-Markov (OU) process this equals R(tau) exactly
        (THEORY.md Prop 4).
        """
        return float(self.R(tau_s))


@dataclass(frozen=True)
class TwoExponentialKernel(Kernel):
    """Two-timescale mixture w*exp(-dt/Tf) + (1-w)*exp(-dt/Ts), Tf < Ts.

    Fallback when a measured (Watermark) autocorrelation is not a single
    exponential (THEORY.md Prop 4 / caveat a). Still Gaussian, so the converse
    survives with rho replaced by rho_star(tau), computed here.
    """

    Tf_s: float
    Ts_s: float
    w: float

    def __post_init__(self):
        if not (0.0 <= self.w <= 1.0):
            raise ValueError(f"w must be in [0,1], got {self.w}")
        if not (self.Tf_s > 0 and self.Ts_s > 0):
            raise ValueError("Tf_s, Ts_s must be > 0")

    def R(self, dt):
        dt = np.asarray(dt, dtype=float)
        return self.w * np.exp(-dt / self.Tf_s) + (1.0 - self.w) * np.exp(-dt / self.Ts_s)

    def rho_star(self, tau_s: float) -> float:
        """corr of value with its best linear predictor from the infinite past.

        For a sum of independent OU components with the SAME innovation-to-
        variance split as the mixture weights, the one-step-ahead-from-tau
        predictor correlation is R(tau)/sqrt(R(0)) = R(tau) since R(0)=1. This is
        a first-order surrogate; the exact value for a general 2-OU state needs
        the joint covariance and is provided by the Kalman filter at runtime.
        We return R(tau) as the documented conservative proxy (>= single-exp).
        """
        return float(self.R(tau_s))


def coherence_time_from_R(kernel: Kernel, level: float = math.exp(-1.0),
                          t_hi_s: float = 1e6) -> float:
    """Invert R(T)=level numerically. Sanity/util; not on any hot path.

    Default level = 1/e recovers Tc for ExponentialKernel (THEORY.md D1).
    """
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0,1)")
    lo, hi = 0.0, t_hi_s
    if float(kernel(hi)) > level:
        raise ValueError("level not reached below t_hi_s; increase t_hi_s")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(kernel(mid)) > level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
