"""alphaSGN(m): stationary alpha-sub-Gaussian noise with memory order m.

Definition (Mahmood & Chitre, OCEANS 2015, eqs. (3), (7)-(10); literature/Mahmood2015OCEANS.pdf):
every window X_{t,m} = [X_{t-m}, ..., X_t] is alpha-sub-Gaussian,
    X_{t,m} = A^{1/2} G,  G ~ N(0, R_m),  E[exp(-s A)] = exp(-s^{alpha/2}),
so that E[exp(i theta^T X)] = exp(-(theta^T R_m theta / 2)^{alpha/2}); R_m is symmetric
Toeplitz with diagonal 2 delta^2, hence each X_t is SaS with scale delta (eq. (10)).

Generation (owner decision 2026-09-30: exact conditional sampling, no discretisation).
The paper generates x_t from the conditional PDF of eq. (11). Given the past m samples p:
  - Gaussian part: mean mu(p) = R21 R11^{-1} p (independent of A), variance A s^2 with
    s^2 = R22 - R21 R11^{-1} R12 (Schur complement);
  - mixing variable: A | p has density proportional to f_A(a) a^{-m/2} exp(-q / (2a)),
    q = p^T R11^{-1} p. Sampled EXACTLY by rejection with two schemes (_posterior_mixing):
    prior proposals when a* = q/m >= 1, tilted-prior proposals when a* < 1. (2026-09-30:
    the prior-only scheme stalled for q ~ 0 -- acceptance collapses ~ (q/m)^(m/2) --
    found by the D-B3 convergence study, seed 1012.)
The first m samples are drawn from the exact m-dimensional marginal (one A, G ~ N(0, R11)).
m = 0 gives i.i.d. SaS(alpha, delta). Real-valued (the paper's model is real).
"""

from __future__ import annotations

import numpy as np

from .stable import positive_stable, sas_real


class _Pool:
    """Pre-drawn prior proposals and uniforms, consumed sequentially (deterministic per seed)."""

    def __init__(self, a_half, rng, size=65536):
        self.a_half, self.rng, self.size = a_half, rng, size
        self._refill()

    def _refill(self):
        self.a = positive_stable(self.a_half, self.size, self.rng)
        self.u = self.rng.random(self.size)
        self.i = 0

    def next(self):
        if self.i == self.size:
            self._refill()
        a, u = self.a[self.i], self.u[self.i]
        self.i += 1
        return a, u


def _kanter_k(u, a):
    return np.sin(a * u) / np.sin(u) ** (1.0 / a) * np.sin((1.0 - a) * u) ** ((1.0 - a) / a)


class _TiltedPool:
    """Exact draws from h(a) ∝ f_A(a) a^(-m/2) (positive a-stable prior tilted by a^(-m/2)).

    Kanter: A = K(U) W^(-(1-a)/a), U ~ Unif(0, pi), W ~ Exp(1). Tilting by A^(-m/2) factorises:
    U has density ∝ K(U)^(-m/2) (rejection from Unif(0, pi); K increases from K0 = a(1-a)^((1-a)/a)
    at 0, so (K/K0)^(-m/2) <= 1; checked numerically 2026-09-30) and W ~ Gamma(1 + m(1-a)/(2a)).
    """

    def __init__(self, a_half, m, rng, size=65536):
        self.a, self.m, self.rng, self.size = a_half, m, rng, size
        self.k0 = a_half * (1.0 - a_half) ** ((1.0 - a_half) / a_half)
        self.shape = 1.0 + m * (1.0 - a_half) / (2.0 * a_half)
        self._refill()

    def _refill(self):
        us = []
        while sum(u.size for u in us) < self.size:
            u = self.rng.uniform(0.0, np.pi, self.size)
            v = self.rng.random(self.size)
            us.append(u[v <= (_kanter_k(u, self.a) / self.k0) ** (-0.5 * self.m)])
        u = np.concatenate(us)[: self.size]
        w = self.rng.gamma(self.shape, 1.0, self.size)
        self.draws = _kanter_k(u, self.a) * w ** (-(1.0 - self.a) / self.a)
        self.acc = self.rng.random(self.size)
        self.i = 0

    def next(self):
        if self.i == self.size:
            self._refill()
        a, v = self.draws[self.i], self.acc[self.i]
        self.i += 1
        return a, v


def _posterior_mixing(q, m, prior_pool, tilted_pool, max_tries, t):
    """Exact draw of A from p(a | q) ∝ f_A(a) a^(-m/2) exp(-q/(2a)).

    Two exact rejection schemes; the choice only affects speed:
    - a* = q/m >= 1: proposals from the prior f_A, accept w.p. L(a)/L(a*) with
      L(a) = a^(-m/2) exp(-q/(2a)) (efficient when the likelihood peak is in the prior bulk);
    - a* < 1: proposals from the tilted prior h(a) ∝ f_A(a) a^(-m/2), accept w.p. exp(-q/(2a))
      (efficient when q is small, where the first scheme's acceptance collapses ~ (q/m)^(m/2)).
    """
    a_star = q / m
    if a_star >= 1.0:
        log_lmax = -0.5 * m * np.log(a_star) - 0.5 * m
        for _ in range(max_tries):
            a, u = prior_pool.next()
            if np.log(u) <= -0.5 * m * np.log(a) - q / (2.0 * a) - log_lmax:
                return a
    else:
        for _ in range(max_tries):
            a, u = tilted_pool.next()
            if u <= np.exp(-q / (2.0 * a)):
                return a
    raise RuntimeError(f"rejection sampler did not accept in {max_tries} tries at t={t}")


def asgn_m(alpha: float, delta: float, rho, n: int, rng: np.random.Generator,
           max_tries: int = 1_000_000) -> np.ndarray:
    """n samples of alphaSGN(m) with m = len(rho) - 1 and rho = r_{1,1+k} / (2 delta^2), rho[0] = 1."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator (explicit seeding)")
    if not 0.0 < alpha < 2.0:
        raise ValueError(f"alpha must be in (0, 2) (alpha = 2 is Gaussian AR), got {alpha}")
    rho = np.asarray(rho, dtype=float)
    if rho.ndim != 1 or rho[0] != 1.0:
        raise ValueError("rho must be a 1-D sequence with rho[0] == 1")
    m = rho.size - 1
    if m == 0:
        return sas_real(alpha, delta, n, rng)
    R = 2.0 * delta ** 2 * np.array([[rho[abs(i - j)] for j in range(m + 1)] for i in range(m + 1)])
    if np.linalg.eigvalsh(R).min() <= 0.0:
        raise ValueError("R_m built from rho is not positive definite")
    R11, R21, R22 = R[:m, :m], R[m, :m], R[m, m]
    R11_inv = np.linalg.inv(R11)
    w = R21 @ R11_inv                      # conditional-mean weights
    s = np.sqrt(R22 - w @ R21)             # conditional std for A = 1
    a_half = alpha / 2.0

    x = np.empty(n)
    head = min(m, n)                       # exact m-dim marginal for the first samples
    a0 = positive_stable(a_half, 1, rng)[0]
    g0 = rng.multivariate_normal(np.zeros(m), R11)
    x[:head] = np.sqrt(a0) * g0[:head]
    pool = _Pool(a_half, rng)
    tilted = _TiltedPool(a_half, m, rng)
    eps = rng.standard_normal(max(n - m, 0))
    for t in range(m, n):
        p = x[t - m:t]
        q = float(p @ R11_inv @ p)
        a = _posterior_mixing(q, m, pool, tilted, max_tries, t)
        x[t] = w @ p + np.sqrt(a) * s * eps[t - m]
    return x


def covariation_ratio(x: np.ndarray, k: int, p: float) -> float:
    """Paper's eq. (18) estimate of r_{1,1+k} / (2 delta^2), valid for 1 <= p < alpha."""
    xp = np.sign(x) * np.abs(x) ** (p - 1.0)
    num = np.mean(x[:-k] * xp[k:]) if k > 0 else np.mean(x * xp)
    return float(num / np.mean(np.abs(x) ** p))
