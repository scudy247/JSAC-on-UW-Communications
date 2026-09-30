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
    q = p^T R11^{-1} p. Sampled EXACTLY by rejection from the prior f_A: the likelihood
    L(a) = a^{-m/2} exp(-q/(2a)) peaks at a* = q/m with L(a*) = (q/m)^{-m/2} exp(-m/2).
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
    eps = rng.standard_normal(max(n - m, 0))
    for t in range(m, n):
        p = x[t - m:t]
        q = float(p @ R11_inv @ p)
        a_star = max(q / m, 1e-300)
        log_lmax = -0.5 * m * np.log(a_star) - 0.5 * m
        for _ in range(max_tries):
            a, u = pool.next()
            if np.log(u) <= -0.5 * m * np.log(a) - q / (2.0 * a) - log_lmax:
                break
        else:
            raise RuntimeError(f"rejection sampler did not accept in {max_tries} tries at t={t}")
        x[t] = w @ p + np.sqrt(a) * s * eps[t - m]
    return x


def covariation_ratio(x: np.ndarray, k: int, p: float) -> float:
    """Paper's eq. (18) estimate of r_{1,1+k} / (2 delta^2), valid for 1 <= p < alpha."""
    xp = np.sign(x) * np.abs(x) ** (p - 1.0)
    num = np.mean(x[:-k] * xp[k:]) if k > 0 else np.mean(x * xp)
    return float(num / np.mean(np.abs(x) ** p))
