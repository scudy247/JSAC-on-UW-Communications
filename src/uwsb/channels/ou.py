"""Ornstein-Uhlenbeck (AR(1)) reward-process generator.

Implements the THEORY.md D4 model of the per-arm expected reward:

    mu_k(t) = m_k + d_k(t),     d_k(t) = b_k * g(t) + e_k(t)

    m_k   static (quasi-static) arm mean            -- learnable, never decays
    g(t)  shared latent OU factor, unit variance    -- the common channel state
    e_k(t) per-arm idiosyncratic OU deviation, var sigma_idio_k^2

Both g and e are exact-discretization OU processes on the slot grid:

    x[t] = alpha * x[t-1] + sqrt(1 - alpha^2) * s * z[t],   alpha = exp(-T_slot/Tc)

initialized from their stationary distribution so the whole trajectory is
stationary (no burn-in bias). Because a shared-slow + idiosyncratic-fast pair of
OU components already sums to a two-timescale autocorrelation, this model
naturally produces the non-single-exponential reward correlation we expect on
real channels (THEORY.md Prop 4) without any extra machinery.

Conventions (project.MD §7.6): every time quantity carries an _s suffix; the
shared vs idiosyncratic split is set once here and exposed via properties so
oracles/metrics read the SAME sigma the generator used (no drift).

NB oracle hygiene (project.MD §7.2): this object owns the *true* mu_k(t). It is
consumed only by the environment (to realize rewards) and by metrics/oracles.
Agents never receive an OUChannel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _ou_trajectory(n: int, alpha: float, stationary_std: float,
                   rng: np.random.Generator) -> np.ndarray:
    """One exact-discretization OU path of length n, stationary from t=0."""
    x = np.empty(n, dtype=float)
    if n == 0:
        return x
    x[0] = rng.normal(0.0, stationary_std)
    if n == 1:
        return x
    innov_std = stationary_std * np.sqrt(max(1.0 - alpha * alpha, 0.0))
    noise = rng.normal(0.0, 1.0, size=n - 1)
    # Sequential recursion (OU is Markov; no closed vectorized form for the path).
    for t in range(1, n):
        x[t] = alpha * x[t - 1] + innov_std * noise[t - 1]
    return x


@dataclass
class OUChannelConfig:
    """Configuration for the OU reward channel.

    m: static arm means, shape (K,).
    sigma_idio: idiosyncratic OU std per arm (scalar broadcast or shape (K,)).
    b: shared-factor loadings per arm (scalar broadcast or shape (K,)); 0 => the
       independent-arms model used by the unstructured theory (THEORY.md D4).
    Tc_s: coherence time of the idiosyncratic components (and of g unless
       g_Tc_s is set). Definition R(Tc)=1/e (D1).
    g_Tc_s: optional distinct coherence time for the shared factor g (e.g. a
       slow common trend); defaults to Tc_s.
    T_slot_s: decision-slot duration (the trajectory grid step).
    """

    m: np.ndarray
    sigma_idio: np.ndarray | float
    Tc_s: float
    T_slot_s: float
    b: np.ndarray | float = 0.0
    g_Tc_s: float | None = None

    K: int = field(init=False)

    def __post_init__(self):
        self.m = np.asarray(self.m, dtype=float).ravel()
        self.K = int(self.m.shape[0])
        if self.K == 0:
            raise ValueError("need at least one arm")
        self.sigma_idio = self._broadcast(self.sigma_idio, "sigma_idio")
        self.b = self._broadcast(self.b, "b")
        if np.any(self.sigma_idio < 0):
            raise ValueError("sigma_idio must be >= 0")
        if not (self.Tc_s > 0 and self.T_slot_s > 0):
            raise ValueError("Tc_s and T_slot_s must be > 0")
        if self.g_Tc_s is not None and not self.g_Tc_s > 0:
            raise ValueError("g_Tc_s must be > 0 when given")

    def _broadcast(self, v, name) -> np.ndarray:
        arr = np.asarray(v, dtype=float)
        if arr.ndim == 0:
            arr = np.full(self.K, float(arr))
        elif arr.shape != (self.K,):
            raise ValueError(f"{name} must be scalar or shape ({self.K},)")
        return arr

    @property
    def sigma_dev(self) -> np.ndarray:
        """Stationary std of the deviation d_k = b_k g + e_k: sqrt(b_k^2+sig^2).

        This is the sigma that enters the converse formula (†) (per arm); for the
        symmetric two-arm analysis sigma_D = sqrt(2)*sigma_dev.
        """
        return np.sqrt(self.b ** 2 + self.sigma_idio ** 2)


class OUChannel:
    """Generates and holds the true reward-mean trajectory mu_k(t)."""

    def __init__(self, config: OUChannelConfig):
        self.cfg = config
        self._mu: np.ndarray | None = None   # shape (T, K), true means (hidden)
        self._g: np.ndarray | None = None    # shared factor path, shape (T,)

    def simulate(self, n_slots: int, rng: np.random.Generator) -> np.ndarray:
        """Draw the full mu_k(t) trajectory. Returns array shape (n_slots, K).

        Deterministic given rng state (project.MD §7.4). The shared factor is
        drawn first, then each idiosyncratic path, so adding/removing arms does
        not perturb g's stream.
        """
        cfg = self.cfg
        n = int(n_slots)
        if n <= 0:
            raise ValueError("n_slots must be > 0")

        g_Tc = cfg.g_Tc_s if cfg.g_Tc_s is not None else cfg.Tc_s
        alpha_g = float(np.exp(-cfg.T_slot_s / g_Tc))
        alpha_e = float(np.exp(-cfg.T_slot_s / cfg.Tc_s))

        g = _ou_trajectory(n, alpha_g, 1.0, rng)  # unit-variance shared factor
        mu = np.empty((n, cfg.K), dtype=float)
        for k in range(cfg.K):
            e_k = _ou_trajectory(n, alpha_e, float(cfg.sigma_idio[k]), rng)
            mu[:, k] = cfg.m[k] + cfg.b[k] * g + e_k

        self._mu = mu
        self._g = g
        return mu

    @property
    def mu(self) -> np.ndarray:
        if self._mu is None:
            raise RuntimeError("call simulate() first")
        return self._mu

    @property
    def g(self) -> np.ndarray:
        if self._g is None:
            raise RuntimeError("call simulate() first")
        return self._g

    @property
    def alpha_slot(self) -> float:
        """Per-slot OU factor of the idiosyncratic component, exp(-T_slot/Tc)."""
        return float(np.exp(-self.cfg.T_slot_s / self.cfg.Tc_s))
