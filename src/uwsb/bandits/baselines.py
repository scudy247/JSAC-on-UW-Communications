"""Baseline agents (THEORY.md §2, project.MD §5).

All agents share the causal interface used by the environment:

    agent.observe(arm: int, tx_time_s: float, reward: float) -> None
    agent.select(now_s: float, rng) -> int

None of them use Tc, tau, or any forward prediction -- that asymmetry is the
point: SW-UCB / D-UCB approximate forgetting with a *hand-tuned* window/discount,
so as staleness grows their per-step regret plateaus strictly above best-
stationary while PUCB degrades to it gracefully (THEORY.md §8 prediction 4).
Their window/discount hyperparameters are meant to be oracle-grid-searched per
scenario by the experiment layer, otherwise the comparison is a straw man.

Oracles (dynamic / best-fixed / best-stationary) are NOT here: they need the true
mu_k(t) and live in metrics.py (project.MD §7.2, no oracle leakage into agents).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class UCB1:
    """Vanilla UCB1 on empirical arm means. Ignores time entirely."""

    def __init__(self, K: int, c: float = math.sqrt(2.0)):
        self.K = int(K)
        self.c = float(c)
        self.sum = np.zeros(self.K)
        self.n = np.zeros(self.K, dtype=int)
        self._t = 0

    def observe(self, arm, tx_time_s, reward):
        self.sum[arm] += reward
        self.n[arm] += 1

    def select(self, now_s, rng=None):
        self._t += 1
        unseen = np.where(self.n == 0)[0]
        if unseen.size:
            return int(unseen[0])
        mean = self.sum / self.n
        bonus = self.c * np.sqrt(np.log(max(self._t, 2)) / self.n)
        return int(np.argmax(mean + bonus))


class SlidingWindowUCB:
    """SW-UCB: UCB on the most recent `window` observations (global count).

    Window is in number of observations retained. Tuning `window` per scenario
    is an experiment-layer job (THEORY.md §8).
    """

    def __init__(self, K: int, window: int, c: float = math.sqrt(2.0)):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.K = int(K)
        self.window = int(window)
        self.c = float(c)
        self.buf: deque[tuple[int, float]] = deque()   # (arm, reward), FIFO
        self._sum = np.zeros(self.K)
        self._n = np.zeros(self.K, dtype=int)
        self._t = 0

    def observe(self, arm, tx_time_s, reward):
        self.buf.append((arm, reward))
        self._sum[arm] += reward
        self._n[arm] += 1
        while len(self.buf) > self.window:
            a0, r0 = self.buf.popleft()
            self._sum[a0] -= r0
            self._n[a0] -= 1

    def select(self, now_s, rng=None):
        self._t += 1
        unseen = np.where(self._n == 0)[0]
        if unseen.size:
            return int(unseen[0])
        mean = self._sum / np.maximum(self._n, 1)
        horizon = max(min(self._t, self.window), 2)
        bonus = self.c * np.sqrt(np.log(horizon) / self._n)
        return int(np.argmax(mean + bonus))


class DiscountedUCB:
    """D-UCB: discounted empirical means with per-decision discount `gamma`."""

    def __init__(self, K: int, gamma: float = 0.99, c: float = math.sqrt(2.0)):
        if not (0.0 < gamma < 1.0):
            raise ValueError("gamma must be in (0,1)")
        self.K = int(K)
        self.gamma = float(gamma)
        self.c = float(c)
        self.X = np.zeros(self.K)      # discounted reward sums
        self.N = np.zeros(self.K)      # discounted counts
        self._t = 0

    def _discount(self):
        self.X *= self.gamma
        self.N *= self.gamma

    def observe(self, arm, tx_time_s, reward):
        self.X[arm] += reward
        self.N[arm] += 1.0

    def select(self, now_s, rng=None):
        self._t += 1
        self._discount()
        unseen = np.where(self.N < 1e-12)[0]
        if unseen.size:
            return int(unseen[0])
        mean = self.X / self.N
        total = self.N.sum()
        bonus = self.c * np.sqrt(np.log(max(total, 2.0)) / self.N)
        return int(np.argmax(mean + bonus))


class GaussianThompson:
    """Stationary Gaussian Thompson sampling on arm means (no forgetting)."""

    def __init__(self, K: int, obs_var: float = 1.0, prior_var: float = 1.0,
                 prior_mean: float = 0.0):
        self.K = int(K)
        self.obs_var = float(obs_var)
        self.prior_var = float(prior_var)
        self.prior_mean = float(prior_mean)
        self.sum = np.zeros(self.K)
        self.n = np.zeros(self.K, dtype=int)

    def observe(self, arm, tx_time_s, reward):
        self.sum[arm] += reward
        self.n[arm] += 1

    def select(self, now_s, rng):
        if rng is None:
            raise ValueError("GaussianThompson.select needs an rng")
        # Conjugate Normal-Normal posterior on each arm mean.
        post_var = 1.0 / (1.0 / self.prior_var + self.n / self.obs_var)
        post_mean = post_var * (self.prior_mean / self.prior_var + self.sum / self.obs_var)
        sample = post_mean + np.sqrt(post_var) * rng.standard_normal(self.K)
        return int(np.argmax(sample))


class LatestSampleGreedy:
    """Myopic 'act on outdated CSI' policy: play argmax of each arm's freshest
    delayed reward, with no forgetting model and no averaging.

    This is the Layer-1 abstraction of outdated-CSI threshold AMC (THEORY.md P4):
    a decision driven purely by the most recent (stale) per-arm observation. The
    full genie SNR->arm-table AMC needs an explicit rate/PER structure and is
    wired at Layer 2 where that structure exists; here the reward IS the utility,
    so 'latest sample' is the faithful reduced form. Serves as the no-learning,
    no-tracking reference.
    """

    def __init__(self, K: int, init: float = np.inf):
        self.K = int(K)
        # inf until seen => every arm is tried once first.
        self.latest = np.full(self.K, float(init))
        self.latest_tx = np.full(self.K, -np.inf)

    def observe(self, arm, tx_time_s, reward):
        # Keep only the freshest sample per arm (by transmit time).
        if tx_time_s >= self.latest_tx[arm]:
            self.latest[arm] = reward
            self.latest_tx[arm] = tx_time_s

    def select(self, now_s, rng=None):
        return int(np.argmax(self.latest))
