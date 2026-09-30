"""Delayed-reward link-adaptation environment (Layer-1 driver).

The simulation loop that makes staleness *real* (project.MD §7.2, §7.3):

  per slot t:
    1. DELIVER feedback for every packet whose round trip has elapsed
       (transmit slot s with s + tau_rt <= t). Delivered age t - s >= tau_rt,
       asserted every time -- a sample younger than tau_rt can never reach an
       agent (project.MD §7.3).
    2. SELECT an arm at wall-clock now = t (the agent sees only `now_s`).
    3. TRANSMIT: the packet is received at channel slot t + tau_ow, so its
       reward is realized from the TRUE mu_k(t + tau_ow); enqueue the feedback
       for release at slot t + tau_rt, tagged with transmit slot t.

Oracle hygiene: the agent object is handed only (arm, tx_time_s, reward) through
`observe` and `now_s` through `select`. It never receives the channel, mu, g,
Tc, or tau. Everything the oracles need (true mu at reception) is logged for
metrics.py, not shown to the agent.

Units (project.MD §7.6): tau_rt_slots / tau_ow_slots are integer slot counts;
seconds passed to the agent are slot_index * T_slot_s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..channels.ou import OUChannel


class RewardModel:
    """Maps a true reward-mean mu to a realized (noisy) observation."""

    def realize(self, mu: float, rng: np.random.Generator) -> float:
        raise NotImplementedError

    def mean(self, mu):
        """Expected observed reward given mu (used by oracles/metrics)."""
        raise NotImplementedError

    @property
    def noise_var(self) -> float:
        raise NotImplementedError


@dataclass
class GaussianReward(RewardModel):
    """Additive Gaussian observation noise. Matches the Gaussian converse model."""

    sigma_n: float = 0.0

    def realize(self, mu, rng):
        if self.sigma_n == 0.0:
            return float(mu)
        return float(mu + self.sigma_n * rng.standard_normal())

    def mean(self, mu):
        return mu

    @property
    def noise_var(self) -> float:
        return float(self.sigma_n ** 2)


@dataclass
class SimResult:
    """Per-slot log. All arrays length T. No agent internals here."""

    chosen_arm: np.ndarray       # arm played at each decision slot t
    reception_slot: np.ndarray   # channel slot t + tau_ow whose mu was earned
    realized_reward: np.ndarray  # observed (noisy) reward earned
    tau_rt_slots: int
    tau_ow_slots: int
    T_slot_s: float
    min_delivered_age_slots: int  # smallest t - s seen at delivery (>= tau_rt)


@dataclass
class LinkAdaptationEnv:
    """Runs one agent against one pre-simulated OU channel."""

    channel: OUChannel
    reward_model: RewardModel
    tau_rt_slots: int
    tau_ow_slots: int = 0
    T_slot_s: float = 1.0

    def __post_init__(self):
        if self.tau_rt_slots < 0 or self.tau_ow_slots < 0:
            raise ValueError("delays must be >= 0")
        if self.tau_ow_slots > self.tau_rt_slots:
            raise ValueError("tau_ow cannot exceed tau_rt (D2)")

    def run(self, agent, T: int, rng: np.random.Generator) -> SimResult:
        mu = self.channel.mu                       # (n_slots, K) TRUE means
        n_slots, K = mu.shape
        if T + self.tau_ow_slots > n_slots:
            raise ValueError(
                f"channel too short: need >= {T + self.tau_ow_slots} slots for "
                f"T={T}, tau_ow={self.tau_ow_slots}, have {n_slots}")

        chosen = np.empty(T, dtype=int)
        recv_slot = np.empty(T, dtype=int)
        realized = np.empty(T, dtype=float)

        # Feedback queue: list of (release_slot, tx_slot, arm, reward), FIFO by
        # release_slot (release_slot = tx_slot + tau_rt is monotone in tx_slot).
        from collections import deque
        queue: deque[tuple[int, int, int, float]] = deque()

        min_age = np.iinfo(np.int64).max

        for t in range(T):
            # 1. DELIVER matured feedback (never younger than tau_rt).
            while queue and queue[0][0] <= t:
                release_slot, tx_slot, arm, reward = queue.popleft()
                age = t - tx_slot
                if age < self.tau_rt_slots:
                    raise AssertionError(
                        f"feedback delivered too early: age {age} < "
                        f"tau_rt {self.tau_rt_slots}")
                min_age = min(min_age, age)
                agent.observe(arm, tx_slot * self.T_slot_s, reward)

            # 2. SELECT (agent sees only now_s).
            arm = int(agent.select(t * self.T_slot_s, rng))
            if not (0 <= arm < K):
                raise ValueError(f"agent chose invalid arm {arm}")

            # 3. TRANSMIT: reward realized at reception slot t + tau_ow.
            rslot = t + self.tau_ow_slots
            reward = self.reward_model.realize(mu[rslot, arm], rng)
            queue.append((t + self.tau_rt_slots, t, arm, reward))

            chosen[t] = arm
            recv_slot[t] = rslot
            realized[t] = reward

        if min_age == np.iinfo(np.int64).max:
            min_age = self.tau_rt_slots  # no deliveries within horizon
        return SimResult(chosen, recv_slot, realized,
                         self.tau_rt_slots, self.tau_ow_slots, self.T_slot_s,
                         int(min_age))
