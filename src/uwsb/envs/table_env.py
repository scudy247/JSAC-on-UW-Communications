"""Delayed-feedback environment over counterfactual outcome tables (Proposal B, plan §6 task 6).

An OutcomeTable holds, for every slot t and every arm k, what the receiver chain would have
produced (e.g. gamma_db[t, k], ack[t, k]); the learner sees only the arm it pulled, after the
round-trip delay. Between packets the receiver observes noise-only samples (the dense noise
stream, THEORY-B §2.1), delivered through `observe_noise`.

Per slot t (time now_s = t * T_slot_s):
  1. DELIVER every queued reward with tx slot s such that s + tau_rt_slots <= t; the age
     t - s >= tau_rt_slots is asserted for each one (project.MD §7.3);
  2. DELIVER the noise-only samples of slot t - 1 - noise_delay_slots (noise observed after
     that slot's packet); noise_delay_slots = 0 models a receiver-side learner (D-B12 default,
     pending), > 0 a transmitter-side learner that gets the side information fed back;
  3. SELECT an arm (the agent sees only now_s);
  4. TRANSMIT: the outcome of (t, arm) is queued for release at slot t + tau_rt_slots.

Convention (same as Proposal 0's envs/link_adaptation.py): a reward can be used at the earliest by
the NEXT decision, so tau_rt_slots = 0 and 1 behave alike (delivered age max(tau, 1) slots).

Oracle hygiene (project.MD §7.2): the agent receives copies of the pulled arm's fields only;
table fields of other arms and `truth` are used by `run` for metrics and never passed on.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class OutcomeTable:
    fields: dict                          # name -> array (n_slots, K), agent-visible per pull
    truth_mean: np.ndarray                # (n_slots, K) expected value used for regret (metrics only)
    noise: np.ndarray | None = None       # (n_slots, nu) noise-only samples after each slot's packet
    T_slot_s: float = 1.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.fields:
            raise ValueError("fields must not be empty")
        shapes = {np.shape(v) for v in self.fields.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2:
            raise ValueError(f"all fields must share one (n_slots, K) shape, got {shapes}")
        if np.shape(self.truth_mean) != next(iter(shapes)):
            raise ValueError("truth_mean must have the fields' (n_slots, K) shape")
        if self.noise is not None and np.shape(self.noise)[0] != self.n_slots:
            raise ValueError("noise must have n_slots rows")
        if not self.T_slot_s > 0:
            raise ValueError("T_slot_s must be > 0")

    @property
    def n_slots(self) -> int:
        return np.shape(self.truth_mean)[0]

    @property
    def n_arms(self) -> int:
        return np.shape(self.truth_mean)[1]


@dataclass
class TableRunResult:
    arms: np.ndarray                      # (T,) arm pulled at each slot
    regret_inst: np.ndarray               # (T,) max_k truth[t] - truth[t, arm_t]
    n_delivered: int
    n_noise_delivered: int
    min_delivered_age_slots: int     # >= max(tau_rt_slots, 1); -1 if nothing delivered

    @property
    def regret(self) -> float:
        return float(self.regret_inst.sum())


class TableEnv:
    def __init__(self, table: OutcomeTable, tau_rt_slots: int, noise_delay_slots: int = 0):
        if int(tau_rt_slots) != tau_rt_slots or tau_rt_slots < 0:
            raise ValueError("tau_rt_slots must be a non-negative integer")
        if int(noise_delay_slots) != noise_delay_slots or noise_delay_slots < 0:
            raise ValueError("noise_delay_slots must be a non-negative integer")
        self.table = table
        self.tau_rt_slots = int(tau_rt_slots)
        self.noise_delay_slots = int(noise_delay_slots)

    def run(self, agent, T: int | None = None) -> TableRunResult:
        tab = self.table
        T = tab.n_slots if T is None else int(T)
        if not 0 < T <= tab.n_slots:
            raise ValueError(f"T must be in (0, {tab.n_slots}]")
        dt = tab.T_slot_s
        queue: deque = deque()            # (tx_slot, arm, reward dict), FIFO in tx order
        arms = np.empty(T, dtype=int)
        regret = np.empty(T)
        n_deliv = n_noise = 0
        min_age = -1
        for t in range(T):
            while queue and queue[0][0] + self.tau_rt_slots <= t:
                s, k, r = queue.popleft()
                assert t - s >= self.tau_rt_slots, "reward younger than the round trip"
                agent.observe(k, s * dt, r)
                min_age = t - s if min_age < 0 else min(min_age, t - s)
                n_deliv += 1
            src = t - 1 - self.noise_delay_slots
            if tab.noise is not None and src >= 0 and hasattr(agent, "observe_noise"):
                agent.observe_noise(t * dt, np.array(tab.noise[src], copy=True))
                n_noise += 1
            k = int(agent.select(t * dt))
            if not 0 <= k < tab.n_arms:
                raise ValueError(f"agent selected invalid arm {k}")
            arms[t] = k
            row = tab.truth_mean[t]
            regret[t] = float(np.max(row) - row[k])
            queue.append((t, k, {name: float(v[t, k]) for name, v in tab.fields.items()}))
        return TableRunResult(arms=arms, regret_inst=regret, n_delivered=n_deliv,
                              n_noise_delivered=n_noise, min_delivered_age_slots=min_age)


def best_fixed_arm_regret(table: OutcomeTable, T: int | None = None) -> np.ndarray:
    """Reference policies (plan §7). Regret is measured against the DYNAMIC oracle
    (max_k truth[t] at every slot), whose regret is 0 by definition. The best fixed arm in
    hindsight (argmax_k sum_t truth[t, k] over the first T slots) has instantaneous regret
    max_k truth[t] - truth[t, k*]: identically 0 for stationary tables, > 0 under regime changes."""
    T = table.n_slots if T is None else int(T)
    tr = np.asarray(table.truth_mean)[:T]
    k = int(np.argmax(tr.sum(axis=0)))
    return tr.max(axis=1) - tr[:, k]
