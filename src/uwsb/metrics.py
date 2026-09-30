"""Regret metrics and the oracle ladder (THEORY.md §2).

Oracles live here, never in agents (project.MD §7.2). They read the TRUE mu_k(t)
from the channel and the per-slot decision log from the environment.

The staleness number S = tau/Tc is computed in exactly one function
(`staleness_number`) so no rogue tau/Tc ratio appears elsewhere (project.MD §7.6).

Regret is *expected* regret: it uses the reward model's mean(mu), not the noisy
realized samples, so curves are not inflated by observation noise.
"""

from __future__ import annotations

import numpy as np

from .channels.ou import OUChannel
from .envs.link_adaptation import RewardModel, SimResult


def _synthetic_result(reception_slot: np.ndarray, tau_rt_slots: int,
                      T_slot_s: float) -> SimResult:
    """A SimResult carrying only reception slots, for agent-independent oracle
    quantities (dyn/caus/stat do not depend on the played arm)."""
    T = reception_slot.shape[0]
    return SimResult(
        chosen_arm=np.zeros(T, dtype=int),
        reception_slot=reception_slot,
        realized_reward=np.zeros(T),
        tau_rt_slots=tau_rt_slots, tau_ow_slots=0, T_slot_s=T_slot_s,
        min_delivered_age_slots=tau_rt_slots,
    )


def oracle_ladder(channel: OUChannel, tau_rt_slots: int, reward_model: RewardModel,
                  Tc_true_s: float, T_slot_s: float = 1.0) -> dict:
    """Agent-independent per-slot dynamic / causal / stationary rewards, computed
    directly from the true mu over every slot with a matured CSI sample.

    Used by the phase-transition sweep: none of the three depend on a learning
    agent, so we skip the env loop entirely (fast, exact).
    """
    n_slots = channel.mu.shape[0]
    if n_slots <= tau_rt_slots:
        raise ValueError("channel too short for this tau_rt")
    reception_slot = np.arange(tau_rt_slots, n_slots)
    res = _synthetic_result(reception_slot, tau_rt_slots, T_slot_s)
    return {
        "dynamic": dynamic_oracle_reward(channel, res, reward_model),
        "causal": best_causal_reward(channel, res, reward_model, Tc_true_s),
        "stationary": best_stationary_reward(channel, res, reward_model),
    }


# --- the one true staleness number ------------------------------------------
def staleness_number(tau_s: float, Tc_s: float) -> float:
    """S = tau / Tc (dimensionless). THEORY.md §2, project.MD §7.6."""
    if Tc_s <= 0:
        raise ValueError("Tc_s must be > 0")
    return float(tau_s) / float(Tc_s)


def staleness_number_slots(tau_rt_slots: int, Tc_s: float, T_slot_s: float) -> float:
    return staleness_number(tau_rt_slots * T_slot_s, Tc_s)


# --- expected reward earned by the agent ------------------------------------
def earned_expected_reward(channel: OUChannel, result: SimResult,
                           reward_model: RewardModel) -> np.ndarray:
    """Expected reward the agent actually earned each slot: mean(mu_A(reception))."""
    mu = channel.mu
    return np.asarray(reward_model.mean(mu[result.reception_slot, result.chosen_arm]),
                      dtype=float)


# --- oracle ladder (all return per-slot expected reward arrays, length T) ----
def dynamic_oracle_reward(channel: OUChannel, result: SimResult,
                          reward_model: RewardModel) -> np.ndarray:
    """pi_dyn: knows mu_k(t+tau_ow) exactly, plays the argmax each slot."""
    mu = channel.mu
    rmu = reward_model.mean(mu[result.reception_slot, :])   # (T, K)
    return np.asarray(rmu, dtype=float).max(axis=1)


def best_stationary_reward(channel: OUChannel, result: SimResult,
                           reward_model: RewardModel) -> np.ndarray:
    """pi_stat: always plays argmax_k m_k (the true static-best arm).

    This is the converse's benchmark (THEORY.md §2). Uses the channel's true
    static means m_k, not a hindsight fit.
    """
    mu = channel.mu
    k_star = int(np.argmax(channel.cfg.m))
    return np.asarray(reward_model.mean(mu[result.reception_slot, k_star]), dtype=float)


def best_fixed_arm_reward(channel: OUChannel, result: SimResult,
                          reward_model: RewardModel) -> np.ndarray:
    """Best-fixed-arm-in-hindsight: the single arm with highest realized-mean
    reward over the actual reception slots."""
    mu = channel.mu
    rmu = np.asarray(reward_model.mean(mu[result.reception_slot, :]), dtype=float)
    k_hind = int(np.argmax(rmu.sum(axis=0)))
    return rmu[:, k_hind]


def best_causal_reward(channel: OUChannel, result: SimResult,
                       reward_model: RewardModel, Tc_true_s: float) -> np.ndarray:
    """pi_caus*: the delayed clairvoyant (THEORY.md §2, the FAIR benchmark).

    Knows the true channel state up to channel-time t - tau_fb (equivalently, the
    reception of the packet sent at t - tau_rt) and forms the exact OU predictor
    forward by tau_rt, then plays its argmax. Its gap-to-oracle is F_delay; its
    gap to a converged PUCB is F_est (THEORY.md Prop 6).
    """
    mu = channel.mu
    m = channel.cfg.m
    tau_rt_slots = result.tau_rt_slots
    tau_s = tau_rt_slots * result.T_slot_s
    R_tau = float(np.exp(-tau_s / Tc_true_s))           # predictability over tau
    rmu = np.asarray(reward_model.mean(mu[result.reception_slot, :]), dtype=float)

    obs_slot = result.reception_slot - tau_rt_slots     # channel-time of freshest CSI
    T = result.chosen_arm.shape[0]
    valid = obs_slot >= 0
    os_clip = np.where(valid, obs_slot, 0)
    pred = m[None, :] + R_tau * (mu[os_clip, :] - m[None, :])   # exact OU predictor
    pred[~valid, :] = m[None, :]                        # no CSI yet -> stationary
    choice = np.argmax(pred, axis=1)
    return rmu[np.arange(T), choice]


# --- regret curves ----------------------------------------------------------
def cumulative_regret(reference: np.ndarray, earned: np.ndarray) -> np.ndarray:
    """Cumulative (reference - earned)."""
    return np.cumsum(np.asarray(reference) - np.asarray(earned))


def regret_report(channel: OUChannel, result: SimResult, reward_model: RewardModel,
                  Tc_true_s: float) -> dict:
    """All benchmark curves + the three-term decomposition (THEORY.md §2).

    Returns per-slot expected rewards and cumulative regrets vs each benchmark,
    plus F_delay and (F_delay+F_est) as running per-step averages.
    """
    earned = earned_expected_reward(channel, result, reward_model)
    dyn = dynamic_oracle_reward(channel, result, reward_model)
    stat = best_stationary_reward(channel, result, reward_model)
    fixed = best_fixed_arm_reward(channel, result, reward_model)
    caus = best_causal_reward(channel, result, reward_model, Tc_true_s)

    T = earned.shape[0]
    slots = np.arange(1, T + 1)
    return {
        "earned": earned,
        "reward_dynamic": dyn,
        "reward_stationary": stat,
        "reward_best_fixed": fixed,
        "reward_causal": caus,
        "regret_vs_dynamic": cumulative_regret(dyn, earned),
        "regret_vs_stationary": cumulative_regret(stat, earned),
        "regret_vs_best_fixed": cumulative_regret(fixed, earned),
        "regret_vs_causal": cumulative_regret(caus, earned),
        # Per-step floors (running averages):
        "F_delay_running": np.cumsum(dyn - caus) / slots,       # oracle - fair
        "F_delay_plus_est_running": np.cumsum(dyn - earned) / slots,
    }
