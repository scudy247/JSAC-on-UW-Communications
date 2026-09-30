"""Layer-1 synthetic experiments (project.MD §5 Layer 1, milestone 2).

One entrypoint, driven by a YAML config; results are written to
results/<name>-<confighash>/ so a figure never reuses a stale run
(project.MD §7.5). Seeds come from the config and are logged.

Modes:
  compare     multi-agent regret curves on one scenario (the learning-term
              shape deliverable of milestone 2).
  delay_tax   symmetric-arm sweep over S = tau/Tc; checks that the best causal
              tracking surplus over stationary decays like e^{-S} (THEORY.md
              §8 prediction 1 -- the milestone-3 kill test, previewed here).

At this milestone the PUCB is given genie (Tc, sigma^2, sigma_n^2). Replacing Tc
with the online estimator is milestone 4; the estimator plugs into the same
constructor argument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from uwsb.channels.ou import OUChannel, OUChannelConfig
from uwsb.envs.link_adaptation import LinkAdaptationEnv, GaussianReward
from uwsb.bandits.predictive_ucb import (
    PredictiveUCB, PredictiveThompson, beta_theory,
)
from uwsb.bandits.baselines import (
    UCB1, SlidingWindowUCB, DiscountedUCB, GaussianThompson, LatestSampleGreedy,
)
from uwsb import metrics
from uwsb import theory

ROOT = Path(__file__).resolve().parent.parent


# --- config plumbing --------------------------------------------------------
def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def result_dir(cfg: dict) -> Path:
    d = ROOT / "results" / f"{cfg.get('name', 'run')}-{config_hash(cfg)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_channel(cfg: dict, n_slots: int, rng) -> OUChannel:
    c = cfg["channel"]
    m = c["m"] if isinstance(c["m"], list) else [c["m"]] * c["K"]
    occ = OUChannelConfig(
        m=np.asarray(m, dtype=float),
        sigma_idio=c.get("sigma_idio", 0.2),
        b=c.get("b", 0.0),
        Tc_s=float(c["Tc_s"]),
        T_slot_s=float(c.get("T_slot_s", 1.0)),
        g_Tc_s=c.get("g_Tc_s"),
    )
    ch = OUChannel(occ)
    ch.simulate(n_slots, rng)
    return ch


def make_agent(spec: dict, channel: OUChannel, reward_model: GaussianReward):
    """Agent factory. Physics inputs (Tc, sigma^2, sigma_n^2) are genie at this
    milestone (see module docstring)."""
    K = channel.cfg.K
    Tc = channel.cfg.Tc_s
    sigma2 = float(np.mean(channel.cfg.sigma_dev ** 2))
    sn2 = reward_model.noise_var
    typ = spec["type"]
    prior_m = float(np.mean(channel.cfg.m))
    if typ == "predictive_ucb":
        beta = spec.get("beta", 1.5)
        if beta == "theory":
            beta = beta_theory(spec.get("beta_c", 1.0))
        return PredictiveUCB(K, Tc, sigma2, sn2, T_slot_s=channel.cfg.T_slot_s,
                             beta=beta, prior_m_mean=prior_m, prior_m_var=1.0,
                             q_m=spec.get("q_m", 0.0))
    if typ == "predictive_thompson":
        return PredictiveThompson(K, Tc, sigma2, sn2, T_slot_s=channel.cfg.T_slot_s,
                                  prior_m_mean=prior_m, prior_m_var=1.0)
    if typ == "ucb1":
        return UCB1(K, c=spec.get("c", math.sqrt(2.0)))
    if typ == "sw_ucb":
        return SlidingWindowUCB(K, window=int(spec["window"]),
                                c=spec.get("c", math.sqrt(2.0)))
    if typ == "d_ucb":
        return DiscountedUCB(K, gamma=float(spec.get("gamma", 0.99)),
                             c=spec.get("c", math.sqrt(2.0)))
    if typ == "gaussian_thompson":
        return GaussianThompson(K, obs_var=max(sigma2 + sn2, 1e-6),
                                prior_var=1.0, prior_mean=prior_m)
    if typ == "latest_greedy":
        return LatestSampleGreedy(K)
    raise ValueError(f"unknown agent type {typ!r}")


# --- modes ------------------------------------------------------------------
def run_compare(cfg: dict, smoke: bool) -> dict:
    T = int(cfg["T"])
    n_seeds = int(cfg["n_seeds"])
    base_seed = int(cfg.get("seed", 0))
    delay = cfg["delay"]
    tau_rt, tau_ow = int(delay["tau_rt_slots"]), int(delay.get("tau_ow_slots", 0))
    rm = GaussianReward(float(cfg["reward"].get("sigma_n", 0.0)))
    n_slots = T + tau_ow + 1

    agent_specs = cfg["agents"]
    curves = {s["name"]: np.zeros(T) for s in agent_specs}
    final_dyn = {s["name"]: [] for s in agent_specs}
    final_stat = {s["name"]: [] for s in agent_specs}
    Tc_true = float(cfg["channel"]["Tc_s"])

    for si in range(n_seeds):
        # Common channel per seed for all agents (CRN variance reduction).
        ch = build_channel(cfg, n_slots, np.random.default_rng(base_seed + si))
        for ai, spec in enumerate(agent_specs):
            agent = make_agent(spec, ch, rm)
            env = LinkAdaptationEnv(ch, rm, tau_rt_slots=tau_rt,
                                    tau_ow_slots=tau_ow, T_slot_s=ch.cfg.T_slot_s)
            rng = np.random.default_rng(10_000 * (base_seed + si) + ai)
            res = env.run(agent, T=T, rng=rng)
            rep = metrics.regret_report(ch, res, rm, Tc_true_s=Tc_true)
            curves[spec["name"]] += rep["regret_vs_dynamic"]
            final_dyn[spec["name"]].append(rep["regret_vs_dynamic"][-1])
            final_stat[spec["name"]].append(rep["regret_vs_stationary"][-1])

    for name in curves:
        curves[name] /= n_seeds

    summary = {
        "S": metrics.staleness_number_slots(tau_rt, Tc_true, ch.cfg.T_slot_s),
        "tau_rt_slots": tau_rt, "T": T, "n_seeds": n_seeds,
        "regret_vs_dynamic": {n: [float(np.mean(v)), float(np.std(v))]
                              for n, v in final_dyn.items()},
        "regret_vs_stationary": {n: [float(np.mean(v)), float(np.std(v))]
                                 for n, v in final_stat.items()},
    }
    if not smoke:
        _plot_compare(cfg, curves, summary)
    return {"summary": summary, "curves": {k: v.tolist() for k, v in curves.items()}}


def _plot_compare(cfg, curves, summary):
    d = result_dir(cfg)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, curve in curves.items():
        ax.plot(curve, label=name, lw=1.6)
    ax.set_xlabel("decision slot t")
    ax.set_ylabel("cumulative regret vs dynamic oracle")
    ax.set_title(f"{cfg.get('name')}  (S = tau/Tc = {summary['S']:.2f})")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "regret_curves.png", dpi=130)
    plt.close(fig)


def run_delay_tax(cfg: dict, smoke: bool) -> dict:
    """Symmetric arms (Delta=0): measure best-causal surplus over stationary vs
    S; theory predicts surplus/oracle-surplus = e^{-S} (THEORY.md Prop 1)."""
    T = int(cfg["T"])
    n_seeds = int(cfg["n_seeds"])
    base_seed = int(cfg.get("seed", 0))
    Tc = float(cfg["channel"]["Tc_s"])
    Tslot = float(cfg["channel"].get("T_slot_s", 1.0))
    rm = GaussianReward(0.0)               # noiseless: isolate the delay tax
    S_grid = cfg["S_grid"]
    tau_list = [int(round(S * Tc / Tslot)) for S in S_grid]

    ratios = []
    for tau_rt in tau_list:
        tau_ow = tau_rt // 2
        n_slots = T + tau_ow + 1
        num = den = 0.0
        for si in range(n_seeds):
            ch = build_channel(cfg, n_slots, np.random.default_rng(base_seed + si))
            # Use a fixed round-robin "agent" only to generate reception slots;
            # the surplus quantities read the true mu directly (oracle-side).
            env = LinkAdaptationEnv(ch, rm, tau_rt_slots=tau_rt, tau_ow_slots=tau_ow,
                                    T_slot_s=Tslot)
            rr = _RoundRobin(ch.cfg.K)
            res = env.run(rr, T=T, rng=np.random.default_rng(si))
            dyn = metrics.dynamic_oracle_reward(ch, res, rm).mean()
            caus = metrics.best_causal_reward(ch, res, rm, Tc_true_s=Tc).mean()
            stat = metrics.best_stationary_reward(ch, res, rm).mean()
            num += (caus - stat)
            den += (dyn - stat)
        ratios.append(num / den if den > 0 else float("nan"))

    predicted = [math.exp(-S) for S in S_grid]
    out = {"S_grid": list(S_grid), "measured_ratio": ratios, "predicted": predicted}
    if not smoke:
        _plot_delay_tax(cfg, out)
    return out


def _plot_delay_tax(cfg, out):
    d = result_dir(cfg)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(out["S_grid"], out["predicted"], "k--", label=r"theory $e^{-S}$")
    ax.plot(out["S_grid"], out["measured_ratio"], "o-", label="measured")
    ax.set_xlabel(r"staleness number $S=\tau/T_c$")
    ax.set_ylabel("causal surplus / oracle surplus")
    ax.set_title("Delay tax (THEORY.md Prop 1)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "delay_tax.png", dpi=130)
    plt.close(fig)


class _RoundRobin:
    def __init__(self, K): self.K = K; self._t = -1
    def observe(self, *a): pass
    def select(self, now_s, rng=None):
        self._t += 1
        return self._t % self.K


def run_phase_transition(cfg: dict, smoke: bool) -> dict:
    """Milestone 3: sweep (u=Delta_bar/sigma_D, S=tau/Tc) and locate the phase
    boundary (THEORY.md Prop 2/3, §8 prediction 2).

    Primary field = best-causal relative gain (caus-stat)/(dyn-stat), which is
    agent-independent and computed directly from mu (fast). We check it against
    the closed form (†), show the data collapse onto z = u*e^S, and confirm the
    compensated-gain contours are slope -1. Optionally overlay a real PUCB run to
    show the algorithm approaches the frontier.
    """
    T = int(cfg["T"])
    n_seeds = int(cfg["n_seeds"])
    base_seed = int(cfg.get("seed", 0))
    Tc = float(cfg["channel"]["Tc_s"])
    Tslot = float(cfg["channel"].get("T_slot_s", 1.0))
    sigma_idio = float(cfg["channel"].get("sigma_idio", 0.3))
    sigma_D = math.sqrt(2.0) * sigma_idio          # independent equal-var arms
    rm = GaussianReward(0.0)                        # noiseless: isolate the limit
    u_grid = np.asarray(cfg["u_grid"], dtype=float)
    S_grid = np.asarray(cfg["S_grid"], dtype=float)
    include_pucb = bool(cfg.get("include_pucb", False))
    pucb_sigma_n = float(cfg.get("pucb_sigma_n", 0.1))
    pucb_seeds = int(cfg.get("pucb_seeds", min(n_seeds, 4)))

    ratio_meas = np.full((len(u_grid), len(S_grid)), np.nan)
    ratio_ana = np.full_like(ratio_meas, np.nan)
    ratio_pucb = np.full_like(ratio_meas, np.nan)

    for j, S in enumerate(S_grid):
        tau_rt = int(round(S * Tc / Tslot))
        for i, u in enumerate(u_grid):
            delta_bar = u * sigma_D
            m = np.array([delta_bar / 2.0, -delta_bar / 2.0])
            occ = OUChannelConfig(m=m, sigma_idio=sigma_idio, b=0.0, Tc_s=Tc,
                                  T_slot_s=Tslot)
            num = den = 0.0
            for si in range(n_seeds):
                ch = OUChannel(occ)
                ch.simulate(T + tau_rt + 1, np.random.default_rng(base_seed + si))
                lad = metrics.oracle_ladder(ch, tau_rt, rm, Tc_true_s=Tc,
                                            T_slot_s=Tslot)
                num += float((lad["causal"] - lad["stationary"]).sum())
                den += float((lad["dynamic"] - lad["stationary"]).sum())
            ratio_meas[i, j] = num / den if den > 0 else np.nan
            ratio_ana[i, j] = float(theory.relative_gain_uS(u, S))

            if include_pucb:
                pn, pd = 0.0, 0.0
                rmn = GaussianReward(pucb_sigma_n)
                for si in range(pucb_seeds):
                    ch = OUChannel(occ)
                    ch.simulate(T + tau_rt // 2 + 1,
                                np.random.default_rng(7_000 + base_seed + si))
                    agent = PredictiveUCB(2, Tc, sigma_D ** 2 / 2.0, rmn.noise_var,
                                          T_slot_s=Tslot, beta=1.5,
                                          prior_m_mean=0.0, prior_m_var=1.0)
                    env = LinkAdaptationEnv(ch, rmn, tau_rt_slots=tau_rt,
                                            tau_ow_slots=tau_rt // 2, T_slot_s=Tslot)
                    res = env.run(agent, T=T, rng=np.random.default_rng(si + 1))
                    earned = metrics.earned_expected_reward(ch, res, rmn)
                    dyn = metrics.dynamic_oracle_reward(ch, res, rmn)
                    stat = metrics.best_stationary_reward(ch, res, rmn)
                    pn += float((earned - stat).sum())
                    pd += float((dyn - stat).sum())
                ratio_pucb[i, j] = pn / pd if pd > 0 else np.nan

    out = {
        "u_grid": u_grid.tolist(), "S_grid": S_grid.tolist(),
        "sigma_D": sigma_D,
        "ratio_measured": ratio_meas.tolist(),
        "ratio_analytic": ratio_ana.tolist(),
        "max_abs_error": float(np.nanmax(np.abs(ratio_meas - ratio_ana))),
        "rms_error": float(np.sqrt(np.nanmean((ratio_meas - ratio_ana) ** 2))),
    }
    if include_pucb:
        out["ratio_pucb"] = ratio_pucb.tolist()
    if not smoke:
        _plot_phase(cfg, u_grid, S_grid, ratio_meas, ratio_ana,
                    ratio_pucb if include_pucb else None)
    return out


def _plot_phase(cfg, u_grid, S_grid, ratio_meas, ratio_ana, ratio_pucb):
    d = result_dir(cfg)

    # (1) Phase diagram: measured field + analytic contours + z=1 boundary.
    fig, ax = plt.subplots(figsize=(7, 5))
    SS, UU = np.meshgrid(S_grid, u_grid)
    pcm = ax.pcolormesh(SS, UU, ratio_meas, shading="nearest",
                        cmap="viridis", vmin=0, vmax=1)
    # analytic contours on a fine grid
    Sf = np.linspace(S_grid.min(), S_grid.max(), 200)
    Uf = np.geomspace(max(u_grid.min(), 1e-3), u_grid.max(), 200)
    RA = theory.relative_gain_uS(Uf[:, None], Sf[None, :])
    cs = ax.contour(*np.meshgrid(Sf, Uf), RA, levels=[0.1, 0.25, 0.5, 0.75],
                    colors="w", linewidths=1.0)
    ax.clabel(cs, fmt="%.2f", fontsize=7)
    ax.plot(Sf, np.exp(-Sf), "r--", lw=1.8, label=r"$z=1$: $\bar\Delta=\rho\sigma_D$")
    ax.set_yscale("log")
    ax.set_xlabel(r"staleness number $S=\tau/T_c$")
    ax.set_ylabel(r"$\bar\Delta/\sigma_D$")
    ax.set_title("Phase diagram: causal tracking gain (measured field, analytic contours)")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(pcm, ax=ax, label=r"relative gain $G(\rho)/G(1)$")
    fig.tight_layout(); fig.savefig(d / "phase_diagram.png", dpi=130); plt.close(fig)

    # (2) Agreement: measured vs analytic.
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.scatter(ratio_ana.ravel(), ratio_meas.ravel(), s=18, alpha=0.7)
    ax.set_xlabel("analytic relative gain (†)")
    ax.set_ylabel("measured relative gain")
    ax.set_title("Simulation vs closed form")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(d / "phase_agreement.png", dpi=130); plt.close(fig)

    # (3) Data collapse onto z = u e^S, with psi(z) overlay.
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    z = (u_grid[:, None] * np.exp(S_grid[None, :])).ravel()
    comp = (ratio_meas * np.exp(S_grid[None, :])).ravel()
    order = np.argsort(z)
    ax.scatter(z, comp, s=16, alpha=0.6, label="measured  $r\\,e^{S}$")
    zc = np.geomspace(max(z.min(), 1e-3), z.max(), 200)
    ax.plot(zc, theory.psi(zc), "r-", lw=1.8, label=r"$\psi(z)=h(z)/h(0)$")
    ax.axvline(1.0, color="gray", ls=":", label="$z=1$")
    ax.set_xscale("log")
    ax.set_xlabel(r"collapse variable $z=\bar\Delta/(\rho\sigma_D)=u\,e^{S}$")
    ax.set_ylabel(r"compensated gain $r\,e^{S}$")
    ax.set_title("Field collapses onto a single variable (THEORY.md Prop 3)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(d / "phase_collapse.png", dpi=130); plt.close(fig)

    # (4) Compensated-gain boundary: ln u*(S) vs S, slope -1.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    Sb = np.linspace(max(S_grid.min(), 1.5), S_grid.max() + 2.0, 30)
    for level in (0.5, 0.25, 0.1):
        ub = theory.phase_boundary_curve(level, Sb)
        slope = np.polyfit(Sb, np.log(ub), 1)[0]
        ax.plot(Sb, ub, label=f"$r e^S$={level}  (slope {slope:.2f})")
    ax.set_yscale("log")
    ax.set_xlabel(r"$S=\tau/T_c$")
    ax.set_ylabel(r"boundary $\bar\Delta/\sigma_D$")
    ax.set_title(r"Phase boundary: slope $\to -1$ in $(\ln\bar\Delta/\sigma_D,\,S)$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(d / "phase_boundary.png", dpi=130); plt.close(fig)

    if ratio_pucb is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.scatter(ratio_ana.ravel(), ratio_pucb.ravel(), s=18, alpha=0.7,
                   color="C3")
        ax.set_xlabel("analytic causal frontier (†)")
        ax.set_ylabel("PUCB achieved relative gain")
        ax.set_title("PUCB approaches the causal frontier")
        ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(d / "phase_pucb.png", dpi=130); plt.close(fig)


# --- main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run, skip plotting, assert finiteness (<60s)")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    mode = cfg.get("mode", "compare")

    if mode == "compare":
        out = run_compare(cfg, smoke=args.smoke)
    elif mode == "delay_tax":
        out = run_delay_tax(cfg, smoke=args.smoke)
    elif mode == "phase_transition":
        out = run_phase_transition(cfg, smoke=args.smoke)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    d = result_dir(cfg)
    (d / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (d / "results.json").write_text(json.dumps(out, indent=2, default=str))

    if args.smoke:
        _assert_finite(out)
        print("SMOKE OK:", d.name)
    else:
        print("wrote", d)


def _assert_finite(out):
    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, (int, float)):
            assert math.isfinite(x), f"non-finite result: {x}"
    walk(out)


if __name__ == "__main__":
    main()
