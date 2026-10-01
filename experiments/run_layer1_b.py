"""Layer 1 (synthetic) regret experiments for Proposal B.

Synthetic counterfactual outcome tables (every arm at every slot, same table for every agent
within a seed: common random numbers), delayed feedback through TableEnv, regret against the
true arm means. Two reward models (THEORY-B §2.3):
  sas    (branch S): reward = mu_k + SaS(alpha, c) noise; noise stream from the same law.
  evm_db (branch E): reward = 10 log10(P_k / P_hat_n), P_hat_n = mean |n|^2 over n_sym pilot symbols of
         complex isotropic SaS noise (the gamma-hat of reports/group_evidence_phase0.md §1); true means
         by a large seeded Monte Carlo; noise stream = real parts of the same noise law.
ACK = 1{reward > ack_threshold}. Oracle baselines (plan §7, no straw men) get their parameters from
the true law. NIR-UCB v0 uses the identity noise->reward map, i.e. it is MIS-SPECIFIED in branch E.

  python -m experiments.run_layer1_b --config experiments/configs/b_layer1_smoke.yaml --smoke
  python -m experiments.run_layer1_b --config ... --dry-run
  python -m experiments.run_layer1_b --config ... --n-workers 4
"""

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

from uwsb.bandits import robust as rb
from uwsb.envs.table_env import OutcomeTable, TableEnv, best_fixed_arm_regret
from uwsb.noise.stable import sas_complex_isotropic, sas_real
from uwsb.runtime import config_hash, result_dir, run_metadata


# --- reward models ----------------------------------------------------------------------------
def _evm_db(snr_db, n_pkt, cfg, rng):
    noise = cfg["noise"]
    n_sym = int(cfg["evm"]["n_sym"])
    z = sas_complex_isotropic(noise["alpha"], noise["c"], (n_pkt, n_sym), rng)
    return 10.0 * np.log10(10 ** (snr_db / 10.0) / np.mean(np.abs(z) ** 2, axis=1))


_CACHE = {}          # per-process memo of the Monte Carlo quantities (keyed by the config)


def _memo(tag, cfg, fn):
    key = (tag, json.dumps({k: cfg[k] for k in ("reward_model", "noise", "arms", "evm") if k in cfg},
                           sort_keys=True))
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def true_means(cfg):
    return _memo("means", cfg, lambda: _true_means(cfg))


def _true_means(cfg):
    if cfg["reward_model"] == "sas":
        return np.asarray(cfg["arms"]["means"], float)
    rng = np.random.default_rng(int(cfg["evm"]["mc_seed"]))
    return np.array([_evm_db(s, int(cfg["evm"]["mc_packets"]), cfg, rng).mean()
                     for s in cfg["arms"]["snr_db"]])


def ack_threshold(cfg):
    """A number, or {between_arms: [i, j], frac: f} = mu_i + f (mu_j - mu_i) on the true means
    (used by the sweeps, where the reward scale moves with alpha and the gaps)."""
    a = cfg["ack_threshold"]
    if isinstance(a, dict):
        mu = true_means(cfg)
        i, j = a["between_arms"]
        return float(mu[i] + float(a["frac"]) * (mu[j] - mu[i]))
    return float(a)


def make_table(cfg, seed):
    rng = np.random.default_rng(seed)
    T, nu = int(cfg["T"]), int(cfg["nu"])
    noise = cfg["noise"]
    mu = true_means(cfg)
    if cfg["reward_model"] == "sas":
        g = mu + sas_real(noise["alpha"], noise["c"], (T, mu.size), rng)
        stream = sas_real(noise["alpha"], noise["c"], (T, nu), rng)
    elif cfg["reward_model"] == "evm_db":
        g = np.column_stack([_evm_db(s, T, cfg, rng) for s in cfg["arms"]["snr_db"]])
        stream = sas_complex_isotropic(noise["alpha"], noise["c"], (T, nu), rng).real
    else:
        raise ValueError(f"unknown reward_model {cfg['reward_model']!r}")
    ack = (g > ack_threshold(cfg)).astype(float)
    return OutcomeTable(fields={"gamma_db": g, "ack": ack}, truth_mean=np.tile(mu, (T, 1)),
                        noise=stream, T_slot_s=float(cfg.get("T_slot_s", 1.0)))


# --- oracle parameters for the baselines ------------------------------------------------------
def _evm_mc_samples(cfg):
    def draw():
        rng = np.random.default_rng(int(cfg["evm"]["mc_seed"]) + 1)
        return [_evm_db(s, int(cfg["evm"]["mc_packets"]), cfg, rng) for s in cfg["arms"]["snr_db"]]
    return _memo("mc_samples", cfg, draw)


def oracle_moments(cfg, eps):
    """(u raw-moment bound, v centred-moment bound) of order 1 + eps of the reward noise."""
    if cfg["reward_model"] == "sas":
        a, c = cfg["noise"]["alpha"], cfg["noise"]["c"]
        m = rb.sas_abs_moment(1 + eps, a, c)             # raises if 1 + eps >= alpha (infinite)
        return 2 ** eps * (np.max(np.abs(true_means(cfg))) ** (1 + eps) + m), m
    xs = _evm_mc_samples(cfg)
    return (max(float(np.mean(np.abs(x) ** (1 + eps))) for x in xs),
            max(float(np.mean(np.abs(x - x.mean()) ** (1 + eps))) for x in xs))


def oracle_sigma(cfg):
    """Gaussian-width scale for the naive agents: sqrt(2) c for SaS (the Gaussian-equivalent
    scale; the variance is infinite for alpha < 2), the true std of gamma-hat in branch E."""
    if cfg["reward_model"] == "sas":
        return math.sqrt(2) * cfg["noise"]["c"]
    return max(float(np.std(x)) for x in _evm_mc_samples(cfg))


def make_agent(spec, cfg, K, seed):
    t, p = spec["type"], dict(spec.get("params", {}))
    rng = np.random.default_rng(seed)
    if t in ("empirical_ucb", "gaussian_ts"):
        sigma = oracle_sigma(cfg) if p.get("sigma") == "oracle" else float(p.get("sigma", 1.0))
        sigma *= float(p.get("sigma_scale", 1.0))        # tuned-width variant (scale swept)
        return rb.EmpiricalUCB(K, sigma=sigma) if t == "empirical_ucb" else rb.GaussianTS(K, sigma=sigma, rng=rng)
    if t == "ack_ucb1":
        return rb.AckUCB1(K)
    if t == "bernoulli_ts":
        return rb.BernoulliTS(K, rng=rng)
    if t in ("trunc_ucb", "mom_ucb"):
        eps = float(p["eps"])
        u, v = oracle_moments(cfg, eps)
        return rb.TruncatedMeanUCB(K, eps=eps, u=u) if t == "trunc_ucb" else rb.MedianOfMeansUCB(K, eps=eps, v=v)
    if t == "catoni_ucb":                                 # eps = 1: needs a finite variance
        return rb.CatoniUCB(K, v=oracle_moments(cfg, 1.0)[1] if p.get("v", "oracle") == "oracle" else float(p["v"]))
    if t == "clipped_ucb":                                # clip to centre +- b, centre = mid-range of the true means
        mu = true_means(cfg)
        centre = 0.5 * (float(mu.min()) + float(mu.max()))
        return rb.ClippedUCB(K, lo=centre - float(p["b"]), hi=centre + float(p["b"]))
    if t == "median_greedy":
        return rb.MedianFilterGreedy(K, window=int(p.get("window", 16)))
    if t == "adar_ucb":
        return rb.AdaRUCB(K)
    if t == "nir_ucb":
        return rb.NIRUCB(K, **p)
    raise ValueError(f"unknown agent type {t!r}")


# --- runs -------------------------------------------------------------------------------------
def agent_seed(cfg, seed, j, name):
    """Agent random stream. Default (configs without `agent_seeding`): seed * 1000 + list position, as
    in all results so far. `agent_seeding: name` (THEORY-B G-7): keyed by the agent's name, so adding or
    reordering agents leaves every other agent's results unchanged."""
    mode = cfg.get("agent_seeding", "position")
    if mode == "position":
        return seed * 1000 + j
    if mode == "name":
        return [int(seed), int.from_bytes(hashlib.sha1(name.encode()).digest()[:4], "little")]
    raise ValueError(f"agent_seeding must be 'position' or 'name', got {mode!r}")


def run_seed(args):
    cfg, s = args
    seed = int(cfg["seed"]) + s
    tab = make_table(cfg, seed)
    idx = checkpoints(cfg)
    out = {"_best_fixed_arm_final": float(best_fixed_arm_regret(tab, int(cfg["T"])).sum())}
    for j, spec in enumerate(cfg["agents"]):
        agent = make_agent(spec, cfg, tab.n_arms, agent_seed(cfg, seed, j, spec["name"]))
        t0 = time.perf_counter()
        res = TableEnv(tab, int(cfg["tau_rt_slots"])).run(agent)
        out[spec["name"]] = {"cum_regret": np.cumsum(res.regret_inst)[idx].tolist(),   # at checkpoints
                             "wall_s": time.perf_counter() - t0}
    return out


def checkpoints(cfg):
    T = int(cfg["T"])
    return np.unique(np.linspace(0, T - 1, min(T, 200)).astype(int))


def summarise(cfg, per_seed):
    idx = checkpoints(cfg)
    summary = {"checkpoints": (idx + 1).tolist(), "agents": {},
               # reference policies: dynamic oracle = 0 by definition (regret is measured against it)
               "best_fixed_arm_final_mean": float(np.mean([ps["_best_fixed_arm_final"] for ps in per_seed]))}
    for spec in cfg["agents"]:
        R = np.array([ps[spec["name"]]["cum_regret"] for ps in per_seed])
        n = R.shape[0]
        half = 1.96 * R.std(axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros(R.shape[1])
        summary["agents"][spec["name"]] = {
            "mean": R.mean(axis=0).tolist(), "ci95_half": half.tolist(),
            "final_mean": float(R[:, -1].mean()), "final_ci95_half": float(half[-1]),
            "final_per_seed": R[:, -1].tolist(),
            "wall_s_mean": float(np.mean([ps[spec["name"]]["wall_s"] for ps in per_seed]))}
    return summary


def plot(cfg, summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = summary["checkpoints"]
    for name, a in summary["agents"].items():
        m, h = np.array(a["mean"]), np.array(a["ci95_half"])
        ax.plot(x, m, label=name)
        ax.fill_between(x, m - h, m + h, alpha=0.2)
    ax.set_xlabel("decision t")
    ax.set_ylabel("cumulative regret")
    ax.set_title(f"{cfg['name']}: {cfg['reward_model']}, alpha={cfg['noise']['alpha']}, "
                 f"tau={cfg['tau_rt_slots']}, {cfg['n_seeds']} seeds (95% CI)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "regret.png", dpi=130)
    plt.close(fig)


def dry_run(cfg, n_workers):
    """Time one seed at two horizons, fit a*T + b*T^2 per agent, extrapolate."""
    probes = [min(int(cfg["T"]), 500), min(int(cfg["T"]), 1000)]
    t0 = time.perf_counter()
    true_means(cfg)
    if cfg["reward_model"] == "evm_db":
        _evm_mc_samples(cfg)
    t_mc = time.perf_counter() - t0
    t0 = time.perf_counter()
    make_table(cfg, 0)
    t_tab = time.perf_counter() - t0
    print(f"one-off Monte Carlo per worker: {t_mc:.1f} s; table generation per seed: {t_tab:.1f} s")
    times = []
    for Tp in probes:
        c2 = dict(cfg, T=Tp)
        times.append({k: v["wall_s"] for k, v in run_seed((c2, 0)).items() if not k.startswith("_")})
    T = int(cfg["T"])
    total = 0.0
    print(f"{'agent':28s} {'s at T=' + str(probes[0]):>12s} {'s at T=' + str(probes[1]):>12s} {'est s/seed':>11s}")
    for spec in cfg["agents"]:
        t1, t2 = times[0][spec["name"]], times[1][spec["name"]]
        A = np.array([[probes[0], probes[0] ** 2], [probes[1], probes[1] ** 2]], float)
        a, b = np.linalg.lstsq(A, np.array([t1, t2]), rcond=None)[0]
        est = max(a * T + b * T * T, t2 * T / probes[1])
        total += est
        print(f"{spec['name']:28s} {t1:12.2f} {t2:12.2f} {est:11.1f}")
    seeds = int(cfg["n_seeds"])
    total += t_tab
    print(f"total ≈ {(total * seeds + t_mc * n_workers) / 60:.1f} CPU-min; "
          f"≈ {(total * math.ceil(seeds / n_workers) + t_mc) / 60:.1f} min wall with {n_workers} workers "
          f"(agent times extrapolated from T = {probes}: rough)")
    print(f"results dir: {result_dir(cfg)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="assert finite results, no plot")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.dry_run:
        dry_run(cfg, args.n_workers)
        return 0
    out_dir = result_dir(cfg)
    if (out_dir / "summary.json").exists():
        print(f"already complete: {out_dir}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with Pool(args.n_workers) as pool:
        per_seed = pool.map(run_seed, [(cfg, s) for s in range(int(cfg["n_seeds"]))])
    summary = summarise(cfg, per_seed)
    summary["true_means"] = true_means(cfg).tolist()
    summary["ack_threshold"] = ack_threshold(cfg)
    summary["wall_total_s"] = time.perf_counter() - t0
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (out_dir / "metadata.json").write_text(json.dumps(run_metadata(), indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    if args.smoke:
        assert all(math.isfinite(a["final_mean"]) for a in summary["agents"].values())
        print("SMOKE OK:", out_dir.name, f"({summary['wall_total_s']:.1f} s)")
    else:
        plot(cfg, summary, out_dir)
        print("wrote", out_dir)
    for name, a in summary["agents"].items():
        print(f"  {name:28s} final regret {a['final_mean']:9.1f} ± {a['final_ci95_half']:.1f}  "
              f"({a['wall_s_mean']:.1f} s/seed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
