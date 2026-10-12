"""
E1 (PLAN: Phase 4), third run: stage-aware bit allocation.

What the transmitter should send depends on what the receiver estimates itself. For each receiver stage, the
interference caused by a unit error in each Legendre coefficient after that stage is computed exactly
(sideinfo.stage_weights), and bits go where they reduce the weighted error most per bit (sideinfo.allocate_weighted;
a coefficient is started with 2 bits). Same setting, budgets and scoring as scripts/e1_allocation.py, whose
size-based allocation is drawn for comparison.
Output: results/e1_sideinfo/stage_allocation.json, sir_vs_bits_stage_alloc.png; printed bits within 1 dB of
pre-compensation with both allocations.
"""
import json
import os

import numpy as np

from uwofdm import placement as P, sideinfo, theory
from uwofdm.experiment import at_times
from uwofdm.paths import output_dir
from e1_allocation import BUDGETS, K_MAX
from e1_sideinfo import FC, N_PACKETS, SCENARIOS, SIGMA_RX, STAGES, STYLE, series


def run_case(s, T, rng):
    rxm = P.ReceiverModel(packet_s=T, stages=STAGES)
    est = {st: P.stage_estimator(st, rxm) for st in STAGES}
    tg, windows = est["none"][0], est["none"][3]
    f, S = theory.spectrum(s["tx_est"], s["fs"])
    std = sideinfo.coefficient_std(f, S, tg[-1], K_MAX)
    ranges = np.abs(np.mean(s["tx_est"])) + sideinfo.LOAD * std
    B = sideinfo.basis(tg, tg[-1], K_MAX)
    alloc = {}
    for st in STAGES:
        w = sideinfo.stage_weights(tg, est[st][1], est[st][2], windows, FC, K_MAX)
        alloc.update({(b, st): sideinfo.allocate_weighted(std, w, b) for b in BUDGETS})
    starts = rng.uniform(s["t"][0] + 30, s["t"][-1] - T - 10, N_PACKETS)
    acc = {}
    for t0 in starts:
        g = {k: at_times(s[k], s["t"], s["fs"], t0 + tg) for k in ("tx_true", "tx_est", "rx_true", "rx_est", "slow")}
        base = g["rx_true"] - g["rx_est"] + g["slow"]
        c = np.linalg.lstsq(B, g["tx_est"], rcond=None)[0]
        noise = {st: SIGMA_RX * rng.standard_normal(len(est[st][1])) for st in STAGES}
        for st in STAGES:
            t_s, W = est[st][1], est[st][2]
            hats = {"rx_alone": np.zeros(len(tg)), "applied": g["tx_est"]}
            hats.update({b: B @ sideinfo.quantise_alloc(c, alloc[(b, st)], ranges) for b in BUDGETS})
            for key, hat in hats.items():
                r0 = g["tx_true"] - hat + base
                r = r0 if W is None else r0 - W @ (np.interp(t_s, tg, r0) + noise[st])
                acc.setdefault((key, st), []).append(sideinfo.interference(r, tg, windows, FC))
    return {k: float(-10 * np.log10(np.mean(v))) for k, v in acc.items()}, alloc


def main():
    out = output_dir("e1_sideinfo")
    old = json.load(open(os.path.join(out, "allocation.json")))
    rows, rng = [], np.random.default_rng(1)
    for label, (make, rx_moves) in SCENARIOS.items():
        s = series(*make(), rx_moves, rng)
        for T in (1.0, 4.0):
            res, alloc = run_case(s, T, rng)
            for (key, st), sir in res.items():
                row = dict(scenario=label, packet_s=T, stage=st, sir_db=sir)
                if isinstance(key, str):
                    row["reference"] = key
                else:
                    a = alloc[(key, st)]
                    row.update(bits=key, coefficients_sent=int(np.sum(a > 0)), allocation=a.tolist())
                rows.append(row)
            for st in STAGES:
                app = res[("applied", st)]
                need = next((b for b in BUDGETS if res[(b, st)] >= app - 1.0), None)
                prev = {r["bits"]: r["sir_db"] for r in old if r["scenario"] == label and r["packet_s"] == T
                        and r["stage"] == st and "bits" in r}
                need_old = next((b for b in BUDGETS if prev.get(b, -99) >= app - 1.0), None)
                first = [int(k) for k in np.nonzero(alloc[(32, st)])[0]]
                print(f"{label:30s} T {T:3.0f} s  {st:5s}: applied {app:5.1f}; 32 bits {res[(32, st)]:5.1f} "
                      f"(size-based {prev.get(32, float('nan')):5.1f}); within 1 dB with {need} bits "
                      f"(size-based {need_old}); coefficients sent at 32 bits {first}", flush=True)
        with open(os.path.join(out, "stage_allocation.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    plot(rows, old, os.path.join(out, "sir_vs_bits_stage_alloc.png"))


def plot(rows, old, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharey=True)
    for j, label in enumerate(SCENARIOS):
        for i, T in enumerate((1.0, 4.0)):
            ax = axes[i, j]
            for st, (col, name) in STYLE.items():
                sel = lambda rr: [r for r in rr if r["scenario"] == label and r["packet_s"] == T and r["stage"] == st]
                q = sorted((r["bits"], r["sir_db"]) for r in sel(rows) if "bits" in r)
                q_old = sorted((r["bits"], r["sir_db"]) for r in sel(old) if "bits" in r)
                ax.plot(*zip(*q_old), color=col, lw=1, alpha=0.5)
                ax.plot(*zip(*q), color=col, lw=2, marker="o", ms=4, label=name)
                ref = {r["reference"]: r["sir_db"] for r in sel(rows) if "reference" in r}
                ax.axhline(ref["applied"], color=col, lw=1, ls="--")
                ax.axhline(ref["rx_alone"], color=col, lw=1, ls=":")
            ax.set_xscale("log", base=2)
            ax.set_title(f"{label}, {T:.0f} s packets", fontsize=9)
            ax.grid(color="#e6e6e6", lw=0.6)
            ax.tick_params(labelsize=8)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 1:
                ax.set_xlabel("side-information bits per packet", fontsize=8)
            if j == 0:
                ax.set_ylabel("SIR between subcarriers [dB]", fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="#555555", lw=1, alpha=0.5), plt.Line2D([], [], color="#555555", ls="--"),
          plt.Line2D([], [], color="#555555", ls=":")]
    l += ["size-based allocation (previous run)", "transmitter applies its estimate (no bits)",
          "receiver alone (no transmitter information)"]
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
