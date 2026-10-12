"""
Figures of the end-to-end results for the deck (PLAN: Phase 8), from the JSON files of the chain experiments.
  fig_sinr_rough.png   SINR vs SNR, buoy-buoy OOI, 1.05 s and 4.05 s packets (results/e2_chain/points.json)
  fig_per_long.png     PER vs SNR, buoy-buoy OOI, 4.05 s packets, QPSK and 16-QAM (per_qpsk_27.json, per_16qam_27.json)
  fig_multipath.png    SINR at 30 dB, buoy-buoy OOI over Bellhop multipath, per receiver and scattered power
                       (multipath.json)
Colours and markers follow the receiver (dataviz default categorical slots, fixed order); one y-axis per panel.
Output: results/figures/. Usage: python scripts/fig_results.py [suffix] (suffix '_hi' reads the high-statistics
files where they exist).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from uwofdm.paths import output_dir  # noqa: E402

STYLE = {  # receiver key -> (label, colour, marker)
    "signal_only": ("Li et al. / Lee et al. front end", "#2a78d6", "o"),
    "cho_kim": ("Cho–Kim 2026", "#eb6834", "s"),
    "signal_kalman": ("signal-only Kalman", "#1baf7a", "^"),
    "fused_32": ("sensor-aided fusion (ours)", "#d55181", "D"),
    "fused": ("sensor-aided fusion (ours)", "#d55181", "D"),
    "full_chain": ("sensor chain, real header (ours)", "#7a3b9e", "v"),
    "tu_grid": ("Tu-like multi-branch", "#8a8a8a", "x"),
    "signal_sic": ("signal-only cancellation", "#0b6e4f", "P"),
    "imu_sic": ("sensor cancellation (ours)", "#d55181", "*"),
}
RES = output_dir("e2_chain")


def load(name, suffix):
    p = os.path.join(RES, name.replace(".json", suffix + ".json"))
    return json.load(open(p if os.path.exists(p) else os.path.join(RES, name)))


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(color="#e6e6e6", lw=0.6)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_sinr(out):
    rows = json.load(open(os.path.join(RES, "points.json")))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, n in zip(axes, (7, 27)):
        for k in ("signal_only", "cho_kim", "signal_kalman", "fused_32"):
            pts = sorted((r["snr_db"], r["sinr_db"]) for r in rows if r["scenario"].startswith("buoy - buoy")
                         and r["n_data"] == n and r["receiver"] == k)
            lab, col, mk = STYLE[k]
            ax.plot(*zip(*pts), color=col, marker=mk, lw=2, ms=6, label=lab)
        ax.plot([5, 30], [5, 30], color="#bbbbbb", lw=1, ls="--", label="noise limit (SINR = SNR)")
        style_axis(ax, f"buoy – buoy, OOI Irminger, {'1.05' if n == 7 else '4.05'} s packets",
                   "per-subcarrier SNR [dB]", "SINR [dB]")
    axes[0].set_ylim(-5, 32)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(out, "fig_sinr_rough.png"), dpi=150)
    plt.close(fig)


def fig_per(out, suffix):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (name, mod) in zip(axes, (("per_qpsk_27.json", "QPSK"), ("per_16qam_27.json", "16-QAM"))):
        rows = [r for r in load(name, suffix) if r["scenario"].startswith("buoy - buoy")]
        for k in ("signal_only", "cho_kim", "signal_kalman", "full_chain"):
            pts = sorted((r["snr_db"], max(r[f"per_{k}"], 1e-3)) for r in rows)
            lab, col, mk = STYLE[k]
            ax.semilogy(*zip(*pts), color=col, marker=mk, lw=2, ms=6, label=lab)
        style_axis(ax, f"buoy – buoy, OOI Irminger, 4.05 s, {mod}, rate-1/2 code", "per-subcarrier SNR [dB]",
                   "packet error rate")
        ax.text(0.98, 0.04, "points at 1e-3: no error observed", transform=ax.transAxes, ha="right", fontsize=7,
                color="#555555")
    axes[0].set_ylim(8e-4, 1.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "fig_per_long.png"), dpi=150)
    plt.close(fig)


def fig_multipath(out, suffix):
    rows = [r for r in load("multipath.json", suffix) if r["scenario"].startswith("buoy - buoy") and r["snr_db"] == 30]
    keys = ("signal_only", "signal_kalman", "tu_grid", "signal_sic", "imu_sic")
    rhos = (None, -30, -20)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    w = 0.16
    for i, k in enumerate(keys):
        vals = [next(r["sinr_db"] for r in rows if r["receiver"] == k and r["rho_db"] == rho) for rho in rhos]
        lab, col, _ = STYLE[k]
        xs = [j + (i - 2) * w for j in range(len(rhos))]
        ax.bar(xs, vals, width=w - 0.02, color=col, label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.3, f"{v:.0f}", ha="center", fontsize=7, color="#333333")
    ax.set_xticks(range(len(rhos)))
    ax.set_xticklabels(["no scattered power", "scattered −30 dB", "scattered −20 dB"], fontsize=8)
    style_axis(ax, "buoy – buoy over Bellhop multipath (direct + bottom bounce, heave on the bounce), SNR 30 dB",
               "", "SINR after pilot channel estimate [dB]")
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
    ax.set_ylim(0, 27)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_multipath.png"), dpi=150)
    plt.close(fig)


def main():
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""
    out = output_dir("figures")
    fig_sinr(out)
    fig_per(out, suffix)
    fig_multipath(out, suffix)
    print("figures in", out)


if __name__ == "__main__":
    main()
