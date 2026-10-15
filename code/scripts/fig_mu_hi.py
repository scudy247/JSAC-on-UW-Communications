"""
Multi-user uplink at 100 packets per point (PLAN: Phase 8; results/b22_multiuser/hi_16qam_buoy.json, 15 Oct 2026).
  fig_mu_hi.png  left: mean SINR per user vs number of users K (buoy users, 16-QAM, 8 guard subcarriers, 30 dB),
                 95 % confidence bars, for the two signal-only receivers, ours and the true-Doppler bound;
                 right: ours minus the better signal-only receiver at each point, 20 and 30 dB, with its 95 %
                 confidence interval (paired per packet).
Colours as scripts/fig_gains.py (signal-only blue, ours orange, bound grey dashed); the second signal-only receiver
(pilot tracking) in aqua. Usage: python scripts/fig_mu_hi.py [guard]. Output: results/figures/fig_mu_hi.png.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from uwofdm.paths import output_dir  # noqa: E402
from fig_gains import AQUA, BLUE, GREY, INK, MUTED, ORANGE, finish, style_axis  # noqa: E402

RECV = (("branch_signal", "Signal-only, one estimate per user (Tu et al.)", BLUE, "o", "-"),
        ("branch_pilot", "Signal-only, pilot tracking per user", AQUA, "^", "-"),
        ("branch_sensor", "Sensor-aided, one branch per user (ours)", ORANGE, "D", "-"),
        ("branch_genie", "True Doppler of each user (upper bound)", GREY, None, "--"))


def best_signal_gain(r):
    """Ours minus the better signal-only receiver (by mean SINR) at this point, with its paired 95 % interval."""
    base = max(("branch_signal", "branch_pilot"), key=lambda k: r[k]["mean_sinr_db"])
    g = r["branch_sensor"]["gain_over_" + base]
    return g["mean_db"], g["ci95"]


def main():
    guard = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = json.load(open(os.path.join(output_dir("b22_multiuser"), "hi_16qam_buoy.json")))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    ax = axes[0]
    sel = sorted((r for r in rows if r["guard"] == guard and r["snr_db"] == 30), key=lambda r: r["K"])
    Ks = [r["K"] for r in sel]
    for key, lab, col, mk, ls in RECV:
        m = [r[key]["mean_sinr_db"] for r in sel]
        e = [r[key]["mean_sinr_ci95"] for r in sel]
        ax.errorbar(Ks, m, yerr=e, color=col, lw=2.2, ls=ls, marker=mk, ms=8 if mk else 0, capsize=4, label=lab,
                    markeredgecolor="white", markeredgewidth=0.8)
    style_axis(ax, f"30 dB SNR, {guard} empty subcarriers between users", "Number of users sharing the band, K",
               "Mean SINR per user (dB)")
    ax.set_xticks((2, 4, 8))
    ax.set_xlim(1.5, 8.5)
    ax = axes[1]
    for snr, mk, ls in ((30, "D", "-"), (20, "s", "--")):
        pts = sorted((r["K"], *best_signal_gain(r)) for r in rows if r["guard"] == guard and r["snr_db"] == snr)
        if pts:
            k, g, c = zip(*pts)
            ax.errorbar(k, g, yerr=c, color=ORANGE, lw=2.2, ls=ls, marker=mk, ms=8, capsize=4,
                        markeredgecolor="white", markeredgewidth=0.8)
            for x, y in zip(k, g):
                ax.text(x + 0.18, y, f"{y:+.1f} dB", fontsize=10.5, fontweight="bold", color=INK, va="center")
            i = k.index(4) if 4 in k else 0  # line label at K = 4: 30 dB above its point, 20 dB below
            ax.text(k[i], g[i] + (0.22 if snr == 30 else -0.22), f"{snr} dB SNR", fontsize=10, color=MUTED,
                    ha="center", va="bottom" if snr == 30 else "top")
    ax.axhline(0, color="#9a9a9a", lw=1.0)
    style_axis(ax, "Ours minus the better signal-only receiver", "Number of users sharing the band, K",
               "SINR gain (dB), 95 % interval")
    ax.set_xticks((2, 4, 8))
    ax.set_xlim(1.5, 9.2)
    finish(fig, axes[0], 2, "Multi-user uplink, 100 packets per point: the gain appears as the band gets shared",
           "fig_mu_hi.png")
    print("written", os.path.join(output_dir("figures"), "fig_mu_hi.png"))


if __name__ == "__main__":
    main()
