"""
Figures of where sensor-aided Doppler estimation gains (PLAN: Phase 8; results of 14-15 Oct 2026).
  fig_multipath_cdf.png  distribution of per-packet SINR at 30 dB SNR, buoy-buoy OOI over a two-path Bellhop channel:
                         1 s QPSK, 1 s 16-QAM, 4 s QPSK (results/b13_multipath/*.json)
  fig_mu_users.png       multi-user uplink, mean SINR vs number of users at 30 dB, 8 guard subcarriers; seabed users
                         and buoy users (results/b22_multiuser/points.json)
  fig_mu_guard.png       multi-user uplink, buoy users, mean SINR vs guard subcarriers (band overhead) at 30 dB
Colours follow the role, the same in every figure (dataviz default categorical slots 1-3, which pass the
colour-blind checks for all pairs): signal-only best baseline blue, sensor-aided (ours) orange, no cancellation aqua;
genie / ideal references grey dashed; each series also has its own marker. One y-axis per panel.
Usage: python scripts/fig_gains.py. Output: results/figures/.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from uwofdm.paths import output_dir  # noqa: E402

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a8a"
INK, MUTED = "#0b0b0b", "#52514e"
ACTIVE = 992  # active subcarriers (waveform.OFDM)
OUT = output_dir("figures")


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=9, color=INK)
    ax.set_xlabel(xlabel, fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.grid(color="#e6e6e6", lw=0.6)
    ax.tick_params(labelsize=8, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbdbd")


def legend_top(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.98))


def fig_multipath_cdf():
    cases = (("bb_iso_1s", "1 s packets, QPSK"), ("bb_iso_1s_16qam", "1 s packets, 16-QAM"),
             ("bb_iso_4s", "4 s packets, QPSK"))
    recv = (("signal_kalman", "signal-only Kalman, no cancellation", AQUA, "^", "-"),
            ("signal_sic", "signal-only path cancellation", BLUE, "o", "-"),
            ("sensor_sic", "sensor-aided path cancellation (ours)", ORANGE, "D", "-"),
            ("ideal_sic", "ideal cancellation (true path Doppler)", GREY, None, "--"))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, (cfg, title) in zip(axes, cases):
        d = json.load(open(os.path.join(output_dir("b13_multipath"), cfg + ".json")))
        for key, lab, col, mk, ls in recv:
            r = [x for x in d["rows"] if x.get("receiver") == key and x["snr_db"] == 30][0]
            v = np.sort(r["per_packet_db"])
            p = np.arange(1, len(v) + 1) / len(v)
            ax.step(v, p, where="post", color=col, lw=2, ls=ls, label=lab)
            if mk:
                ax.plot(v, p, ls="none", marker=mk, ms=4, color=col, markeredgecolor="white", markeredgewidth=0.6)
        style_axis(ax, f"{title} ({d['n_packets']} packets)", "SINR per packet [dB]",
                   "fraction of packets" if ax is axes[0] else "")
        ax.set_xlim(left=0)
    legend_top(fig, axes[0], 4)
    fig.suptitle("Two-path buoy-buoy link, 30 dB SNR: the sensor prior lifts the worst packets", fontsize=10,
                 color=INK, y=1.12)
    fig.savefig(os.path.join(OUT, "fig_multipath_cdf.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


MU = (("branch_signal", "per-user signal-only branch (Tu et al.)", BLUE, "o", "-"),
      ("branch_sensor", "per-user sensor-aided branch (ours)", ORANGE, "D", "-"),
      ("branch_genie", "per-user branch with true Doppler", GREY, None, "--"))


def fig_mu_users(rows):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, case, title in zip(axes, ("seabed users", "buoy users"),
                               ("Seabed users (gateway IMU only)", "Buoy users (side information + gateway IMU)")):
        for key, lab, col, mk, ls in MU:
            pts = sorted((r["K"], r[key]["mean_sinr_db"]) for r in rows
                         if r["scenario"] == case and r["guard"] == 8 and r["snr_db"] == 30)
            ax.plot(*zip(*pts), color=col, lw=2, ls=ls, marker=mk, ms=7 if mk else 0, label=lab,
                    markeredgecolor="white", markeredgewidth=0.8)
        style_axis(ax, title, "users sharing the band, K", "mean SINR per user [dB]" if ax is axes[0] else "")
        ax.set_xticks((2, 4, 8))
    legend_top(fig, axes[0], 3)
    fig.suptitle("Multi-user uplink, 30 dB SNR, 8 guard subcarriers (8 packets x K users per point)", fontsize=10,
                 color=INK, y=1.12)
    fig.savefig(os.path.join(OUT, "fig_mu_users.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_mu_guard(rows):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, K in zip(axes, (2, 4, 8)):
        sel = [r for r in rows if r["scenario"] == "buoy users" and r["K"] == K and r["snr_db"] == 30]
        for key, lab, col, mk, ls in MU:
            pts = [(i, r[key]["mean_sinr_db"]) for i, g in enumerate((0, 8, 32)) for r in sel if r["guard"] == g]
            ax.plot(*zip(*pts), color=col, lw=2, ls=ls, marker=mk, ms=7 if mk else 0, label=lab,
                    markeredgecolor="white", markeredgewidth=0.8)
        ref = [r["branch_signal"]["mean_sinr_db"] for r in sel if r["guard"] == 32][0]
        ax.axhline(ref, color=BLUE, lw=0.8, ls=":")
        guards = (0, 8, 32)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{g}\n({100 * (K - 1) * g / ACTIVE:.0f}% of band)" for g in guards])
        ax.set_xlim(-0.3, 2.3)
        style_axis(ax, f"K = {K} buoy users", "guard subcarriers between users",
                   "mean SINR per user [dB]" if ax is axes[0] else "")
    legend_top(fig, axes[0], 3)
    fig.suptitle("Buoy users, 30 dB SNR: with no guard, the sensor-aided branch reaches what signal-only gets with 32\n"
                 "guard subcarriers (dotted) for K = 2 (+4.0 dB) and K = 4 (-0.4 dB), not quite for K = 8 (-0.6 dB)",
                 fontsize=10, color=INK, y=1.16)
    fig.savefig(os.path.join(OUT, "fig_mu_guard.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_multipath_cdf()
    rows = json.load(open(os.path.join(output_dir("b22_multiuser"), "points.json")))
    fig_mu_users(rows)
    fig_mu_guard(rows)
    print("written to", OUT)


if __name__ == "__main__":
    main()
