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
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from uwofdm.paths import output_dir  # noqa: E402

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a8a"
INK, MUTED = "#0b0b0b", "#3d3c3a"
ACTIVE = 992  # active subcarriers (waveform.OFDM)
OUT = output_dir("figures")


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11.5, color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=11, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=11, color=MUTED)
    ax.grid(color="#e6e6e6", lw=0.6)
    ax.tick_params(labelsize=10, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#bdbdbd")


def finish(fig, ax, ncol, title, caption, name, wrap=150):
    """Headline (the claim) and legend above the panels, caption (how to read it, setup) below."""
    fig.tight_layout(rect=(0, 0, 1, 0.78), w_pad=2.0)
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=14, fontweight="bold", color=INK)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.925), ncol=ncol, fontsize=10.5, frameon=False,
               handlelength=2.6, columnspacing=2.0)
    fig.text(0.01, -0.01, textwrap.fill(caption, wrap), ha="left", va="top", fontsize=10, color=MUTED,
             linespacing=1.4)
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def fig_multipath_cdf():
    cases = (("bb_iso_1s", "1 s packets, QPSK"), ("bb_iso_1s_16qam", "1 s packets, 16-QAM"),
             ("bb_iso_4s", "4 s packets, QPSK"))
    recv = (("signal_kalman", "No path cancellation (signal-only Kalman)", AQUA, "^", "-"),
            ("signal_sic", "Path cancellation, signal-only estimates", BLUE, "o", "-"),
            ("sensor_sic", "Path cancellation, sensor-aided estimates (ours)", ORANGE, "D", "-"),
            ("ideal_sic", "Ideal: true Doppler of each path", GREY, None, "--"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.9), sharey=True)
    for ax, (cfg, title) in zip(axes, cases):
        d = json.load(open(os.path.join(output_dir("b13_multipath"), cfg + ".json")))
        for key, lab, col, mk, ls in recv:
            r = [x for x in d["rows"] if x.get("receiver") == key and x["snr_db"] == 30][0]
            v = np.sort(r["per_packet_db"])
            p = np.arange(1, len(v) + 1) / len(v)
            ax.step(v, p, where="post", color=col, lw=2.2, ls=ls, label=lab)
            if mk:
                ax.plot(v, p, ls="none", marker=mk, ms=5, color=col, markeredgecolor="white", markeredgewidth=0.6)
        style_axis(ax, f"{title} ({d['n_packets']} packets)", "SINR of the packet (dB)",
                   "Fraction of packets at or below" if ax is axes[0] else "")
        ax.set_xlim(left=0)
    finish(fig, axes[0], 2, "Multipath: sensor-aided path cancellation lifts the worst packets",
           "How to read: each curve gives, for every SINR on the x-axis, the fraction of packets that did no better; "
           "further right is better, and the bottom of each curve is the worst packets. Setup: buoy-to-buoy link "
           "at 1 km (real buoy motion, OOI Irminger), direct path plus a bottom bounce 10.7 ms later whose Doppler "
           "differs by the buoys' heave, 30 dB SNR, industrial-grade motion sensors.", "fig_multipath_cdf.png", 170)


MU = (("branch_signal", "Signal-only, one Doppler estimate per user (Tu et al.)", BLUE, "o", "-"),
      ("branch_sensor", "Sensor-aided, one branch per user (ours)", ORANGE, "D", "-"),
      ("branch_genie", "True Doppler of each user (upper bound)", GREY, None, "--"))


def fig_mu_users(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), sharey=True)
    for ax, case, title in zip(axes, ("seabed users", "buoy users"),
                               ("Users on the seabed", "Users on buoys")):
        for key, lab, col, mk, ls in MU:
            pts = sorted((r["K"], r[key]["mean_sinr_db"]) for r in rows
                         if r["scenario"] == case and r["guard"] == 8 and r["snr_db"] == 30)
            ax.plot(*zip(*pts), color=col, lw=2.2, ls=ls, marker=mk, ms=8 if mk else 0, label=lab,
                    markeredgecolor="white", markeredgewidth=0.8)
        style_axis(ax, title, "Number of users sharing the band, K", "Mean SINR per user (dB)" if ax is axes[0] else "")
        ax.set_xticks((2, 4, 8))
    finish(fig, axes[0], 2, "Multi-user uplink: sensor-aided Doppler per user beats the signal-only receiver",
           "Setup: a gateway on a buoy receives K users at 1 km, each on its own equal share of the 8 kHz band, "
           "with 8 empty subcarriers between neighbours; 30 dB SNR; 8 packets per user and point. Left: only the "
           "gateway moves, so its own motion sensor predicts every user's Doppler (no side information needed). "
           "Right: every user is on a buoy and sends its motion as side information.", "fig_mu_users.png", 125)


def fig_mu_guard(rows):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.9), sharey=True)
    for ax, K in zip(axes, (2, 4, 8)):
        sel = [r for r in rows if r["scenario"] == "buoy users" and r["K"] == K and r["snr_db"] == 30]
        for key, lab, col, mk, ls in MU:
            pts = [(i, r[key]["mean_sinr_db"]) for i, g in enumerate((0, 8, 32)) for r in sel if r["guard"] == g]
            ax.plot(*zip(*pts), color=col, lw=2.2, ls=ls, marker=mk, ms=8 if mk else 0, label=lab,
                    markeredgecolor="white", markeredgewidth=0.8)
        ref = [r["branch_signal"]["mean_sinr_db"] for r in sel if r["guard"] == 32][0]
        ax.axhline(ref, color=BLUE, lw=1.0, ls=":")
        guards = (0, 8, 32)
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{g}\n({100 * (K - 1) * g / ACTIVE:.0f}% of band)" for g in guards])
        ax.set_xlim(-0.3, 2.3)
        style_axis(ax, f"K = {K} users", "Empty subcarriers between users", "Mean SINR per user (dB)" if ax is axes[0] else "")
    finish(fig, axes[0], 2, "Multi-user uplink: sensors can replace guard bands",
           "Users on buoys, 30 dB SNR, 8 packets per user and point. The dotted line marks what the signal-only "
           "receiver reaches with the widest guard (32 subcarriers). With no guard at all, the sensor-aided "
           "receiver reaches that level for K = 2 (+4.0 dB) and K = 4 (-0.4 dB) and falls 0.6 dB short for K = 8, "
           "saving the guard's share of the band shown under each tick.", "fig_mu_guard.png", 170)


def main():
    fig_multipath_cdf()
    rows = json.load(open(os.path.join(output_dir("b22_multiuser"), "points.json")))
    fig_mu_users(rows)
    fig_mu_guard(rows)
    print("written to", OUT)


if __name__ == "__main__":
    main()
