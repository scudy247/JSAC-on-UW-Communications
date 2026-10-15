"""
Figures of where sensor-aided Doppler estimation gains (PLAN: Phase 8; results of 14-15 Oct 2026).
  fig_multipath_cdf.png  distribution of per-packet SINR at 30 dB SNR, buoy-buoy OOI over a two-path Bellhop channel:
                         1 s QPSK, 1 s 16-QAM, 4 s QPSK (results/b13_multipath/*.json)
  fig_mu_users.png       multi-user uplink, mean SINR vs number of users at 30 dB, 8 guard subcarriers; seabed users
                         and buoy users (results/b22_multiuser/points.json)
  fig_mu_guard.png       multi-user uplink, buoy users, mean SINR vs guard subcarriers (band overhead) at 30 dB
Colours follow the role, the same in every figure (dataviz default categorical slots 1-3, which pass the
colour-blind checks for all pairs): signal-only best baseline blue, sensor-aided (ours) orange, no cancellation aqua;
genie / ideal references grey dashed; each series also has its own marker. One y-axis per panel. The gain of ours over
the signal-only baseline is written on the plot.
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


def finish(fig, ax, ncol, title, name):
    """Headline (the claim) and legend above the panels."""
    fig.tight_layout(rect=(0, 0, 1, 0.80), w_pad=2.0)
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=14, fontweight="bold", color=INK)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.005, 0.925), ncol=ncol, fontsize=10.5, frameon=False,
               handlelength=2.6, columnspacing=2.0)
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)


def gap_label(ax, x, y_lo, y_hi, dx):
    """Double arrow from y_lo to y_hi beside x, with the difference written next to it (no arrow under 1 dB: the
    label is placed above the point instead)."""
    if abs(y_hi - y_lo) < 1.0:
        ax.text(x + dx, max(y_lo, y_hi) + 0.6, f"{y_hi - y_lo:+.1f} dB", fontsize=10.5, fontweight="bold",
                color=INK, ha="left", va="bottom", bbox=BOX)
        return
    ax.annotate("", xy=(x + dx, y_hi), xytext=(x + dx, y_lo),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0, shrinkA=3, shrinkB=3))
    ax.text(x + 1.6 * dx, (y_lo + y_hi) / 2, f"{y_hi - y_lo:+.1f} dB", fontsize=10.5, fontweight="bold", color=INK,
            ha="left" if dx > 0 else "right", va="center", bbox=BOX)


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
        packets = {}
        for key, lab, col, mk, ls in recv:
            v = np.sort([x for x in d["rows"] if x.get("receiver") == key and x["snr_db"] == 30][0]["per_packet_db"])
            packets[key] = v
            p = np.arange(1, len(v) + 1) / len(v)
            ax.step(v, p, where="post", color=col, lw=2.2, ls=ls, label=lab)
            if mk:
                ax.plot(v, p, ls="none", marker=mk, ms=5, color=col, markeredgecolor="white", markeredgewidth=0.6)
        # worst-10 % gap: 10th percentile of each cancellation
        a, b = (float(np.percentile(packets[k], 10)) for k in ("signal_sic", "sensor_sic"))
        ax.axhline(0.1, color="#9a9a9a", lw=0.8, ls=":")
        ax.annotate("", xy=(b, 0.1), xytext=(a, 0.1), arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
        ax.text(0.03, 0.97, f"Worst 10 % of packets (dotted):\nours {b - a:+.1f} dB", transform=ax.transAxes,
                fontsize=10.5, color=INK, ha="left", va="top", fontweight="bold")
        style_axis(ax, f"{title} ({d['n_packets']} packets)", "SINR of the packet (dB)",
                   "Fraction of packets at or below" if ax is axes[0] else "")
        ax.set_xlim(left=0)
        ax.set_ylim(-0.03, 1.28)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
    finish(fig, axes[0], 2, "Multipath: sensor-aided path cancellation lifts the worst packets",
           "fig_multipath_cdf.png")


MU = (("branch_signal", "Signal-only, one Doppler estimate per user (Tu et al.)", BLUE, "o", "-"),
      ("branch_sensor", "Sensor-aided, one branch per user (ours)", ORANGE, "D", "-"),
      ("branch_genie", "True Doppler of each user (upper bound)", GREY, None, "--"))


def plot_mu(ax, xs, sel_by_x):
    for key, lab, col, mk, ls in MU:
        ax.plot(xs, [sel_by_x[x][key]["mean_sinr_db"] for x in xs], color=col, lw=2.2, ls=ls, marker=mk,
                ms=8 if mk else 0, label=lab, markeredgecolor="white", markeredgewidth=0.8)


def fig_mu_users(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.9), sharey=True)
    for ax, case, title in zip(axes, ("seabed users", "buoy users"), ("Users on the seabed", "Users on buoys")):
        sel = {r["K"]: r for r in rows if r["scenario"] == case and r["guard"] == 8 and r["snr_db"] == 30}
        plot_mu(ax, (2, 4, 8), sel)
        for K, r in sel.items():
            gap_label(ax, K, r["branch_signal"]["mean_sinr_db"], r["branch_sensor"]["mean_sinr_db"],
                      0.15 if K < 8 else -0.15)
        style_axis(ax, title, "Number of users sharing the band, K",
                   "Mean SINR per user (dB)" if ax is axes[0] else "")
        ax.set_xticks((2, 4, 8))
        ax.set_xlim(1.5, 8.5)
    finish(fig, axes[0], 2, "Multi-user uplink: sensor-aided Doppler per user beats the signal-only receiver",
           "fig_mu_users.png")


def fig_mu_guard(rows):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.9), sharey=True)
    guards = (0, 8, 32)
    for ax, K in zip(axes, (2, 4, 8)):
        sel = {r["guard"]: r for r in rows if r["scenario"] == "buoy users" and r["K"] == K and r["snr_db"] == 30}
        plot_mu(ax, range(3), {i: sel[g] for i, g in enumerate(guards)})
        ref = sel[32]["branch_signal"]["mean_sinr_db"]
        ax.axhline(ref, color=BLUE, lw=1.0, ls=":")
        ax.text(2.3, ref - 0.35, "signal-only,\n32-subcarrier guard", fontsize=9.5, color=BLUE, ha="right", va="top",
                bbox=BOX)
        g0 = sel[0]["branch_sensor"]["mean_sinr_db"] - ref
        ax.text(0.03, 0.97, f"Ours with no guard vs\nsignal-only with 32: {g0:+.1f} dB", transform=ax.transAxes,
                fontsize=10.5, color=INK, ha="left", va="top", fontweight="bold")
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{g}\n({100 * (K - 1) * g / ACTIVE:.0f}% of band)" for g in guards])
        ax.set_xlim(-0.3, 2.35)
        ax.set_ylim(16, 31.5)
        style_axis(ax, f"K = {K} users", "Empty subcarriers between users",
                   "Mean SINR per user (dB)" if ax is axes[0] else "")
    finish(fig, axes[0], 2, "Multi-user uplink: with no guard band, sensor-aided ≈ signal-only with the widest guard",
           "fig_mu_guard.png")


def main():
    fig_multipath_cdf()
    rows = json.load(open(os.path.join(output_dir("b22_multiuser"), "points.json")))
    fig_mu_users(rows)
    fig_mu_guard(rows)
    print("written to", OUT)


if __name__ == "__main__":
    main()
