"""Figure of the Layer-1 alpha x gap sweep and the T = 1e5 runs (results/b_l1*), for the group.

Left: regret of NIR-UCB v0 divided by the regret of the best simple baseline (naive UCB/TS with oracle
sigma, ACK UCB1, ACK Bernoulli TS) at T = 1e4, per (alpha, gap scale); the cell names that baseline.
Right: cumulative regret vs t at alpha = 1.6, gap x1, T = 1e5 (20 seeds, 95 % CI), log-log; the dotted
line has slope 1 (linear regret). Rows: branch S, branch E. No randomness; reads summary.json files.

  python experiments/analysis/layer1_sweep_figure_20261001.py   # -> reports/figures/layer1_sweep.{pdf,png}
"""
import glob
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

ROOT = Path(__file__).resolve().parents[2]
ALPHAS = [1.2, 1.4, 1.6, 1.8, 1.95]
GAPS = [0.5, 1, 2]
# agent -> (short label, colour); fixed categorical order (dataviz reference palette, slots 1-8)
AGENTS = [("nUCB", "naive UCB (oracle σ)", "#2a78d6"), ("nTS", "naive TS (oracle σ)", "#eb6834"),
          ("aUCB", "ACK UCB1", "#1baf7a"), ("aTS", "ACK Bernoulli TS", "#eda100"),
          ("Trunc", "Trunc-mean UCB (oracle)", "#e87ba4"), ("MoM", "MoM UCB (oracle)", "#008300"),
          ("AdaR", "AdaR-UCB", "#4a3aa7"), ("NIR", "NIR-UCB v0", "#e34948")]
SHORT = {"UCB (naive, oracle sigma)": "nUCB", "Gaussian TS (naive)": "nTS", "ACK UCB1": "aUCB",
         "ACK Bernoulli TS": "aTS", "Trunc-mean UCB (oracle)": "Trunc", "Trunc-mean UCB (oracle, eps=1)": "Trunc",
         "MoM UCB (oracle)": "MoM", "MoM UCB (oracle, eps=1)": "MoM", "AdaR-UCB": "AdaR",
         "NIR-UCB v0": "NIR", "NIR-UCB v0 (misspecified)": "NIR"}
SIMPLE = ["nUCB", "nTS", "aUCB", "aTS"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])


def load():
    out = {}
    for d in glob.glob(str(ROOT / "results" / "b_l1*")):
        cfg = json.load(open(Path(d) / "config.json"))
        s = json.load(open(Path(d) / "summary.json"))
        s["agents"] = {SHORT[k]: v for k, v in s["agents"].items()}
        out[cfg["name"]] = s
    return out


def heat(ax, rows, br, norm):
    R = np.zeros((len(ALPHAS), len(GAPS)))
    best = [[None] * len(GAPS) for _ in ALPHAS]
    for i, a in enumerate(ALPHAS):
        for j, g in enumerate(GAPS):
            ag = rows[f"b_l1sw_{br}_a{a:g}_g{g:g}"]["agents"]
            b = min(SIMPLE, key=lambda k: ag[k]["final_mean"])
            R[i, j], best[i][j] = ag["NIR"]["final_mean"] / ag[b]["final_mean"], b
    im = ax.imshow(R, cmap=SEQ, norm=norm, aspect="auto", origin="lower")
    for i in range(len(ALPHAS)):
        for j in range(len(GAPS)):
            dark = norm(R[i, j]) > 0.35
            ax.text(j, i + 0.08, f"{R[i, j]:.1f}×", ha="center", va="center", fontsize=10,
                    color="white" if dark else TEXT, fontweight="bold")
            ax.text(j, i - 0.25, f"vs {best[i][j]}", ha="center", va="center", fontsize=7,
                    color="white" if dark else MUTED)
    ax.set_xticks(range(len(GAPS)), [f"×{g:g}" for g in GAPS])
    ax.set_yticks(range(len(ALPHAS)), [f"{a:g}" for a in ALPHAS])
    ax.set_xlabel("gap scale")
    ax.set_ylabel("noise index α")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def curves(ax, rows, br):
    s = rows[f"b_l1long_{br}_a1.6"]
    t = np.array(s["checkpoints"])               # 200 checkpoints, linear spacing
    ends = []
    for key, label, col in AGENTS:
        a = s["agents"][key]
        m, h = np.array(a["mean"]), np.array(a["ci95_half"])
        lw = 2.6 if key == "NIR" else 1.6
        ax.plot(t, np.maximum(m, 1), color=col, lw=lw, label=label, zorder=3 if key == "NIR" else 2)
        ax.fill_between(t, np.maximum(m - h, 1), m + h, color=col, alpha=0.12, lw=0)
        ends.append((m[-1], key, col))
    ref = 0.5 * t                                # slope-1 reference
    ax.plot(t, ref, ls=":", color=MUTED, lw=1.2)
    ax.text(t[-1] * 0.55, ref[-1] * 0.55 * 1.25, "∝ t", color=MUTED, fontsize=8, ha="right")
    for v, key, col in ends:                     # selective direct labels: NIR and ACK TS
        if key in ("NIR", "aTS"):
            ax.annotate(key, (t[-1], v), xytext=(4, 0), textcoords="offset points", fontsize=8,
                        color=TEXT, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(t[1], t[-1] * 1.9)               # first checkpoint after t = 1 (~500)
    ax.set_ylim(30, 2e5)
    ax.set_xlabel("decision t")
    ax.set_ylabel("cumulative regret")
    ax.grid(color=GRID, lw=0.6, which="major")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    rows = load()
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), gridspec_kw={"width_ratios": [1, 1.35]})
    norm = LogNorm(vmin=1, vmax=30)
    titles = {"S": "Branch S — α-stable reward noise", "E": "Branch E — γ̂ in dB (NIR map misspecified)"}
    for r, br in enumerate("SE"):
        im = heat(axes[r, 0], rows, br, norm)
        axes[r, 0].set_title(f"{titles[br]}\nNIR-UCB v0 regret ÷ best simple baseline, T = 10⁴",
                             fontsize=9, loc="left", color=TEXT)
        curves(axes[r, 1], rows, br)
        axes[r, 1].set_title(f"{titles[br]}\nα = 1.6, gap ×1, T = 10⁵, 20 seeds (95 % CI)",
                             fontsize=9, loc="left", color=TEXT)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.17, hspace=0.42, wspace=0.28)
    cb = fig.colorbar(im, cax=fig.add_axes([0.10, 0.075, 0.31, 0.015]), orientation="horizontal")
    cb.set_label("regret ratio (1 = as good as the best simple baseline)", color=TEXT)
    cb.set_ticks([1, 2, 5, 10, 20, 30], labels=["1", "2", "5", "10", "20", "30"])
    cb.outline.set_visible(False)
    cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower right", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.99, 0.0))
    out = ROOT / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "layer1_sweep.pdf")
    fig.savefig(out / "layer1_sweep.png", dpi=150)
    print("wrote", out / "layer1_sweep.{pdf,png}")


if __name__ == "__main__":
    main()
