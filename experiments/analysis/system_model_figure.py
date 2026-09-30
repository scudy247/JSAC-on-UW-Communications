"""System-model figure for Proposal B (draft for the group meeting, 2026-09-30).

Panels: (a) block diagram with key quantities, implementing modules and data sources;
(b) timing: sparse delayed rewards vs dense noise stream; (c) which blocks each layer uses.
Regenerate:  .venv/bin/python experiments/analysis/system_model_figure.py
Output:      reports/figures/system_model.pdf and .png
Pending group decisions are in red (D-B1, D-B6, D-B12, Q13). Module names under
src/uwsb/ are the planned layout (PLAN §3, hybrid draft §3); most do not exist yet.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy.stats import levy_stable

from uwsb.runtime import REPO_ROOT

OUT = REPO_ROOT / "reports" / "figures"      # reports/ is not published
OUT.mkdir(exist_ok=True)

C_PY, C_LR, C_DS, C_OR = "#dbe9f6", "#dff0d8", "#fde5c8", "#eeeeee"
EDGE, PEND, MOD = "#333333", "#b22222", "#1f4e79"

fig = plt.figure(figsize=(16, 13.2))
gs = fig.add_gridspec(2, 2, height_ratios=[2.25, 1.0], width_ratios=[1.75, 1.0],
                      hspace=0.08, wspace=0.04)
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 160)
ax.set_ylim(-3.5, 88)
ax.axis("off")


def box(x, y, w, h, num, title, lines, module, color, pend=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=color, ec=EDGE, lw=1.2, zorder=2))
    ax.text(x + 1.0, y + h - 1.4, f"({num})", fontsize=9.5, fontweight="bold", va="top", zorder=3)
    ax.text(x + 4.8, y + h - 1.4, title, fontsize=10, fontweight="bold", va="top", zorder=3)
    ax.text(x + 1.2, y + h - 5.3, "\n".join(lines), fontsize=8.3, va="top",
            linespacing=1.38, zorder=3)
    ax.text(x + 1.2, y + 1.0, module, fontsize=7.4, color=MOD, style="italic",
            va="bottom", zorder=3, linespacing=1.3)
    if pend:
        ax.text(x + w - 0.8, y + h - 1.4, pend, fontsize=8, color=PEND, ha="right",
                va="top", fontweight="bold", zorder=3)


def arrow(p, q, label=None, lpos=None, ls="-", color=EDGE, rad=0.0, fs=8.3):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.3, ls=ls,
                                 color=color, connectionstyle=f"arc3,rad={rad}", zorder=4))
    if label:
        x, y = lpos if lpos else ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 1.2)
        ax.text(x, y, label, fontsize=fs, ha="center", va="bottom", color=color, zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.3))


# environment boundary (delayed-feedback environment)
ax.add_patch(FancyBboxPatch((30.5, -2.6), 90.5, 88.1, boxstyle="round,pad=0.3,rounding_size=2",
                            fc="none", ec="#888888", lw=1.1, ls=(0, (5, 3)), zorder=1))
ax.text(31.5, 84.6, "delayed-feedback environment: agents see only delayed rewards and the "
        "noise stream (envs/table_env.py; project.MD §7.2–7.3)", fontsize=8.2,
        color="#555555", va="top", style="italic")

# row 1: forward link
box(2, 52, 26, 29, 1, "Transmitter", [
    r"arm $k_t \in \{1..K\}$ = PHY config",
    r"(sub-band $\times$ power $\times$ MCS)",
    "packet: preamble (LFM/HFM)", "+ pilots + coded data",
    r"slot length $T_{slot}$, duration $T_p$"],
    "phy/waveform.py", C_PY, pend="D-B6")
box(34, 52, 27, 29, 2, "UWA channel replay", [
    "measured time-varying IR", r"$y(t)=\int h(\tau,t)\,s(t-\tau)\,d\tau$",
    "multipath, Doppler, delay drift", r"one-way delay $\tau_{ow}$",
    "data: channel library", "(Red 20–30 kHz, Blue 10.5–15.5)"],
    "channels/uwa_library.py\n(uwa-channels 0.7.1)", C_PY)
box(34, 13, 27, 31, 3, "Ambient noise", [
    "impulsive, regime-switching", r"regime $Z_t$: ambient / shrimp / ship", "",
    "real: SanctSound FK01, HI01", "(shrimp, 84–131 events/s)",
    r"models: S$\alpha$S ($\alpha$=1.7), Class A,", r"$\alpha$SGN(m) with memory"],
    "noise/recordings.py, stable.py,\nclass_a.py, asgn.py", C_PY)
ax.add_patch(plt.Circle((66.5, 66.5), 2.2, fc="white", ec=EDGE, lw=1.3, zorder=3))
ax.text(66.5, 66.5, "+", fontsize=14, ha="center", va="center", zorder=4)
box(73, 52, 30, 29, 4, "Receiver chain", [
    "sync, Doppler compensation,", "channel estimation, equaliser,",
    r"decode + CRC $\rightarrow$ ACK/NACK", "",
    r"pilot EVM: $\hat P_n=(1/N)\,\Sigma_i\,|r_i-\hat h s_i|^2$",
    r"soft feedback: $\hat\gamma = P_s/\hat P_n$ (dB)"],
    "phy/receiver.py, phy/chain.py\n→ phy/outcome_tables.py", C_PY, pend="D-B1")

# row 2: noise stream + learner
box(73, 13, 30, 31, 5, "Noise-state estimator", [
    "noise-only samples between", r"packets: $\nu \gg 1$ per slot,",
    "shared by all arms", "",
    r"O(1) updates: tail $\hat\kappa$ (or $\hat\alpha$),",
    r"scale $\hat c$, regime $\hat Z$", "(exceedance-driven, THEORY-B T4)"],
    "estimation/noise_state.py,\ntail_index.py", C_LR)
box(125, 13, 33, 68, 6, "NIR-UCB learner", [
    r"inputs: delayed $\hat\gamma$, ACK;", r"noise state $(\hat\kappa,\hat c,\hat Z)$", "",
    "robust mean per arm (truncated):",
    r"$\hat\mu_k=(1/n)\,\Sigma_s\,X_s\,\mathbb{1}\{|X_s-a|\leq\psi_s\}$",
    r"$\psi_s$ set by $(\hat\kappa,\hat c)$ (BCL Lemma 1)", "",
    r"index: $B_k=\hat\mu_k + w(\hat\kappa,\hat c, n_k, t)$",
    r"decision: argmax$_k$ rate$_k\times$PSR$_k(\hat\mu_k)$", "",
    r"regime switch in $\hat Z$:", "inflate widths (no restart)", "",
    r"baselines: UCB/TS on $\hat\gamma$, ACK-only,", "trunc./MoM/Catoni UCB, AdaR-UCB,",
    "fixed clip, threshold AMC", "",
    "location: receiver (default)", "or transmitter"],
    "bandits/robust.py", C_LR, pend="D-B12")

# row 3: feedback link + metrics
box(34, -1.0, 69, 10.5, 7, "Feedback link (DESERT, Layer 3)", [
    r"reverse UWA link: arm index / $\hat\gamma$ / ACK-NACK;  delay $\tau_{fb}$, loss, MAC, multi-node"],
    "desert/addon (uwtracephy + MAC), desert/bridge.py", C_DS, pend="Q13")
box(2, 13, 26, 31, 8, "Metrics & oracles", [
    r"true $\mu_k$, oracle arm $k^*$", r"regret $R_T=\Sigma_k\,\Delta_k\,E[T_k]$",
    "goodput, compute/memory", "", "ground truth logged apart;", "never visible to (6)"],
    "metrics.py", C_OR)

# arrows
arrow((28.8, 66.5), (33.6, 66.5), r"$s(t)$")
arrow((61.6, 66.5), (64.1, 66.5))
arrow((61.6, 33), (66.6, 64.1), r"$n(t)$", lpos=(66.4, 46.5), rad=0.3)
arrow((68.9, 66.5), (72.6, 66.5), r"$r(t)$", lpos=(70.7, 67.8))
arrow((88, 51.6), (88, 44.6), "noise-only\nsegments", lpos=(81.2, 46.0))
arrow((103.6, 66.5), (124.6, 66.5), r"$\hat\gamma_t$, ACK$_t$ (after $\tau$)", lpos=(114, 67.8))
arrow((103.6, 28.5), (124.6, 28.5), r"$(\hat\kappa,\hat c,\hat Z)$ every slot", lpos=(114, 29.8))
arrow((141.5, 12.6), (103.6, 4.2), r"arm $k_{t+1}$", lpos=(123, 6.4), rad=-0.1)
ax.plot([33.6, 31.5, 31.5], [4.2, 4.2, 49.5], color=EDGE, lw=1.3, zorder=4)
arrow((31.5, 49.3), (24, 51.6))
ax.text(30.2, 26, "next arm", fontsize=8.3, rotation=90, ha="center", va="center",
        bbox=dict(fc="white", ec="none", pad=0.3), zorder=5)
arrow((34, 55), (20, 44.6), r"true $\mu_k$ (log)", lpos=(13, 47.3), ls="--", color="#777777", rad=0.15)
ax.text(0.5, 87.5, "(a)", fontsize=13, fontweight="bold", va="top")

# ---- (b) timing panel ---------------------------------------------------------------
bx = fig.add_subplot(gs[1, 0])
bx.set_xlim(-0.2, 8.2)
bx.set_ylim(-0.75, 4.4)
bx.set_yticks([3.5, 2.2, 0.9])
bx.set_yticklabels(["TX: packets\n(arm colours)", "RX: received signal\n+ impulsive noise",
                    "learner: rewards\n(sparse, delayed)"], fontsize=8.5)
bx.set_xlabel(r"time [slots of length $T_{slot}$]", fontsize=9)
for sp in ("top", "right"):
    bx.spines[sp].set_visible(False)
arm_col = ["#4c72b0", "#dd8452", "#55a868"]
arms = [0, 1, 0, 2, 0, 0, 1, 0]
tau_ow, tau_fb, tp = 0.55, 0.65, 0.3
rng = np.random.default_rng(3)
t = np.linspace(0, 8.2, 4000)
z = levy_stable.rvs(1.5, 0.0, size=t.size, random_state=rng) * 0.035
z = np.clip(z, -0.55, 0.55)
bx.plot(t, 2.2 + z, color="#777777", lw=0.5)
for i, k in enumerate(arms):
    bx.add_patch(Rectangle((i, 3.3), tp, 0.4, fc=arm_col[k], ec="none"))
    bx.add_patch(Rectangle((i + tau_ow, 1.95), tp, 0.5, fc=arm_col[k], ec="none", alpha=0.55))
    if i + tau_ow + tau_fb + tp < 8.2:
        bx.plot(i + tau_ow + tp + tau_fb, 0.9, "v", color=arm_col[k], ms=8)
for i in range(8):           # noise-only gaps (dense side information)
    bx.add_patch(Rectangle((i + tau_ow + tp, 1.72), 1 - tp, 0.96, fc="#dff0d8", ec="none",
                           alpha=0.55, zorder=0))
bx.annotate("", xy=(0 + tau_ow, 3.1), xytext=(0, 3.1),
            arrowprops=dict(arrowstyle="<->", lw=1))
bx.text(tau_ow / 2, 2.95, r"$\tau_{ow}$", ha="center", va="top", fontsize=9)
bx.annotate("", xy=(1 + tau_ow + tp + tau_fb, 0.55), xytext=(1, 0.55),
            arrowprops=dict(arrowstyle="<->", lw=1))
bx.text(1 + (tau_ow + tp + tau_fb) / 2, 0.4, r"$\tau=\tau_{ow}+T_p+\tau_{fb}$ (reward delay)",
        ha="center", va="top", fontsize=8.5)
bx.annotate("", xy=(1 + tau_ow + tp + tau_fb, 1.05), xytext=(1 + tau_ow + tp, 1.95),
            arrowprops=dict(arrowstyle="->", lw=0.9, color=arm_col[1]))
bx.text(5.35, 2.9, r"green = noise-only stream: $\nu$ samples per slot, shared by all arms",
        fontsize=8.5, color="#2e6b2e", ha="center")
bx.text(5.6, -0.2, "one reward per pull, delayed by τ; impulses hit rewards and noise stream alike",
        fontsize=8.5, ha="center", va="top")
bx.text(-0.15, 4.35, "(b)", fontsize=13, fontweight="bold", va="top")

# ---- (c) layer map --------------------------------------------------------------------
cx = fig.add_subplot(gs[1, 1])
cx.axis("off")
cx.set_xlim(0, 10)
cx.set_ylim(0, 10)
rows = [("", ["(2)\nchannel", "(3)\nnoise", "(4)\nreceiver", "(7)\nfeedback"]),
        ("Layer 1  synthetic", ["model", "models", "reward model", "const. τ"]),
        ("Gate  (Phase 2)", ["replay", "real rec.", "full chain", "—"]),
        ("Layer 2  replay", ["replay", "real + models", "full chain", "const. τ"]),
        ("Layer 3  DESERT", ["tables", "tables", "tables", "DESERT"])]
xs = [0.0, 3.35, 5.05, 6.85, 8.9]
for r, (name, cells) in enumerate(rows):
    y = 8.6 - r * 1.55
    cx.text(xs[0], y, name, fontsize=8.8, fontweight="bold" if r else "normal", va="center")
    for j, c in enumerate(cells):
        cx.text(xs[j + 1], y, c, fontsize=8.0, va="center", ha="center",
                fontweight="bold" if r == 0 else "normal")
    if r == 0:
        cx.plot([0, 10], [y - 0.7, y - 0.7], color=EDGE, lw=0.8)
cx.text(0.0, 0.75, "learner (6) and noise-state estimator (5): the same Python\n"
        "code in every layer (THEORY-B T-1).  'tables' = per-slot × per-arm\n"
        "outcomes from the Python chain, replayed by DESERT (uwtracephy).",
        fontsize=7.9, va="center", color="#444444")
cx.text(0.0, 9.95, "(c)", fontsize=13, fontweight="bold", va="top")

fig.suptitle("Proposal B, Noise-Informed Robust Bandits: system model (draft, 2026-09-30).  "
             "Red = pending group decision; blue italics = implementing module (src/uwsb/…).",
             fontsize=10.5, style="italic", y=0.905)
fig.savefig(OUT / "system_model.pdf", bbox_inches="tight")
fig.savefig(OUT / "system_model.png", dpi=140, bbox_inches="tight")
print("wrote", OUT / "system_model.pdf", OUT / "system_model.png")
