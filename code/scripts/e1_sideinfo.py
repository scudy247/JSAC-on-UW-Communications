"""
E1 (PLAN: Phase 4): how many bits of the transmitter's own motion does a packet need?

The transmitter's IMU estimate of its share over the packet (drift 0.01 m/s, 50 ms latency bridged, lever arm
corrected) is sent as K Legendre coefficients with b bits each (sideinfo.py); the receiver subtracts the rebuilt
share, its own IMU share if it moves, and fits the remainder from the signal:
  stage 'none'  sensors only (no signal estimate)
  stage 'line2' line through 2 signal estimates per packet (first / last symbol)
  stage 'sym'   one signal estimate per symbol, piecewise constant
Signal estimates have error 3e-6 (as in the C2 runs). References per stage: the receiver alone (no transmitter
information) and the transmitter applying its own estimate (pre-compensation: no bits, no representation error).
Scored per packet in the time domain (sideinfo.interference), power-averaged over 200 packets of real motion.
Scenarios: buoy -> seabed (CDIP 029 swell; OOI Irminger, 2 m lever arm), buoy -> buoy (OOI); packets 1 and 4 s.
Output: results/e1_sideinfo/points.json, sirs_vs_bits.png; printed summary (bits needed to come within 1 dB of
pre-compensation). Single thread, a few minutes.
"""
import json
import os

import numpy as np

from uwofdm import estimates, motion, placement as P, sideinfo, theory
from uwofdm.experiment import at_times
from uwofdm.paths import output_dir
from c2_sweeps import CDIP_DAY, OOI_DAY, buoy_buoy_link, seabed_link

N_PACKETS, SIGMA_RX, FC = 200, 3e-6, 16e3
KS, BITS = (1, 2, 3, 4, 5, 6, 8), (2, 3, 4, 5, 6, 8, 10)
STAGES = ("none", "line2", "sym")
SCENARIOS = {"buoy -> seabed, CDIP swell": (lambda: seabed_link("cdip", CDIP_DAY), False),
             "buoy -> seabed, OOI Irminger": (lambda: seabed_link("ooi", OOI_DAY, (0.0, 0.0, 2.0)), False),
             "buoy - buoy, OOI Irminger": (buoy_buoy_link, True)}
STYLE = {"none": ("#2a78d6", "sensors only"), "line2": ("#eb6834", "+ 2 signal estimates"),
         "sym": ("#1baf7a", "+ 1 signal estimate per symbol")}


def series(tx, rx, rx_moves, rng):
    """Link time grid and the per-share true / IMU-estimated Doppler series."""
    sen = P.NodeSensors()
    fs, comps = P.link_components(tx, rx, sen, sen if rx_moves else None)
    t_day = motion.link_doppler(tx, rx)["times"]
    c = {k.name: k for k in comps}
    n = len(t_day)
    drift = lambda: estimates.gauss_markov(n, fs, sen.sigma, sen.tau, rng) / motion.C
    out = dict(t=t_day, fs=fs, tx_true=c["tx"].true, tx_est=c["tx"].true - c["tx"].after_node + drift(),
               slow=c["slow"].true, rx_true=c["rx"].true)
    out["rx_est"] = (c["rx"].true - c["rx"].after_node + drift()) if rx_moves else np.zeros(n)
    return out


def run_case(s, T, rng):
    rxm = P.ReceiverModel(packet_s=T, stages=STAGES)
    est = {st: P.stage_estimator(st, rxm) for st in STAGES}
    tg, windows = est["none"][0], est["none"][3]
    f, S = theory.spectrum(s["tx_est"], s["fs"])
    std = sideinfo.coefficient_std(f, S, tg[-1], max(KS))
    ranges = np.abs(np.mean(s["tx_est"])) + sideinfo.LOAD * std
    bases = {K: sideinfo.basis(tg, tg[-1], K) for K in KS}
    starts = rng.uniform(s["t"][0] + 30, s["t"][-1] - T - 10, N_PACKETS)
    acc = {}
    for t0 in starts:
        g = {k: at_times(s[k], s["t"], s["fs"], t0 + tg) for k in ("tx_true", "tx_est", "rx_true", "rx_est", "slow")}
        base = g["rx_true"] - g["rx_est"] + g["slow"]
        c_full = np.linalg.lstsq(bases[max(KS)], g["tx_est"], rcond=None)[0]
        noise = {st: SIGMA_RX * rng.standard_normal(len(est[st][1])) for st in STAGES}
        hats = {"rx_alone": np.zeros(len(tg)), "applied": g["tx_est"]}
        for K in KS:
            for b in BITS:
                q = sideinfo.quantise(c_full[:K], b, ranges[:K])  # orthonormal basis: first K coefficients
                hats[(K, b)] = bases[K] @ q
        for key, hat in hats.items():
            r0 = g["tx_true"] - hat + base
            for st in STAGES:
                t_s, W = est[st][1], est[st][2]
                r = r0 if W is None else r0 - W @ (np.interp(t_s, tg, r0) + noise[st])
                acc.setdefault((key, st), []).append(sideinfo.interference(r, tg, windows, FC))
    return {k: float(-10 * np.log10(np.mean(v))) for k, v in acc.items()}, std


def main():
    out = output_dir("e1_sideinfo")
    rows, rng = [], np.random.default_rng(0)
    for label, (make, rx_moves) in SCENARIOS.items():
        s = series(*make(), rx_moves, rng)
        for T in (1.0, 4.0):
            res, std = run_case(s, T, rng)
            for (key, st), sir in res.items():
                row = dict(scenario=label, packet_s=T, stage=st, sir_db=sir)
                if isinstance(key, tuple):
                    row.update(K=key[0], b=key[1], bits=key[0] * key[1])
                else:
                    row.update(reference=key)
                rows.append(row)
            for st in STAGES:
                app = res[("applied", st)]
                front = sorted((K * b, v) for (k, s2), v in res.items() if s2 == st and isinstance(k, tuple)
                               for K, b in [k])
                need = next((B for B, v in front if v >= app - 1.0), None)
                print(f"{label:30s} T {T:3.0f} s  {st:5s}: receiver alone {res[('rx_alone', st)]:5.1f}, applied "
                      f"{app:5.1f} dB; within 1 dB of applied with {need} bits", flush=True)
            print(f"   coefficient std (Doppler scale): {np.array2string(std, precision=2)}", flush=True)
        with open(os.path.join(out, "points.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    plot(rows, os.path.join(out, "sir_vs_bits.png"))


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(SCENARIOS)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharey=True)
    for j, label in enumerate(labels):
        for i, T in enumerate((1.0, 4.0)):
            ax = axes[i, j]
            for st, (col, name) in STYLE.items():
                pts = [r for r in rows if r["scenario"] == label and r["packet_s"] == T and r["stage"] == st]
                q = sorted((r["bits"], r["sir_db"]) for r in pts if "bits" in r)
                front, best = [], -np.inf
                for B, v in q:
                    if v > best:
                        front.append((B, v))
                        best = v
                ax.plot(*zip(*front), color=col, lw=2, marker="o", ms=4, label=name)
                ref = {r["reference"]: r["sir_db"] for r in pts if "reference" in r}
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
    h += [plt.Line2D([], [], color="#555555", ls="--"), plt.Line2D([], [], color="#555555", ls=":")]
    l += ["transmitter applies its estimate (no bits)", "receiver alone (no transmitter information)"]
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
