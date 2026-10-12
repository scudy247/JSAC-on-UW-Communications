"""
Placement map in dimensionless terms (PLAN: Phase 3, placement rule).

Which placement wins depends only on ratios: packet length and latency in wave periods (T f_p, L f_p), IMU drift and
receiver estimate error relative to the motion (sigma_imu / sigma_v, sigma_rx c / sigma_v), and the symbol length in
wave periods (T_s f_p, fixed here at 0.015). Absolute SIR also depends on sigma_v f_c T_s / c.
Link: a transmitter heaving and surging with a JONSWAP sea (peak f_p = 0.1 Hz, gamma 3.3, velocity along the path
sigma_v = 0.3 m/s, about a CDIP swell day), towards a static receiver without an IMU; the transmitter has one.
Maps (placement.evaluate at every point, SIR from the interference kernel, closed forms only):
  A  T f_p x sigma_imu / sigma_v, latency 50 ms extrapolated (L f_p = 0.005)
  B  T f_p x L f_p, sigma_imu / sigma_v = 0.03, extrapolated
each for two estimate budgets at the receiver: 'two' (first and last symbol: preamble / postamble) and 'per_symbol'
(one estimate per symbol: pilots or null subcarriers in every symbol).
Output: results/c2_map/map_{A,B}_{budget}.json and .png (winning placement, contours of its SIR).
Usage: python scripts/c2_map.py [A_two A_per_symbol B_two B_per_symbol]  (default: all).
"""
import json
import os
import sys

import numpy as np

from uwofdm import motion, placement as P
from uwofdm.paths import output_dir

FP, SIGMA_V, FS, HOURS = 0.1, 0.3, 10.0, 2.0
SIGMA_RX = 3e-6
T_FP = np.geomspace(0.015, 1.5, 34)  # packet length in wave periods (T from 0.15 to 15 s)


def jonswap_velocity(fs, n, fp, sigma, rng, gamma=3.3):
    """Velocity along the path with a JONSWAP shape (random phases), scaled to std sigma."""
    f = np.fft.rfftfreq(n, 1 / fs)
    f[0] = 1e-9
    s = np.where(f <= fp, 0.07, 0.09)
    S_eta = f ** -5 * np.exp(-1.25 * (fp / f) ** 4) * gamma ** np.exp(-((f - fp) ** 2) / (2 * s ** 2 * fp ** 2))
    S_v = (2 * np.pi * f) ** 2 * S_eta
    S_v[0] = 0.0
    X = np.sqrt(S_v) * np.exp(2j * np.pi * rng.uniform(size=len(f)))
    v = np.fft.irfft(X, n)
    return v * sigma / v.std()


def tracks(rng):
    n = int(HOURS * 3600 * FS)
    v = np.zeros((n, 3))
    v[:, 0] = jonswap_velocity(FS, n, FP, SIGMA_V, rng)
    tx = motion.Track(FS, v, np.zeros((n, 3)), np.array([0.0, 0.0, -5.0]))
    rx = motion.static_track(FS, n, (1000.0, 0.0, -5.0))
    return tx, rx


BUDGETS = {"two": ("none", "const2", "line2"), "per_symbol": ("none", "sym", "const2", "line2", "quad")}
NAMES = {"rx_sym": "receiver per symbol", "rx_pkt2": "receiver quadratic (all symbols)",
         "own[none]+rx[line2]": "receiver line (2 estimates)", "own[none]+rx[const2]": "receiver constant (2 est.)",
         "own[none]+rx[none]": "nothing", "tx_own": "tx IMU only",
         "own[tx]+rx[const2]": "tx IMU + receiver constant (2 est.)",
         "own[tx]+rx[line2]": "split: tx IMU + receiver line (2 est.)",
         "own[tx]+rx[sym]": "tx IMU + receiver per symbol", "own[tx]+rx[quad]": "tx IMU + receiver quadratic (all)"}
COLOURS = dict(zip(NAMES, ["#9ecae1", "#3182bd", "#c6dbef", "#deebf7", "#f0f0f0", "#fdae6b", "#a1d99b", "#31a354",
                           "#e6550d", "#756bb1"]))


def evaluate_grid(tx, rx, ys, sensors_of_y, rebuild, stages):
    """best[i][j] (name, sir) and all SIRs at every (y_i, T_j)."""
    rows = []
    comps_cache = None
    for y in ys:
        sen = sensors_of_y(y)
        if rebuild or comps_cache is None:
            comps_cache = P.link_components(tx, rx, sen, None)
        fs, comps = comps_cache
        for c in comps:  # sigma changes only the analytic drift term
            if c.drift is not None:
                c.drift = (sen.sigma / motion.C, sen.tau)
        row = []
        for tf in T_FP:
            rxm = P.ReceiverModel(packet_s=tf / FP, sigma=SIGMA_RX, stages=stages)
            ev = P.evaluate(fs, comps, rxm)
            row.append({"best": ev[0]["name"], "sir": ev[0]["sir_db"], "all": {r["name"]: r["sir_db"] for r in ev}})
        rows.append(row)
        print(f"y {y:.4g} done", flush=True)
    return rows


def plot(rows, ys, ylabel, title, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    names = [n for n in NAMES if any(c["best"] == n for r in rows for c in r)]
    names += sorted({c["best"] for r in rows for c in r} - set(names))
    idx = np.array([[names.index(c["best"]) for c in r] for r in rows])
    sir = np.array([[c["sir"] for c in r] for r in rows])
    cmap = ListedColormap([COLOURS.get(n, "#bbbbbb") for n in names])
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.pcolormesh(T_FP, ys, idx, cmap=cmap, vmin=-0.5, vmax=len(names) - 0.5, shading="nearest")
    cs = ax.contour(T_FP, ys, sir, levels=[15, 20, 25, 30, 35, 40, 45], colors="k", linewidths=0.8)
    ax.clabel(cs, fmt="%d dB", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("packet length / wave period  (T f_p)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOURS.get(n, "#bbbbbb")) for n in names],
              labels=[NAMES.get(n, n) for n in names], fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    out = output_dir("c2_map")
    tx, rx = tracks(np.random.default_rng(0))
    note = f"JONSWAP f_p {FP} Hz, sigma_v {SIGMA_V} m/s, T_s f_p 0.015, sigma_rx c / sigma_v {SIGMA_RX * motion.C / SIGMA_V:.3f}"
    maps = {"A": (np.geomspace(1e-3, 1.0, 31), "IMU drift / motion  (sigma_imu / sigma_v)",
                  lambda y: P.NodeSensors(sigma=y * SIGMA_V, latency=0.005 / FP), False, "latency 50 ms (L f_p 0.005)"),
            "B": (np.geomspace(1e-3, 0.1, 25), "latency / wave period  (L f_p)",
                  lambda y: P.NodeSensors(sigma=0.03 * SIGMA_V, latency=y / FP), True, "sigma_imu / sigma_v 0.03")}
    only = sys.argv[1:]  # e.g. B_per_symbol; default: all four maps
    for budget, stages in BUDGETS.items():
        for m, (ys, ylabel, sen, rebuild, fixed) in maps.items():
            tag = f"map_{m}_{budget}"
            if only and f"{m}_{budget}" not in only:
                continue
            rows = evaluate_grid(tx, rx, ys, sen, rebuild, stages)
            with open(os.path.join(out, tag + ".json"), "w", encoding="utf-8") as fh:
                json.dump({"note": note, "budget": budget, "T_fp": T_FP.tolist(), "y": ys.tolist(), "rows": rows}, fh)
            plot(rows, ys, ylabel, f"{m}, receiver estimates: {budget.replace('_', ' ')}; {fixed}\n{note}",
                 os.path.join(out, tag + ".png"))
            print("==", tag, flush=True)


if __name__ == "__main__":
    main()
