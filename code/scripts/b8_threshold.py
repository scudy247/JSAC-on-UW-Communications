"""
Threshold analysis (PLAN: Phase 5, theory T3). The mean-square bound (b7_bcrb.py) says every receiver could be
noise-limited; the simulated sensor gain lives in the tail. Hypothesis: a receiver fails when the Doppler residual it
faces before fine estimation (signal-only: after the chirps' mean-scale correction; sensor-aided: after the sensor
prior) leaves the linear range of its per-symbol measurements somewhere in the packet.

Step 1, calibration (this script, part "calibrate"): the packets of scripts/b2_baselines.py are regenerated from the
same seeds (t0 = first draw of the case's generator); per packet, the residual statistic
    x = max over data FFT windows of |mean residual in the window| x fc / df     [subcarrier spacings]
is computed for the signal-only front end (link Doppler minus its mean between the chirps) and for the sensor prior
(industrial MEMS, as in b2), and set against the packet's simulated SINR (results/b1_baselines/compare.json).
A packet "fails" if its SINR is > 3 dB below the SNR (30 dB).
Usage: python scripts/b8_threshold.py calibrate. Output: results/b8_threshold/calibration.json.
"""
import json
import os
import sys

import numpy as np

from uwofdm import frame, motion, sideinfo
from uwofdm.experiment import at_times
from uwofdm.paths import output_dir, results_dir
from e2_chain import SCENARIOS, SIGMA_IMU, allocation
from e2_imu_grades import node
from b2_baselines import CASES

FAIL_DB = 3.0


def residual_stats(label, n_data, n, seed, grade="industrial"):
    """Per packet: (x_signal, x_sensor) for the b2 packets of one case (same seed => same start times)."""
    tx_track, rx_track = SCENARIOS[label]()
    d = motion.link_doppler(tx_track, rx_track)
    F = frame.Frame(n_data=n_data)
    o = F.o
    T = F.length / o.fs
    tx, rx = node(tx_track, d["u"], +1, 0.0, grade, seed + 1), node(rx_track, d["u"], -1, d["tau"], grade, seed + 2)
    ranges, alloc = allocation(F, tx, tx_track.fs, budgets=(32,))
    a = alloc[(32, "none")]
    rng = np.random.default_rng(seed)
    wins = [(F.data_start + k * o.n_symbol + o.n_cp, F.data_start + k * o.n_symbol + o.n_cp + o.n_fft)
            for k in range(F.n_data)]
    out = []
    for t0 in rng.uniform(d["times"][0] + 240, d["times"][-1] - T - 20, n):
        tf = t0 + np.arange(F.length) / o.fs
        eps = at_times(d["eps"], d["times"], tx_track.fs, tf)
        c = sideinfo.project(tx.estimate(tf, SIGMA_IMU), tf - t0, T, 24)
        prior = sideinfo.reconstruct(sideinfo.quantise_alloc(c, a, ranges), tf - t0, T) + rx.estimate(tf, SIGMA_IMU)
        x = []
        for res in (eps - np.mean(eps[:F.gap]), eps - prior):
            x.append(max(abs(np.mean(res[i:j])) for i, j in wins) * o.fc / o.df)
        out.append(x)
    return np.array(out)


def calibrate():
    comp = json.load(open(os.path.join(results_dir(), "b1_baselines", "compare.json")))
    n = comp["n_packets"]
    rows = []
    for i, (label, n_data) in enumerate(CASES):
        x = residual_stats(label, n_data, n, 700 + i)
        sinr = {r["receiver"]: np.array(r["per_packet_db"]) for r in comp["rows"]
                if r["scenario"] == label and r["n_data"] == n_data and r["snr_db"] == 30}
        for k in range(n):
            rows.append(dict(scenario=label, n_data=n_data, x_signal=float(x[k, 0]), x_sensor=float(x[k, 1]),
                             **{f"sinr_{r}": float(v[k]) for r, v in sinr.items()}))
        order = np.argsort(-x[:, 0])
        print(f"{label:28s} {n_data:2d} sym: packets by signal-only residual x (subcarriers) -> SINR signal Kalman / "
              f"+pFFT8 / sensor fusion", flush=True)
        for k in order[:8]:
            print(f"    x_sig {x[k, 0]:5.2f}  x_sen {x[k, 1]:5.2f}  ->  {sinr['signal_kalman'][k]:5.1f} / "
                  f"{sinr['pfft8_kalman'][k]:5.1f} / {sinr['safe_industrial'][k]:5.1f}", flush=True)
    xs = np.array([r["x_signal"] for r in rows])
    for rec in ("signal_kalman", "pfft8_kalman"):
        fail = np.array([r[f"sinr_{rec}"] < 30 - FAIL_DB for r in rows])
        print(f"{rec}: {fail.sum()} failures of {len(rows)};", "  ".join(
            f"x > {th:.1f}: {np.mean(fail[xs > th]) if np.any(xs > th) else float('nan'):.2f} fail "
            f"({np.sum(xs > th)} pk), x <= {th:.1f}: {np.mean(fail[xs <= th]):.2f}" for th in (0.5, 1.0, 1.5, 2.0)))
    with open(os.path.join(output_dir("b8_threshold"), "calibration.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(fail_db=FAIL_DB, rows=rows), fh, indent=1)


if __name__ == "__main__":
    {"calibrate": calibrate}[sys.argv[1] if len(sys.argv) > 1 else "calibrate"]()
