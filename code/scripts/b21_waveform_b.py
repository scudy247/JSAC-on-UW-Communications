"""
B at waveform level (PLAN: Phase 5; check of scripts/b20_long_integration.py). Long DSSS packets (uwofdm.longint)
warped by real link Doppler; signal-only vs sensor-aided Doppler estimation with real maximum-likelihood frequency
estimates per segment (the low-SNR threshold happens instead of being modelled), Kalman smoothing with outlier
gating, compensation; coherent-integration efficiency on the compensated noise-free signal.
Fairness: the signal-only receiver's coarse correction is the TRUE mean Doppler of the packet (an ideal preamble /
postamble), and its result is the better of tracking and of that mean correction alone (fallback when tracking fails
at low CNR; chosen per configuration, i.e. generously); the sensor-aided one starts from its own prior (side information + own IMU, industrial MEMS), mean error
included. Each receiver uses its best segment length. Frequency search window per receiver (longint step 2):
+-BAND_K times the RMS of its own residual over the record - signal-only: the link Doppler around the packet mean
(motion statistics); sensor-aided: the prior's error (a calibration-level statistic of the sensor). Same cases,
carriers and CNR grid as the phase-level study.
Usage: python scripts/b21_waveform_b.py [n_packets] [ml|hmm] [--cases 0,1,2] [--fcs 16000,4000] [--cnrs 5,10]
[--grade industrial] [--out name]; python scripts/b21_waveform_b.py merge <out> <part> ... Output: results/b21_waveform_b/points.json
(points_hmm.json with the Viterbi frequency-line tracker for both receivers, the lower-threshold signal-only
baseline).
"""
import argparse
import json
import os
import sys

import numpy as np

from uwofdm import chain, longint, motion
from uwofdm.waveform import unwarp
from uwofdm.experiment import at_times
from uwofdm.paths import output_dir
from uwofdm.stats import seed_of
from e2_chain import SCENARIOS, SIGMA_IMU
from e2_imu_grades import node

CASES = ("buoy - buoy, OOI Irminger", "buoy -> seabed, OOI Irminger", "buoy -> seabed, CDIP swell")
FCS = (16e3, 4e3)
CNR_DBHZ = (5, 10, 15, 20, 25)
T_SEGS = (0.25, 0.5, 1.0, 2.0)
T_INT = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
T_PKT, CHIP_RATE, FS = 16.0, 1000.0, 4000.0
N_MAX = 1000  # packet times drawn once per case for this many packets
BAND_K, DT = 4.0, 0.05


def window_rms(d, tx_track, tx, rx):
    """RMS over the record of each receiver's residual Doppler scale: (signal-only: eps minus its mean over
    T_PKT blocks; sensor-aided: eps minus the prior)."""
    tr = np.arange(d["times"][0] + 240, d["times"][-1] - 20, DT)
    eps = at_times(d["eps"], d["times"], tx_track.fs, tr)
    prior = tx.estimate(tr, SIGMA_IMU) + rx.estimate(tr, SIGMA_IMU)
    L = int(round(T_PKT / DT))
    blocks = eps[:len(eps) // L * L].reshape(-1, L)
    return float(np.sqrt(np.mean((blocks - blocks.mean(1, keepdims=True)) ** 2))), float(np.sqrt(np.mean((eps - prior) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=40, help="packets per point")
    ap.add_argument("tracker", nargs="?", default="ml", choices=("ml", "hmm"),
                    help="hmm: Viterbi frequency-line tracker (both receivers)")
    ap.add_argument("--cases", default=",".join(map(str, range(len(CASES)))), help="indices into CASES")
    ap.add_argument("--fcs", default=",".join(str(int(f)) for f in FCS))
    ap.add_argument("--cnrs", default=",".join(map(str, CNR_DBHZ)))
    ap.add_argument("--grade", default="industrial", help="sensor grade of both moving ends (imu.GRADES)")
    ap.add_argument("--out", default=None, help="output name (default points[_hmm][_grade])")
    a = ap.parse_args()
    out = a.out or "points" + ("" if a.tracker == "ml" else "_" + a.tracker) + \
        ("" if a.grade == "industrial" else "_" + a.grade)
    n, tracker = a.n, a.tracker
    rows = []
    sig_set = dict(chain.SIGNAL_FUSION)
    sen_set = {k: chain.FUSION[k] for k in ("sigma_r0", "sigma_dr0", "q")}
    for label in [CASES[int(i)] for i in a.cases.split(",")]:
        tx_track, rx_track = SCENARIOS[label]()
        d = motion.link_doppler(tx_track, rx_track)
        tx = node(tx_track, d["u"], +1, 0.0, a.grade, 21)
        rx = node(rx_track, d["u"], -1, d["tau"], a.grade, 22)
        n_s = int(T_PKT * FS)
        rms_sig, rms_sen = window_rms(d, tx_track, tx, rx)
        # per-case packet times and per-packet seeds (stats.seed_of): jobs split by case / carrier / CNR draw the
        # same packets as one run; tracker and grade are not in the seed, so their variants are paired
        starts = np.random.default_rng(seed_of("b21", label)).uniform(d["times"][0] + 240,
                                                                       d["times"][-1] - T_PKT - 20, N_MAX)[:n]
        for fc in [float(f) for f in a.fcs.split(",")]:
            for cnr in [int(c) for c in a.cnrs.split(",")]:
                acc = {k: {(ts, t): [] for ts in T_SEGS for t in T_INT} for k in ("signal", "sensor")}
                none = {t: [] for t in T_INT}
                mean_only = {t: [] for t in T_INT}  # signal-only fallback: the packet's mean Doppler, no tracking
                for i, t0 in enumerate(starts):
                    rp = np.random.default_rng(seed_of("b21", label, int(fc), cnr, i))
                    tg = t0 + np.arange(n_s) / FS
                    eps = at_times(d["eps"], d["times"], tx_track.fs, tg)
                    prior = tx.estimate(tg, SIGMA_IMU) + rx.estimate(tg, SIGMA_IMU)
                    c = longint.dsss(int(T_PKT * CHIP_RATE), CHIP_RATE, FS, rp)
                    y0, y = longint.received(c, FS, fc, eps, cnr, rp)
                    z_mean = unwarp(y0, FS, np.full(n_s, np.mean(eps)), fc)
                    for t in T_INT:
                        none[t].append(longint.efficiency(y0, c, FS, t))
                        mean_only[t].append(longint.efficiency(z_mean, c, FS, t))
                    for ts in T_SEGS:
                        for name, coarse, st, rms in (("signal", np.full(n_s, np.mean(eps)), sig_set, rms_sig),
                                                      ("sensor", prior, sen_set, rms_sen)):
                            eff, _ = longint.receive(y, y0, c, FS, fc, coarse, cnr, ts, st, band_hz=BAND_K * rms * fc,
                                                     tracker=tracker)
                            for t in T_INT:
                                acc[name][(ts, t)].append(eff(t))
                res = {}
                for name, by in acc.items():
                    best = max(((10 * np.log10(t * max(np.mean(v), 1e-12)), ts, t) for (ts, t), v in by.items()))
                    res[name] = dict(best_gain_db=best[0], t_seg=best[1], best_t_int=best[2])
                g_none = max(10 * np.log10(t * max(np.mean(v), 1e-12)) for t, v in none.items())
                g_mean = max(10 * np.log10(t * max(np.mean(v), 1e-12)) for t, v in mean_only.items())
                res["signal_tracked"] = dict(res["signal"])
                if g_mean > res["signal"]["best_gain_db"]:  # best signal-only = better of tracking and mean only
                    res["signal"] = dict(best_gain_db=g_mean, t_seg=None, best_t_int=None, fallback="mean only")
                rows.append(dict(scenario=label, fc_hz=fc, cnr_dbhz=cnr, n_packets=n, tracker=tracker, grade=a.grade,
                                 none_db=g_none, mean_only_db=g_mean,
                                 band_hz=dict(signal=BAND_K * rms_sig * fc, sensor=BAND_K * rms_sen * fc), **res))
                print(f"{label:28s} fc {fc / 1e3:4.0f} kHz CNR {cnr:2d} dB-Hz | best coherent gain [dB re 1 s]: none "
                      f"{g_none:5.1f}  mean only {g_mean:5.1f}  signal {res['signal']['best_gain_db']:5.1f} (seg {res['signal']['t_seg']})  sensor "
                      f"{res['sensor']['best_gain_db']:5.1f} (seg {res['sensor']['t_seg']:g})", flush=True)
            with open(os.path.join(output_dir("b21_waveform_b"), out + ".json"), "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=1)


def merge(out, parts):
    """Concatenate the rows of split jobs (results/b21_waveform_b/<part>.json) into <out>.json."""
    d = output_dir("b21_waveform_b")
    rows = [r for p in parts for r in json.load(open(os.path.join(d, p + ".json")))]
    with open(os.path.join(d, out + ".json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    print(f"merged {len(parts)} parts, {len(rows)} rows -> {out}.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "merge":  # b21_waveform_b.py merge <out> <part> ...
        merge(sys.argv[2], sys.argv[3:])
    else:
        main()
