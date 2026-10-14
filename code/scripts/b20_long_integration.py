"""
B: low SNR / long coherent integration (PLAN: Phase 5; candidate B of 14 Oct 2026).

A long-range or low-rate link gains processing gain by integrating coherently over a window T_int; the residual
Doppler r(t) turns the phase theta(t) = 2 pi fc int r dt during the window and erodes that gain. Signal-based Doppler
estimates degrade as the signal gets weaker; a sensor prior does not depend on the SNR.

Phase-level Monte Carlo on real motion (no waveforms): packets of T_PKT seconds on a DT grid, n per case.
  signal-only    linear MMSE estimate of the link Doppler from per-segment frequency measurements (segments of t_seg,
                 known-symbol frequency CRB at carrier-to-noise density CNR: var f = 3 / (2 pi^2 CNR T_seg^3)), with
                 the motion statistics as prior (covariances from the record's correlations); a segment is usable
                 only above SEG_THRESHOLD_DB of segment SNR (the ML frequency-estimation threshold); each receiver
                 uses its best segment length; still optimistic for signal-only above the threshold
  sensor-aided   the same plus the sensor prior s (industrial MEMS at both moving ends, imu.py): MMSE of eps given
                 (s, z) from the joint covariances; at very low CNR it tends to the sensor prior alone
Metric, on the true residual phase: coherent-integration efficiency L(T_int) = E |mean over the window exp(j theta)|^2
over windows tiling the packet; effective coherent gain G = 10 log10(T_int L) (dB relative to 1 s); the best G over
T_int and the T_int reaching it. fc = 16 kHz and 4 kHz (long-range links).
Usage: python scripts/b20_long_integration.py [n_packets]. Output: results/b20_long_integration/points.json.
"""
import json
import os
import sys

import numpy as np

from uwofdm import motion
from uwofdm.experiment import at_times
from uwofdm.paths import output_dir
from e2_chain import SCENARIOS, SIGMA_IMU
from e2_imu_grades import node
from b7_bcrb import _toeplitz, _xcorr

CASES = ("buoy - buoy, OOI Irminger", "buoy -> seabed, OOI Irminger", "buoy -> seabed, CDIP swell")
T_PKT, DT = 16.0, 0.05
T_SEGS = (0.25, 0.5, 1.0, 2.0)  # each receiver uses its best segment length (fairness)
SEG_THRESHOLD_DB = 10.0  # a segment's frequency estimate is usable only above this segment SNR (ML threshold)
CNR_DBHZ = (5, 10, 15, 20, 25, 30, 40)
T_INT = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
FCS = (16e3, 4e3)


def link_series(label):
    """Link Doppler and the sensor prior (both moving ends' industrial IMUs) on a DT grid over the record."""
    tx_track, rx_track = SCENARIOS[label]()
    d = motion.link_doppler(tx_track, rx_track)
    tx = node(tx_track, d["u"], +1, 0.0, "industrial", 11)
    rx = node(rx_track, d["u"], -1, d["tau"], "industrial", 12)
    tr = np.arange(d["times"][0] + 240, d["times"][-1] - 20, DT)
    eps = at_times(d["eps"], d["times"], tx_track.fs, tr)
    s = tx.estimate(tr, SIGMA_IMU) + rx.estimate(tr, SIGMA_IMU)
    return tr, eps, s


def covariances(eps, s, L):
    lags = np.arange(L)[:, None] - np.arange(L)[None, :]
    K = _toeplitz(_xcorr(eps, eps, L)[L:], L) + np.mean(eps) ** 2
    Kss = _toeplitz(_xcorr(s, s, L)[L:], L) + np.mean(s) ** 2
    Kes = _xcorr(eps, s, L)[lags + L] + np.mean(eps) * np.mean(s)
    return K, Kss, Kes


def seg_matrix(n, per_seg):
    H = np.zeros((n // per_seg, n))
    for k in range(n // per_seg):
        H[k, k * per_seg:(k + 1) * per_seg] = 1.0 / per_seg
    return H


def efficiency(err, fc, t_int):
    """E |mean exp(j theta)|^2 over windows of t_int tiling the packet, theta = 2 pi fc int err dt."""
    th = 2 * np.pi * fc * np.cumsum(err) * DT
    w = max(1, int(round(t_int / DT)))
    vals = [abs(np.mean(np.exp(1j * th[i:i + w]))) ** 2 for i in range(0, len(th) - w + 1, w)]
    return float(np.mean(vals))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    L = int(round(T_PKT / DT))
    rows = []
    rng = np.random.default_rng(5)
    for label in CASES:
        tr, eps, s = link_series(label)
        K, Kss, Kes = covariances(eps, s, L)
        starts = rng.integers(0, len(tr) - L, n)
        for fc in FCS:
            for cnr_db in CNR_DBHZ:
                cnr = 10 ** (cnr_db / 10)
                res = {}
                for t_seg in T_SEGS:
                    usable = 10 * np.log10(cnr * t_seg) >= SEG_THRESHOLD_DB
                    H = seg_matrix(L, int(round(t_seg / DT))) if usable else np.zeros((0, L))
                    var_r = 3 / (2 * np.pi ** 2 * cnr * t_seg ** 3) / fc ** 2  # Doppler-scale variance per segment
                    R = var_r * np.eye(len(H))
                    Kj = Kss + 1e-6 * np.trace(Kss) / L * np.eye(L)
                    if len(H):  # signal-only MMSE: eps_hat = K H^T (H K H^T + R)^-1 z; sensor-aided: o = [s; z]
                        G_sig = K @ H.T @ np.linalg.inv(H @ K @ H.T + R)
                        G_sen = np.hstack([Kes, K @ H.T]) @ np.linalg.inv(
                            np.block([[Kj, Kes.T @ H.T], [H @ Kes, H @ K @ H.T + R]]))
                    else:  # below the threshold: no usable signal measurement
                        G_sig, G_sen = np.zeros((L, 0)), Kes @ np.linalg.inv(Kj)
                    acc = {k: {t: [] for t in T_INT} for k in ("none", "signal", "sensor_alone", "sensor")}
                    for i0 in starts:
                        e, sp = eps[i0:i0 + L], s[i0:i0 + L]
                        z = H @ e + np.sqrt(var_r) * rng.standard_normal(len(H))
                        est = {"none": np.zeros(L), "signal": G_sig @ z, "sensor_alone": sp,
                               "sensor": G_sen @ np.concatenate([sp, z])}
                        for k, eh in est.items():
                            for t in T_INT:
                                acc[k][t].append(efficiency(e - eh, fc, t))
                    for k, by_t in acc.items():
                        eff = {t: float(np.mean(v)) for t, v in by_t.items()}
                        gain = {t: 10 * np.log10(t * max(eff[t], 1e-12)) for t in T_INT}
                        best = max(gain, key=gain.get)
                        if k not in res or gain[best] > res[k]["best_gain_db"]:
                            res[k] = dict(efficiency=eff, best_gain_db=gain[best], best_t_int=best, t_seg=t_seg)
                rows.append(dict(scenario=label, fc_hz=fc, cnr_dbhz=cnr_db, **res))
                print(f"{label:28s} fc {fc / 1e3:4.0f} kHz CNR {cnr_db} dB-Hz | best coherent gain [dB re 1 s] (T_int): "
                      + "  ".join(f"{k} {v['best_gain_db']:5.1f} ({v['best_t_int']:g} s, seg {v['t_seg']:g})"
                                  for k, v in res.items()),
                      flush=True)
        with open(os.path.join(output_dir("b20_long_integration"), "points.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
