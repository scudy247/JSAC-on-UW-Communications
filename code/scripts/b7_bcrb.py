"""
T3 on real motion (PLAN: Phase 5, theory): Bayesian Cramer-Rao bound (bcrb.py) on the Doppler trajectory over a
packet, signal-only vs with the sensor prior, against SNR and pilot count.

Prior covariances on a 10 ms grid over the frame, from the auto- and cross-correlations of up to one hour of real
motion per case (stationarity assumed; per-packet sample covariances overfit the 435 x 435 matrices):
  signal-only        the link Doppler itself
  sensor (grade g)   link Doppler given the sensor prior s (32-bit side information + own IMU, as in the chain;
                     grades old / industrial / consumer, imu.py): K_ee - K_es K_ss^-1 K_se from the joint samples
                     (Gaussian conditioning: the receiver keeps what it knows of the motion statistics and adds s;
                     using only the covariance of eps - s would discard the former and can be worse than no prior)
Measurement sets: none (prior alone), 'cfo+timing' (per-symbol null-subcarrier offsets + postamble timing, what the
signal Kalman and the sensor fusion use), 'cfo+timing+pilot' (+ pilot carrier-phase increments, the pilot tracker).
Outputs per case, prior, measurement set and SNR: SIR bound (ICI only) and SINR bound = 1 / (1/snr + 1/SIR); pilot
sweep at 30 dB: SIR bound with 1/8 ... all of the pilots in the phase increments.
Usage: python scripts/b7_bcrb.py [n_packets]. Output: results/b7_bcrb/points.json.
"""
import json
import os
import sys

import numpy as np

from uwofdm import bcrb, frame, motion, sideinfo
from uwofdm.experiment import at_times
from uwofdm.paths import output_dir
from e2_chain import SCENARIOS, SIGMA_IMU, allocation
from e2_imu_grades import node

CASES = (("buoy - buoy, OOI Irminger", 7), ("buoy - buoy, OOI Irminger", 27), ("buoy -> seabed, OOI Irminger", 27),
         ("buoy -> seabed, CDIP swell", 7))
GRADES = ("old", "industrial", "consumer")
SNRS_DB = (0, 5, 10, 15, 20, 25, 30, 35)
SETS = {"none": (), "cfo+timing": ("cfo", "timing"), "cfo+timing+pilot": ("cfo", "timing", "pilot")}


def _toeplitz(acf, n):
    idx = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return acf[idx]


def _xcorr(a, b, max_lag):
    """c[l] = E[a(t + l) b(t)] for l = -max_lag ... max_lag (FFT, biased, means removed)."""
    a, b = a - a.mean(), b - b.mean()
    n = len(a)
    nf = 1 << int(np.ceil(np.log2(2 * n)))
    c = np.fft.ifft(np.fft.fft(a, nf) * np.conj(np.fft.fft(b, nf))).real / n
    return np.concatenate([c[nf - max_lag:], c[:max_lag + 1]])


def covariances(label, n_data, n, seed, dt=0.01, max_record_s=3600.0):
    """Prior covariances on the 10 ms frame grid from the auto- and cross-correlations of the whole record
    (stationary motion and sensor errors; the 32-bit quantisation of the side information is left out of the
    prior, a small error at 32 bits). Packet-wise sample covariances overfit: 435 x 435 matrices from a few hundred
    packets. `n` is unused (kept for the command line)."""
    tx_track, rx_track = SCENARIOS[label]()
    d = motion.link_doppler(tx_track, rx_track)
    F = frame.Frame(n_data=n_data)
    t = bcrb.grid(F, dt)
    nodes = {g: (node(tx_track, d["u"], +1, 0.0, g, seed + 1), node(rx_track, d["u"], -1, d["tau"], g, seed + 2))
             for g in GRADES}
    t0, t1 = d["times"][0] + 240, min(d["times"][-1] - 20, d["times"][0] + 240 + max_record_s)
    tr = np.arange(t0, t1, dt)
    eps = at_times(d["eps"], d["times"], tx_track.fs, tr)
    L = len(t)
    acf = _xcorr(eps, eps, L)[L:]
    K = {"signal": _toeplitz(acf, L) + np.mean(eps) ** 2}
    for g, (tx, rx) in nodes.items():
        s_ = tx.estimate(tr, SIGMA_IMU) + rx.estimate(tr, SIGMA_IMU)  # the sensor prior as a stationary series
        Kss = _toeplitz(_xcorr(s_, s_, L)[L:], L)
        c = _xcorr(eps, s_, L)  # E[eps(t + l) s(t)], l = -L..L
        lags = np.arange(L)[:, None] - np.arange(L)[None, :]
        Kes = c[lags + L]  # Kes[i, j] = E[eps(t_i) s(t_j)]
        lam = 1e-6 * np.trace(Kss) / L
        Kc = K["signal"] - Kes @ np.linalg.solve(Kss + lam * np.eye(L), Kes.T)
        K[g] = 0.5 * (Kc + Kc.T)
    return F, t, K


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    rows = []
    for i, (label, n_data) in enumerate(CASES):
        F, t, K = covariances(label, n_data, n, 1500 + i)
        for prior, Kp in K.items():
            for name, kinds in SETS.items():
                if prior == "signal" and name == "none":
                    continue
                for snr in SNRS_DB:
                    if kinds:
                        H, R = bcrb.measurements(F, t, snr, kinds)
                        P = bcrb.bound(Kp, H, R)
                    else:
                        P = Kp
                    sir = bcrb.sir_db(P, F, t)
                    sinr = -10 * np.log10(10 ** (-snr / 10) + 10 ** (-sir / 10))
                    rows.append(dict(scenario=label, n_data=n_data, prior=prior, measurements=name, snr_db=snr,
                                     sir_bound_db=sir, sinr_bound_db=float(sinr),
                                     rms_error=float(np.sqrt(np.trace(P) / len(t)))))
            print(f"{label:28s} {n_data:2d} sym  prior {prior:10s}: " + "  ".join(
                f"{r['measurements']} @30 dB SIR {r['sir_bound_db']:5.1f}" for r in rows
                if r['scenario'] == label and r['n_data'] == n_data and r['prior'] == prior and r['snr_db'] == 30),
                flush=True)
        o = F.o
        n_all = int(np.sum(o.pilots() & o.active))
        for prior in ("signal", "industrial"):
            for frac in (8, 4, 2, 1):
                H, R = bcrb.measurements(F, t, 30, SETS["cfo+timing+pilot"], n_pilots=n_all // frac)
                sir = bcrb.sir_db(bcrb.bound(K[prior], H, R), F, t)
                rows.append(dict(scenario=label, n_data=n_data, prior=prior, measurements="pilot_sweep", snr_db=30,
                                 n_pilots=n_all // frac, sir_bound_db=sir))
            print(f"{'':28s} pilot sweep @30 dB, prior {prior}: " + "  ".join(
                f"{r['n_pilots']} pilots {r['sir_bound_db']:5.1f}" for r in rows if r.get('n_pilots') and
                r['scenario'] == label and r['n_data'] == n_data and r['prior'] == prior), flush=True)
        with open(os.path.join(output_dir("b7_bcrb"), "points.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(n_packets=n, rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
