"""
Fair baselines, step 1 (PLAN: Phase 5): tune the signal-only Kalman (chain.receive_signal_fused), whose settings
SIGNAL_FUSION = (sigma_r0 2e-4, sigma_dr0 2e-3, q (4e-3)^2) were set by hand.

Grid over sigma_dr0 (prior uncertainty of the Doppler rate) and q (process noise of the rate, i.e. how fast the
receiver lets the Doppler wander); sigma_r0 stays (the mean comes from the chirps). Hard cases where the signal-only
receiver lags: buoy-buoy OOI 1.05 / 4.05 s, buoy -> seabed OOI 4.05 s, plus the moderate CDIP 1.05 s as a check;
per-subcarrier SNR 20 and 30 dB; same packets and noise for every setting. Metric: genie SINR on data symbols,
power-averaged. The setting with the best mean over the cases is the tuned baseline (one setting for all links,
as for the sensor fusion).
Usage: python scripts/b1_tune_signal_kalman.py [n_packets]. Output: results/b1_baselines/tune_signal_kalman.json.
"""
import itertools
import json
import os
import sys

import numpy as np

from uwofdm import chain, frame, motion, waveform
from uwofdm.experiment import at_times, genie_powers
from uwofdm.paths import output_dir
from e2_chain import SCENARIOS

CASES = (("buoy - buoy, OOI Irminger", 7), ("buoy - buoy, OOI Irminger", 27), ("buoy -> seabed, OOI Irminger", 27),
         ("buoy -> seabed, CDIP swell", 7))
SNRS_DB = (20, 30)
SIGMA_DR0 = (5e-4, 2e-3, 8e-3)
Q_STD = (1e-3, 4e-3, 1.6e-2, 6.4e-2)


def packets(label, n_data, n, seed):
    """Received packets (y, X, search) at each SNR for one case: real motion, single path, unknown start."""
    tx, rx = SCENARIOS[label]()
    d = motion.link_doppler(tx, rx)
    F = frame.Frame(n_data=n_data)
    o = F.o
    T = F.length / o.fs
    rng = np.random.default_rng(seed)
    out = []
    for t0 in rng.uniform(d["times"][0] + 240, d["times"][-1] - T - 20, n):
        rp = np.random.default_rng(rng.integers(1 << 30))
        X = o.random_symbols(F.n_data, rp)
        s = F.build(o.random_symbols(F.n_header, rp), X)
        lead = int(rp.integers(500, 3000))
        x = np.concatenate([np.zeros(lead, complex), s, np.zeros(8000, complex)])
        tg = t0 - lead / o.fs + np.arange(len(x)) / o.fs
        y0 = waveform.warp(x, o.fs, at_times(d["eps"], d["times"], tx.fs, tg), o.fc)
        for snr in SNRS_DB:
            sig = np.sqrt(4.0 / 10 ** (snr / 10) / 2)
            y = y0 + sig * (rp.standard_normal(len(y0)) + 1j * rp.standard_normal(len(y0)))
            out.append((snr, y, X, (0, lead + 1500)))
    return F, out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = []
    for i, (label, n_data) in enumerate(CASES):
        F, pk = packets(label, n_data, n, seed=500 + i)
        for sdr, qs in itertools.product(SIGMA_DR0, Q_STD):
            acc = {}
            for snr, y, X, search in pk:
                Y = chain.receive_signal_fused(y, F, search, params=dict(sigma_dr0=sdr, q=qs ** 2))[0]
                acc.setdefault(snr, []).append(genie_powers(Y, X, F.used))
            for snr, pw in acc.items():
                pw = np.array(pw)
                rows.append(dict(scenario=label, n_data=n_data, snr_db=snr, sigma_dr0=sdr, q_std=qs,
                                 sinr_db=float(10 * np.log10(pw[:, 0].sum() / pw[:, 1].sum()))))
            print(f"{label:28s} {n_data:2d} sym  sigma_dr0 {sdr:.0e} q_std {qs:.1e}: " +
                  "  ".join(f"{r['snr_db']} dB {r['sinr_db']:5.1f}" for r in rows[-len(acc):]), flush=True)
    mean = {}
    for r in rows:
        mean.setdefault((r["sigma_dr0"], r["q_std"]), []).append(r["sinr_db"])
    best = max(mean, key=lambda k: np.mean(mean[k]))
    print("mean SINR over cases and SNRs per setting:")
    for k, v in sorted(mean.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  sigma_dr0 {k[0]:.0e} q_std {k[1]:.1e}: {np.mean(v):5.2f} dB")
    print(f"best: sigma_dr0 {best[0]:.0e}, q_std {best[1]:.1e} (current 2e-3, 4.0e-3)")
    out = output_dir("b1_baselines")
    with open(os.path.join(out, "tune_signal_kalman.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(n_packets=n, rows=rows, best=dict(sigma_dr0=best[0], q_std=best[1])), fh, indent=1)


if __name__ == "__main__":
    main()
