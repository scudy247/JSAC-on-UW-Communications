"""
Fair baselines, step 2 (PLAN: Phase 5): the best signal-only receivers against sensor fusion, with more packets and
per-packet statistics.

Receivers (single path, real motion, unknown start, same packets and noise for all):
  lilee               Li et al. 2008 / Lee et al. 2026 front end
  signal_kalman       signal-only Kalman, tuned settings (scripts/b1_tune_signal_kalman.py)
  pfft<P>_kalman      signal-only Kalman resampling + partial-FFT demodulation with P segments (Yerramalli et al. 2012)
  safe_<grade>        sensor fusion (32-bit side information + own IMU; chain.receive_fused_safe), IMU grade 'old'
                      (original model) and 'industrial' (GNSS-aided MEMS, imu.py)
  safe_<grade>_pfft<P> the same + partial-FFT demodulation (pFFT is a demodulator: both sides may use it)
Pilots are known to every receiver (the pFFT weights use them). Metric: genie SINR on data symbols per packet;
reported power-averaged (as before), median and 10th percentile, since a few bad packets dominate the power average
(8-20 packets moved buoy-buoy 4 s signal-only results by ~10 dB between packet sets).
Usage: python scripts/b2_baselines.py [n_packets] [sigma_dr0] [q_std]. Output: results/b1_baselines/compare.json.
"""
import json
import os
import sys

import numpy as np

from uwofdm import chain, frame, motion, sideinfo, waveform
from uwofdm.experiment import at_times, genie_powers
from uwofdm.paths import output_dir
from e2_chain import SCENARIOS, SIGMA_IMU, allocation
from e2_imu_grades import node

CASES = (("buoy - buoy, OOI Irminger", 7), ("buoy - buoy, OOI Irminger", 27), ("buoy -> seabed, OOI Irminger", 27),
         ("buoy -> seabed, CDIP swell", 7))
SNRS_DB = (20, 30)
GRADES = ("old", "industrial")
N_SEG = (4, 8)


def run_case(label, n_data, n, seed, sk_params):
    tx_track, rx_track = SCENARIOS[label]()
    return run_link(tx_track, rx_track, n_data, n, seed, sk_params)


def run_link(tx_track, rx_track, n_data, n, seed, sk_params, grades=GRADES, n_seg=N_SEG, snrs=SNRS_DB, ofdm=None,
             pilot_track=False, only_new=False):
    """Per-packet SINR of every receiver on one link (tracks), n packets; returns {(snr, receiver): stats}."""
    d = motion.link_doppler(tx_track, rx_track)
    F = frame.Frame(n_data=n_data) if ofdm is None else frame.Frame(o=ofdm, n_data=n_data)
    o = F.o
    T = F.length / o.fs
    nodes = {g: (node(tx_track, d["u"], +1, 0.0, g, seed + 1), node(rx_track, d["u"], -1, d["tau"], g, seed + 2))
             for g in grades}
    if grades:
        ranges, alloc = allocation(F, nodes[grades[0]][0], tx_track.fs, budgets=(32,))
        a = alloc[(32, "none")]
    rng = np.random.default_rng(seed)
    per = {}
    for t0 in rng.uniform(d["times"][0] + 240, d["times"][-1] - T - 20, n):
        rp = np.random.default_rng(rng.integers(1 << 30))
        X = o.random_symbols(F.n_data, rp)
        X[:, F.nulls] = 0
        s = F.build(o.random_symbols(F.n_header, rp), X)
        lead = int(rp.integers(500, 3000))
        x = np.concatenate([np.zeros(lead, complex), s, np.zeros(8000, complex)])
        tg = t0 - lead / o.fs + np.arange(len(x)) / o.fs
        y0 = waveform.warp(x, o.fs, at_times(d["eps"], d["times"], tx_track.fs, tg), o.fc)
        te = t0 + np.arange(F.length + 4000) / o.fs
        priors = {}
        for g, (tx, rx) in nodes.items():
            c = sideinfo.project(tx.estimate(te[:F.length], SIGMA_IMU), te[:F.length] - t0, T, 24)
            priors[g] = sideinfo.reconstruct(sideinfo.quantise_alloc(c, a, ranges), np.minimum(te - t0, T), T) \
                + rx.estimate(te, SIGMA_IMU)
        for snr in snrs:
            sig = np.sqrt(4.0 / 10 ** (snr / 10) / 2)
            y = y0 + sig * (rp.standard_normal(len(y0)) + 1j * rp.standard_normal(len(y0)))
            search = (0, lead + 1500)
            out = {}
            if not only_new:  # only_new: skip the receivers already run on the same packets (same seed)
                out["lilee"] = chain.receive_signal_only(y, F, search)[0]
                Ys, inf = chain.receive_signal_fused(y, F, search, params=sk_params)
                out["signal_kalman"] = Ys
                for P in n_seg:
                    out[f"pfft{P}_kalman"] = chain._demod_pfft(inf["z"], F, X, P)[0]
            if pilot_track:
                Yp, inf_p = chain.receive_pilot_track(y, F, X, search, params=sk_params)
                out["pilot_track"] = Yp
                for P in n_seg:
                    out[f"pfft{P}_pilot_track"] = chain._demod_pfft(inf_p["z"], F, X, P)[0]
            for g, pr in priors.items():
                Yf, inf_f = chain.receive_fused_safe(y, F, pr, search)
                out[f"safe_{g}"] = Yf
                for P in n_seg:
                    out[f"safe_{g}_pfft{P}"] = chain._demod_pfft(inf_f["z"], F, X, P)[0]
            for k, Y in out.items():
                per.setdefault((snr, k), []).append(genie_powers(Y, X, F.used))
    res = {}
    for (snr, k), pw in per.items():
        pw = np.array(pw)
        db = 10 * np.log10(pw[:, 0] / pw[:, 1])
        res[(snr, k)] = dict(power_avg_db=float(10 * np.log10(pw[:, 0].sum() / pw[:, 1].sum())),
                             median_db=float(np.median(db)), p10_db=float(np.percentile(db, 10)),
                             per_packet_db=[float(v) for v in db])
    return res


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    sk = dict(sigma_dr0=float(sys.argv[2]), q=float(sys.argv[3]) ** 2) if len(sys.argv) > 3 else None
    out_dir = output_dir("b1_baselines")
    rows = []
    for i, (label, n_data) in enumerate(CASES):
        res = run_case(label, n_data, n, 700 + i, sk)
        for (snr, k), v in sorted(res.items()):
            rows.append(dict(scenario=label, n_data=n_data, snr_db=snr, receiver=k, **v))
        for snr in SNRS_DB:
            keys = [k for (s, k) in res if s == snr]
            print(f"{label:28s} {n_data:2d} sym SNR {snr} (power avg / median / p10):", flush=True)
            for k in keys:
                v = res[(snr, k)]
                print(f"    {k:22s} {v['power_avg_db']:5.1f} / {v['median_db']:5.1f} / {v['p10_db']:5.1f}", flush=True)
        with open(os.path.join(out_dir, "compare.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(n_packets=n, signal_kalman_params=sk, rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
