"""
Tune the sensor fusion's Kalman settings (PLAN: Phase 5), as the signal-only ones were tuned
(scripts/b1_tune_signal_kalman.py): grid over sigma_dr0 and q of chain.FUSION, used by receive_fused_safe as its
starting point (it moves towards the signal-only settings when the prior is poor). Same cases and SNRs as b1;
industrial MEMS, 32-bit side information. The setting with the best mean is the tuned default.
Usage: python scripts/b14_tune_fusion.py [n_packets]. Output: results/b1_baselines/tune_fusion.json.
"""
import itertools
import json
import os
import sys

import numpy as np

from uwofdm import chain, frame, motion, sideinfo, waveform
from uwofdm.experiment import at_times, genie_powers
from uwofdm.paths import output_dir
from b1_tune_signal_kalman import CASES, SNRS_DB
from e2_chain import SCENARIOS, SIGMA_IMU, allocation
from e2_imu_grades import node

SIGMA_DR0 = (1e-4, 3e-4, 1e-3)
Q_STD = (5e-5, 1e-4, 2e-4, 4e-4, 1e-3)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rows = []
    for i, (label, n_data) in enumerate(CASES):
        tx_track, rx_track = SCENARIOS[label]()
        d = motion.link_doppler(tx_track, rx_track)
        F = frame.Frame(n_data=n_data)
        o = F.o
        T = F.length / o.fs
        tx = node(tx_track, d["u"], +1, 0.0, "industrial", 600 + i)
        rx = node(rx_track, d["u"], -1, d["tau"], "industrial", 650 + i)
        ranges, alloc = allocation(F, tx, tx_track.fs, budgets=(32,))
        a = alloc[(32, "none")]
        rng = np.random.default_rng(500 + i)
        pk = []
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
            c = sideinfo.project(tx.estimate(te[:F.length], SIGMA_IMU), te[:F.length] - t0, T, 24)
            prior = sideinfo.reconstruct(sideinfo.quantise_alloc(c, a, ranges), np.minimum(te - t0, T), T) \
                + rx.estimate(te, SIGMA_IMU)
            for snr in SNRS_DB:
                sig = np.sqrt(4.0 / 10 ** (snr / 10) / 2)
                pk.append((snr, y0 + sig * (rp.standard_normal(len(y0)) + 1j * rp.standard_normal(len(y0))), X,
                           (0, lead + 1500), prior))
        default = dict(chain.FUSION)
        for sdr, qs in itertools.product(SIGMA_DR0, Q_STD):
            chain.FUSION.update(sigma_dr0=sdr, q=qs ** 2)
            acc = {}
            for snr, y, X, se, prior in pk:
                acc.setdefault(snr, []).append(genie_powers(chain.receive_fused_safe(y, F, prior, se)[0], X, F.used))
            for snr, pw in acc.items():
                pw = np.array(pw)
                rows.append(dict(scenario=label, n_data=n_data, snr_db=snr, sigma_dr0=sdr, q_std=qs,
                                 sinr_db=float(10 * np.log10(pw[:, 0].sum() / pw[:, 1].sum()))))
            print(f"{label:28s} {n_data:2d} sym sigma_dr0 {sdr:.0e} q_std {qs:.0e}: " +
                  "  ".join(f"{r['snr_db']} dB {r['sinr_db']:5.2f}" for r in rows[-len(acc):]), flush=True)
        chain.FUSION.clear()
        chain.FUSION.update(default)
    mean = {}
    for r in rows:
        mean.setdefault((r["sigma_dr0"], r["q_std"]), []).append(r["sinr_db"])
    for k, v in sorted(mean.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  sigma_dr0 {k[0]:.0e} q_std {k[1]:.0e}: {np.mean(v):6.3f} dB", flush=True)
    best = max(mean, key=lambda k: np.mean(mean[k]))
    print(f"best: sigma_dr0 {best[0]:.0e}, q_std {best[1]:.0e} (current 3e-4, 2e-4)")
    with open(os.path.join(output_dir("b1_baselines"), "tune_fusion.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(n_packets=n, rows=rows, best=dict(sigma_dr0=best[0], q_std=best[1])), fh, indent=1)


if __name__ == "__main__":
    main()
