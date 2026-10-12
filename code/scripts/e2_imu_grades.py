"""
Does the sensor gain survive realistic motion sensors? (review of 12 Oct 2026; PLAN: Phases 1 and 5)

Same chain as scripts/e2_chain.py (real buoy motion, single path, unknown start, complex noise; 32-bit side
information, allocation for no signal estimate), with the moving nodes' IMUs from imu.RealisticIMU instead of the
original model (true velocity + 50 ms latency bridged by the true acceleration + Gauss-Markov drift 0.01 m/s).
Grades: 'old' (original model), nav / industrial / consumer (GNSS-aided, surface buoys) and nav_unaided /
industrial_unaided (no aiding: a subsurface node). A static seabed receiver uses no IMU (its share is exactly 0).
Packets and noise are identical across grades (paired comparison); the signal-only receivers do not depend on the
grade and are run once.
Receivers: signal_only (Li et al. 2008 / Lee et al. 2026 front end), signal_kalman (best signal-only), fused_32
(sensor prior + Kalman fusion), prior_32 (sensor prior alone). Metric: genie SINR on data symbols, power-averaged over
packets; also the rms error of the sensor prior's Doppler over the frame (the estimation error itself).
Usage: python scripts/e2_imu_grades.py [n_packets]. Output: results/e2_imu_grades/points.json.
"""
import json
import os
import sys

import numpy as np

from uwofdm import chain, frame, imu, motion, sideinfo, waveform
from uwofdm.experiment import NodeIMU, at_times, genie_powers
from uwofdm.paths import output_dir
from e2_chain import SCENARIOS, SIGMA_IMU, allocation

SNRS_DB = (20, 30)
BITS = 32
GRADES = ("old", "nav", "industrial", "consumer", "nav_unaided", "industrial_unaided")


def _static(track):
    return np.allclose(track.vel_fast, 0) and np.allclose(track.vel_slow, 0)


def node(track, u, sign, shift, grade, seed):
    """NodeIMU of one node for a grade; a static node knows its share is 0 (no sensor)."""
    kw = dict(sigmas=(SIGMA_IMU,), tau_imu=20.0, latency=0.05, extrapolate=True, lever_corrected=True)
    if _static(track):
        return NodeIMU(track, u, sign, shift, np.random.default_rng(seed), grade=imu.ZERO, **dict(kw, latency=0.0))
    g = None if grade == "old" else imu.GRADES[grade]
    lever_m = 2.0 if track.vel_lever is not None else None
    return NodeIMU(track, u, sign, shift, np.random.default_rng(seed), grade=g, lever_m=lever_m, **kw)


def run(F, tx_track, rx_track, n_packets, seed):
    o = F.o
    d = motion.link_doppler(tx_track, rx_track)
    nodes = {g: (node(tx_track, d["u"], +1, 0.0, g, seed + 1), node(rx_track, d["u"], -1, d["tau"], g, seed + 2))
             for g in GRADES}
    ref_tx = nodes["old"][0]
    ranges, alloc = allocation(F, ref_tx, ref_tx.fs, budgets=(BITS,))
    a = alloc[(BITS, "none")]
    T = F.length / o.fs
    rng = np.random.default_rng(seed)
    acc, perr = {}, {}
    for t0 in rng.uniform(d["times"][0] + 240, d["times"][-1] - T - 20, n_packets):  # 240 s: filter start-up
        rp = np.random.default_rng(rng.integers(1 << 30))
        X = o.random_symbols(F.n_data, rp)
        s = F.build(o.random_symbols(F.n_header, rp), X)
        lead, tail = int(rp.integers(500, 3000)), 8000
        x = np.concatenate([np.zeros(lead, complex), s, np.zeros(tail, complex)])
        tg = t0 - lead / o.fs + np.arange(len(x)) / o.fs
        y0 = waveform.warp(x, o.fs, at_times(d["eps"], d["times"], ref_tx.fs, tg), o.fc)
        te = t0 + np.arange(F.length + tail // 2) / o.fs
        true = at_times(d["eps"], d["times"], ref_tx.fs, te[:F.length])
        priors = {}
        for g, (tx, rx) in nodes.items():
            tx_est, rx_est = tx.estimate(te, SIGMA_IMU), rx.estimate(te, SIGMA_IMU)
            c = sideinfo.project(tx_est[:F.length], te[:F.length] - t0, T, 24)
            q = sideinfo.quantise_alloc(c, a, ranges)
            priors[g] = sideinfo.reconstruct(q, np.minimum(te - t0, T), T) + rx_est
            perr.setdefault(g, []).append(float(np.sqrt(np.mean((priors[g][:F.length] - true) ** 2))))
        for snr in SNRS_DB:
            sigma = np.sqrt(4.0 / 10 ** (snr / 10) / 2)
            y = y0 + sigma * (rp.standard_normal(len(y0)) + 1j * rp.standard_normal(len(y0)))
            search = (0, lead + 1500)
            out = {"signal_only": chain.receive_signal_only(y, F, search)[0],
                   "signal_kalman": chain.receive_signal_fused(y, F, search)[0]}
            for g, pr in priors.items():
                out[f"fused_{g}"] = chain.receive_fused(y, F, pr, search)[0]
                out[f"prior_{g}"] = chain.receive_aided(y, F, pr, search, prepost=False, cfo=False)[0]
            for k, Y in out.items():
                acc.setdefault((snr, k), []).append(genie_powers(Y, X, F.used))
    res = {}
    for (snr, k), pw in acc.items():
        pw = np.array(pw)
        res[(snr, k)] = float(10 * np.log10(pw[:, 0].sum() / pw[:, 1].sum()))
    return res, {g: float(np.sqrt(np.mean(np.square(v)))) for g, v in perr.items()}


def main():
    n_packets = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = output_dir("e2_imu_grades")
    rows = []
    for i, (label, make) in enumerate(SCENARIOS.items()):
        tx_track, rx_track = make()
        for n_data in (7, 27):
            F = frame.Frame(n_data=n_data)
            res, perr = run(F, tx_track, rx_track, n_packets, seed=100 + 10 * i + n_data)
            for (snr, k), v in sorted(res.items()):
                rows.append(dict(scenario=label, n_data=n_data, snr_db=snr, receiver=k, sinr_db=v))
            for g, e in perr.items():
                rows.append(dict(scenario=label, n_data=n_data, grade=g, prior_rms_eps=e))
            for snr in SNRS_DB:
                fused = "  ".join(f"{g} {res[(snr, 'fused_' + g)]:5.1f}" for g in GRADES)
                print(f"{label:28s} {n_data:2d} sym SNR {snr}: Li/Lee {res[(snr, 'signal_only')]:5.1f}  "
                      f"signal Kalman {res[(snr, 'signal_kalman')]:5.1f} | fused: {fused}", flush=True)
            print(f"{'':28s} prior rms eps: " + "  ".join(f"{g} {e:.1e}" for g, e in perr.items()), flush=True)
            with open(os.path.join(out, "points.json"), "w", encoding="utf-8") as fh:
                json.dump(dict(n_packets=n_packets, snrs_db=SNRS_DB, bits=BITS, grades=GRADES, rows=rows), fh,
                          indent=1)


if __name__ == "__main__":
    main()
