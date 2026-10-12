"""
Acquisition with the sensor prior (PLAN: Phase 5, "acquisition: prior-centred Doppler search window; probability of
wrong acquisition vs window and SNR").

The frame's chirps tolerate Doppler, so the preamble is found without a Doppler search; the Doppler search is where
the receiver looks for the postamble: a feature emitted D samples after the preamble arrives D / (1 + eps) later.
A signal-only receiver must search the whole Doppler range (+-3e-3: about +-400 samples on a 4 s packet); with the
sensor prior (side information + own IMU) the window can be centred on the prior's mean Doppler over the packet and
sized from the sensor quality. At low SNR a wider window offers more noise peaks that can beat the true postamble.

Per packet (real motion, single path, unknown start, complex noise at per-subcarrier SNR -12 ... -4 dB, where chirp detection breaks):
  preamble miss      detected preamble > 1 ms from the true position
  wrong acquisition  |eps_hat - eps_true| > 1e-4, eps_true the mean link Doppler between preamble and postamble
                     (1e-4 = 1.6 Hz = 0.2 subcarrier: beyond it the residual search of every receiver fails),
                     counted on packets whose preamble was found
Windows: full +-3e-3 around 0 (signal-only); prior-centred +-4 sigma for the industrial MEMS (sigma = the prior's
rms error measured on these packets, at least 2e-5) and for the original IMU model. Cost: correlation lags searched.
Usage: python scripts/b6_acquisition.py [n_packets]. Output: results/b6_acquisition/points.json.
"""
import json
import os
import sys

import numpy as np

from uwofdm import estimators, frame, motion, sideinfo, waveform
from uwofdm.experiment import at_times
from uwofdm.paths import output_dir
from e2_chain import SCENARIOS, SIGMA_IMU, allocation
from e2_imu_grades import node

CASES = (("buoy - buoy, OOI Irminger", 7), ("buoy - buoy, OOI Irminger", 27), ("buoy -> seabed, OOI Irminger", 27))
SNRS_DB = (-12, -11, -10, -9, -8, -7, -6, -4)
GRADES = ("industrial", "old")
EPS_MAX = 3e-3
WRONG = 1e-4


def window(n_pre, F, center, half):
    """Postamble search window (samples) for scales in [center - half, center + half]."""
    lo = n_pre + F.gap / (1 + center + half)
    hi = n_pre + F.gap / (1 + center - half)
    return int(np.floor(lo)) - 2, int(np.ceil(hi)) + 2


def run_case(label, n_data, n, seed):
    tx_track, rx_track = SCENARIOS[label]()
    d = motion.link_doppler(tx_track, rx_track)
    F = frame.Frame(n_data=n_data)
    o = F.o
    T = F.length / o.fs
    nodes = {g: (node(tx_track, d["u"], +1, 0.0, g, seed + 1), node(rx_track, d["u"], -1, d["tau"], g, seed + 2))
             for g in GRADES}
    ranges, alloc = allocation(F, nodes[GRADES[0]][0], tx_track.fs, budgets=(32,))
    a = alloc[(32, "none")]
    rng = np.random.default_rng(seed)
    pk = []
    for t0 in rng.uniform(d["times"][0] + 240, d["times"][-1] - T - 20, n):
        rp = np.random.default_rng(rng.integers(1 << 30))
        s = F.build(o.random_symbols(F.n_header, rp), o.random_symbols(F.n_data, rp))
        lead = int(rp.integers(500, 3000))
        x = np.concatenate([np.zeros(lead, complex), s, np.zeros(8000, complex)])
        tg = t0 - lead / o.fs + np.arange(len(x)) / o.fs
        eps_t = at_times(d["eps"], d["times"], tx_track.fs, tg)
        y0 = waveform.warp(x, o.fs, eps_t, o.fc)
        true_pre = lead - waveform.phase_track(eps_t, o.fs)[lead] * o.fs  # arrival position of the preamble start
        eps_true = float(np.mean(eps_t[lead:lead + F.gap]))
        te = t0 + np.arange(F.length + 4000) / o.fs
        centers = {}
        for g, (tx, rx) in nodes.items():
            c = sideinfo.project(tx.estimate(te[:F.length], SIGMA_IMU), te[:F.length] - t0, T, 24)
            prior = sideinfo.reconstruct(sideinfo.quantise_alloc(c, a, ranges), np.minimum(te - t0, T), T) \
                + rx.estimate(te, SIGMA_IMU)
            centers[g] = float(np.mean(prior[:F.gap]))
        pk.append((y0, lead, true_pre, eps_true, centers, rp))
    sig = {g: max(2e-5, float(np.sqrt(np.mean([(p[4][g] - p[3]) ** 2 for p in pk])))) for g in GRADES}
    wins = {"full": (lambda p: 0.0, EPS_MAX)}
    wins.update({f"prior_{g}": ((lambda p, g=g: p[4][g]), 4 * sig[g]) for g in GRADES})
    res = []
    for snr in SNRS_DB:
        miss, wrong = 0, {w: 0 for w in wins}
        for y0, lead, true_pre, eps_true, centers, rp in pk:
            sigma = np.sqrt(4.0 / 10 ** (snr / 10) / 2)
            y = y0 + sigma * (rp.standard_normal(len(y0)) + 1j * rp.standard_normal(len(y0)))
            n_pre, _ = estimators.detect_chirp(y, F, 0, lead + 1500)
            if abs(n_pre - true_pre) > 1e-3 * o.fs:
                miss += 1
                continue
            for w, (cen, half) in wins.items():
                c0 = cen((y0, lead, true_pre, eps_true, centers))
                n_post, _ = estimators.detect_chirp(y, F, *window(n_pre, F, c0, half))
                wrong[w] += abs(estimators.scale_pre_post(n_pre, n_post, F.gap) - eps_true) > WRONG
        found = len(pk) - miss
        res.append(dict(snr_db=snr, preamble_miss=miss / len(pk),
                        **{f"wrong_{w}": (wrong[w] / found if found else float("nan")) for w in wins},
                        **{f"lags_{w}": int(np.diff(window(0, F, 0.0, h))[0]) for w, (_, h) in wins.items()}))
    return res, sig


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rows = []
    for i, (label, n_data) in enumerate(CASES):
        res, sig = run_case(label, n_data, n, 1300 + i)
        for r in res:
            rows.append(dict(scenario=label, n_data=n_data, prior_sigma=sig, **r))
            print(f"{label:28s} {n_data:2d} sym SNR {r['snr_db']:4d}: preamble miss {r['preamble_miss']:.3f} | wrong "
                  f"acquisition: full {r['wrong_full']:.3f} ({r['lags_full']} lags)  " +
                  "  ".join(f"{g} {r[f'wrong_prior_{g}']:.3f} ({r[f'lags_prior_{g}']} lags)" for g in GRADES),
                  flush=True)
        with open(os.path.join(output_dir("b6_acquisition"), "points.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(n_packets=n, wrong_threshold=WRONG, rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
