"""
Sea-state breadth (PLAN: Phase 5): the chain on every downloaded day instead of one extreme day.

Days: 9 CDIP Waverider days (3 sites x swell / mixed / wind) and 6 OOI days (CE04OSSM Oregon, GI01SUMO Irminger;
swell / mixed / wind), as in the compensation-phase sea sweep (c2_sweeps.sweep_sea). Links: buoy -> seabed on every
day; buoy - buoy on the OOI days (two 20-minute bursts of the same day, 2 m lever arms, as scripts/c2_sweeps
.buoy_buoy_link; the CDIP records are one buoy per site). Packets 1.05 and 4.05 s, per-subcarrier SNR 30 dB,
n packets per point. Receivers (scripts/b2_baselines.run_link): Li/Lee, tuned signal-only Kalman, signal Kalman +
partial FFT (8 segments), safe sensor fusion with GNSS-aided industrial MEMS (imu.py), the same + partial FFT.
Each day is described by Hs (4 x std of the vertical displacement, from the vertical velocity spectrum above 0.03 Hz)
and the rms link Doppler. Output: results/b3_sea_states/points.json (per-packet SINR kept).
Usage: python scripts/b3_sea_states.py [n_packets]
"""
import glob
import json
import os
import sys

import numpy as np

from uwofdm import motion, theory
from uwofdm.paths import data_dir, output_dir
from b2_baselines import run_link
from c2_sweeps import seabed_link

LEVER = (0.0, 0.0, 2.0)


def days():
    out = [("cdip", os.path.basename(p)[:-4], None)
           for p in sorted(glob.glob(str(data_dir("buoy", "cdip", "*_*_20*.npz"))))]
    chosen = json.loads(data_dir("buoy", "ooi", "chosen_days.json").read_text())
    out += [("ooi", f"{m}/{v['day']}", LEVER) for m, kinds in chosen.items() for v in kinds.values()]
    return out


def hs(track):
    f, S = theory.spectrum(track.vel_fast[:, 2], track.fs)
    k = f > 0.03
    return float(4 * np.sqrt(np.sum(S[k] / (2 * np.pi * f[k]) ** 2) * (f[1] - f[0])))


def buoy_buoy(name):
    a = motion.buoy_track("ooi", name, (0.0, 0.0, -10.0), segment=3, lever=LEVER)
    b = motion.buoy_track("ooi", name, (1000.0, 0.0, -10.0), segment=10, lever=LEVER)
    n = min(a.n, b.n)
    for tr in (a, b):
        tr.vel_fast, tr.vel_slow, tr.vel_lever = tr.vel_fast[:n], tr.vel_slow[:n], tr.vel_lever[:n]
    return a, b


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out_dir = output_dir("b3_sea_states")
    rows = []
    for i, (source, name, lever) in enumerate(days()):
        links = [("buoy -> seabed", seabed_link(source, name, lever))]
        if source == "ooi":
            links.append(("buoy - buoy", buoy_buoy(name)))
        for j, (kind, (tx, rx)) in enumerate(links):
            d = motion.link_doppler(tx, rx)
            desc = dict(source=source, day=name, link=kind, hs_m=hs(tx), doppler_rms=float(np.std(d["eps"])))
            for n_data in (7, 27):
                res = run_link(tx, rx, n_data, n, 900 + 37 * i + 7 * j + n_data, None, grades=("industrial",),
                               n_seg=(8,), snrs=(30,))
                for (snr, k), v in sorted(res.items()):
                    rows.append(dict(desc, n_data=n_data, snr_db=snr, receiver=k, **v))
                line = "  ".join(f"{k} {v['power_avg_db']:5.1f}/{v['median_db']:5.1f}" for (s, k), v in sorted(res.items()))
                print(f"{source}:{name:22s} {kind:15s} Hs {desc['hs_m']:4.1f} m  {n_data:2d} sym  (power avg/median) "
                      f"{line}", flush=True)
            with open(os.path.join(out_dir, "points.json"), "w", encoding="utf-8") as fh:
                json.dump(dict(n_packets=n, rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
