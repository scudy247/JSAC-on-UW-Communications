"""
C2 sweeps (PLAN: Phase 3): one factor at a time around the default point, to see where each strategy wins.

Default point: IMU drift 0.01 m/s (Gauss-Markov, 20 s), latency 50 ms extrapolated with the measured acceleration,
lever arm 2 m (OOI, gyro-corrected), 1 s packets, range 1 km, receiver estimate error 3e-6, 10 packets per point.
Sweeps:
  imu      IMU drift 0.005 ... 0.1 m/s          buoy -> seabed (CDIP swell, OOI swell), buoy -> buoy (OOI)
  latency  0 ... 200 ms, with / without extrapolation          buoy -> seabed (OOI)
  lever    0 ... 5 m, corrected / uncorrected                   buoy -> seabed (OOI)
  packet   0.5 ... 8 s                                          buoy -> buoy (OOI)
  range    0.5 ... 4 km, stale estimate = previous round trip   buoy -> buoy (OOI)
  sea      all downloaded days (9 CDIP, 6 OOI), 1 and 4 s        buoy -> seabed
Output: results/c2_sweeps/<sweep>.json (one row per point and strategy), printed progress.
Usage: python scripts/c2_sweeps.py [sweep ...]   (default: all). Single thread, low priority, in the background.
"""
import glob
import json
import os
import sys
from dataclasses import replace

import numpy as np

from uwofdm import motion
from uwofdm.experiment import LinkConfig, run_link
from uwofdm.paths import data_dir, output_dir

DEFAULT = LinkConfig(packet_s=1.0, n_packets=10, sigmas_imu=(0.01,))
OOI_DAY = "GI01SUMO/20230919"
CDIP_DAY = "029p1_swell_2022-01-27"
SEABED = (1000.0, 0.0, -100.0)


def seabed_link(source, name, lever=None, segment=3):
    tx = motion.buoy_track(source, name, (0.0, 0.0, 0.0), segment=segment, lever=lever)
    return tx, motion.static_track(tx.fs, tx.n, SEABED)


def buoy_buoy_link(range_m=1000.0, lever=(0.0, 0.0, 2.0)):
    a = motion.buoy_track("ooi", OOI_DAY, (0.0, 0.0, -10.0), segment=3, lever=lever)
    b = motion.buoy_track("ooi", OOI_DAY, (range_m, 0.0, -10.0), segment=10, lever=lever)
    n = min(a.n, b.n)
    for tr in (a, b):
        tr.vel_fast, tr.vel_slow = tr.vel_fast[:n], tr.vel_slow[:n]
        tr.vel_lever = None if tr.vel_lever is None else tr.vel_lever[:n]
    return a, b


def rows_of(res, **point):
    return [dict(point, strategy=k, **v) for k, v in res.items()]


def sweep_imu(rng):
    sig = (0.005, 0.01, 0.02, 0.05, 0.1)
    cfg = replace(DEFAULT, sigmas_imu=sig, strategies=("tx_own", "both", "split", "split0"))
    out = []
    for label, link in (("seabed_cdip", lambda: seabed_link("cdip", CDIP_DAY)),
                        ("seabed_ooi", lambda: seabed_link("ooi", OOI_DAY, (0.0, 0.0, 2.0))),
                        ("buoy_buoy_ooi", buoy_buoy_link)):
        out += rows_of(run_link(*link(), cfg, rng), scenario=label)
        print("imu", label, flush=True)
    return out


def sweep_latency(rng):
    out = []
    tx, rx = seabed_link("ooi", OOI_DAY, (0.0, 0.0, 2.0))
    for lat, ext in [(0.0, True), (0.05, True), (0.1, True), (0.2, True), (0.05, False), (0.1, False)]:
        cfg = replace(DEFAULT, latency=lat, extrapolate=ext, strategies=("tx_own", "split"))
        out += rows_of(run_link(tx, rx, cfg, rng), latency_s=lat, extrapolate=ext)
        print("latency", lat, ext, flush=True)
    return out


def sweep_lever(rng):
    out = []
    for arm in (0.0, 0.5, 1.0, 2.0, 5.0):
        tx, rx = seabed_link("ooi", OOI_DAY, (0.0, 0.0, arm))
        for corr in (True, False):
            cfg = replace(DEFAULT, lever_corrected=corr, strategies=("tx_own", "split"))
            out += rows_of(run_link(tx, rx, cfg, rng), lever_m=arm, corrected=corr)
        print("lever", arm, flush=True)
    return out


def sweep_packet(rng):
    out = []
    a, b = buoy_buoy_link()
    for T in (0.5, 1.0, 2.0, 4.0, 8.0):
        cfg = replace(DEFAULT, packet_s=T, strategies=("rx_sym", "rx_pkt2", "both", "split", "split0"))
        out += rows_of(run_link(a, b, cfg, rng), packet_s=T)
        print("packet", T, flush=True)
    return out


def sweep_range(rng):
    out = []
    for r in (500.0, 1000.0, 2000.0, 4000.0):
        a, b = buoy_buoy_link(range_m=r)
        tau = np.hypot(r, 0.0) / motion.C
        cfg = replace(DEFAULT, stale_h=2 * tau + DEFAULT.packet_s, strategies=("rx_stale", "rx_sym", "both", "split"))
        out += rows_of(run_link(a, b, cfg, rng), range_m=r, stale_h=cfg.stale_h)
        print("range", r, flush=True)
    return out


def sweep_sea(rng):
    out = []
    days = [("cdip", os.path.basename(p)[:-4], None) for p in sorted(glob.glob(str(data_dir("buoy", "cdip", "*_*_20*.npz"))))]
    chosen = json.loads(data_dir("buoy", "ooi", "chosen_days.json").read_text())
    days += [("ooi", f"{m}/{v['day']}", (0.0, 0.0, 2.0)) for m, kinds in chosen.items() for v in kinds.values()]
    for source, name, lever in days:
        tx, rx = seabed_link(source, name, lever)
        for T in (1.0, 4.0):
            cfg = replace(DEFAULT, packet_s=T, strategies=("rx_sym", "rx_pkt2", "tx_own", "split"))
            out += rows_of(run_link(tx, rx, cfg, rng), day=f"{source}:{name}", packet_s=T)
        print("sea", name, flush=True)
    return out


SWEEPS = {"imu": sweep_imu, "latency": sweep_latency, "lever": sweep_lever, "packet": sweep_packet,
          "range": sweep_range, "sea": sweep_sea}


def main():
    todo = sys.argv[1:] or list(SWEEPS)
    out = output_dir("c2_sweeps")
    for name in todo:
        rng = np.random.default_rng(0)
        rows = SWEEPS[name](rng)
        with open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        print(f"== {name}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
