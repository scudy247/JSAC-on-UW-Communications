"""
C2 robustness checks (PLAN: Phase 3), before deriving the placement rule:

multipath   Does path-specific Doppler set a floor that hides the differences between strategies? Three channel
            cases with a pilot-based receiver (experiment.pilot_powers): single path; multipath without path-specific
            Doppler; multipath where surface-bounced paths add 2 sin(grazing) w(t) / c (experiment.Multipath).
direction   How sensitive are the IMU-based strategies to the path direction the nodes use to project their own
            velocity? Azimuth error 0 ... 20 deg (single path, genie metric).
Scenarios: buoy -> seabed (CDIP 029 swell; OOI Irminger swell, 2 m lever arm corrected), buoy -> buoy (OOI); 1 and 4 s
packets (multipath), 1 s (direction); IMU 0.01 m/s, 50 ms latency extrapolated; 10 packets per point.
Output: results/c2_robustness/<check>.json; printed table. Usage: python scripts/c2_robustness.py [check ...].
"""
import json
import os
import sys
from dataclasses import replace

import numpy as np

from uwofdm.experiment import LinkConfig, Multipath, run_link
from uwofdm.paths import output_dir
from c2_sweeps import CDIP_DAY, OOI_DAY, buoy_buoy_link, seabed_link

DEFAULT = LinkConfig(packet_s=1.0, n_packets=10, sigmas_imu=(0.01,))
SCENARIOS = (("seabed_cdip", lambda: seabed_link("cdip", CDIP_DAY)),
             ("seabed_ooi", lambda: seabed_link("ooi", OOI_DAY, (0.0, 0.0, 2.0))),
             ("buoy_buoy_ooi", buoy_buoy_link))


def show(rows):
    for r in rows:
        point = {k: v for k, v in r.items() if k not in ("strategy", "sir_sim_db", "sir_sim_p10_db", "sir_theory_db")}
        print(f"{json.dumps(point):70s} {r['strategy']:12s} sim {r['sir_sim_db']:6.1f} dB (p10 {r['sir_sim_p10_db']:6.1f})"
              f"  theory {r['sir_theory_db']:6.1f}", flush=True)


def check_multipath(rng):
    out = []
    cases = (("single", None), ("multi_no_path_doppler", Multipath(path_doppler=False)), ("multi_path_doppler", Multipath()))
    for label, link in SCENARIOS:
        tx, rx = link()
        for T in (1.0, 4.0):
            for case, mp in cases:
                cfg = replace(DEFAULT, packet_s=T, multipath=mp, metric="pilot",
                              strategies=("rx_sym", "rx_pkt2", "tx_own", "both", "split"))
                res = run_link(tx, rx, cfg, rng)
                rows = [dict(scenario=label, packet_s=T, channel=case, strategy=k, **v) for k, v in res.items()]
                show(rows)
                out += rows
    return out


def check_direction(rng):
    out = []
    for label, link in SCENARIOS:
        tx, rx = link()
        for deg in (0.0, 2.0, 5.0, 10.0, 20.0):
            cfg = replace(DEFAULT, u_error_deg=deg, strategies=("tx_own", "both", "split"))
            res = run_link(tx, rx, cfg, rng)
            rows = [dict(scenario=label, u_error_deg=deg, strategy=k, **v) for k, v in res.items()]
            show(rows)
            out += rows
    return out


CHECKS = {"multipath": check_multipath, "direction": check_direction}


def main():
    todo = sys.argv[1:] or list(CHECKS)
    out = output_dir("c2_robustness")
    for name in todo:
        rows = CHECKS[name](np.random.default_rng(0))
        with open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        print(f"== {name}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
