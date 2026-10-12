"""
New placements suggested by the evaluator (PLAN: Phase 3, placement rule), simulated beside split and the receiver.

The evaluator (placement.py) ranks above split: the node IMU stage(s) followed by a receiver stage that uses one
estimate per symbol (per symbol, or a quadratic over all of them), and, on a seabed link, no IMU stage at the static
receiver. Here they are simulated with the harness (generic names own[...]+rx[...], experiment.py).
Scenarios: buoy -> seabed (CDIP 029 swell; OOI Irminger swell, 2 m lever arm), buoy -> buoy (OOI); packets 1 and 4 s;
path-direction error 0, 5, 20 deg; IMU 0.01 m/s, 50 ms latency extrapolated; 30 packets per point (10 leave +-2-3 dB
between strategies that differ only by estimate noise).
Output: results/c2_new_placements/points.json; printed table.
"""
import json
import os
from dataclasses import replace

import numpy as np

from uwofdm.experiment import LinkConfig, run_link
from uwofdm.paths import output_dir
from c2_sweeps import CDIP_DAY, OOI_DAY, buoy_buoy_link, seabed_link

BASE = LinkConfig(packet_s=1.0, n_packets=30, sigmas_imu=(0.01,))
COMMON = ("rx_sym", "rx_pkt2", "split", "own[tx+rx]+rx[sym]", "own[tx+rx]+rx[quad]")
SEABED = ("own[tx]+rx[line2]", "own[tx]+rx[sym]", "own[tx]+rx[quad]")  # static receiver: no IMU stage there
SCENARIOS = (("seabed_cdip", lambda: seabed_link("cdip", CDIP_DAY), COMMON + SEABED),
             ("seabed_ooi", lambda: seabed_link("ooi", OOI_DAY, (0.0, 0.0, 2.0)), COMMON + SEABED),
             ("buoy_buoy_ooi", buoy_buoy_link, COMMON))


def main():
    out, rows = output_dir("c2_new_placements"), []
    for label, link, strategies in SCENARIOS:
        tx, rx = link()
        for T in (1.0, 4.0):
            for deg in (0.0, 5.0, 20.0):
                cfg = replace(BASE, packet_s=T, u_error_deg=deg, strategies=strategies)
                res = run_link(tx, rx, cfg, np.random.default_rng(0))
                for k, v in res.items():
                    rows.append(dict(scenario=label, packet_s=T, u_error_deg=deg, strategy=k, **v))
                    print(f"{label:14s} T {T:3.0f} s  {deg:4.0f} deg  {k:28s} sim {v['sir_sim_db']:5.1f} "
                          f"(p10 {v['sir_sim_p10_db']:5.1f})  theory {v['sir_theory_db']:5.1f}", flush=True)
        with open(os.path.join(out, "points.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
