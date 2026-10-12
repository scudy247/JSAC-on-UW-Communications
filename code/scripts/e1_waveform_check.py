"""
E1 (PLAN: Phase 4): waveform check of the side-information results.

The time-domain scoring of scripts/e1_stage_alloc.py is checked with the OFDM waveform harness: side information
(si<B>+own[..]+rx[..], experiment.py) with 8, 16, 32, 64 bits, against the receiver alone and the transmitter applying
its estimate, for the three receiver stages; three scenarios, 1 and 4 s packets, 30 packets per point, genie metric.
Output: results/e1_sideinfo/waveform_check.json; printed table with the time-domain value next to each simulated one.
"""
import json
import os
from dataclasses import replace

import numpy as np

from uwofdm.experiment import LinkConfig, run_link
from uwofdm.paths import output_dir
from c2_sweeps import CDIP_DAY, OOI_DAY, buoy_buoy_link, seabed_link

BITS = (8, 16, 32, 64)
SCENARIOS = {"buoy -> seabed, CDIP swell": (lambda: seabed_link("cdip", CDIP_DAY), "none"),
             "buoy -> seabed, OOI Irminger": (lambda: seabed_link("ooi", OOI_DAY, (0.0, 0.0, 2.0)), "none"),
             "buoy - buoy, OOI Irminger": (buoy_buoy_link, "rx")}


def main():
    out = output_dir("e1_sideinfo")
    td = json.load(open(os.path.join(out, "stage_allocation.json")))
    rows = []
    for label, (make, own_rx) in SCENARIOS.items():
        tx, rx = make()
        for T in (1.0, 4.0):
            for stage in ("none", "line2", "sym"):
                rx_part = "rx" if own_rx == "rx" else "none"
                both = "tx+rx" if own_rx == "rx" else "tx"
                strategies = [f"own[{rx_part}]+rx[{stage}]", f"own[{both}]+rx[{stage}]"] + \
                             [f"si{b}+own[{rx_part}]+rx[{stage}]" for b in BITS]
                cfg = LinkConfig(packet_s=T, n_packets=30, sigmas_imu=(0.01,), strategies=tuple(strategies))
                res = run_link(tx, rx, cfg, np.random.default_rng(0))
                ref = {r.get("reference", r.get("bits")): r["sir_db"] for r in td
                       if r["scenario"] == label and r["packet_s"] == T and r["stage"] == stage}
                for k, v in res.items():
                    name = k.rsplit("_", 1)[0]
                    key = ("rx_alone" if name.startswith(f"own[{rx_part}]") and not name.startswith("si")
                           else "applied" if name.startswith(f"own[{both}]") else int(name[2:name.index("+")]))
                    rows.append(dict(scenario=label, packet_s=T, stage=stage, strategy=name, key=key,
                                     sir_sim_db=v["sir_sim_db"], sir_time_domain_db=ref.get(key)))
                    print(f"{label:30s} T {T:3.0f} s {stage:5s} {str(key):9s} waveform {v['sir_sim_db']:5.1f}  "
                          f"time-domain {ref.get(key, float('nan')):5.1f}", flush=True)
            with open(os.path.join(out, "waveform_check.json"), "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=1)
    d = np.array([r["sir_sim_db"] - r["sir_time_domain_db"] for r in rows if r["sir_time_domain_db"] is not None])
    print(f"\nwaveform - time-domain: median {np.median(d):.1f} dB, |.| p90 {np.percentile(abs(d), 90):.1f} dB, "
          f"max {np.max(abs(d)):.1f} dB (n={len(d)})")


if __name__ == "__main__":
    main()
