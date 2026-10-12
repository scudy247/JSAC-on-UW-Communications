"""
Pilot-tracking baseline on real motion (PLAN: Phase 5, fair baselines; chain.receive_pilot_track, closed loop).
Same cases, seeds, packets and noise as scripts/b2_baselines.py (results/b1_baselines/compare.json), so the numbers
line up with that table; only the pilot-tracking receivers (with and without partial FFT, 8 segments) are run.
Usage: python scripts/b5_pilot_track.py [n_packets]. Output: results/b1_baselines/pilot_track.json.
"""
import json
import os
import sys

from uwofdm.paths import output_dir
from b2_baselines import CASES, SNRS_DB, run_link
from e2_chain import SCENARIOS


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rows = []
    for i, (label, n_data) in enumerate(CASES):
        res = run_link(*SCENARIOS[label](), n_data, n, 700 + i, None, grades=(), n_seg=(8,), pilot_track=True,
                       only_new=True)
        for (snr, k), v in sorted(res.items()):
            rows.append(dict(scenario=label, n_data=n_data, snr_db=snr, receiver=k, **v))
            print(f"{label:28s} {n_data:2d} sym SNR {snr} {k:22s} (power avg / median / p10) {v['power_avg_db']:5.1f} / "
                  f"{v['median_db']:5.1f} / {v['p10_db']:5.1f}", flush=True)
        with open(os.path.join(output_dir("b1_baselines"), "pilot_track.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(n_packets=n, rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
