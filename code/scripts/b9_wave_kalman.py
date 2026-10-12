"""
Wave-state Kalman model on real motion (PLAN: Phase 5, fusion.WaveState). Same cases, seeds, packets and noise as
scripts/b2_baselines.py (CFO search +-4), so the rows line up with results/b1_baselines/compare.json; only the
wave-model receivers are run: signal-only Kalman with the wave model (+ partial FFT, 8 segments) and the safe sensor
fusion with the wave model (sensor residual parameters; its signal-only guard also uses the wave model). Wave
parameters (peak, damping, std) from the whole record: the receiver knows the sea state from past packets.
Usage: python scripts/b9_wave_kalman.py [n_packets]. Output: results/b1_baselines/wave_kalman.json.
"""
import json
import os
import sys

from uwofdm.paths import output_dir
from b2_baselines import CASES, run_link
from e2_chain import SCENARIOS


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rows = []
    for i, (label, n_data) in enumerate(CASES):
        res = run_link(*SCENARIOS[label](), n_data, n, 700 + i, None, grades=("old", "industrial"), n_seg=(8,),
                       only_new=True, wave=True)
        for (snr, k), v in sorted(res.items()):
            rows.append(dict(scenario=label, n_data=n_data, snr_db=snr, receiver=k, **v))
            print(f"{label:28s} {n_data:2d} sym SNR {snr} {k:22s} (power avg / median / p10) {v['power_avg_db']:5.1f} / "
                  f"{v['median_db']:5.1f} / {v['p10_db']:5.1f}", flush=True)
        with open(os.path.join(output_dir("b1_baselines"), "wave_kalman.json"), "w", encoding="utf-8") as fh:
            json.dump(dict(n_packets=n, rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
