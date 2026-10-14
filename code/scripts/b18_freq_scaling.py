"""
Frequency scaling of the E3 rule (PLAN: Phase 3; candidate E of 14 Oct 2026). Interference from a Doppler residual
scales as (fc eps T)^2, while the per-symbol estimate noise as a Doppler scale shrinks as 1 / fc (null-subcarrier
offset error 0.5 Hz / sqrt(snr) + 0.01 Hz). The validated evaluator (placement.py, fair stages, industrial MEMS) is run
at fc = 8, 16, 32, 64 kHz with the same symbol timing (128 ms FFT, 22 ms prefix), 30 dB per-subcarrier SNR, on the
four cases of the fair comparison; SIR per receiver cost level, signal-only vs with sensors.
Output: results/b18_freq_scaling/points.json.
"""
import json
import os

import numpy as np

from uwofdm import imu, placement as P
from uwofdm.paths import output_dir
from e2_chain import SCENARIOS
from b2_baselines import CASES

FCS = (8e3, 16e3, 32e3, 64e3)
STAGES = ("line2", "sym", "smooth", "smooth+pfft8", "smooth_s", "smooth_s+pfft8")
SNR = 30.0


def main():
    rows = []
    for label, n_data in CASES:
        tx, rx = SCENARIOS[label]()
        static = np.allclose(rx.vel_fast, 0)
        st = P.NodeSensors(grade=imu.GRADES["industrial"], lever_m=2.0 if tx.vel_lever is not None else None, seed=1)
        sr = None if static else P.NodeSensors(grade=imu.GRADES["industrial"], lever_m=2.0, seed=2)
        fs, comps = P.link_components(tx, rx, st, sr)
        for fc in FCS:
            sigma = (0.5 / np.sqrt(10 ** (SNR / 10)) + 0.01) / fc
            rxm = P.ReceiverModel(packet_s=n_data * 0.15, fc=fc, sigma=sigma, stages=STAGES)
            P._ICI.clear()
            ev = {(r["tx_imu"], r["stage"]): r["sir_db"] for r in P.evaluate(fs, comps, rxm)
                  if r["rx_imu"] == (r["tx_imu"] and not static)}
            get = lambda sen, st_: ev.get((sen, st_), np.nan)
            row = dict(scenario=label, n_data=n_data, fc_hz=fc,
                       signal={s: get(False, s) for s in ("line2", "sym", "smooth", "smooth+pfft8")},
                       sensor={"line2": get(True, "line2"), "sym": get(True, "sym"), "smooth": get(True, "smooth_s"),
                               "smooth+pfft8": get(True, "smooth_s+pfft8")})
            rows.append(row)
            print(f"{label:28s} {n_data:2d} sym fc {fc / 1e3:4.0f} kHz | SIR signal / sensors: " + "  ".join(
                f"{k} {row['signal'][k]:5.1f}/{row['sensor'][k]:5.1f}" for k in ("line2", "sym", "smooth", "smooth+pfft8")),
                flush=True)
    with open(os.path.join(output_dir("b18_freq_scaling"), "points.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
