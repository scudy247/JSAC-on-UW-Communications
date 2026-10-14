"""
E3 reframing (PLAN: Phase 3): does the placement evaluator, with the fair receiver stages (Kalman smoother, partial
FFT) and realistic sensors (imu.RealisticIMU, industrial grade), predict the fair end-to-end comparison
(results/b1_baselines/compare.json, 30 packets per case)? Receivers -> evaluator placements:
  signal_kalman          own[none]+rx[smooth]           pfft8_kalman           own[none]+rx[smooth+pfft8]
  safe_industrial        own[tx(+rx)]+rx[smooth_s]      safe_industrial_pfft8  own[tx(+rx)]+rx[smooth_s+pfft8]
(the seabed receiver is static: no IMU). Per-symbol estimate noise as the chain measures it: 0.5 Hz / sqrt(snr)
plus 0.01 Hz, over fc. Predicted SINR = 1 / (1/snr + 1/SIR). Output: results/b16_e3/validation.json.
"""
import json
import os

import numpy as np

from uwofdm import imu, placement as P
from uwofdm.paths import output_dir, results_dir
from e2_chain import SCENARIOS
from b2_baselines import CASES

MAP = {"signal_kalman": (False, "smooth"), "pfft8_kalman": (False, "smooth+pfft8"),
       "safe_industrial": (True, "smooth_s"), "safe_industrial_pfft8": (True, "smooth_s+pfft8")}


def main():
    sim = json.load(open(os.path.join(results_dir(), "b1_baselines", "compare.json")))["rows"]
    rows, err = [], []
    for label, n_data in CASES:
        tx, rx = SCENARIOS[label]()
        static = np.allclose(rx.vel_fast, 0)
        sen_tx = P.NodeSensors(grade=imu.GRADES["industrial"], lever_m=2.0 if tx.vel_lever is not None else None, seed=1)
        sen_rx = None if static else P.NodeSensors(grade=imu.GRADES["industrial"], lever_m=2.0, seed=2)
        fs, comps = P.link_components(tx, rx, sen_tx, sen_rx)
        T = n_data * 0.15
        for snr in (20, 30):
            sigma = (0.5 / np.sqrt(10 ** (snr / 10)) + 0.01) / 16e3
            ev = {(r["tx_imu"], r["rx_imu"], r["stage"]): r["sir_db"]
                  for r in P.evaluate(fs, comps, P.ReceiverModel(packet_s=T, sigma=sigma, stages=P.STAGES_FAIR))}
            line = []
            for rec, (sensors, stage) in MAP.items():
                sir = ev[(sensors, sensors and not static, stage)]
                pred = float(-10 * np.log10(10 ** (-snr / 10) + 10 ** (-sir / 10)))
                s = next(r for r in sim if r["scenario"] == label and r["n_data"] == n_data and r["snr_db"] == snr
                         and r["receiver"] == rec)
                rows.append(dict(scenario=label, n_data=n_data, snr_db=snr, receiver=rec, sir_pred_db=sir,
                                 sinr_pred_db=pred, sinr_sim_power_avg_db=s["power_avg_db"],
                                 sinr_sim_median_db=s["median_db"], sinr_sim_p10_db=s["p10_db"]))
                err.append((pred - s["median_db"], pred - s["power_avg_db"]))
                line.append(f"{rec} pred {pred:4.1f} (SIR {sir:4.1f}) sim {s['median_db']:4.1f}/{s['power_avg_db']:4.1f}")
            print(f"{label:28s} {n_data:2d} sym {snr} dB: " + " | ".join(line), flush=True)
    e = np.array(err)
    print(f"prediction - simulated median: median {np.median(e[:, 0]):+.2f} dB, |.| max {np.max(np.abs(e[:, 0])):.2f}; "
          f"- power average: median {np.median(e[:, 1]):+.2f}, |.| max {np.max(np.abs(e[:, 1])):.2f} dB")
    with open(os.path.join(output_dir("b16_e3"), "validation.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
