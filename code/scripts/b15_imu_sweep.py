"""
Motion-sensor sensitivity (PLAN: Phase 1): wave-band (> 0.04 Hz) velocity error of imu.RealisticIMU against the
attitude (tilt) error and the crossover of the complementary filter with GNSS velocity, industrial grade otherwise,
on real buoy motion (OOI Irminger with 2 m lever arm, CDIP swell). Tilt leaks gravity into the horizontal axes; a
higher crossover hands more of the band to GNSS (its noise grows, the tilt leakage shrinks).
Output: results/b15_imu_sweep/points.json and a printed table.
"""
import json
import os
from dataclasses import replace

import numpy as np
from scipy.signal import butter, sosfiltfilt

from uwofdm import imu, motion
from uwofdm.paths import output_dir
from c2_sweeps import CDIP_DAY, OOI_DAY

TILT_DEG = (0.01, 0.05, 0.1, 0.2, 0.5, 1.0)
CROSS_HZ = (0.01, 0.03, 0.1, 0.3)


def main():
    rows = []
    for label, tr, arm in (("OOI Irminger", motion.buoy_track("ooi", OOI_DAY, (0, 0, 0), segment=3, lever=(0, 0, 2.0)), 2.0),
                           ("CDIP swell", motion.buoy_track("cdip", CDIP_DAY, (0, 0, 0)), None)):
        tt = tr.t[int(200 * tr.fs):-int(60 * tr.fs)]
        true = tr.velocity_at(tt, "total")
        sos = butter(4, 0.04, "highpass", fs=tr.fs, output="sos")
        for tilt in TILT_DEG:
            line = []
            for fc in CROSS_HZ:
                g = replace(imu.GRADES["industrial"], tilt_std=tilt * imu.DEG, hp_hz=fc)
                e = imu.RealisticIMU(tr, g, np.random.default_rng(0), lever_m=arm).velocity(tt) - true
                eb = sosfiltfilt(sos, e, axis=0)
                h = float(np.sqrt(np.mean(eb[:, 0] ** 2 + eb[:, 1] ** 2) / 2))
                rows.append(dict(record=label, tilt_deg=tilt, crossover_hz=fc, wave_band_h_err=h,
                                 wave_band_z_err=float(np.std(eb[:, 2])), total_h_err=float(np.sqrt(np.mean(e[:, :2] ** 2)))))
                line.append(f"{fc:.2f} Hz {h:.4f}")
            print(f"{label:12s} tilt {tilt:4.2f} deg: wave-band horizontal error [m/s] by crossover: " + "  ".join(line), flush=True)
    with open(os.path.join(output_dir("b15_imu_sweep"), "points.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
