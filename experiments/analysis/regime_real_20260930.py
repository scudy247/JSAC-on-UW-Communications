"""Regime detector (estimation/regime.py) on real recordings: how often does it alarm, and when?

FK01 (SanctSound Florida Keys, 6 h continuous, 48 kHz) in the Blue band, and HI01 (Hawaii, 4 x 15 min
joined, 96 kHz) in the Red band. Streaming: FLAC blocks of 60 s -> causal band-pass -> detector blocks
of `DET_BLOCK_S`. Reports the alarm times and the level (log robust sigma) around each alarm. The
detector's defaults (warm-up 30 blocks, k = 0.5, h = 8) were set on synthetic data; this checks them
on real noise. Seeded? No randomness. 1 core.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import time

import numpy as np
import soundfile as sf

from uwsb.estimation.regime import RegimeDetector, block_statistics
from uwsb.noise.impulsiveness import BandpassStream
from uwsb.runtime import data_dir

DET_BLOCK_S = 0.5
SKIP_S = 5.0


def stream(path, band, det, levels, alarms, t_offset_s):
    with sf.SoundFile(str(path)) as f:
        fs = f.samplerate
        bp = BandpassStream(fs, band)
        nb = int(DET_BLOCK_S * fs)
        f.seek(int(SKIP_S * fs))
        t_s = t_offset_s
        carry = np.empty(0)
        first = True
        for blk in f.blocks(blocksize=int(60 * fs), dtype="float64"):
            y = bp(blk)
            if first:                       # drop the filter transient
                y, first = y[int(fs):], False
            y = np.concatenate([carry, y])
            n_full = y.size // nb
            for i in range(n_full):
                b = y[i * nb:(i + 1) * nb]
                levels.append((t_s, block_statistics(b)[0]))
                out = det.update(b)
                if out["alarm"]:
                    alarms.append(t_s)
                t_s += DET_BLOCK_S
            carry = y[n_full * nb:]
    return t_s


if __name__ == "__main__":
    d = data_dir() / "noise_census"
    for name, paths, band in [
        ("FK01 blue", [d / "S1_fk01" / "SanctSound_FK01_01_671359016_20181218T214011Z.flac"], (10_500.0, 15_500.0)),
        ("HI01 red", sorted((d / "S2_hi01").glob("*.flac")), (20_000.0, 30_000.0)),
    ]:
        t0 = time.perf_counter()
        det, levels, det_alarm_times = RegimeDetector(), [], []
        t_end = 0.0
        for p in paths:
            t_end = stream(p, band, det, levels, det_alarm_times, t_end)   # one detector across joined segments
        lv = np.array(levels)
        print(f"== {name}: {t_end / 60:.1f} min analysed, {lv.shape[0]} blocks of {DET_BLOCK_S} s, "
              f"{len(det_alarm_times)} alarms ({len(det_alarm_times) / (t_end / 3600):.1f} per hour), "
              f"{time.perf_counter() - t0:.0f} s")
        for ta in det_alarm_times[:12]:
            before = lv[(lv[:, 0] >= ta - 60) & (lv[:, 0] < ta), 1]
            after = lv[(lv[:, 0] >= ta) & (lv[:, 0] < ta + 60), 1]
            print(f"   alarm at {ta / 60:7.2f} min: level change {20 / np.log(10) * (after.mean() - before.mean()):+.1f} dB "
                  f"(1 min before vs after)")
        if len(det_alarm_times) > 12:
            print(f"   ... {len(det_alarm_times) - 12} more")
