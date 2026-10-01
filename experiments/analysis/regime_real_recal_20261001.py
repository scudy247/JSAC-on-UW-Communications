"""Recalibrated regime detector (estimation/regime.py, 2026-10-01) on real recordings, with an ablation.

Same streams as regime_real_20260930.py (FK01 Blue band 6 h; HI01 Red band 4 x 15 min joined; blocks of
0.5 s): the block statistics are computed once, then each detector setting runs on them
(RegimeDetector.update_stats). Reports alarms per hour and the level change at each alarm (mean log
robust sigma 1 min after vs 1 min before, in dB), plus a rough count of real level changes: consecutive
1-min means differing by >= 2 dB. No randomness. 1 core.
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
SETTINGS = {
    "original (warmup 30)": dict(),
    "warmup 120 only": dict(warmup=120),
    "+ long-run sd": dict(warmup=120, long_run=True),
    "+ min shift 2 dB / 0.2": dict(warmup=120, min_shift_db=2.0, min_shift_impuls=0.2),
    "REAL_NOISE (all three)": RegimeDetector.REAL_NOISE,
    "REAL_NOISE, min shift 1 dB / 0.1": {**RegimeDetector.REAL_NOISE, "min_shift_db": 1.0, "min_shift_impuls": 0.1},
    "REAL_NOISE, min shift 3 dB / 0.3": {**RegimeDetector.REAL_NOISE, "min_shift_db": 3.0, "min_shift_impuls": 0.3},
}


def stream_stats(paths, band):
    """Block statistics over the joined files (same blocking as regime_real_20260930.py)."""
    stats = []
    for path in paths:
        with sf.SoundFile(str(path)) as f:
            fs = f.samplerate
            bp = BandpassStream(fs, band)
            nb = int(DET_BLOCK_S * fs)
            f.seek(int(SKIP_S * fs))
            carry, first = np.empty(0), True
            for blk in f.blocks(blocksize=int(60 * fs), dtype="float64"):
                y = bp(blk)
                if first:
                    y, first = y[int(fs):], False
                y = np.concatenate([carry, y])
                n_full = y.size // nb
                for i in range(n_full):
                    stats.append(block_statistics(y[i * nb:(i + 1) * nb]))
                carry = y[n_full * nb:]
    return np.array(stats)


if __name__ == "__main__":
    d = data_dir() / "noise_census"
    to_db = 20 / np.log(10)
    for name, paths, band in [
        ("FK01 blue", [d / "S1_fk01" / "SanctSound_FK01_01_671359016_20181218T214011Z.flac"], (10_500.0, 15_500.0)),
        ("HI01 red", sorted((d / "S2_hi01").glob("*.flac")), (20_000.0, 30_000.0)),
    ]:
        t0 = time.perf_counter()
        S = stream_stats(paths, band)
        hours = S.shape[0] * DET_BLOCK_S / 3600
        lev = S[:, 0]
        per_min = int(60 / DET_BLOCK_S)
        m = lev[: lev.size // per_min * per_min].reshape(-1, per_min).mean(axis=1) * to_db
        real = int(np.sum(np.abs(np.diff(m)) >= 2.0))
        r1 = [float(np.corrcoef(S[:-1, j], S[1:, j])[0, 1]) for j in range(2)]
        print(f"== {name}: {hours * 60:.1f} min, {S.shape[0]} blocks ({time.perf_counter() - t0:.0f} s); "
              f"lag-1 autocorrelation level {r1[0]:.2f}, impulsiveness {r1[1]:.2f}; "
              f"consecutive 1-min level changes >= 2 dB: {real}")
        for label, kw in SETTINGS.items():
            det = RegimeDetector(**kw)
            for s in S:
                det.update_stats(s)
            ch, ci = [], []
            for a in det.alarms:
                sl_b, sl_a = slice(max(a - per_min, 0), a), slice(a, a + per_min)
                if a - per_min >= 0 and a + per_min <= S.shape[0]:
                    ch.append((S[sl_a, 0].mean() - S[sl_b, 0].mean()) * to_db)
                    ci.append(S[sl_a, 1].mean() - S[sl_b, 1].mean())
            ch, ci = np.abs(np.array(ch)), np.abs(np.array(ci))
            kinds = {k: det.alarm_stat.count(k) for k in sorted(set(det.alarm_stat))}
            txt = (f"|level change| median {np.median(ch):4.1f} dB (>= 1 dB: {np.mean(ch >= 1) * 100:3.0f} %), "
                   f"|impulsiveness change| median {np.median(ci):.2f}" if ch.size else "no alarms")
            print(f"   {label:34s} {len(det.alarms):4d} alarms ({len(det.alarms) / hours:5.1f}/h) {kinds}; {txt}")
        sd_i = float(np.std(S[:, 1]))
        print(f"   (impulsiveness statistic: overall sd {sd_i:.2f}; a change of 0.2 is alpha 1.8 -> 1.5 for SaS)")
