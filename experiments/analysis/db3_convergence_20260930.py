"""Evidence for D-B3: convergence of the two recursive alpha trackers (THEORY-B §2, T4).

RMSE of alpha-hat vs number of noise samples n for LogMomentTracker and QuantileTracker
(blocks of 1000, no forgetting), 50 repetitions, on (i) i.i.d. SaS alpha = 1.5 and (ii)
alphaSGN(4) with the Mahmood-Chitre D1 parameters (alpha = 1.715, memory). Also the cost per
sample of each tracker. Seeded; 1 core.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import time

import numpy as np

from uwsb.estimation.noise_state import LogMomentTracker, QuantileTracker
from uwsb.noise.asgn import asgn_m
from uwsb.noise.stable import sas_real

CHECKS = (1_000, 3_000, 10_000, 30_000, 100_000)
REPS, BLOCK = 50, 1_000
SOURCES = {
    "iid SaS a=1.5": (1.5, lambda r: sas_real(1.5, 1.0, CHECKS[-1], r)),
    "aSGN(4) a=1.715": (1.715, lambda r: asgn_m(1.715, 1.0, [1.0, 0.621, 0.237, 0.161, -0.054],
                                                CHECKS[-1], r)),
}


def run():
    print(f"{'source':17s} {'tracker':11s} " + " ".join(f"n={n:>7d}" for n in CHECKS))
    for name, (alpha, gen) in SOURCES.items():
        est = {"log_moment": np.zeros((REPS, len(CHECKS))), "quantile": np.zeros((REPS, len(CHECKS)))}
        for rep in range(REPS):
            x = gen(np.random.default_rng(1000 + rep))
            lm, qt = LogMomentTracker(), QuantileTracker()
            j = 0
            for i in range(0, x.size, BLOCK):
                lm.update(x[i:i + BLOCK])
                qt.update(x[i:i + BLOCK])
                if i + BLOCK == CHECKS[j]:
                    est["log_moment"][rep, j] = lm.estimate()[0]
                    est["quantile"][rep, j] = qt.estimate()[0]
                    j += 1
        for tr, e in est.items():
            rmse = np.sqrt(np.mean((e - alpha) ** 2, axis=0))
            print(f"{name:17s} {tr:11s} " + " ".join(f"{v:9.4f}" for v in rmse))
    x = sas_real(1.5, 1.0, 1_000_000, np.random.default_rng(1))
    for tr in (LogMomentTracker(), QuantileTracker()):
        t0 = time.perf_counter()
        for i in range(0, x.size, BLOCK):
            tr.update(x[i:i + BLOCK])
        dt = time.perf_counter() - t0
        print(f"cost {type(tr).__name__}: {dt / x.size * 1e9:.1f} ns per sample (blocks of {BLOCK})")


if __name__ == "__main__":
    run()
