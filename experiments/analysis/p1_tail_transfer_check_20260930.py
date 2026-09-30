"""Numerical check of THEORY-B P1(c): if |n|^2 is regularly varying with index kappa, the
drop of gamma-hat in dB below its median has an exponential tail whose mean excess tends
to 10 / (kappa ln 10) dB, for any packet length N.

Noise per symbol: n = sqrt(A) * CN(0,1), A ~ Pareto(kappa, x_m = 1), so |n|^2 = A * Exp(1)
has tail index exactly kappa (Breiman). gamma_lin = 1 / mean_i |n_i|^2 (unit pilots).
Reuses the mean-excess function and its known-case checks from gamma_tail_mc_20260930.py.
"""
import importlib.util
import os
import sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mc", HERE / "gamma_tail_mc_20260930.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

KAPPAS = (0.6, 0.85, 1.2, 1.5, 2.0)
NS = (64, 256, 1024)
N_PKT = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
CHUNK = 5_000
rng = np.random.default_rng(20260930)


def packet_noise_power(kappa, n_sym, n_pkt):
    out = np.empty(n_pkt)
    for a in range(0, n_pkt, CHUNK):
        m = min(CHUNK, n_pkt - a)
        amp = (1.0 - rng.random((m, n_sym))) ** (-1.0 / kappa)      # Pareto(kappa), x_m = 1
        pw = amp * rng.exponential(1.0, (m, n_sym))                  # |sqrt(A) CN(0,1)|^2
        out[a:a + m] = pw.mean(axis=1)
    return out


if __name__ == "__main__":
    mc.known_case_checks()
    print(f"{'kappa':>5} {'N':>5} {'pred dB':>8} {'e(q99) dB':>10} {'e(q99.9) dB':>12} "
          f"{'e(q99.99) dB':>12} {'ratio q99.99':>12} {'n > q99.99':>10}")
    for kappa in KAPPAS:
        pred = 10.0 / (kappa * np.log(10.0))
        for n_sym in NS:
            pn = packet_noise_power(kappa, n_sym, N_PKT)
            g_db = -10.0 * np.log10(pn)
            drop = np.median(g_db) - g_db
            (u1, e1), (u2, e2), (u3, e3) = mc.mean_excess(drop, qs=(0.99, 0.999, 0.9999))
            n3 = int(np.sum(drop > u3))
            print(f"{kappa:5.2f} {n_sym:5d} {pred:8.2f} {e1:10.2f} {e2:12.2f} "
                  f"{e3:12.2f} {e3 / pred:12.2f} {n3:10d}")
