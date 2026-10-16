"""
Best-numerology comparison (PLAN: Phase 5, fair baselines). A signal-only designer facing rough seas would shorten
the OFDM symbols (fewer, wider subcarriers): less Doppler change within a symbol, less ICI, but more cyclic-prefix
overhead, since the prefix (22 ms) is set by the channel's delay spread, not by the Doppler. Does a signal-only
receiver on its best numerology close the gap to sensor fusion, at equal throughput?

Numerologies: 1024 subcarriers (7.8 Hz, 128 ms, the default), 512 (15.6 Hz, 64 ms), 256 (31.2 Hz, 32 ms); same
8 kHz band, same 22 ms prefix, guard band fixed in Hz (n_sc / 64 subcarriers each side), pilots every 4th and nulls
every 16th subcarrier (same fractions). Packet duration fixed: data part ~1.05 s or ~4.05 s (n_data = round(D /
symbol time)). Receivers (scripts/b2_baselines.run_link): Li/Lee, tuned signal Kalman, signal Kalman + partial FFT
(8 segments), safe sensor fusion (GNSS-aided industrial MEMS), the same + partial FFT. Links: buoy-buoy OOI,
buoy -> seabed OOI and CDIP; 20 and 30 dB per-subcarrier SNR (same per-subcarrier SNR = same transmit power per Hz).
Metrics: genie SINR per packet (power average, median, 10th percentile) and the equal-throughput proxy
    rate = eta * log2(1 + SINR)  [bit/s/Hz on data],  eta = T_fft / (T_fft + T_cp)   (prefix efficiency),
per packet (median, 10th percentile). Limitation: single path; with the Bellhop delay spreads (~11 ms) the 256-
subcarrier numerology would also need denser pilots (125 Hz pilot spacing aliases delays beyond 8 ms), which is not
charged here (favours short symbols).
Usage: python scripts/b4_numerology.py [n_packets] [out_name]. Output: results/b1_baselines/numerology.json.
"""
import json
import os
import sys

import numpy as np

from uwofdm.waveform import OFDM
from uwofdm.paths import output_dir
from b2_baselines import run_link
from e2_chain import SCENARIOS

CASES = (("buoy - buoy, OOI Irminger", 1.05), ("buoy - buoy, OOI Irminger", 4.05),
         ("buoy -> seabed, OOI Irminger", 4.05), ("buoy -> seabed, CDIP swell", 1.05))
N_SC = (1024, 512, 256)
SNRS_DB = (20, 30)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out_dir = output_dir("b1_baselines")
    rows = []
    for i, (label, dur) in enumerate(CASES):
        tx, rx = SCENARIOS[label]()
        for nsc in N_SC:
            o = OFDM(n_sc=nsc, guard_sc=nsc // 64)
            t_sym = o.n_symbol / o.fs
            n_data = max(1, round(dur / t_sym))
            eta = (o.n_fft / o.fs) / t_sym
            res = run_link(tx, rx, n_data, n, 1100 + 10 * i, None, grades=("industrial",), n_seg=(8,), snrs=SNRS_DB,
                           ofdm=o)
            for (snr, k), v in sorted(res.items()):
                rate = eta * np.log2(1 + 10 ** (np.array(v["per_packet_db"]) / 10))
                rows.append(dict(scenario=label, duration_s=dur, n_sc=nsc, df_hz=o.df, n_data=n_data, eta=eta,
                                 snr_db=snr, receiver=k, rate_median=float(np.median(rate)),
                                 rate_p10=float(np.percentile(rate, 10)), **v))
            for snr in SNRS_DB:
                line = "  ".join(f"{k} {v['median_db']:4.1f}/{v['p10_db']:4.1f}dB "
                                 f"{eta * np.log2(1 + 10 ** (v['median_db'] / 10)):.2f}"
                                 for (s, k), v in sorted(res.items()) if s == snr)
                print(f"{label:28s} {dur:4.2f} s  {nsc:4d} sc ({n_data:2d} sym, eta {eta:.2f})  SNR {snr}: "
                      f"(median/p10 SINR, median rate) {line}", flush=True)
            with open(os.path.join(out_dir, (sys.argv[2] if len(sys.argv) > 2 else "numerology") + ".json"), "w", encoding="utf-8") as fh:
                json.dump(dict(n_packets=n, rows=rows), fh, indent=1)


if __name__ == "__main__":
    main()
