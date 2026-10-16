"""
High-statistics runs for the paper figures (PLAN: Phase 8). Same experiments as scripts/e2_per.py and
scripts/e2_multipath.py, with more packets at the SNRs where the receivers differ; results go to new files
(suffix _hi), the earlier runs are kept. Usage: python scripts/paper_stats.py A | B (two queues, run in parallel).
  A: PER QPSK 1.05 s (300 packets), 16-QAM 1.05 s (200), 16-QAM 4.05 s (150)        about 7 h
  B: PER QPSK 4.05 s (200 packets), multipath 1.05 s (60 packets, rho none / -30 / -20 dB)   about 8 h
Receiver version: sparse channel estimate + prior-quality-aware fusion (12 Oct 2026); SNR grids moved 3-6 dB lower
than the first launch, since the old channel estimate lost ~4 dB (waterfalls now at lower SNR).
"""
import sys

import e2_multipath
from e2_per import run_per


def queue_a():
    run_per("qpsk", 7, 32, (0, 3, 6, 9, 12, 20), n_packets=300, tag="_qpsk_7_hi")
    run_per("16qam", 7, 32, (9, 12, 15, 18, 22, 30), n_packets=200, tag="_16qam_7_hi")
    run_per("16qam", 27, 96, (12, 15, 18, 22, 26, 30), n_packets=150, tag="_16qam_27_hi")


def queue_b():
    run_per("qpsk", 27, 96, (3, 6, 9, 12, 15, 20, 25), n_packets=200, tag="_qpsk_27_hi")
    e2_multipath.N_PACKETS = 60
    e2_multipath.main("multipath_hi.json")


PER_RUNS = {"qpsk_7": ("qpsk", 7, 32, (0, 3, 6, 9, 12, 20), 300),  # one PER curve per campaign job (16 Oct 2026)
            "16qam_7": ("16qam", 7, 32, (9, 12, 15, 18, 22, 30), 200),
            "16qam_27": ("16qam", 27, 96, (12, 15, 18, 22, 26, 30), 150),
            "qpsk_27": ("qpsk", 27, 96, (3, 6, 9, 12, 15, 20, 25), 200)}


if __name__ == "__main__":
    # paper_stats.py A | B            the original two queues
    # paper_stats.py per <key> [tag] [n]  one curve of PER_RUNS (tag default _<key>_hi; n overrides the packets)
    if sys.argv[1] == "per":
        mod, n_data, bits, snrs, n = PER_RUNS[sys.argv[2]]
        n = int(sys.argv[4]) if len(sys.argv) > 4 else n
        run_per(mod, n_data, bits, snrs, n_packets=n, tag=sys.argv[3] if len(sys.argv) > 3 else f"_{sys.argv[2]}_hi")
    else:
        {"A": queue_a, "B": queue_b}[sys.argv[1]]()
