"""
High-statistics runs for the paper figures (PLAN: Phase 8). Same experiments as scripts/e2_per.py and
scripts/e2_multipath.py, with more packets at the SNRs where the receivers differ; results go to new files
(suffix _hi), the earlier runs are kept. Usage: python scripts/paper_stats.py A | B (two queues, run in parallel).
  A: PER QPSK 1.05 s (300 packets), 16-QAM 1.05 s (200), 16-QAM 4.05 s (150)        about 5 h
  B: PER QPSK 4.05 s (200 packets), multipath 1.05 s (60 packets, rho none / -30 / -20 dB)   about 6 h
"""
import sys

import e2_multipath
from e2_per import run_per


def queue_a():
    run_per("qpsk", 7, 32, (6, 9, 12, 15, 20), n_packets=300, tag="_qpsk_7_hi")
    run_per("16qam", 7, 32, (15, 18, 22, 26, 30), n_packets=200, tag="_16qam_7_hi")
    run_per("16qam", 27, 96, (18, 22, 26, 30), n_packets=150, tag="_16qam_27_hi")


def queue_b():
    run_per("qpsk", 27, 96, (9, 12, 15, 20, 25), n_packets=200, tag="_qpsk_27_hi")
    e2_multipath.N_PACKETS = 60
    e2_multipath.main("multipath_hi.json")


if __name__ == "__main__":
    {"A": queue_a, "B": queue_b}[sys.argv[1]]()
