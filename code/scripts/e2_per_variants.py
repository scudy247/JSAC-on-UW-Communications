"""
Coded PER where Doppler should matter (PLAN: Phases 2 and 5): 16-QAM and longer packets, same chain and receivers as
scripts/e2_per.py (signal-only front end, full sensor-aided chain with the real header, ideal side information).
Variants: 16-QAM / 7 symbols (1.05 s, 32 side-information bits); QPSK / 27 symbols (4.05 s, 96 bits, ≈ 25 bits/s);
16-QAM / 27 symbols (96 bits). 40 packets per point. Output: results/e2_chain/per_<variant>.json.
"""
from e2_per import run_per

VARIANTS = (("16qam", 7, 32, (6, 9, 12, 15, 18, 22, 26, 30), "_16qam_7"),
            ("qpsk", 27, 96, (0, 3, 6, 9, 12, 15, 20, 25), "_qpsk_27"),
            ("16qam", 27, 96, (6, 9, 12, 15, 18, 22, 26, 30), "_16qam_27"))

if __name__ == "__main__":
    for mod, n_data, bits, snrs, tag in VARIANTS:
        run_per(mod, n_data, bits, snrs, n_packets=40, tag=tag)
