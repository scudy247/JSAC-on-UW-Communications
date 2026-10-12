"""
Side-information header on real motion (PLAN: Phases 4-5): is it read before the Doppler is known, and what does the
whole chain deliver with it?

Per packet: the transmitter's IMU share over the frame is projected, quantised (allocation for no signal estimate: level
and slope first, 16 / 32 / 64 bits) and packed with a CRC-8 into the header symbol (header.py: DBPSK
across subcarriers, repetition over the band); the frame goes through the link Doppler of real buoy motion with an
unknown start and noise. Receiver: chain.receive_with_header (own IMU -> mean scale from preamble / postamble ->
header CFO -> bits -> CRC; fallback to the signal-only front end if the CRC fails).
Reported per scenario, bits and per-subcarrier SNR (-5 ... 30 dB): header success rate, SINR of the full chain, of the
chain with ideally delivered side information (same bits), and of the signal-only front end.
Output: results/e1_sideinfo/header.json; printed table. 20 packets per point.
"""
import json
import os

import numpy as np

from uwofdm import chain, frame, header, sideinfo, waveform
from uwofdm.experiment import at_times, genie_powers
from uwofdm.paths import output_dir
from e2_chain import BITS, SCENARIOS, SIGMA_IMU, allocation, setup

SNRS_DB = (-5, 0, 5, 10, 20, 30)
N_PACKETS = 20


def main():
    out = output_dir("e1_sideinfo")
    rows, rng = [], np.random.default_rng(5)
    for label, make in SCENARIOS.items():
        d, tx, rx = setup(*make(), rng)
        F = frame.Frame(n_data=7)
        o = F.o
        T = F.length / o.fs
        ranges, alloc = allocation(F, tx, tx.fs)
        basis_fn = lambda cc, mm: sideinfo.reconstruct(cc, np.minimum(mm / o.fs, T), T)
        acc = {}
        for t0 in rng.uniform(d["times"][0] + 40, d["times"][-1] - T - 20, N_PACKETS):
            rp = np.random.default_rng(rng.integers(1 << 30))
            te = t0 + np.arange(F.length + 4000) / o.fs
            tx_est, rx_est = tx.estimate(te, SIGMA_IMU), rx.estimate(te, SIGMA_IMU)
            c = sideinfo.project(tx_est[:F.length], te[:F.length] - t0, T, 24)
            X = o.random_symbols(F.n_data, rp)
            lead = int(rp.integers(500, 3000))
            for b in BITS:
                a = alloc[(b, "none")]
                s = F.build(header.modulate(header.pack(c, a, ranges), o, F.used), X)
                x = np.concatenate([np.zeros(lead, complex), s, np.zeros(8000, complex)])
                tg = t0 - lead / o.fs + np.arange(len(x)) / o.fs
                y0 = waveform.warp(x, o.fs, at_times(d["eps"], d["times"], tx.fs, tg), o.fc)
                ideal_prior = basis_fn(sideinfo.quantise_alloc(c, a, ranges), np.arange(len(rx_est))) + rx_est
                for snr in SNRS_DB:
                    sig = np.sqrt(4.0 / 10 ** (snr / 10) / 2)
                    y = y0 + sig * (rp.standard_normal(len(y0)) + 1j * rp.standard_normal(len(y0)))
                    search = (0, lead + 1500)
                    Y, info = chain.receive_with_header(y, F, rx_est, a, ranges, basis_fn, search)
                    Yi = chain.receive_aided(y, F, ideal_prior, search)[0]
                    Ys = chain.receive_signal_only(y, F, search)[0]
                    key = (b, snr)
                    e = acc.setdefault(key, dict(ok=[], full=[], ideal=[], signal=[]))
                    e["ok"].append(info["header_ok"])
                    e["full"].append(genie_powers(Y, X, F.used))
                    e["ideal"].append(genie_powers(Yi, X, F.used))
                    e["signal"].append(genie_powers(Ys, X, F.used))
        for (b, snr), e in sorted(acc.items()):
            db = lambda pw: float(10 * np.log10(np.sum([p[0] for p in pw]) / np.sum([p[1] for p in pw])))
            row = dict(scenario=label, bits=b, snr_db=snr, header_success=float(np.mean(e["ok"])),
                       sinr_full_db=db(e["full"]), sinr_ideal_side_info_db=db(e["ideal"]),
                       sinr_signal_only_db=db(e["signal"]))
            rows.append(row)
            print(f"{label:30s} {b:3d} bits SNR {snr:3d}: header ok {row['header_success']:4.0%}  full chain "
                  f"{row['sinr_full_db']:5.1f}  ideal side info {row['sinr_ideal_side_info_db']:5.1f}  signal only "
                  f"{row['sinr_signal_only_db']:5.1f}", flush=True)
        with open(os.path.join(out, "header.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
