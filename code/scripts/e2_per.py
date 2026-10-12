"""
Coded PER of the end-to-end chain on real motion (PLAN: Phases 2 and 5) — the metric of Lee et al. 2026.

Frame: chirp | header (side information, 32 bits + CRC) | 7 data symbols (rate-1/2 K=7 code, interleaver, QPSK,
pilot channel estimate, LLRs weighted by the noise + interference measured on in-band nulls) | chirp; 4768 info
bits per packet (≈ 4.5 kbit/s). Link Doppler of real buoy motion, unknown start, complex noise.
Receivers: signal_only (Li et al. 2008 / Lee et al. 2026 front end), signal_kalman (signal-only Kalman smoother),
full_chain (real header, then the prior-quality-aware Kalman fusion chain.receive_fused_safe; fallback to the
signal-only Kalman if the CRC fails; before 12 Oct 2026: receive_aided, fallback Li/Lee), ideal_side_info
(sensor-aided with the side information delivered, receive_aided), fused (ideal side information +
receive_fused_safe; before 12 Oct: fixed-setting receive_fused). Channel estimate: receiver.channel_estimates (sparse
SOMP since 12 Oct 2026; the earlier per-symbol LS lost ~4 dB).
Scenarios: buoy -> seabed (CDIP swell, OOI Irminger), buoy -> buoy (OOI); per-subcarrier SNR 0 ... 25 dB;
100 packets per point (PER resolution 0.01).
Output: results/e2_chain/per.json; printed table.
"""
import json
import os

import numpy as np

from uwofdm import chain, frame, header, receiver, sideinfo, waveform
from uwofdm.experiment import at_times
from uwofdm.paths import output_dir
from e2_chain import SCENARIOS, SIGMA_IMU, allocation, setup

SNRS_DB = (0, 3, 6, 9, 12, 15, 20, 25)
N_PACKETS = 100
BITS = 32


def run_per(mod="qpsk", n_data=7, bits=BITS, snrs=SNRS_DB, n_packets=N_PACKETS, tag=""):
    """PER of the three receivers for one modulation / packet length / side-information size; writes per<tag>.json."""
    out = output_dir("e2_chain")
    rows, rng = [], np.random.default_rng(11)
    for label, make in SCENARIOS.items():
        d, tx, rx = setup(*make(), rng)
        F = frame.Frame(n_data=n_data)
        o = F.o
        T = F.length / o.fs
        ranges, alloc = allocation(F, tx, tx.fs, budgets=(bits,))
        a = alloc[(bits, "none")]
        basis_fn = lambda cc, mm: sideinfo.reconstruct(cc, np.minimum(mm / o.fs, T), T)
        err = {}
        hdr_ok = {}
        for t0 in rng.uniform(d["times"][0] + 40, d["times"][-1] - T - 20, n_packets):
            rp = np.random.default_rng(rng.integers(1 << 30))
            te = t0 + np.arange(F.length + 4000) / o.fs
            tx_est, rx_est = tx.estimate(te, SIGMA_IMU), rx.estimate(te, SIGMA_IMU)
            c = sideinfo.project(tx_est[:F.length], te[:F.length] - t0, T, 24)
            info = rp.integers(0, 2, receiver.n_info_bits(F, mod))
            s = F.build(header.modulate(header.pack(c, a, ranges), o, F.used), receiver.build_data(info, F, mod))
            lead = int(rp.integers(500, 3000))
            x = np.concatenate([np.zeros(lead, complex), s, np.zeros(8000, complex)])
            tg = t0 - lead / o.fs + np.arange(len(x)) / o.fs
            y0 = waveform.warp(x, o.fs, at_times(d["eps"], d["times"], tx.fs, tg), o.fc)
            ideal = basis_fn(sideinfo.quantise_alloc(c, a, ranges), np.arange(len(rx_est))) + rx_est
            for snr in snrs:
                sig = np.sqrt(4.0 / 10 ** (snr / 10) / 2)
                y = y0 + sig * (rp.standard_normal(len(y0)) + 1j * rp.standard_normal(len(y0)))
                search = (0, lead + 1500)
                Yf, inf = chain.receive_with_header(y, F, rx_est, a, ranges, basis_fn, search, receiver="safe")
                outs = {"signal_only": chain.receive_signal_only(y, F, search)[0], "full_chain": Yf,
                        "ideal_side_info": chain.receive_aided(y, F, ideal, search)[0],
                        "signal_kalman": chain.receive_signal_fused(y, F, search)[0],
                        "cho_kim": chain.receive_cho_kim(y, F, search)[0],
                        "fused": chain.receive_fused_safe(y, F, ideal, search)[0]}
                for k, Y in outs.items():
                    err.setdefault((snr, k), []).append(not np.array_equal(receiver.decode(Y, F, mod), info))
                hdr_ok.setdefault(snr, []).append(inf["header_ok"])
        for snr in snrs:
            row = dict(receiver_version="somp+safe_fusion", scenario=label, modulation=mod, n_data=n_data, side_info_bits=bits, snr_db=snr, header_success=float(np.mean(hdr_ok[snr])),
                       **{f"per_{k}": float(np.mean(err[(snr, k)])) for k in ("signal_only", "full_chain",
                                                                               "ideal_side_info", "signal_kalman",
                                                                               "cho_kim", "fused")})
            rows.append(row)
            print(f"{label:30s} {mod:5s} {n_data:2d} sym SNR {snr:2d}: PER signal-only {row['per_signal_only']:.2f}  "
                  f"signal Kalman {row['per_signal_kalman']:.2f}  Cho-Kim {row['per_cho_kim']:.2f}  full chain {row['per_full_chain']:.2f}  ideal side "
                  f"info {row['per_ideal_side_info']:.2f}  fused {row['per_fused']:.2f}  header ok "
                  f"{row['header_success']:.0%}", flush=True)
        with open(os.path.join(out, f"per{tag}.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)


def main():
    run_per()


if __name__ == "__main__":
    main()
