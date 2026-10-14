"""
Side-information header in multipath, and the cost of a wrong header (PLAN: Phase 4).

Part 1, header read rate: buoy-buoy OOI, 1.05 s frames carrying 32 side-information bits + CRC-8 in the header
symbol (DBPSK across subcarriers with repetition, chain.read_header), industrial MEMS; single path vs Bellhop multipath
(isovelocity, measured surface roughness, path clusters (5, 0.3 deg, 0.05 ms), fractional delays: direct + bottom
bounce 10.7 ms apart, the bounce with its own heave Doppler). Per-subcarrier SNR -3 ... 10 dB, n packets each.
Counted: header accepted (CRC passes) and accepted but wrong (undetected error).
Part 2, cost of a wrong header: what the safe sensor fusion (chain.receive_fused_safe) does when the CRC passes on a
corrupted header (random quantiser indices): single path, 1 and 4 s, 30 dB, against the correct header and the
signal-only Kalman.
Usage: python scripts/b12_header.py [n_packets]. Output: results/b12_header/points.json.
"""
import json
import os
import sys

import numpy as np

from uwofdm import chain, channel, frame, header, motion, multipath, sideinfo, waveform
from uwofdm.experiment import at_times, genie_powers
from uwofdm.paths import output_dir
from e2_chain import SIGMA_IMU, allocation
from e2_imu_grades import node
from e2_multipath import sigma_eta
from c2_sweeps import buoy_buoy_link

SNRS_DB = (-3, 0, 3, 6, 10)
GEO = (1000.0, 10.0, 10.0, 100.0)


def setup(n_data, seed):
    tx, rx = buoy_buoy_link()
    d = motion.link_doppler(tx, rx)
    F = frame.Frame(n_data=n_data)
    ntx, nrx = node(tx, d["u"], +1, 0.0, "industrial", seed + 1), node(rx, d["u"], -1, d["tau"], "industrial", seed + 2)
    ranges, alloc = allocation(F, ntx, tx.fs, budgets=(32,))
    return tx, rx, d, F, ntx, nrx, ranges, alloc[(32, "none")]


def make_packet(F, d, tx, ntx, nrx, ranges, a, t0, rp, link=None, coherent=False):
    o = F.o
    T = F.length / o.fs
    te = t0 + np.arange(F.length + 4000) / o.fs
    c = sideinfo.project(ntx.estimate(te[:F.length], SIGMA_IMU), te[:F.length] - t0, T, 24)
    bits = header.pack(c, a, ranges)
    X = o.random_symbols(F.n_data, rp)
    X[:, F.nulls] = 0
    from uwofdm import receiver
    hsym = header.modulate_coherent(bits, o, F.used, receiver.masks(F)[1]) if coherent else header.modulate(bits, o, F.used)
    s = F.build(hsym, X)
    lead = int(rp.integers(500, 3000))
    x = np.concatenate([np.zeros(lead, complex), s, np.zeros(9000, complex)])
    extra = int(0.09 * o.fs)
    tg = t0 - lead / o.fs + np.arange(len(x) + extra) / o.fs
    if link is None:
        y0 = waveform.warp(np.pad(x, (0, extra)), o.fs, at_times(d["eps"], d["times"], tx.fs, tg), o.fc)
    else:
        bulk, paths = link.build(tg, rp)
        y0 = channel.apply(x, o.fs, o.fc, bulk, paths, extra=extra)
    return y0, X, bits, nrx.estimate(te, SIGMA_IMU), lead, c


def part1(n):
    rows = []
    tx, rx, d, F, ntx, nrx, ranges, a = setup(7, 10)
    o = F.o
    T = F.length / o.fs
    links = {"single path": None,
             "multipath": multipath.LinkMultipath(tx, rx, GEO, sigma_eta(tx), tx.vel_fast[:, 2], tx.fs,
                                                  cluster=(5, 0.3, 5e-5))}
    for (name, link), coh in [(kv, c) for kv in links.items() for c in (False, True)]:
        name = f"{name}, {'coherent' if coh else 'DBPSK'}"
        rng = np.random.default_rng(42)
        acc = {s: [0, 0] for s in SNRS_DB}
        for t0 in rng.uniform(d["times"][0] + 240, d["times"][-1] - T - 60, n):
            rp = np.random.default_rng(rng.integers(1 << 30))
            y0, X, bits, rx_est, lead, _ = make_packet(F, d, tx, ntx, nrx, ranges, a, t0, rp, link, coh)
            for snr in SNRS_DB:
                sig = np.sqrt(4.0 / 10 ** (snr / 10) / 2)
                y = y0 + sig * (rp.standard_normal(len(y0)) + 1j * rp.standard_normal(len(y0)))
                ok, payload, _ = chain.read_header(y, F, rx_est, a, (0, lead + 1500), coherent=coh)
                acc[snr][0] += ok
                acc[snr][1] += ok and not np.array_equal(payload, bits[:-8])
        for snr, (ok, wrong) in acc.items():
            rows.append(dict(part="read_rate", channel=name, snr_db=snr, accepted=ok / n, undetected_wrong=wrong / n))
            print(f"header, {name:25s} SNR {snr:3d} dB: accepted {ok / n:.2f}, accepted but wrong {wrong / n:.3f}",
                  flush=True)
    return rows


def part2(n):
    rows = []
    for n_data in (7, 27):
        tx, rx, d, F, ntx, nrx, ranges, a = setup(n_data, 20)
        o = F.o
        T = F.length / o.fs
        rng = np.random.default_rng(43)
        acc = {}
        for t0 in rng.uniform(d["times"][0] + 240, d["times"][-1] - T - 60, n):
            rp = np.random.default_rng(rng.integers(1 << 30))
            y0, X, bits, rx_est, lead, c = make_packet(F, d, tx, ntx, nrx, ranges, a, t0, rp)
            y = y0 + np.sqrt(4e-3 / 2) * (rp.standard_normal(len(y0)) + 1j * rp.standard_normal(len(y0)))
            m = np.arange(len(rx_est))
            rebuild = lambda cc: sideinfo.reconstruct(cc, np.minimum(m / o.fs, T), T) + rx_est
            good = header.unpack(bits[:-8], a, ranges)
            wrong = header.unpack(rp.integers(0, 2, len(bits) - 8), a, ranges)
            se = (0, lead + 1500)
            out = {"signal_kalman": chain.receive_signal_fused(y, F, se)[0],
                   "fusion_correct_header": chain.receive_fused_safe(y, F, rebuild(good), se)[0],
                   "fusion_wrong_header": chain.receive_fused_safe(y, F, rebuild(wrong), se)[0],
                   "fusion_fixed_wrong_header": chain.receive_fused(y, F, rebuild(wrong), se)[0]}
            for k, Y in out.items():
                acc.setdefault(k, []).append(genie_powers(Y, X, F.used))
        for k, pw in acc.items():
            pw = np.array(pw)
            db = 10 * np.log10(pw[:, 0] / pw[:, 1])
            rows.append(dict(part="wrong_header", n_data=n_data, receiver=k, median_db=float(np.median(db)),
                             p10_db=float(np.percentile(db, 10)),
                             power_avg_db=float(10 * np.log10(pw[:, 0].sum() / pw[:, 1].sum()))))
            print(f"wrong header, {n_data:2d} sym, 30 dB, {k:26s}: power avg {rows[-1]['power_avg_db']:5.1f}, "
                  f"median {rows[-1]['median_db']:5.1f}, p10 {rows[-1]['p10_db']:5.1f}", flush=True)
    return rows


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    rows = part1(n) + (part2(max(10, n // 5)) if "--no-part2" not in sys.argv else [])
    with open(os.path.join(output_dir("b12_header"), "points.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
