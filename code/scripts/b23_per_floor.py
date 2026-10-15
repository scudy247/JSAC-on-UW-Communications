"""
Diagnostic of the packet-error floor of the ideal multipath receiver (PLAN: Phase 2; 15 Oct 2026). In
b13_multipath.py bb_iso_1s_16qam the ideal cancellation (true path trajectories and delays) loses 7-10 % of packets
at 23-31 dB SINR, where coded 16-QAM should decode. Hypothesis: the decoder's noise variance comes from the in-band
null subcarriers (receiver.combine -> channel_estimates NV), which do not see all of what is left on the data
subcarriers after cancellation, so the LLRs are overconfident.
Same packets as b13 (same seeds, same channel and noise), ideal_sic only. Per packet: decoded or not; per OFDM symbol,
the decoder's effective noise (mean NVe on data subcarriers) vs the true error |Z - X|^2 after MRC; and a re-decode with the true per-symbol error variance in place of NVe (oracle noise).
If the oracle re-decode removes the failures, the floor is the noise estimate, not Doppler estimation.
Usage: python scripts/b23_per_floor.py [n_packets] [snr_db]. Output: results/b23_per_floor/points.json.
"""
import json
import os
import sys

import numpy as np

from uwofdm import channel, coding, frame, imu, mpchain, multipath, receiver, sensors
from uwofdm.paths import output_dir
from b13_multipath import CONFIGS, LINKS
from e2_multipath import sigma_eta


def decode_with(Z, NVe, F, mod):
    data, _ = receiver.masks(F)
    llr = np.concatenate([coding.MODS[mod][2](Z[k][data], np.ones(int(data.sum())), NVe[k][data])
                          for k in range(F.n_data)])
    perm = coding.interleaver(len(llr))
    deint = np.empty_like(llr)
    deint[perm] = llr
    return coding.viterbi(deint, receiver.n_info_bits(F, mod))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    snr = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    config = "bb_iso_1s_16qam"
    env, rho, n_data, mod, snrs = CONFIGS[config]
    make, geo, ssp = LINKS[env]
    tx_track, rx_track = make()
    rng = np.random.default_rng(31)  # as b13: same packet times and seeds
    F = frame.Frame(n_data=n_data)
    o = F.o
    T = F.length / o.fs
    L = multipath.LinkMultipath(tx_track, rx_track, geo, sigma_eta(tx_track), tx_track.vel_fast[:, 2], tx_track.fs,
                                rho_db=rho, ssp=ssp, cluster=(5, 0.3, 5e-5))
    rng.standard_normal(2)  # b13 draws the nominal-geometry errors here, then builds the two IMUs from rng
    lever = lambda tr: 2.0 if tr.vel_lever is not None else None
    sensors.IMU3(tx_track, rng, grade=imu.GRADES["industrial"], lever_m=lever(tx_track))
    sensors.IMU3(rx_track, rng, grade=imu.GRADES["industrial"], lever_m=lever(rx_track))
    data, _ = receiver.masks(F)
    rows = []
    for i, t0 in enumerate(rng.uniform(tx_track.t[0] + 240, tx_track.t[-1] - T - 60, n)):
        rp = np.random.default_rng(rng.integers(1 << 30))
        info = rp.integers(0, 2, receiver.n_info_bits(F, mod))
        X = receiver.build_data(info, F, mod)
        s = F.build(o.random_symbols(F.n_header, rp), X)
        lead, extra = int(rp.integers(500, 3000)), int(0.09 * o.fs)
        x = np.concatenate([np.zeros(lead, complex), s, np.zeros(9000, complex)])
        tg = t0 - lead / o.fs + np.arange(len(x) + extra) / o.fs
        bulk, paths = L.build(tg, rp)
        y0 = channel.apply(x, o.fs, o.fc, bulk, paths, extra=extra)
        trs, dl = [], []
        for grp_idx in L.groups:
            p_ = max([paths[j] for j in grp_idx], key=lambda p: np.mean(np.abs(p.gain)))
            trs.append(bulk[lead:] + (p_.delta_eps[lead:] if np.ndim(p_.delta_eps) else p_.delta_eps))
            dl.append(p_.delay)
        dl = [d - min(dl) for d in dl]
        for s_db in snrs:  # b13 draws the noise of every SNR in turn from rp; keep the requested one
            sig = np.sqrt(4.0 / 10 ** (s_db / 10) / 2)
            w = sig * (rp.standard_normal(len(y0)) + 1j * rp.standard_normal(len(y0)))
            if s_db == snr:
                y = y0 + w
        br = mpchain.sic(y, F, trs, dl, (0, lead + 1500), mod=mod)[0]
        Z, NVe = receiver.combine(br, F)
        err_true = np.abs(Z[:, data] - X[:, data]) ** 2
        nv_dec = NVe[:, data].mean(1)
        nv_true = err_true.mean(1)
        ok = np.array_equal(receiver.decode_combined(br, F, mod), info)
        NVo = np.repeat(nv_true[:, None], o.n_sc, 1)
        ok_oracle = np.array_equal(decode_with(Z, NVo, F, mod), info)
        sinr_db = 10 * np.log10(np.sum(np.abs(X[:, data]) ** 2) / np.sum(err_true))
        rows.append(dict(packet=i, ok=ok, ok_oracle_noise=ok_oracle, sinr_db=float(sinr_db),
                         nv_decoder_db=[float(10 * np.log10(v)) for v in nv_dec],
                         nv_true_db=[float(10 * np.log10(v)) for v in nv_true]))
        print(f"packet {i:3d} SINR {sinr_db:5.1f} dB ok {ok} oracle-noise ok {ok_oracle} | per symbol, decoder noise "
              f"- true error [dB]: " + " ".join(f"{10 * np.log10(a / b):+.1f}" for a, b in zip(nv_dec, nv_true)),
              flush=True)
    fails = [r for r in rows if not r["ok"]]
    print(f"failures {len(fails)}/{n}; fixed by oracle noise {sum(r['ok_oracle_noise'] for r in fails)}", flush=True)
    with open(os.path.join(output_dir("b23_per_floor"), "points.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
