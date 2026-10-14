"""
Multipath evaluation on the realistic channel (PLAN: Phases 1 and 5; 14 Oct 2026). Everything realistic on both
sides: Bellhop paths with fading clusters (5 sub-paths, 0.3 deg, 0.05 ms) and fractional delays, optional diffuse
surface scattering (rho); industrial MEMS at both ends (imu.py), two-component side information (32 + 32 bits,
delivered; header read rates in scripts/b12_header.py); path angles from a Bellhop run on a nominal geometry with
random errors (water depth +-3 m, range +-2 %), never the true ones.
Environments: buoy - buoy OOI, isovelocity (direct + bottom bounce 10.7 ms: the differential-heave case); buoy - buoy
OOI, summer profile (one coherent path: control); buoy -> seabed OOI, isovelocity.
Receivers on identical packets and noise:
  signal-only   lilee, signal_kalman, pfft8_kalman (Kalman + partial FFT), tu_grid (multi-branch MRC),
                signal_sic (mpchain.receive_sic_signal: soft cancellation, time-varying gains, sub-sample delays)
  sensor-aided  fused_safe (+ _pfft8; one resampling), sensor_sic (mpchain.receive_sic_sensor, guard against
                signal_sic), sensor_sic_noguard
  reference     ideal_sic (true per-path trajectories and delays)
Metrics: SINR after the receiver's own channel estimate and MRC (receiver.sinr_combined), coded QPSK PER; per-packet
SINR kept. Usage: python scripts/b13_multipath.py <config> [n_packets], config in CONFIGS.
Output: results/b13_multipath/<config>.json.
"""
import json
import os
import sys

import numpy as np

from uwofdm import chain, channel, frame, imu, motion, mpchain, multipath, receiver, sensors
from uwofdm.paths import output_dir
from c2_sweeps import OOI_DAY, buoy_buoy_link, seabed_link
from e2_multipath import sigma_eta

LINKS = {"bb_iso": (buoy_buoy_link, (1000.0, 10.0, 10.0, 100.0), None),
         "bb_summer": (buoy_buoy_link, (1000.0, 10.0, 10.0, 100.0), multipath.SUMMER_SSP),
         "seabed_iso": (lambda: seabed_link("ooi", OOI_DAY, (0.0, 0.0, 2.0)), (1000.0, 2.0, 99.5, 100.0), None)}
CONFIGS = {"bb_iso_1s": ("bb_iso", None, 7), "bb_iso_1s_rho20": ("bb_iso", -20.0, 7), "bb_iso_4s": ("bb_iso", None, 27),
           "bb_summer_1s": ("bb_summer", None, 7), "seabed_iso_1s": ("seabed_iso", None, 7),
           "bb_iso_1s_16qam": ("bb_iso", None, 7, "16qam", (20, 25, 30))}
SNRS = (10, 20, 30)


def run(config, n):
    env, rho, n_data = CONFIGS[config][:3]
    mod = CONFIGS[config][3] if len(CONFIGS[config]) > 3 else "qpsk"
    snrs = CONFIGS[config][4] if len(CONFIGS[config]) > 4 else SNRS
    make, geo, ssp = LINKS[env]
    tx_track, rx_track = make()
    rng = np.random.default_rng(31)
    F = frame.Frame(n_data=n_data)
    o = F.o
    T = F.length / o.fs
    se = sigma_eta(tx_track)
    L = multipath.LinkMultipath(tx_track, rx_track, geo, se, tx_track.vel_fast[:, 2], tx_track.fs, rho_db=rho,
                                ssp=ssp, cluster=(5, 0.3, 5e-5))
    nom_geo = (geo[0] * (1 + 0.02 * rng.standard_normal()), geo[1], geo[2], geo[3] + 3.0 * rng.standard_normal())
    nominal = multipath.nominal_angles(nom_geo, ssp, surface_rms=se)
    angles = [(a, b) for a, b, _ in nominal]
    pred_delays = [dd for _, _, dd in nominal]
    lever = lambda tr: 2.0 if tr.vel_lever is not None else None
    imu_tx = sensors.IMU3(tx_track, rng, grade=imu.GRADES["industrial"], lever_m=lever(tx_track))
    imu_rx = sensors.IMU3(rx_track, rng, grade=imu.GRADES["industrial"], lever_m=lever(rx_track))
    si2 = sensors.TwoComponentSideInfo(imu_tx, L.u, T, 32, 32)
    print(f"{config}: coherent paths {[(round(p['delay_ms'], 2), round(p['power_db'], 1), round(p['src_deg'], 1)) for p in L.summary()]}; "
          f"nominal {[(round(a, 1), round(d * 1e3, 2)) for a, _, d in nominal]}", flush=True)
    acc = {}
    for t0 in rng.uniform(tx_track.t[0] + 240, tx_track.t[-1] - T - 60, n):
        rp = np.random.default_rng(rng.integers(1 << 30))
        info = rp.integers(0, 2, receiver.n_info_bits(F, mod))
        X = receiver.build_data(info, F, mod)
        Xf = X.copy()
        Xf[:, F.nulls] = 0
        s = F.build(o.random_symbols(F.n_header, rp), X)
        lead, extra = int(rp.integers(500, 3000)), int(0.09 * o.fs)
        x = np.concatenate([np.zeros(lead, complex), s, np.zeros(9000, complex)])
        tg = t0 - lead / o.fs + np.arange(len(x) + extra) / o.fs
        bulk, paths = L.build(tg, rp)
        y0 = channel.apply(x, o.fs, o.fc, bulk, paths, extra=extra)
        m = np.arange(len(x) + extra - lead)
        te = t0 + m / o.fs
        q = si2.encode(imu_tx.velocity(te[:F.length]), te[:F.length] - t0)
        h, zv = si2.decode(q, te - t0)
        pred = sensors.predict_paths(h, zv, imu_rx.velocity(te + L.tau), L.u, angles)
        rx_share = -(imu_rx.velocity(te + L.tau) @ L.u) / motion.C
        prior = h * (L.u @ sensors.horizontal(L.u)) / motion.C + zv * L.u[2] / motion.C + rx_share
        # the single-resampling sensor prior along the link direction (side information + own IMU)
        ideal_trs, ideal_d = [], []
        for grp_idx in L.groups:  # power-weighted centre of each arrival's sub-paths (ideal knowledge)
            grp = [paths[i] for i in grp_idx]
            p_ = max(grp, key=lambda p: np.mean(np.abs(p.gain)))
            ideal_trs.append(bulk[lead:] + (p_.delta_eps[lead:] if np.ndim(p_.delta_eps) else p_.delta_eps))
            ideal_d.append(p_.delay)
        ideal_d = [dd - min(ideal_d) for dd in ideal_d]
        for snr in snrs:
            sig = np.sqrt(4.0 / 10 ** (snr / 10) / 2)
            y = y0 + sig * (rp.standard_normal(len(y0)) + 1j * rp.standard_normal(len(y0)))
            se_ = (0, lead + 1500)
            out = {"lilee": [chain.receive_signal_only(y, F, se_)[0]]}
            Ys, inf = chain.receive_signal_fused(y, F, se_)
            out["signal_kalman"] = [Ys]
            out["pfft8_kalman"] = [chain._demod_pfft(inf["z"], F, mpchain.pilot_frame(F), 8)[0]]
            out["tu_grid"] = chain.receive_branches(y, F, chain.grid_trajectories(y, F, search=se_), se_)
            br_sig, _ = mpchain.receive_sic_signal(y, F, se_, mod=mod)
            out["signal_sic"] = br_sig
            Yf, inf_f = chain.receive_fused_safe(y, F, prior, se_)
            out["fused_safe"] = [Yf]
            out["fused_safe_pfft8"] = [chain._demod_pfft(inf_f["z"], F, mpchain.pilot_frame(F), 8)[0]]
            br_sen, si = mpchain.receive_sic_sensor(y, F, pred, pred_delays, se_, mod=mod, signal_branches=br_sig)
            out["sensor_sic"] = br_sen
            out["sensor_sic_noguard"] = si["sensor_branches"]
            if len(ideal_trs) > 1:
                out["ideal_sic"] = mpchain.sic(y, F, ideal_trs, ideal_d, se_, mod=mod)[0]
            for k, br in out.items():
                e = acc.setdefault((snr, k), dict(pw=[], err=[]))
                e["pw"].append(receiver.sinr_combined(br, Xf, F))
                e["err"].append(not np.array_equal(receiver.decode_combined(br, F, mod), info))
            acc.setdefault((snr, "_sensor_choice"), dict(c=[]))["c"].append(si.get("choice"))
    rows = []
    for (snr, k), e in sorted(acc.items()):
        if k.startswith("_"):
            rows.append(dict(config=config, snr_db=snr, sensor_guard_choice_signal=float(np.mean([c == "signal" for c in e["c"]]))))
            continue
        pw = np.array(e["pw"])
        db = 10 * np.log10(pw[:, 0] / pw[:, 1])
        rows.append(dict(config=config, snr_db=snr, receiver=k, sinr_power_avg_db=float(10 * np.log10(pw[:, 0].sum() / pw[:, 1].sum())),
                         sinr_median_db=float(np.median(db)), sinr_p10_db=float(np.percentile(db, 10)),
                         per=float(np.mean(e["err"])), per_packet_db=[float(v) for v in db]))
    for snr in snrs:
        print(f"{config} SNR {snr} (SINR power avg / median / p10, PER):", flush=True)
        for r in rows:
            if r["snr_db"] == snr and "receiver" in r:
                print(f"    {r['receiver']:20s} {r['sinr_power_avg_db']:5.1f} / {r['sinr_median_db']:5.1f} / {r['sinr_p10_db']:5.1f}  PER {r['per']:.2f}", flush=True)
            elif r["snr_db"] == snr:
                print(f"    sensor guard chose signal-only in {r['sensor_guard_choice_signal']:.0%} of packets", flush=True)
    with open(os.path.join(output_dir("b13_multipath"), f"{config}.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(config=config, n_packets=n, nominal=nominal, summary=L.summary(), rows=rows), fh, indent=1)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30)
