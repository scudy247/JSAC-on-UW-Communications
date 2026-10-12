"""
End-to-end chain over Bellhop multipath (PLAN: Phase 1 Bellhop paths in the chain; Phase 5).

Channel: coherent Bellhop arrivals with the measured rough-surface loss, each path with its own Doppler from its
departure / arrival angles (multipath.LinkMultipath), plus optionally diffusely scattered surface power rho (-30,
-20 dB) with per-patch surface Doppler; complex noise at per-subcarrier SNR 20 and 30 dB.
Receivers with one resampling for the bulk (strongest path) Doppler: Li/Lee signal-only, signal-only Kalman,
Cho-Kim, sensor-aided (ideal side information, 32 bits), sensor fusion. Multi-branch: Tu-like grid (signal-only
Kalman bulk +-1.5e-4, MRC), sensor_mrc (one branch per coherent Bellhop path at its Doppler from ideal sensors, MRC),
sensor_sic (decoder-aided per-path cancellation, chain.receive_sic), imu_sic (the same with per-path Doppler
predicted from realistic sensors: 3-axis IMUs with 0.01 m/s drift and 50 ms latency, 64 bits of two-component side
information, Bellhop path angles; common residual from preamble / postamble), signal_sic (signal-only
counterpart: path delays and mean Doppler from the chirps, every path's Doppler refined by a search on the
least-squares fit of the regenerated paths, two passes; chain.receive_signal_sic), imu_sic_search (realistic
sensors with path delays measured from the preamble peaks and a narrow ±2e-5 search around the prediction;
chain.receive_sensor_sic). All scored after MRC.
Metrics: SINR from the receiver's own pilot channel estimate (receiver.sinr) and coded QPSK PER (rate 1/2).
Links: buoy -> seabed (CDIP swell, OOI Irminger; 1 km, 100 m, 2 m -> 99.5 m), buoy -> buoy (OOI; 10 m -> 10 m);
7-symbol packets; 20 packets per point. Output: results/e2_chain/multipath.json; printed table and path summaries.
"""
import json
import os

import numpy as np

from uwofdm import chain, channel, frame, multipath, receiver, sensors, sideinfo, theory
from uwofdm.paths import output_dir
from e2_chain import SIGMA_IMU, allocation, setup
from c2_sweeps import CDIP_DAY, OOI_DAY, buoy_buoy_link, seabed_link

LINKS = {"buoy -> seabed, CDIP swell": (lambda: seabed_link("cdip", CDIP_DAY), (1000.0, 2.0, 99.5, 100.0)),
         "buoy -> seabed, OOI Irminger": (lambda: seabed_link("ooi", OOI_DAY, (0.0, 0.0, 2.0)),
                                          (1000.0, 2.0, 99.5, 100.0)),
         "buoy - buoy, OOI Irminger": (buoy_buoy_link, (1000.0, 10.0, 10.0, 100.0))}
RHOS = (None, -30, -20)
SNRS = (20, 30)
N_PACKETS = 20


def sigma_eta(track):
    """rms surface elevation from the buoy's heave velocity spectrum (above 0.03 Hz)."""
    f, S = theory.spectrum(track.vel_fast[:, 2], track.fs)
    m = f > 0.03
    return float(np.sqrt(np.sum(S[m] / (2 * np.pi * f[m]) ** 2) * (f[1] - f[0])))


def main(out_name="multipath.json"):
    out = output_dir("e2_chain")
    rows, rng = [], np.random.default_rng(21)
    for label, (make, geo) in LINKS.items():
        tx_track, rx_track = make()
        d, tx, rx = setup(tx_track, rx_track, rng)
        F = frame.Frame(n_data=7)
        o = F.o
        T = F.length / o.fs
        ranges, alloc = allocation(F, tx, tx.fs, budgets=(32,))
        a = alloc[(32, "none")]
        basis_fn = lambda cc, mm: sideinfo.reconstruct(cc, np.minimum(mm / o.fs, T), T)
        se = sigma_eta(tx_track)
        for rho in RHOS:
            L = multipath.LinkMultipath(tx_track, rx_track, geo, se, tx_track.vel_fast[:, 2], tx_track.fs,
                                        rho_db=rho)
            summ = L.summary()
            rel_geo = [r["delay_ms"] / 1e3 for r in summ]
            angles = [(r["src_deg"], r["rcv_deg"]) for r in summ]
            imu_tx, imu_rx = sensors.IMU3(tx_track, rng), sensors.IMU3(rx_track, rng)
            si2 = sensors.TwoComponentSideInfo(imu_tx, L.u, T, 32, 32)
            if rho is None:
                print(label, "sigma_eta %.2f m; coherent paths:" % se, flush=True)
                for p in L.summary():
                    print("   ", {k: (round(float(v), 2) if not isinstance(v, int) else v) for k, v in p.items()})
            acc = {}
            for t0 in rng.uniform(tx_track.t[0] + 60, tx_track.t[-1] - T - 60, N_PACKETS):
                rp = np.random.default_rng(rng.integers(1 << 30))
                info = rp.integers(0, 2, receiver.n_info_bits(F))
                X = receiver.build_data(info, F)
                Xf = X.copy()
                Xf[:, F.nulls] = 0
                s = F.build(o.random_symbols(F.n_header, rp), X)
                lead = int(rp.integers(500, 3000))
                x = np.concatenate([np.zeros(lead, complex), s, np.zeros(9000, complex)])
                extra = int(0.09 * o.fs)
                tg = t0 - lead / o.fs + np.arange(len(x) + extra) / o.fs
                bulk, paths = L.build(tg, rp)
                y0 = channel.apply(x, o.fs, o.fc, bulk, paths, extra=extra)
                te = t0 + np.arange(F.length + 4000) / o.fs
                tx_est, rx_est = tx.estimate(te, SIGMA_IMU), rx.estimate(te, SIGMA_IMU)
                c = sideinfo.project(tx_est[:F.length], te[:F.length] - t0, T, 24)
                prior = basis_fn(sideinfo.quantise_alloc(c, a, ranges), np.arange(len(rx_est))) + rx_est
                for snr in SNRS:
                    sig = np.sqrt(4.0 / 10 ** (snr / 10) / 2)
                    y = y0 + sig * (rp.standard_normal(len(y0)) + 1j * rp.standard_normal(len(y0)))
                    search = (0, lead + 1500)
                    outs = {"signal_only": [chain.receive_signal_only(y, F, search)[0]],
                            "signal_kalman": [chain.receive_signal_fused(y, F, search)[0]],
                            "cho_kim": [chain.receive_cho_kim(y, F, search)[0]],
                            "aided_32": [chain.receive_aided(y, F, prior, search)[0]],
                            "fused_32": [chain.receive_fused(y, F, prior, search)[0]],
                            "tu_grid": chain.receive_branches(y, F, chain.grid_trajectories(y, F, search=search),
                                                              search)}
                    coh = paths[:len(L.coh)]  # per-path Doppler from ideal sensors (own motion + path angles)
                    sens = [bulk[lead:] + p.delta_eps[lead:] if np.ndim(p.delta_eps) else bulk[lead:] for p in coh]
                    rel = [p.delay for p in coh]
                    if len(coh) > 1:
                        outs["sensor_mrc"] = chain.receive_branches(y, F, sens, search)
                        outs["sensor_sic"] = chain.receive_sic(y, F, sens, rel, search, iters=3)[0]
                        m = np.arange(len(x) + extra - lead)
                        te_p = t0 + m / o.fs
                        q = si2.encode(imu_tx.velocity(te_p[:F.length]), te_p[:F.length] - t0)
                        h, zv = si2.decode(q, te_p - t0)
                        pred = sensors.predict_paths(h, zv, imu_rx.velocity(te_p + L.tau), L.u, angles)
                        outs["imu_sic"] = chain.receive_sic(y, F, pred, rel_geo, search, iters=3,
                                                            refine_common=True)[0]
                        _, rel_meas, _ = chain.signal_path_estimates(y, F, search)
                        if len(rel_meas) >= len(pred):  # delays from the signal, angles from the geometry, search
                            outs["imu_sic_search"] = chain.receive_sensor_sic(y, F, pred, rel_meas[:len(pred)],
                                                                              search)[0]
                        outs["signal_sic"] = chain.receive_signal_sic(y, F, search)[0]
                    for k, Ys in outs.items():
                        e = acc.setdefault((snr, k), dict(pw=[], err=[]))
                        e["pw"].append(receiver.sinr_combined(Ys, Xf, F))
                        e["err"].append(not np.array_equal(receiver.decode_combined(Ys, F), info))
            for (snr, k), e in sorted(acc.items()):
                pw = np.array(e["pw"])
                row = dict(scenario=label, rho_db=rho, snr_db=snr, receiver=k,
                           sinr_db=float(10 * np.log10(pw[:, 0].sum() / pw[:, 1].sum())), per=float(np.mean(e["err"])))
                rows.append(row)
            for snr in SNRS:
                line = "  ".join(f"{k} {next(r['sinr_db'] for r in rows if r['scenario'] == label and r['rho_db'] == rho and r['snr_db'] == snr and r['receiver'] == k):5.1f}/"
                                 f"{next(r['per'] for r in rows if r['scenario'] == label and r['rho_db'] == rho and r['snr_db'] == snr and r['receiver'] == k):.2f}"
                                 for k in ("signal_only", "signal_kalman", "cho_kim", "aided_32", "fused_32",
                                           "tu_grid", "sensor_mrc", "signal_sic", "sensor_sic", "imu_sic", "imu_sic_search")
                                 if any(r['receiver'] == k for r in rows if r['scenario'] == label and r['rho_db'] == rho))
                print(f"{label:30s} rho {str(rho):5s} SNR {snr}: SINR/PER {line}", flush=True)
            with open(os.path.join(out, out_name), "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
