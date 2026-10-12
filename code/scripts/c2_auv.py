"""
C2, scenario 3: AUV to buoy (uplink to the gateway). The AUV's motion is smooth but large (cruise, turns, speed
changes; motion.auv_maneuvering, modelled: no logs available) and fully known to its navigation system (INS + DVL);
the buoy's is wave-driven (real records) and known to its IMU.

Link: AUV 50 m deep, 1 km north of the buoy at the start, heading 45 deg at 1.5 m/s; buoy transducer 10 m deep (OOI:
2 m lever arm, gyro-corrected). The path direction is kept fixed over a 20-minute window (the AUV moves about 1.8 km:
a limitation of the fixed-geometry model, noted in PLAN.md).
Estimates and strategies as in c2_buoy_buoy.py: receiver only (per symbol, quadratic per packet, stale), transmitter
only (AUV INS), receiver only with the buoy IMU, both, split (both + receiver line / constant from two estimates).
The closed form ignores the mean Doppler of the AUV cruise (the spectra are of the fluctuations), so it is optimistic
for strategies that leave the AUV's motion uncompensated.
Outputs: results/c2_auv/metrics.json and a printed table. Single thread, low priority.
"""
import json
import os

import numpy as np

from uwofdm import motion, theory, waveform
from uwofdm.compensate import compensate, fit_trajectory
from uwofdm.paths import output_dir
from c1_residual import genie_powers
from c2_placement import at_times
from c2_buoy_buoy import LATENCY, N_PACKETS, NodeIMU, PACKET_S, SIGMA_IMU, SIGMA_RX, TAU_IMU

PAD = 4096  # zero samples appended to each packet (128 ms): a stretched or pre-compensated packet must not be
#             cut (with bulk Doppler 1e-3, a 4 s packet stretches by 4 ms)
DAYS = [("cdip", "029p1_swell_2022-01-27", 0, None), ("cdip", "029p1_wind_2022-07-08", 0, None),
        ("ooi", "GI01SUMO/20230919", 3, (0.0, 0.0, 2.0))]
WINDOW_S = 1140.0
AUV_START = (1000.0, 0.0, -50.0)
BUOY_DEPTH = 10.0


def main():
    o = waveform.OFDM()
    t_sym, t_fft = o.n_symbol / o.fs, o.n_fft / o.fs
    rng = np.random.default_rng(0)
    res = {"auv_start": AUV_START, "window_s": WINDOW_S, "sigma_rx": SIGMA_RX, "tau_imu": TAU_IMU,
           "latency": LATENCY, "rows": []}
    for source, name, seg, lever in DAYS:
        buoy = motion.buoy_track(source, name, (0.0, 0.0, -BUOY_DEPTH), segment=seg, lever=lever)
        n = min(buoy.n, int(WINDOW_S * buoy.fs) + 1)
        buoy.vel_fast, buoy.vel_slow = buoy.vel_fast[:n], buoy.vel_slow[:n]
        buoy.vel_lever = None if buoy.vel_lever is None else buoy.vel_lever[:n]
        auv = motion.auv_maneuvering(buoy.fs, n, AUV_START, rng)
        d = motion.link_doppler(auv, buoy)
        t_day, eps, u, tau = d["times"], d["eps"], d["u"], d["tau"]
        tx, rx = NodeIMU(auv, u, +1, 0.0, rng, with_slow=True), NodeIMU(buoy, u, -1, tau, rng)
        f, S = theory.spectrum(eps * motion.C, buoy.fs)
        for T in PACKET_S:
            n_sym = int(round(T / t_sym))
            starts = np.linspace(t_day[0] + 30, t_day[-1] - T - 5, N_PACKETS)
            sims = {}
            for t0 in starts:
                X = o.random_symbols(n_sym, rng)
                s = np.pad(o.modulate(X), (0, PAD))  # room for stretching by Doppler or by pre-compensation
                tg = t0 + np.arange(len(s)) / o.fs
                e_true = at_times(eps, t_day, buoy.fs, tg)
                t_est = t0 + (np.arange(n_sym) + 0.5) * t_sym
                e_est = at_times(eps, t_day, buoy.fs, t_est) + SIGMA_RX * rng.standard_normal(n_sym)
                y = waveform.warp(s, o.fs, e_true, o.fc)

                def run(key, z):
                    sims.setdefault(key, []).append(genie_powers(o.demodulate(z, n_sym), X, o.active))

                run("rx_sym", compensate(y, o.fs, o.fc, fit_trajectory(t_est, e_est, tg, "piece")))
                run("rx_pkt2", compensate(y, o.fs, o.fc, fit_trajectory(t_est, e_est, tg, 2)))
                old = at_times(eps, t_day, buoy.fs, [t0 - 1.0])[0] + SIGMA_RX * rng.standard_normal()
                run("rx_stale_1", compensate(y, o.fs, o.fc, np.full(len(tg), old)))
                for sg in SIGMA_IMU:
                    pre_tx = tx.estimate(tg, sg)
                    y_tx = waveform.warp(waveform.unwarp(s, o.fs, pre_tx, o.fc), o.fs, e_true, o.fc)
                    rx_own = rx.estimate(tg, sg)
                    if sg == SIGMA_IMU[0]:
                        run("tx_ins", y_tx)
                        run("rx_imu", compensate(y, o.fs, o.fc, rx_own))
                    run(f"both_{sg:g}", compensate(y_tx, o.fs, o.fc, rx_own))
                    t2 = t_est[[0, -1]]
                    r2 = at_times(eps, t_day, buoy.fs, t2) - tx.estimate(t2, sg) - rx.estimate(t2, sg) \
                        + SIGMA_RX * rng.standard_normal(2)
                    for key, order in ((f"split_{sg:g}", 1), (f"split0_{sg:g}", 0)):
                        run(key, compensate(y_tx, o.fs, o.fc, rx_own + fit_trajectory(t2, r2, tg, order)))
            noise = SIGMA_RX ** 2
            th = {"rx_sym": theory.residual_eps_var(f, S, theory.kernel_poly(f, t_sym, 0)) + noise,
                  "rx_pkt2": theory.residual_eps_var(f, S, theory.kernel_poly(f, T, 2)) + theory.noise_term(SIGMA_RX, 2, n_sym),
                  "rx_stale_1": theory.residual_eps_var(f, S, theory.kernel_stale_packet(f, 1.0, T)) + noise}
            s0 = SIGMA_IMU[0]
            th["tx_ins"] = theory.residual_eps_var(f, tx.residual_spectrum(f, s0) + rx.full_spectrum(), np.ones_like(f))
            th["rx_imu"] = theory.residual_eps_var(f, rx.residual_spectrum(f, s0) + tx.full_spectrum(), np.ones_like(f))
            for sg in SIGMA_IMU:
                R = tx.residual_spectrum(f, sg) + rx.residual_spectrum(f, sg)
                th[f"both_{sg:g}"] = theory.residual_eps_var(f, R, np.ones_like(f))
                th[f"split_{sg:g}"] = theory.residual_eps_var(f, R, theory.kernel_poly(f, T, 1)) + noise
                th[f"split0_{sg:g}"] = theory.residual_eps_var(f, R, theory.kernel_poly(f, T, 0)) + noise / 2
            for key, pw in sims.items():
                pw = np.array(pw)
                per = 10 * np.log10(pw[:, 0] / pw[:, 1])
                row = {"day": f"{source}:{name}", "packet_s": T, "strategy": key,
                       "sir_sim_db": float(10 * np.log10(pw[:, 0].sum() / pw[:, 1].sum())),
                       "sir_sim_p10_db": float(np.percentile(per, 10)),
                       "sir_theory_db": theory.ici_sir_db(th[key], o.fc, t_fft)}
                res["rows"].append(row)
                print(f"{name:24s} T {T:.0f} s {key:12s}: SIR sim {row['sir_sim_db']:6.1f} dB (p10 {row['sir_sim_p10_db']:6.1f})"
                      f"  theory {row['sir_theory_db']:6.1f} dB", flush=True)
    with open(os.path.join(output_dir("c2_auv"), "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
