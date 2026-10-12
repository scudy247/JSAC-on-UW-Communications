"""
C2 / R2 on real motion: where to compensate the buoy's Doppler (receiver, transmitter, split), simulated vs closed form.

Link (scenario 1, decided 9 Oct 2026): a surface buoy (recorded motion) transmits to a static seabed node 1 km away,
100 m deep, so the Doppler is the buoy's transducer velocity projected on the path. On the OOI buoy the transducer is
2 m below the motion sensor: its rotation adds a lever-arm velocity (real gyro data) that the IMU-based correction
does not see. Packets of 1 and 4 s of CP-OFDM (waveform.OFDM defaults).

Estimates (README ground rule: good estimates with stated error and age):
  receiver, in-packet  eps at chosen instants + white error SIGMA_RX
  receiver, stale      one eps value from H seconds before the packet, + SIGMA_RX
  transmitter, IMU     own velocity projected on the path + slow drift (Gauss-Markov, std sigma_imu, TAU_IMU), seen
                       with latency LATENCY and extrapolated across it with the measured acceleration (v + L a);
                       on the OOI buoy, with ('lc') or without ('nolc') the lever-arm velocity from the gyro
Strategies:
  rx_sym        receiver, piecewise per symbol (one estimate per symbol)                  [best C1]
  rx_pkt2       receiver, one quadratic per packet from per-symbol estimates             [C1]
  rx_stale_H    receiver, one stale value held over the packet (H = 1, 2.5 s)            [staleness]
  tx_imu_s      transmitter pre-compensates with its IMU; receiver does nothing
  split_s       tx_imu_s + receiver straight line from two estimates (first and last symbol)
  split0_s      tx_imu_s + receiver constant (mean of the same two estimates)
  (OOI: suffix _lc with lever-arm correction, _nolc without)
Metric and genie as in c1_residual.py (power-averaged SIR after per-symbol common phase and timing ramp).
Theory: theory.ici_sir_db with the residual variance from the spectra: true Doppler (S), IMU drift (Gauss-Markov),
latency (S x stale kernel), lever arm (its own spectrum), filtered by the kernel of each strategy.

Outputs: results/c2_placement/metrics.json and a printed table. Run single-threaded at low priority.
"""
import json
import os

import numpy as np

from uwofdm import estimates, motion, theory, waveform
from uwofdm.compensate import compensate, fit_trajectory
from uwofdm.paths import output_dir
from c1_residual import genie_powers

PAD = 4096  # zero samples appended to each packet (128 ms): a stretched or pre-compensated packet must not be
#             cut (with bulk Doppler 1e-3, a 4 s packet stretches by 4 ms)
DAYS = [("cdip", "029p1_swell_2022-01-27", 0, None), ("cdip", "029p1_wind_2022-07-08", 0, None),
        ("ooi", "GI01SUMO/20230919", 3, (0.0, 0.0, 2.0))]
PACKET_S = (1.0, 4.0)
STALE_H = (1.0, 2.5)
SIGMA_IMU = (0.01, 0.05)  # m/s
TAU_IMU = 20.0
LATENCY = 0.05
SIGMA_RX = 3e-6
N_PACKETS = 20
RX_POS = (1000.0, 0.0, -100.0)


def at_times(x, t_day, fs, times):
    """Band-limited interpolation of a motion series sampled at fs (starting at t_day[0]) at arbitrary times; linear
    interpolation would put kinks every sample (0.78 s for CDIP) that real motion does not have."""
    return np.real(waveform.resample_at(np.asarray(x, float), (np.asarray(times, float) - t_day[0]) * fs))


def spectral_derivative(x, fs):
    """Time derivative by FFT (the IMU measures acceleration directly; finite differences at 1.28 Hz would not)."""
    n = len(x)
    X = np.fft.rfft(x - x.mean())
    return np.fft.irfft(X * 2j * np.pi * np.fft.rfftfreq(n, 1 / fs), n)


def main():
    o = waveform.OFDM()
    t_sym = o.n_symbol / o.fs
    t_fft = o.n_fft / o.fs
    rng = np.random.default_rng(0)
    res = {"rx_pos": RX_POS, "sigma_rx": SIGMA_RX, "tau_imu": TAU_IMU, "latency": LATENCY, "rows": []}
    for source, name, seg, lever in DAYS:
        tr = motion.buoy_track(source, name, (0.0, 0.0, 0.0), segment=seg, lever=lever)
        rx = motion.static_track(tr.fs, tr.n, RX_POS)
        d = motion.link_doppler(tr, rx)
        t_day, eps = d["times"], d["eps"]
        u = d["u"]
        eps_fast = tr.velocity_at(t_day, "fast") @ u / motion.C  # what the IMU sees (no lever arm)
        eps_lever = tr.velocity_at(t_day, "lever") @ u / motion.C
        imu_err = {s: estimates.gauss_markov(len(t_day), tr.fs, s, TAU_IMU, rng) / motion.C for s in SIGMA_IMU}
        d_fast, d_lever = spectral_derivative(eps_fast, tr.fs), spectral_derivative(eps_lever, tr.fs)  # IMU measures
        def seen(times, lc):
            """IMU reading used at `times`: motion LATENCY seconds old, extrapolated with the measured acceleration,
            band-limited (the IMU samples fast); with lc, plus the gyro-based lever-arm velocity."""
            tl = np.asarray(times, float) - LATENCY
            x = at_times(eps_fast, t_day, tr.fs, tl) + LATENCY * at_times(d_fast, t_day, tr.fs, tl)
            if lc:
                x = x + at_times(eps_lever, t_day, tr.fs, tl) + LATENCY * at_times(d_lever, t_day, tr.fs, tl)
            return x

        variants = [("", False)] if lever is None else [("_nolc", False), ("_lc", True)]
        eps_imu = {(sg, v): lc for sg in SIGMA_IMU for v, lc in variants}  # IMU drift added per packet below
        f, S = theory.spectrum(eps * motion.C, tr.fs)
        _, S_fast = theory.spectrum(eps_fast * motion.C, tr.fs)
        _, S_lev = theory.spectrum(eps_lever * motion.C, tr.fs) if lever is not None else (f, np.zeros_like(f))
        for T in PACKET_S:
            n_sym = int(round(T / t_sym))
            starts = np.linspace(t_day[0] + 10, t_day[-1] - T - 5, N_PACKETS)
            sims = {}
            for t0 in starts:
                X = o.random_symbols(n_sym, rng)
                s = np.pad(o.modulate(X), (0, PAD))  # room for stretching by Doppler or by pre-compensation
                tg = t0 + np.arange(len(s)) / o.fs
                e_true = at_times(eps, t_day, tr.fs, tg)
                y = waveform.warp(s, o.fs, e_true, o.fc)
                t_est = t0 + (np.arange(n_sym) + 0.5) * t_sym
                e_est = at_times(eps, t_day, tr.fs, t_est) + SIGMA_RX * rng.standard_normal(n_sym)

                def run(key, z):
                    sims.setdefault(key, []).append(genie_powers(o.demodulate(z, n_sym), X, o.active))

                run("rx_sym", compensate(y, o.fs, o.fc, fit_trajectory(t_est, e_est, tg, "piece")))
                run("rx_pkt2", compensate(y, o.fs, o.fc, fit_trajectory(t_est, e_est, tg, 2)))
                for H in STALE_H:
                    old = at_times(eps, t_day, tr.fs, [t0 - H])[0] + SIGMA_RX * rng.standard_normal()
                    run(f"rx_stale_{H:g}", compensate(y, o.fs, o.fc, np.full(len(tg), old)))
                for (sg, v), lc in eps_imu.items():
                    est = lambda times: seen(times, lc) + np.interp(times, t_day, imu_err[sg])  # drift: slow, linear ok
                    pre = est(tg)
                    y_tx = waveform.warp(waveform.unwarp(s, o.fs, pre, o.fc), o.fs, e_true, o.fc)
                    run(f"tx_imu_{sg:g}{v}", y_tx)
                    t2 = t_est[[0, -1]]
                    r2 = at_times(eps, t_day, tr.fs, t2) - est(t2) + SIGMA_RX * rng.standard_normal(2)
                    run(f"split_{sg:g}{v}", compensate(y_tx, o.fs, o.fc, fit_trajectory(t2, r2, tg, 1)))
                    run(f"split0_{sg:g}{v}", compensate(y_tx, o.fs, o.fc, fit_trajectory(t2, r2, tg, 0)))
            # theory per strategy
            noise = SIGMA_RX ** 2
            th = {"rx_sym": theory.residual_eps_var(f, S, theory.kernel_poly(f, t_sym, 0)) + noise,
                  "rx_pkt2": theory.residual_eps_var(f, S, theory.kernel_poly(f, T, 2)) + theory.noise_term(SIGMA_RX, 2, n_sym)}
            for H in STALE_H:
                th[f"rx_stale_{H:g}"] = theory.residual_eps_var(f, S, theory.kernel_stale_packet(f, H, T)) + noise
            for (sg, v) in eps_imu:
                # independent parts: IMU drift, latency extrapolation error, uncorrected lever arm (cross terms ignored)
                S_res = theory.gauss_markov_spectrum(f, sg, TAU_IMU) + S_fast * theory.kernel_extrapolate(f, LATENCY) \
                    + (S_lev if v == "_nolc" else S_lev * theory.kernel_extrapolate(f, LATENCY))
                th[f"tx_imu_{sg:g}{v}"] = theory.residual_eps_var(f, S_res, np.ones_like(f))
                th[f"split_{sg:g}{v}"] = theory.residual_eps_var(f, S_res, theory.kernel_poly(f, T, 1)) + noise
                th[f"split0_{sg:g}{v}"] = theory.residual_eps_var(f, S_res, theory.kernel_poly(f, T, 0)) + noise / 2
            for key, pw in sims.items():
                pw = np.array(pw)
                per = 10 * np.log10(pw[:, 0] / pw[:, 1])
                row = {"day": f"{source}:{name}", "packet_s": T, "strategy": key,
                       "sir_sim_db": float(10 * np.log10(pw[:, 0].sum() / pw[:, 1].sum())),
                       "sir_sim_p10_db": float(np.percentile(per, 10)),
                       "sir_theory_db": theory.ici_sir_db(th[key], o.fc, t_fft)}
                res["rows"].append(row)
                print(f"{name:24s} T {T:.0f} s {key:14s}: SIR sim {row['sir_sim_db']:6.1f} dB (p10 {row['sir_sim_p10_db']:6.1f})"
                      f"  theory {row['sir_theory_db']:6.1f} dB", flush=True)
    with open(os.path.join(output_dir("c2_placement"), "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
