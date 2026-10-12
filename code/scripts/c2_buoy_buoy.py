"""
C2, scenario 2: buoy to buoy. Both ends are wave-driven, so the link Doppler is the difference of two independent
motions, and each node can only measure its own (README: split compensation).

Link: two surface buoys 1 km apart (north), transducers 10 m deep, so the path is horizontal and the buoys' north
velocities set the Doppler (receiver's taken at reception time, tau = 0.67 s later). Independent buoys in the same sea
state: CDIP, the same day 3 h apart; OOI, two bursts of the same day (with 2 m lever arms, real gyro rates).

Estimates (good, with stated error and age): receiver in-packet SIGMA_RX; each node's IMU as in c2_placement.py (slow
Gauss-Markov drift, latency extrapolated with the measured acceleration, lever arm from the gyro), independent drifts.
Strategies:
  rx_sym, rx_pkt2, rx_stale_1   receiver only (as in c2_placement.py)
  tx_imu                        transmitter removes its own motion while emitting; receiver nothing
  rx_imu                        receiver removes its own motion while receiving; transmitter nothing
  both_imu                      each node removes its own motion
  split, split0                 both_imu + receiver line / constant from two estimates per packet
Theory: residual spectra per node (IMU drift + latency extrapolation + lever-arm extrapolation), plus the full motion
of a node that does not compensate, filtered by each strategy's kernel; cross terms ignored.
Outputs: results/c2_buoy_buoy/metrics.json and a printed table. Single thread, low priority.
"""
import json
import os

import numpy as np

from uwofdm import estimates, motion, theory, waveform
from uwofdm.compensate import compensate, fit_trajectory
from uwofdm.paths import output_dir
from c1_residual import genie_powers
from c2_placement import at_times, spectral_derivative

PAD = 4096  # zero samples appended to each packet (128 ms): a stretched or pre-compensated packet must not be
#             cut (with bulk Doppler 1e-3, a 4 s packet stretches by 4 ms)
DAYS = [("cdip", "029p1_swell_2022-01-27", None), ("cdip", "029p1_wind_2022-07-08", None),
        ("ooi", "GI01SUMO/20230919", (0.0, 0.0, 2.0))]
OOI_SEGMENTS = (3, 10)
CDIP_OFFSET_S = 3 * 3600
PACKET_S = (1.0, 4.0)
SIGMA_IMU = (0.01, 0.05)
TAU_IMU = 20.0
LATENCY = 0.05
SIGMA_RX = 3e-6
N_PACKETS = 20
DEPTH = 10.0
RANGE = 1000.0


class NodeIMU:
    """What a node's IMU tells it about its own share of the link Doppler, on the emission-time grid. sign = +1 for
    the transmitter (eps_tx = +v.u/c), -1 for the receiver (eps_rx = -v(t + tau).u/c, read tau later)."""

    def __init__(self, track, u, sign, shift, rng, with_slow=False):
        """with_slow: the node's navigation system also measures its slow motion (e.g. an AUV's INS + DVL)."""
        self.t = track.t
        self.fs = track.fs
        self.shift = shift
        v = track.velocity_at(track.t, "fast") + (track.velocity_at(track.t, "slow") if with_slow else 0.0)
        self.own = sign * (v @ u) / motion.C
        self.lev = sign * (track.velocity_at(track.t, "lever") @ u) / motion.C
        self.d_own, self.d_lev = spectral_derivative(self.own, self.fs), spectral_derivative(self.lev, self.fs)
        self.drift = {s: estimates.gauss_markov(len(self.t), self.fs, s, TAU_IMU, rng) / motion.C for s in SIGMA_IMU}

    def true_share(self, times):
        return at_times(self.own + self.lev, self.t, self.fs, np.asarray(times) + self.shift)

    def estimate(self, times, sigma):
        tl = np.asarray(times, float) + self.shift - LATENCY
        x = at_times(self.own + self.lev, self.t, self.fs, tl) + LATENCY * at_times(self.d_own + self.d_lev, self.t, self.fs, tl)
        return x + np.interp(np.asarray(times) + self.shift, self.t, self.drift[sigma])

    def residual_spectrum(self, f, sigma):
        _, S_own = theory.spectrum(self.own * motion.C, self.fs)
        _, S_lev = theory.spectrum(self.lev * motion.C, self.fs)
        return theory.gauss_markov_spectrum(f, sigma, TAU_IMU) + (S_own + S_lev) * theory.kernel_extrapolate(f, LATENCY)

    def full_spectrum(self):
        return theory.spectrum((self.own + self.lev) * motion.C, self.fs)[1]


def tracks(source, name, lever):
    a_pos, b_pos = (0.0, 0.0, -DEPTH), (RANGE, 0.0, -DEPTH)
    if source == "ooi":
        a = motion.buoy_track("ooi", name, a_pos, segment=OOI_SEGMENTS[0], lever=lever)
        b = motion.buoy_track("ooi", name, b_pos, segment=OOI_SEGMENTS[1], lever=lever)
        n = min(a.n, b.n)
        for tr in (a, b):
            tr.vel_fast, tr.vel_slow = tr.vel_fast[:n], tr.vel_slow[:n]
            tr.vel_lever = None if tr.vel_lever is None else tr.vel_lever[:n]
        return a, b
    a = motion.buoy_track("cdip", name, a_pos)
    k = int(CDIP_OFFSET_S * a.fs)
    b = motion.Track(a.fs, np.roll(a.vel_fast, -k, axis=0), a.vel_slow.copy(), np.array(b_pos))
    return a, b


def main():
    o = waveform.OFDM()
    t_sym, t_fft = o.n_symbol / o.fs, o.n_fft / o.fs
    rng = np.random.default_rng(0)
    res = {"range": RANGE, "depth": DEPTH, "sigma_rx": SIGMA_RX, "tau_imu": TAU_IMU, "latency": LATENCY, "rows": []}
    for source, name, lever in DAYS:
        a, b = tracks(source, name, lever)
        d = motion.link_doppler(a, b)
        t_day, eps, u, tau = d["times"], d["eps"], d["u"], d["tau"]
        tx, rx = NodeIMU(a, u, +1, 0.0, rng), NodeIMU(b, u, -1, tau, rng)
        f, S = theory.spectrum(eps * motion.C, a.fs)
        for T in PACKET_S:
            n_sym = int(round(T / t_sym))
            starts = np.linspace(t_day[0] + 30, t_day[-1] - T - 5, N_PACKETS)
            sims = {}
            for t0 in starts:
                X = o.random_symbols(n_sym, rng)
                s = np.pad(o.modulate(X), (0, PAD))  # room for stretching by Doppler or by pre-compensation
                tg = t0 + np.arange(len(s)) / o.fs
                e_true = at_times(eps, t_day, a.fs, tg)
                t_est = t0 + (np.arange(n_sym) + 0.5) * t_sym
                e_est = at_times(eps, t_day, a.fs, t_est) + SIGMA_RX * rng.standard_normal(n_sym)
                y = waveform.warp(s, o.fs, e_true, o.fc)

                def run(key, z):
                    sims.setdefault(key, []).append(genie_powers(o.demodulate(z, n_sym), X, o.active))

                run("rx_sym", compensate(y, o.fs, o.fc, fit_trajectory(t_est, e_est, tg, "piece")))
                run("rx_pkt2", compensate(y, o.fs, o.fc, fit_trajectory(t_est, e_est, tg, 2)))
                old = at_times(eps, t_day, a.fs, [t0 - 1.0])[0] + SIGMA_RX * rng.standard_normal()
                run("rx_stale_1", compensate(y, o.fs, o.fc, np.full(len(tg), old)))
                for sg in SIGMA_IMU:
                    pre_tx = tx.estimate(tg, sg)
                    y_tx = waveform.warp(waveform.unwarp(s, o.fs, pre_tx, o.fc), o.fs, e_true, o.fc)
                    rx_own = rx.estimate(tg, sg)
                    if sg == SIGMA_IMU[0]:
                        run("tx_imu", y_tx)
                        run("rx_imu", compensate(y, o.fs, o.fc, rx_own))
                    z = compensate(y_tx, o.fs, o.fc, rx_own)
                    run(f"both_imu_{sg:g}", z)
                    t2 = t_est[[0, -1]]
                    r2 = at_times(eps, t_day, a.fs, t2) - tx.estimate(t2, sg) - rx.estimate(t2, sg) \
                        + SIGMA_RX * rng.standard_normal(2)
                    for key, order in ((f"split_{sg:g}", 1), (f"split0_{sg:g}", 0)):
                        run(key, compensate(y_tx, o.fs, o.fc, rx_own + fit_trajectory(t2, r2, tg, order)))
            noise = SIGMA_RX ** 2
            th = {"rx_sym": theory.residual_eps_var(f, S, theory.kernel_poly(f, t_sym, 0)) + noise,
                  "rx_pkt2": theory.residual_eps_var(f, S, theory.kernel_poly(f, T, 2)) + theory.noise_term(SIGMA_RX, 2, n_sym),
                  "rx_stale_1": theory.residual_eps_var(f, S, theory.kernel_stale_packet(f, 1.0, T)) + noise}
            s0 = SIGMA_IMU[0]
            th["tx_imu"] = theory.residual_eps_var(f, tx.residual_spectrum(f, s0) + rx.full_spectrum(), np.ones_like(f))
            th["rx_imu"] = theory.residual_eps_var(f, rx.residual_spectrum(f, s0) + tx.full_spectrum(), np.ones_like(f))
            for sg in SIGMA_IMU:
                R = tx.residual_spectrum(f, sg) + rx.residual_spectrum(f, sg)
                th[f"both_imu_{sg:g}"] = theory.residual_eps_var(f, R, np.ones_like(f))
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
                print(f"{name:24s} T {T:.0f} s {key:14s}: SIR sim {row['sir_sim_db']:6.1f} dB (p10 {row['sir_sim_p10_db']:6.1f})"
                      f"  theory {row['sir_theory_db']:6.1f} dB", flush=True)
    with open(os.path.join(output_dir("c2_buoy_buoy"), "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
