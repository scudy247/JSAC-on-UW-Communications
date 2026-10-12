"""
C1 / R1 on real motion: interference left by intra-packet Doppler compensation, simulated vs closed form.

Link: a surface buoy (recorded motion, Phase 0 data) transmits to a static seabed receiver 1 km away, 100 m deep; the
Doppler is the buoy's velocity projected on the path. Packets of 1, 2 and 4 s of CP-OFDM (waveform.OFDM defaults:
16 kHz, 8 kHz band, 1024 subcarriers). The receiver has one Doppler estimate per symbol (true value + white error
sigma, README ground rule), fits a trajectory model (constant, linear, quadratic, piecewise constant) and resamples
along it. No noise and a single path, so that only the Doppler residual remains.
Measured: signal-to-interference ratio on the active subcarriers after a genie per-symbol common phase and timing
ramp (what channel estimation removes), i.e. the interference between subcarriers; power-averaged over all symbols
and packets (sir_sim_db) and 10th percentile of the per-packet SIR (sir_sim_p10_db).
Predicted: theory.ici_sir_db with the residual variance from the velocity spectrum of the day, kernel_poly for the
model (piecewise constant: order 0 over one symbol) plus theory.noise_term.

Outputs: results/c1_residual/metrics.json and a printed table. Light: single thread recommended
(OMP_NUM_THREADS=1 nice -n 19).
"""
import json
import os

import numpy as np

from uwofdm import motion, theory, waveform
from uwofdm.compensate import compensate, fit_trajectory
from uwofdm.experiment import genie_powers  # one implementation (ML phase ramp, 9 Oct 2026)
from uwofdm.paths import output_dir

PAD = 4096  # zero samples appended to each packet (128 ms): a stretched or pre-compensated packet must not be
#             cut (with bulk Doppler 1e-3, a 4 s packet stretches by 4 ms)
DAYS = [("cdip", "029p1_swell_2022-01-27", 0), ("cdip", "029p1_wind_2022-07-08", 0), ("ooi", "GI01SUMO/20230919", 3)]
PACKET_S = (1.0, 2.0, 4.0)
MODELS = (0, 1, 2, "piece")
SIGMAS = (0.0, 3e-6)
N_PACKETS = 20
RX_POS = (1000.0, 0.0, -100.0)


def at_times(x, t_day, fs, times):
    """Band-limited interpolation of a motion series sampled at fs (starting at t_day[0]) at arbitrary times; linear
    interpolation would put kinks every sample (0.78 s for CDIP) that real motion does not have."""
    return np.real(waveform.resample_at(np.asarray(x, float), (np.asarray(times, float) - t_day[0]) * fs))


def sir_after_genie(Y, X, active):
    """SIR over a packet [dB] (see genie_powers)."""
    sig, err = genie_powers(Y, X, active)
    return 10 * np.log10(sig / err)


def main():
    o = waveform.OFDM()
    t_sym = o.n_symbol / o.fs
    rng = np.random.default_rng(0)
    res = {"rx_pos": RX_POS, "t_symbol_s": t_sym, "rows": []}
    for source, name, seg in DAYS:
        tr = motion.buoy_track(source, name, (0.0, 0.0, 0.0), segment=seg)
        rx = motion.static_track(tr.fs, tr.n, RX_POS)
        d = motion.link_doppler(tr, rx)
        eps_day, t_day = d["eps"], d["times"]
        f, S = theory.spectrum(eps_day * motion.C, tr.fs)
        for T in PACKET_S:
            n_sym = int(round(T / t_sym))
            starts = np.linspace(t_day[0] + 5, t_day[-1] - T - 5, N_PACKETS)
            for sigma in SIGMAS:
                sims = {str(m): [] for m in MODELS}
                for t0 in starts:
                    X = o.random_symbols(n_sym, rng)
                    s = np.pad(o.modulate(X), (0, PAD))  # room for stretching by Doppler
                    tg = t0 + np.arange(len(s)) / o.fs
                    eps = at_times(eps_day, t_day, tr.fs, tg)
                    y = waveform.warp(s, o.fs, eps, o.fc)
                    t_est = t0 + (np.arange(n_sym) + 0.5) * t_sym
                    e_est = at_times(eps_day, t_day, tr.fs, t_est) + sigma * rng.standard_normal(n_sym)
                    for m in MODELS:
                        z = compensate(y, o.fs, o.fc, fit_trajectory(t_est, e_est, tg, m))
                        sims[str(m)].append(genie_powers(o.demodulate(z, n_sym), X, o.active))
                for m in MODELS:
                    order, span = (0, t_sym) if m == "piece" else (int(m), T)
                    var = theory.residual_eps_var(f, S, theory.kernel_poly(f, span, order)) \
                        + theory.noise_term(sigma, order, 1 if m == "piece" else n_sym)
                    pw = np.array(sims[str(m)])
                    per_packet = 10 * np.log10(pw[:, 0] / pw[:, 1])
                    row = {"day": f"{source}:{name}", "packet_s": T, "n_sym": n_sym, "sigma": sigma, "model": str(m),
                           "sir_sim_db": float(10 * np.log10(pw[:, 0].sum() / pw[:, 1].sum())),
                           "sir_sim_p10_db": float(np.percentile(per_packet, 10)),
                           "sir_theory_db": theory.ici_sir_db(var, o.fc, o.n_fft / o.fs)}
                    res["rows"].append(row)
                    print(f"{name:24s} T {T:.0f} s sigma {sigma:.0e} model {str(m):5s}: "
                          f"SIR sim {row['sir_sim_db']:6.1f} dB (p10 {row['sir_sim_p10_db']:6.1f})  theory {row['sir_theory_db']:6.1f} dB",
                          flush=True)
    with open(os.path.join(output_dir("c1_residual"), "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
