"""
End-to-end chain on real motion (PLAN: Phase 5): signal-only front end vs sensor-aided receiver, against SNR.

Frame (frame.Frame): chirp | guard | 1 header symbol | N data symbols | guard | chirp, sent over the link Doppler of
real buoy motion (wideband warp, single path), with an unknown start (random lead) and complex noise at a given
per-subcarrier SNR. Receivers (chain.py):
  signal_only        preamble + postamble over +-3e-3, one scale per packet, per-symbol residual CFO searched over +-2 subcarriers (Li et al. 2008,
                     the front end of Lee et al. 2026)
  signal_scale_only  the same without the per-symbol CFO step
  aided_<B>          prior = transmitter side information (B bits, allocation for no signal estimate, so that level
                     and slope are sent and the narrow residual search below stays valid;
                     'exact' = the transmitter's unquantised IMU estimate) + the receiver's own IMU; then residual
                     scale from preamble / postamble in a +-3e-4 window and per-symbol residual CFO
  aided_<B>_prior    the prior alone (no signal estimate)
  signal_kalman      signal-only Kalman baseline: preamble/postamble scale, then the smoother of fusion.py over the
                     per-symbol offsets and postamble timing, with wide uncertainty (chain.receive_signal_fused)
  cho_kim            Cho and Kim 2026: mean scale + constant acceleration from cyclic-prefix offsets (best version)
  fused_<B>          sensor-aided Kalman fusion: prior as aided_<B>, then the smoother (chain.receive_fused)
IMUs: 0.01 m/s drift, 50 ms latency bridged, lever arm corrected. The side information is assumed decoded (header
decoding is the next step). Metric: genie SINR on the data symbols (noise included), power-averaged over packets.
Scenarios: buoy -> seabed (CDIP swell, OOI Irminger), buoy -> buoy (OOI); 7 and 27 data symbols (1.05 / 4.05 s);
per-subcarrier SNR 5, 10, 20, 30 dB; 20 packets per point.
Output: results/e2_chain/points.json; printed table.
"""
import json
import os

import numpy as np

from uwofdm import chain, frame, motion, placement as P, sideinfo, theory, waveform
from uwofdm.experiment import NodeIMU, at_times, genie_powers
from uwofdm.paths import output_dir
from c2_sweeps import CDIP_DAY, OOI_DAY, buoy_buoy_link, seabed_link

SNRS_DB = (5, 10, 20, 30)
N_PACKETS = 20
BITS = (16, 32, 64)
SCENARIOS = {"buoy -> seabed, CDIP swell": lambda: seabed_link("cdip", CDIP_DAY),
             "buoy -> seabed, OOI Irminger": lambda: seabed_link("ooi", OOI_DAY, (0.0, 0.0, 2.0)),
             "buoy - buoy, OOI Irminger": buoy_buoy_link}
SIGMA_IMU = 0.01


def setup(tx_track, rx_track, rng):
    d = motion.link_doppler(tx_track, rx_track)
    kw = dict(sigmas=(SIGMA_IMU,), tau_imu=20.0, latency=0.05, extrapolate=True, lever_corrected=True)
    tx = NodeIMU(tx_track, d["u"], +1, 0.0, rng, **kw)
    rx = NodeIMU(rx_track, d["u"], -1, d["tau"], rng, **kw)
    return d, tx, rx


def allocation(F, tx, fs_track, budgets=BITS):
    """Quantiser ranges and stage-aware allocation (per-symbol signal estimates) for the frame's data part."""
    o = F.o
    T = F.length / o.fs
    f, S = theory.spectrum(tx.own + tx.lev, fs_track)
    std = sideinfo.coefficient_std(f, S, T, 24)
    rxm = P.ReceiverModel(packet_s=F.n_data * o.n_symbol / o.fs, t_sym=o.n_symbol / o.fs, t_fft=o.n_fft / o.fs,
                          fc=o.fc)
    alloc = {}
    for stage in ("sym", "none"):  # per-symbol signal estimates follow / prior used alone
        t, ts, W, win = P.stage_estimator(stage, rxm)
        w = sideinfo.stage_weights(t, ts, W, win, o.fc, 24)
        alloc.update({(b, stage): sideinfo.allocate_weighted(std, w, b) for b in budgets})
    return sideinfo.LOAD * std, alloc


def run(label, F, d, tx, rx, rng):
    o = F.o
    fs_track, t_day, eps = tx.fs, d["times"], d["eps"]
    ranges, alloc = allocation(F, tx, fs_track)
    T = F.length / o.fs
    acc = {}
    starts = rng.uniform(t_day[0] + 40, t_day[-1] - T - 20, N_PACKETS)
    for t0 in starts:
        rng_p = np.random.default_rng(rng.integers(1 << 30))
        X = o.random_symbols(F.n_data, rng_p)
        s = F.build(o.random_symbols(F.n_header, rng_p), X)
        lead, tail = int(rng_p.integers(500, 3000)), 8000
        x = np.concatenate([np.zeros(lead, complex), s, np.zeros(tail, complex)])
        tg = t0 - lead / o.fs + np.arange(len(x)) / o.fs  # emission times of every sample
        y0 = waveform.warp(x, o.fs, at_times(eps, t_day, fs_track, tg), o.fc)
        te = t0 + np.arange(F.length + tail // 2) / o.fs  # emission grid of the frame (prior index 0 = preamble)
        tx_est, rx_est = tx.estimate(te, SIGMA_IMU), rx.estimate(te, SIGMA_IMU)
        c = sideinfo.project(tx_est[:F.length], te[:F.length] - t0, T, 24)
        priors = {("exact", st): tx_est + rx_est for st in ("sym", "none")}
        for b in BITS:
            for st in ("sym", "none"):
                q = sideinfo.quantise_alloc(c, alloc[(b, st)], ranges)
                priors[(str(b), st)] = sideinfo.reconstruct(q, np.minimum(te - t0, T), T) + rx_est
        for snr in SNRS_DB:
            sigma = np.sqrt(4.0 / 10 ** (snr / 10) / 2)  # per-subcarrier SNR (demodulated noise = var / 4)
            y = y0 + sigma * (rng_p.standard_normal(len(y0)) + 1j * rng_p.standard_normal(len(y0)))
            search = (0, lead + 1500)
            out = {"signal_only": chain.receive_signal_only(y, F, search)[0],
                   "signal_scale_only": chain.receive_signal_only(y, F, search, cfo=False)[0],
                   "signal_kalman": chain.receive_signal_fused(y, F, search)[0],
                   "cho_kim": chain.receive_cho_kim(y, F, search)[0]}
            for (name, st), pr in priors.items():  # 'none' allocation: level and slope first, so the narrow
                if st == "none":                       # residual search of receive_aided stays valid
                    out[f"aided_{name}"] = chain.receive_aided(y, F, pr, search)[0]
                    out[f"aided_{name}_prior"] = chain.receive_aided(y, F, pr, search, prepost=False, cfo=False)[0]
                    out[f"fused_{name}"] = chain.receive_fused(y, F, pr, search)[0]
            for k, Y in out.items():
                acc.setdefault((snr, k), []).append(genie_powers(Y, X, F.used))
    res = {}
    for (snr, k), pw in acc.items():
        pw = np.array(pw)
        res[(snr, k)] = float(10 * np.log10(pw[:, 0].sum() / pw[:, 1].sum()))
    return res


def main():
    out = output_dir("e2_chain")
    rows, rng = [], np.random.default_rng(0)
    for label, make in SCENARIOS.items():
        d, tx, rx = setup(*make(), rng)
        for n_data in (7, 27):
            F = frame.Frame(n_data=n_data)
            res = run(label, F, d, tx, rx, rng)
            for (snr, k), v in sorted(res.items()):
                rows.append(dict(scenario=label, n_data=n_data, snr_db=snr, receiver=k, sinr_db=v))
            for snr in SNRS_DB:
                line = ", ".join(f"{k} {res[(snr, k)]:5.1f}" for k in
                                 ("signal_only", "signal_kalman", "cho_kim", "aided_32", "fused_32", "fused_exact",
                                  "aided_32_prior"))
                print(f"{label:30s} {n_data:2d} sym  SNR {snr:2d}: {line}", flush=True)
            with open(os.path.join(out, "points.json"), "w", encoding="utf-8") as fh:
                json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
