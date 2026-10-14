"""
Complexity of each receiver (PLAN: Phase 2, complexity counters; candidate C of 14 Oct 2026). Operation counts, not
timings (the shared machine is loaded): the building blocks every receiver calls are instrumented (call counts and
sizes) and converted to real flops with standard cost models:
  FFT / IFFT of N points            5 N log2 N
  band-limited resampling           ~260 flops per output sample (32-tap polyphase interpolator, tap interpolation)
  null-subcarrier energy (CFO grid) 26 flops per input sample (rotation) + its FFT
  correlation (np.correlate)        8 flops per lag and reference sample
  least squares / solve / pinv      ~8 m n^2 (complex)
  Viterbi / max-log-MAP             6 / 18 flops per trellis transition (64 states x 2 per step)
Each receiver runs on the same real-motion packet (buoy-buoy OOI, 30 dB; 1.05 and 4.05 s); the decoder and the
sparse channel estimate are common to all and reported separately. Multipath: the signal-only and sensor-aided
per-path cancellations (mpchain) on one Bellhop two-path packet. Sensor side: the transmitter's side-information
encoding is counted too. Output: results/b19_complexity/points.json.
"""
import json
import os
from contextlib import contextmanager

import numpy as np

from uwofdm import chain, coding, estimators, frame, motion, receiver, sideinfo, waveform
from uwofdm.experiment import at_times
from uwofdm.paths import output_dir
from e2_chain import SCENARIOS, SIGMA_IMU, allocation
from e2_imu_grades import node

FLOPS = {}


def _add(k, v):
    FLOPS[k] = FLOPS.get(k, 0.0) + float(v)


@contextmanager
def instrumented():
    """Wrap the building blocks with counters; restored on exit."""
    saved = dict(fft=np.fft.fft, ifft=np.fft.ifft, rs=waveform.resample_at, ne=estimators.null_energy,
                 corr=np.correlate, lstsq=np.linalg.lstsq, solve=np.linalg.solve, pinv=np.linalg.pinv,
                 vit=coding.viterbi, bcjr=coding.bcjr)

    def fft(a, n=None, *args, **kw):
        N = n or np.shape(a)[-1]
        _add("fft", 5 * N * np.log2(max(N, 2)) * (np.size(a) // max(np.shape(a)[-1], 1)))
        return saved["fft"](a, n, *args, **kw)

    def ifft(a, n=None, *args, **kw):
        N = n or np.shape(a)[-1]
        _add("fft", 5 * N * np.log2(max(N, 2)) * (np.size(a) // max(np.shape(a)[-1], 1)))
        return saved["ifft"](a, n, *args, **kw)

    def rs(s, pos, *a, **k):
        _add("resampling", 260 * len(pos))
        return saved["rs"](s, pos, *a, **k)

    def ne(seg, *a, **k):
        _add("cfo_rotation", 26 * len(seg))
        return saved["ne"](seg, *a, **k)

    def corr(a, v, *x, **k):
        _add("correlation", 8 * len(a) * len(v))
        return saved["corr"](a, v, *x, **k)

    def ls(A, *x, **k):
        m, n_ = np.shape(A)[0], (np.shape(A)[1] if np.ndim(A) > 1 else 1)
        _add("least_squares", 8 * m * n_ * n_)
        return saved["lstsq"](A, *x, **k)

    def sv(A, *x, **k):
        _add("least_squares", 8 * np.shape(A)[0] ** 3 / 3)
        return saved["solve"](A, *x, **k)

    def pv(A, *x, **k):
        m, n_ = np.shape(A)[0], np.shape(A)[1]
        _add("least_squares", 8 * m * n_ * min(m, n_))
        return saved["pinv"](A, *x, **k)

    def vit(llr, n_info):
        _add("decoder", 6 * 128 * (n_info + coding.K - 1))
        return saved["vit"](llr, n_info)

    def bc(llr, n_info):
        _add("decoder", 18 * 128 * (n_info + coding.K - 1))
        return saved["bcjr"](llr, n_info)

    np.fft.fft, np.fft.ifft, waveform.resample_at, estimators.null_energy = fft, ifft, rs, ne
    np.correlate, np.linalg.lstsq, np.linalg.solve, np.linalg.pinv = corr, ls, sv, pv
    coding.viterbi, coding.bcjr = vit, bc
    try:
        yield
    finally:
        np.fft.fft, np.fft.ifft, waveform.resample_at = saved["fft"], saved["ifft"], saved["rs"]
        estimators.null_energy, np.correlate = saved["ne"], saved["corr"]
        np.linalg.lstsq, np.linalg.solve, np.linalg.pinv = saved["lstsq"], saved["solve"], saved["pinv"]
        coding.viterbi, coding.bcjr = saved["vit"], saved["bcjr"]


def measure(fn):
    FLOPS.clear()
    with instrumented():
        fn()
    return dict(FLOPS)


def single_path(n_data):
    tx_track, rx_track = SCENARIOS["buoy - buoy, OOI Irminger"]()
    d = motion.link_doppler(tx_track, rx_track)
    F = frame.Frame(n_data=n_data)
    o = F.o
    T = F.length / o.fs
    tx, rx = node(tx_track, d["u"], +1, 0.0, "industrial", 1), node(rx_track, d["u"], -1, d["tau"], "industrial", 2)
    ranges, alloc = allocation(F, tx, tx_track.fs, budgets=(32,))
    a = alloc[(32, "none")]
    rp = np.random.default_rng(5)
    t0 = d["times"][0] + 400
    info = rp.integers(0, 2, receiver.n_info_bits(F))
    X = receiver.build_data(info, F)
    s = F.build(o.random_symbols(F.n_header, rp), X)
    lead = 1500
    x = np.concatenate([np.zeros(lead, complex), s, np.zeros(8000, complex)])
    tg = t0 - lead / o.fs + np.arange(len(x)) / o.fs
    y = waveform.warp(x, o.fs, at_times(d["eps"], d["times"], tx_track.fs, tg), o.fc)
    y = y + np.sqrt(4e-3 / 2) * (rp.standard_normal(len(y)) + 1j * rp.standard_normal(len(y)))
    te = t0 + np.arange(F.length + 4000) / o.fs
    tx_est = tx.estimate(te[:F.length], SIGMA_IMU)
    c = sideinfo.project(tx_est, te[:F.length] - t0, T, 24)
    prior = sideinfo.reconstruct(sideinfo.quantise_alloc(c, a, ranges), np.minimum(te - t0, T), T) \
        + rx.estimate(te, SIGMA_IMU)
    se = (0, lead + 1500)
    P = np.zeros((F.n_data, o.n_sc), complex)
    P[:, receiver.masks(F)[1]] = receiver.pilot_symbols(F)
    rec = {
        "Li/Lee": lambda: chain.receive_signal_only(y, F, se),
        "Cho-Kim": lambda: chain.receive_cho_kim(y, F, se),
        "signal Kalman": lambda: chain.receive_signal_fused(y, F, se),
        "signal Kalman + pFFT8": lambda: chain._demod_pfft(chain.receive_signal_fused(y, F, se)[1]["z"], F, P, 8),
        "pilot tracking (3 rounds)": lambda: chain.receive_pilot_track(y, F, P, se),
        "sensors + 2 chirp estimates": lambda: chain.receive_aided(y, F, prior, se, cfo=False),
        "sensors + per-symbol CFO": lambda: chain.receive_aided(y, F, prior, se),
        "sensor fusion (Kalman)": lambda: chain.receive_fused(y, F, prior, se),
        "safe sensor fusion (with guard)": lambda: chain.receive_fused_safe(y, F, prior, se),
    }
    out = {k: measure(f) for k, f in rec.items()}
    Y = chain.receive_signal_fused(y, F, se)[0]
    out["common: channel estimate + decoder"] = measure(lambda: receiver.decode(Y, F))
    out["transmitter: side-information encoding"] = measure(lambda: sideinfo.project(tx_est, te[:F.length] - t0, T, 24))
    return out


def multipath():
    from uwofdm import channel, imu, mpchain, multipath as mp, sensors
    from c2_sweeps import buoy_buoy_link
    from e2_multipath import sigma_eta
    tx_track, rx_track = buoy_buoy_link()
    F = frame.Frame(n_data=7)
    o = F.o
    T = F.length / o.fs
    rng = np.random.default_rng(31)
    geo = (1000.0, 10.0, 10.0, 100.0)
    L = mp.LinkMultipath(tx_track, rx_track, geo, sigma_eta(tx_track), tx_track.vel_fast[:, 2], tx_track.fs,
                         cluster=(5, 0.3, 5e-5))
    nominal = mp.nominal_angles(geo, None, surface_rms=sigma_eta(tx_track))
    imu_tx = sensors.IMU3(tx_track, rng, grade=imu.GRADES["industrial"], lever_m=2.0)
    imu_rx = sensors.IMU3(rx_track, rng, grade=imu.GRADES["industrial"], lever_m=2.0)
    si2 = sensors.TwoComponentSideInfo(imu_tx, L.u, T, 32, 32)
    t0 = tx_track.t[0] + 500
    rp = np.random.default_rng(7)
    X = receiver.build_data(rp.integers(0, 2, receiver.n_info_bits(F)), F)
    s = F.build(o.random_symbols(F.n_header, rp), X)
    lead, extra = 1500, int(0.09 * o.fs)
    x = np.concatenate([np.zeros(lead, complex), s, np.zeros(9000, complex)])
    tg = t0 - lead / o.fs + np.arange(len(x) + extra) / o.fs
    bulk, paths = L.build(tg, rp)
    y = channel.apply(x, o.fs, o.fc, bulk, paths, extra=extra)
    y = y + np.sqrt(4e-3 / 2) * (rp.standard_normal(len(y)) + 1j * rp.standard_normal(len(y)))
    te = t0 + np.arange(len(x) + extra - lead) / o.fs
    h, zv = si2.decode(si2.encode(imu_tx.velocity(te[:F.length]), te[:F.length] - t0), te - t0)
    pred = sensors.predict_paths(h, zv, imu_rx.velocity(te + L.tau), L.u, [(a, b) for a, b, _ in nominal])
    se = (0, lead + 1500)
    return {"multipath: signal-only cancellation": measure(lambda: mpchain.receive_sic_signal(y, F, se)),
            "multipath: sensor cancellation (no guard)":
                measure(lambda: mpchain.receive_sic_sensor(y, F, pred, [d for _, _, d in nominal], se, guard=False))}


def main():
    rows = []
    for n_data in (7, 27):
        for k, v in single_path(n_data).items():
            rows.append(dict(case=f"single path, {n_data} symbols", receiver=k, mflops=sum(v.values()) / 1e6,
                             parts={a: b / 1e6 for a, b in v.items()}))
    for k, v in multipath().items():
        rows.append(dict(case="multipath (buoy-buoy two-path), 7 symbols", receiver=k, mflops=sum(v.values()) / 1e6,
                         parts={a: b / 1e6 for a, b in v.items()}))
    for r in rows:
        top = sorted(r["parts"].items(), key=lambda kv: -kv[1])[:3]
        print(f"{r['case']:42s} {r['receiver']:40s} {r['mflops']:9.1f} Mflop  ("
              + ", ".join(f"{a} {b:.0f}" for a, b in top) + ")", flush=True)
    with open(os.path.join(output_dir("b19_complexity"), "points.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
