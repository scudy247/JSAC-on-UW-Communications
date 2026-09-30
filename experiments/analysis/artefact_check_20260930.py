"""Diagnostic: are SanctSound exceedances shrimp-like or an instrument artefact?"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import soundfile as sf
from scipy import signal

from uwsb.noise.impulsiveness import BandpassStream, robust_sigma

from uwsb.runtime import data_dir

D = str(data_dir() / "noise_census")
FILES = [
    ("S1_fk01", f"{D}/S1_fk01/SanctSound_FK01_01_671359016_20181218T214011Z.flac", (10500.0, 15500.0)),
    ("S2_hi01", f"{D}/S2_hi01/SanctSound_HI01_01_671129638_20181115T000002Z.flac", (20000.0, 30000.0)),
]
DUR_S = 600.0
GAP_S = 1e-3        # exceedances closer than this belong to one event
LSB = 1.0 / 32768.0

for sid, path, band in FILES:
    with sf.SoundFile(path) as f:
        fs = f.samplerate
        x = f.read(int(DUR_S * fs), dtype="float64")
    y = BandpassStream(fs, band)(x)[int(fs):]           # drop 1 s transient
    s = robust_sigma(y)
    idx = np.flatnonzero(np.abs(y - np.median(y)) > 5 * s)
    # cluster exceedances into events
    brk = np.flatnonzero(np.diff(idx) > GAP_S * fs)
    starts = np.r_[idx[0], idx[brk + 1]]
    ends = np.r_[idx[brk], idx[-1]]
    t_ev = starts / fs
    dur_ms = (ends - starts + 1) / fs * 1e3
    peak = np.array([np.max(np.abs(y[a:b + 1])) for a, b in zip(starts, ends)]) / s
    iei = np.diff(t_ev)
    cv = iei.std() / iei.mean()
    # periodicity: autocorrelation of event counts in 1 ms bins up to 2 s
    counts = np.bincount((t_ev * 1000).astype(int))
    c = counts - counts.mean()
    ac = signal.correlate(c, c, mode="full", method="fft")[c.size - 1:]
    ac /= ac[0]
    lag_peak = np.argmax(ac[5:2000]) + 5
    # event spectrum vs quiet spectrum on the raw (unfiltered) signal
    win = int(0.004 * fs)
    ev_mask = np.zeros(x.size - int(fs), bool)
    for a in starts[:5000]:
        ev_mask[max(a - win // 4, 0):a + win] = True
    raw = x[int(fs):]
    nper = 1024
    f_ev, p_ev = signal.welch(raw[ev_mask], fs, nperseg=nper)
    quiet = raw[~ev_mask]
    f_q, p_q = signal.welch(quiet, fs, nperseg=nper)
    p_quant = (LSB ** 2 / 12) / (fs / 2)                  # one-sided quantisation PSD
    sel = lambda f, lo, hi: (f >= lo) & (f <= hi)
    print(f"== {sid} fs={fs} band={band} first {DUR_S:.0f} s")
    print(f"  robust sigma = {s / LSB:.2f} LSB; events = {starts.size} ({starts.size / (DUR_S - 1):.1f}/s)")
    print(f"  inter-event interval: mean {iei.mean() * 1e3:.1f} ms, CV = {cv:.2f} (Poisson 1, periodic << 1)")
    print(f"  event duration ms p50/p90 = {np.percentile(dur_ms, 50):.2f}/{np.percentile(dur_ms, 90):.2f}")
    print(f"  peak/robust sigma p50/p90/p99/max = " + "/".join(f"{v:.0f}" for v in np.percentile(peak, [50, 90, 99, 100])))
    print(f"  count autocorr: max over lags 5 ms-2 s = {ac[lag_peak]:.3f} at {lag_peak} ms")
    for lo, hi in [(1e3, 5e3), (5e3, 10e3), (10e3, 16e3), (16e3, 22e3), (22e3, 30e3), (30e3, 45e3)]:
        if hi < fs / 2:
            m1, m2 = sel(f_ev, lo, hi), sel(f_q, lo, hi)
            print(f"  {lo/1e3:4.0f}-{hi/1e3:2.0f} kHz: events/quiet PSD = {10*np.log10(p_ev[m1].mean()/p_q[m2].mean()):5.1f} dB;"
                  f" quiet/quantisation = {10*np.log10(p_q[m2].mean()/p_quant):5.1f} dB")
