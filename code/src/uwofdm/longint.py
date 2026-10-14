"""
Long coherent integration at low SNR, at waveform level (PLAN: Phase 5, candidate B; check of the phase-level study
scripts/b20_long_integration.py, where the frequency-estimation threshold was modelled).

Waveform: a long direct-sequence (DSSS) transmission, BPSK chips at chip_rate, rectangular chips, in complex baseband
at fs, warped by the link Doppler (waveform.warp, wideband). Noise from the carrier-to-noise density: unit signal
power, complex noise variance fs / CNR per sample.

Receivers (identical except for the prior):
  1. coarse correction: resample along a mean Doppler over the packet (what a preamble / postamble pair gives; the
     same value for both receivers) — signal-only — or along the sensor prior (side information + own IMU) —
     sensor-aided;
  2. per segment of t_seg: despread with the known chips (y conj(c)), maximum-likelihood frequency = peak of a
     zero-padded FFT of the despread segment (at low SNR this gives genuine outliers: the threshold is not
     modelled, it happens); residual Doppler scale = frequency / fc;
  3. Kalman / RTS smoother (fusion.smooth) over the segment measurements with their Cramer-Rao variance, then
     outlier gating (measurements beyond GATE sigma of the smoothed track are dropped, smoothing repeated);
  4. final resampling along the estimated trajectory.
Metric: coherent-integration efficiency of the compensated noise-free signal over windows of t_int,
E |mean over the window of z0 conj(c)|^2 — the quality of the Doppler estimate alone, comparable with the
phase-level study.
"""
import numpy as np

from . import fusion
from .waveform import unwarp, warp

GATE = 4.0


def dsss(n_chips, chip_rate, fs, rng):
    """Unit-power BPSK chips, rectangular, at fs (an integer number of samples per chip)."""
    sps = int(round(fs / chip_rate))
    chips = 1.0 - 2.0 * rng.integers(0, 2, n_chips)
    return np.repeat(chips, sps).astype(complex)


def received(c, fs, fc, eps, cnr_dbhz, rng):
    """Noise-free and noisy received baseband for the transmitted chips c and the Doppler trajectory eps
    (per sample)."""
    y0 = warp(c, fs, eps[:len(c)], fc)
    sigma = np.sqrt(fs / 10 ** (cnr_dbhz / 10) / 2)
    return y0, y0 + sigma * (rng.standard_normal(len(y0)) + 1j * rng.standard_normal(len(y0)))


def segment_freqs(z, c, fs, t_seg, pad=8):
    """Per segment: (centre time [s], ML frequency [Hz], peak-to-mean ratio of the periodogram) of the despread
    signal z conj(c)."""
    n = int(round(t_seg * fs))
    out = []
    for a in range(0, len(z) - n + 1, n):
        d = z[a:a + n] * np.conj(c[a:a + n])
        P = np.abs(np.fft.fft(d, pad * n)) ** 2
        k = int(np.argmax(P))
        if 0 < k < len(P) - 1:  # parabolic refinement on the log periodogram
            l_, c_, r_ = np.log(P[k - 1] + 1e-30), np.log(P[k] + 1e-30), np.log(P[(k + 1) % len(P)] + 1e-30)
            den = l_ - 2 * c_ + r_
            k = k + (float(np.clip(0.5 * (l_ - r_) / den, -0.5, 0.5)) if den != 0 else 0.0)
        N = pad * n
        freq = float((k if k < N / 2 else k - N) * fs / N)  # bin -> signed frequency
        out.append(((a + n / 2) / fs, freq, float(P.max() / max(P.mean(), 1e-30))))
    return out


def track(meas, fc, cnr_dbhz, t_seg, t_grid, settings):
    """Kalman / RTS smoother over the segment frequencies (Cramer-Rao variance at the CNR), with one round of
    outlier gating; returns the residual Doppler trajectory on t_grid."""
    cnr = 10 ** (cnr_dbhz / 10)
    var = 3 / (2 * np.pi ** 2 * cnr * t_seg ** 3) / fc ** 2 + (1e-8) ** 2
    ev = [(t, "r", f / fc, var) for t, f, _ in meas]
    if not ev:
        return np.zeros(len(t_grid))
    s = (settings["sigma_r0"], settings["sigma_dr0"], settings["q"])
    r_hat, r_std = fusion.smooth(ev, t_grid, *s)
    pred = np.interp([e[0] for e in ev], t_grid, r_hat)
    sd = np.interp([e[0] for e in ev], t_grid, r_std)
    keep = [e for e, p, d in zip(ev, pred, sd) if abs(e[2] - p) <= GATE * np.sqrt(var + d ** 2)]
    if len(keep) < len(ev):
        r_hat, _ = fusion.smooth(keep, t_grid, *s) if keep else (np.zeros(len(t_grid)), None)
    return r_hat


def efficiency(z0c, c, fs, t_int):
    """E |mean over windows of t_int of z0c conj(c)|^2 (noise-free compensated signal; 1 = perfect)."""
    w = int(round(t_int * fs))
    d = z0c[:len(c)] * np.conj(c[:len(z0c)])
    return float(np.mean([abs(np.mean(d[a:a + w])) ** 2 for a in range(0, len(d) - w + 1, w)]))


def receive(y, y0, c, fs, fc, coarse, cnr_dbhz, t_seg, settings, grid_step=0.05):
    """Steps 1-4 for one receiver. coarse: per-sample Doppler of the coarse correction (a constant mean, or the
    sensor prior). Returns (efficiency function of t_int, estimated trajectory per sample)."""
    z = unwarp(y, fs, coarse[:len(y)], fc)
    meas = segment_freqs(z, c, fs, t_seg)
    t_grid = np.arange(0, len(y) / fs, grid_step)
    r_hat = track(meas, fc, cnr_dbhz, t_seg, t_grid, settings)
    traj = coarse[:len(y)] + np.interp(np.arange(len(y)) / fs, t_grid, r_hat)
    z0c = unwarp(y0, fs, traj, fc)
    return (lambda t_int: efficiency(z0c, c, fs, t_int)), traj
