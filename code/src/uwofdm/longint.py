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
     zero-padded FFT of the despread segment, searched only within the receiver's own uncertainty window +-band_hz
     (signal-only: the spread of the motion around the packet mean; sensor-aided: the prior's error) - a real
     receiver does not search the whole band, and a narrower window means fewer noise peaks, i.e. a lower threshold;
     at low SNR this gives genuine outliers (the threshold is not modelled, it happens); residual Doppler scale =
     frequency / fc;
     the code timing drifts with the residual Doppler (0.8 chip mid-packet for a 3e-4 swing at 0.12 Hz), so each
     segment also searches the chip timing over +-max_shift samples (a code-tracking loop's job), keeping the
     strongest peak;
     tracker="hmm" (15 Oct 2026, a lower-threshold signal-only baseline): instead of deciding each segment alone,
     Viterbi over the segments' periodograms within the window finds the frequency path that maximises the summed
     normalised periodogram minus a Gaussian penalty on the change between segments (std: the rate prior times
     t_seg, times fc) - the classic low-SNR frequency-line tracker (hidden-Markov / Streit-Barrett), whose threshold
     lies several dB below the per-segment maximum's; both receivers can use it;
  3. Kalman / RTS smoother (fusion.smooth) over the segment measurements with their Cramer-Rao variance, then
     outlier gating (measurements beyond GATE sigma of the smoothed track are dropped, smoothing repeated);
  4. second pass (rounds=2): resample along the estimated trajectory, measure what is left, smooth, add (corrects
     the timing drift the first pass worked through); final resampling along the estimated trajectory.
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


def segment_freqs(z, c, fs, t_seg, pad=8, band_hz=None, max_shift=0):
    """Per segment: (centre time [s], ML frequency [Hz], peak-to-mean ratio of the periodogram within the window)
    of the despread signal z conj(c shifted by d samples), d in +-max_shift chosen for the strongest peak; the peak
    is searched within +-band_hz (None: the whole band), widened to at least two resolution cells (2 / t_seg)."""
    n = int(round(t_seg * fs))
    N = pad * n
    f = np.fft.fftfreq(N, 1 / fs)
    inside = np.ones(N, bool) if band_hz is None else np.abs(f) <= max(band_hz, 2 / t_seg)
    out = []
    for a in range(0, len(z) - n + 1, n):
        best = None
        for sh in range(-max_shift, max_shift + 1):
            if a + sh < 0 or a + sh + n > len(c):
                continue
            Pd = np.abs(np.fft.fft(z[a:a + n] * np.conj(c[a + sh:a + sh + n]), N)) ** 2
            if best is None or np.max(Pd[inside]) > np.max(best[inside]):
                best = Pd
        P = best
        Pw = np.where(inside, P, 0.0)
        k = int(np.argmax(Pw))
        freq = float(f[k])
        if inside[k - 1] and inside[(k + 1) % N]:  # parabolic refinement on the log periodogram
            l_, c_, r_ = np.log(P[k - 1] + 1e-30), np.log(P[k] + 1e-30), np.log(P[(k + 1) % N] + 1e-30)
            den = l_ - 2 * c_ + r_
            freq += (float(np.clip(0.5 * (l_ - r_) / den, -0.5, 0.5)) if den != 0 else 0.0) * fs / N
        out.append(((a + n / 2) / fs, freq, float(Pw[k] / max(P[inside].mean(), 1e-30))))
    return out


def segment_spectra(z, c, fs, t_seg, pad=8, band_hz=None, max_shift=0):
    """Per segment the despread periodogram within the search window (the code shift with the strongest peak),
    normalised by its mean over the window: (centre times, frequency grid of the window, (n_seg, n_bins))."""
    n = int(round(t_seg * fs))
    N = pad * n
    f = np.fft.fftfreq(N, 1 / fs)
    inside = np.ones(N, bool) if band_hz is None else np.abs(f) <= max(band_hz, 2 / t_seg)
    order = np.argsort(f[inside])
    times, rows = [], []
    for a in range(0, len(z) - n + 1, n):
        best = None
        for sh in range(-max_shift, max_shift + 1):
            if a + sh < 0 or a + sh + n > len(c):
                continue
            Pd = (np.abs(np.fft.fft(z[a:a + n] * np.conj(c[a + sh:a + sh + n]), N)) ** 2)[inside][order]
            if best is None or Pd.max() > best.max():
                best = Pd
        times.append((a + n / 2) / fs)
        rows.append(best / max(best.mean(), 1e-30))
    return np.array(times), f[inside][order], np.array(rows)


def hmm_freqs(times, f, P, step_std_hz, max_bins=1200):
    """Viterbi frequency path through the normalised periodograms P (n_seg, n_bins): maximise sum_k P[k, path_k] -
    sum_k (f_k - f_(k-1))^2 / (2 step_std^2). The grid is decimated (max-pooled) to at most max_bins. Returns the
    measurement list of segment_freqs: (time, frequency, peak-to-mean)."""
    if len(P) == 0:
        return []
    d = int(np.ceil(P.shape[1] / max_bins))
    if d > 1:
        m = P.shape[1] // d * d
        P = P[:, :m].reshape(len(P), -1, d).max(2)
        f = f[:m].reshape(-1, d).mean(1)
    T = -(f[:, None] - f[None, :]) ** 2 / (2 * max(step_std_hz, 1e-6) ** 2)  # T[prev, next]
    delta = P[0].copy()
    back = np.zeros(P.shape, int)
    for k in range(1, len(P)):
        M = delta[:, None] + T
        back[k] = np.argmax(M, 0)
        delta = M[back[k], np.arange(len(f))] + P[k]
    path = [int(np.argmax(delta))]
    for k in range(len(P) - 1, 0, -1):
        path.append(back[k, path[-1]])
    path = path[::-1]
    out = []
    for k, (t, j) in enumerate(zip(times, path)):  # Viterbi picks the peak; parabolic refinement as segment_freqs
        fk = float(f[j])
        if 0 < j < P.shape[1] - 1:
            l_, c_, r_ = (np.log(P[k, i] + 1e-30) for i in (j - 1, j, j + 1))
            den = l_ - 2 * c_ + r_
            if den != 0:
                fk += float(np.clip(0.5 * (l_ - r_) / den, -0.5, 0.5)) * (f[1] - f[0])
        out.append((float(t), fk, float(P[k, j])))
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


def receive(y, y0, c, fs, fc, coarse, cnr_dbhz, t_seg, settings, band_hz=None, grid_step=0.05, rounds=2,
            max_shift=3, tracker="ml"):
    """Steps 1-4 for one receiver. coarse: per-sample Doppler of the coarse correction (a constant mean, or the
    sensor prior); band_hz: frequency search window around it. Returns (efficiency function of t_int, estimated
    trajectory per sample)."""
    t_grid = np.arange(0, len(y) / fs, grid_step)
    traj = np.array(coarse[:len(y)], float)
    for _ in range(rounds):
        z = unwarp(y, fs, traj, fc)
        if tracker == "hmm":
            ts, f, P = segment_spectra(z, c, fs, t_seg, band_hz=band_hz, max_shift=max_shift)
            meas = hmm_freqs(ts, f, P, fc * settings["sigma_dr0"] * t_seg)
        else:
            meas = segment_freqs(z, c, fs, t_seg, band_hz=band_hz, max_shift=max_shift)
        r_hat = track(meas, fc, cnr_dbhz, t_seg, t_grid, settings)
        traj = traj + np.interp(np.arange(len(y)) / fs, t_grid, r_hat)
    z0c = unwarp(y0, fs, traj, fc)
    return (lambda t_int: efficiency(z0c, c, fs, t_int)), traj
