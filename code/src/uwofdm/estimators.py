"""
Signal-based Doppler estimators for the end-to-end chain (PLAN: Phase 5): the baselines that sensor information is
compared with, and the signal part of the sensor-aided receiver.

- detect: matched-filter peak of a known waveform in a search window, with sub-sample (parabolic) refinement.
- scale_pre_post: one Doppler scale per packet from the arrival times of preamble and postamble (Sharif et al.
  2000): with the convention of waveform.warp, a feature emitted D samples after the preamble arrives D / (1 + eps)
  samples after it.
- residual_cfo: per OFDM symbol, the frequency offset that minimises the energy on null subcarriers (Li et al. 2008).
"""
import numpy as np


FIRST_PATH = 0.5  # first-arrival rule: earliest correlation peak within this fraction of the maximum


def detect(y, ref, start=0, stop=None, first_path=FIRST_PATH):
    """Position (float sample index, start of ref) and magnitude of the matched-filter peak of ref in y[start:stop].
    first_path: lock on the earliest local peak that reaches this fraction of the maximum (the first significant
    arrival), not on the strongest one, so that preamble and postamble lock on the same path in multipath (with two
    near-equal paths 10.7 ms apart, locking on different paths made a 10.7 ms error in the scale estimate);
    None: strongest peak."""
    stop = len(y) if stop is None else min(stop, len(y))
    seg = y[max(start, 0):stop + len(ref)]
    n = len(seg) + len(ref)
    nfft = 1 << int(np.ceil(np.log2(n)))
    corr = np.fft.ifft(np.fft.fft(seg, nfft) * np.conj(np.fft.fft(ref, nfft)))[:len(seg) - len(ref) + 1]
    a = np.abs(corr)
    i = int(np.argmax(a))
    if first_path is not None and i > 0:
        cand = np.where(a >= first_path * a[i])[0]
        j = int(cand[0])
        w = min(len(a), j + 8)  # local maximum of the first group
        i = j + int(np.argmax(a[j:w]))
    d = 0.0
    if 0 < i < len(a) - 1:
        l, c, r = a[i - 1], a[i], a[i + 1]
        den = l - 2 * c + r
        d = 0.5 * (l - r) / den if den != 0 else 0.0
    return max(start, 0) + i + d, float(a[i] / np.linalg.norm(ref) ** 2)


def scale_pre_post(n_pre, n_post, gap):
    """Mean Doppler scale over the packet from the arrival positions of preamble and postamble."""
    return gap / (n_post - n_pre) - 1.0


def null_energy(seg, o, delta_hz, null_bins=None):
    """Energy on the null bins of one FFT window after removing delta_hz. null_bins: in-band nulls and guards
    (frame.Frame.null_bins); default: every unused bin (guards and out-of-band), which is noise-dominated at low SNR."""
    n = np.arange(len(seg))
    F = np.fft.fft(seg * np.exp(-2j * np.pi * delta_hz * n / o.fs))
    if null_bins is None:
        used = np.zeros(len(F), bool)
        used[o.bins()[o.active]] = True
        return float(np.sum(np.abs(F[~used]) ** 2))
    return float(np.sum(np.abs(F[null_bins]) ** 2))


def residual_cfo(seg, o, span=0.5, n_grid=21, null_bins=None):
    """Frequency offset [Hz] of one FFT window minimising the null-subcarrier energy: grid over +-span subcarrier
    spacings, then a parabola through the best point and its neighbours."""
    grid = np.linspace(-span, span, n_grid) * o.df
    e = np.array([null_energy(seg, o, d, null_bins) for d in grid])
    i = int(np.argmin(e))
    if 0 < i < n_grid - 1:
        l, c, r = e[i - 1], e[i], e[i + 1]
        den = l - 2 * c + r
        if den > 0:
            return float(grid[i] + 0.5 * (l - r) / den * (grid[1] - grid[0]))
    return float(grid[i])


def cp_offset(z, start, o, search=4):
    """Cyclic-prefix correlation of one OFDM symbol starting at sample `start` (prefix first): correlate the prefix
    with the last n_cp samples of the symbol, over +-search samples of timing; returns the residual frequency offset
    [Hz] from the phase (unambiguous within +-df/2 for an FFT length of 1/df) and the normalised correlation."""
    best = (0.0, 0j)
    for d in range(-search, search + 1):
        a = start + d
        if a < 0 or a + o.n_symbol > len(z):
            continue
        cp, tail = z[a:a + o.n_cp], z[a + o.n_fft:a + o.n_fft + o.n_cp]
        c = np.vdot(cp, tail)  # sum conj(cp) * tail: phase = 2 pi delta T_fft
        rho = abs(c) / max(np.sqrt(np.vdot(cp, cp).real * np.vdot(tail, tail).real), 1e-30)
        if rho > best[0]:
            best = (rho, c)
    rho, c = best
    return float(np.angle(c) / (2 * np.pi * o.n_fft / o.fs)), float(rho)


def detect_chirp(y, F, start=0, stop=None):
    """Position of a frame chirp (start of its up-sweep): up- and down-sweeps detected separately, positions averaged
    (their range-Doppler coupling shifts are opposite). Returns (position, mean normalised peak)."""
    n_u, a_u = detect(y, F.chirp_up(), start, stop)
    lo = int(np.floor(n_u + F.n_half)) - 16
    n_d, a_d = detect(y, F.chirp_down(), lo, lo + 33)
    return 0.5 * (n_u + n_d - F.n_half), 0.5 * (a_u + a_d)
