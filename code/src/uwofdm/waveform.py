"""
OFDM waveform in complex baseband and the time warp produced by motion (PLAN: Phase 1).

Wideband Doppler model. A passband signal x(t) = Re{s(t) exp(j 2 pi fc t)} received through motion with Doppler scale
eps(t) is x(w(t)), with w(t) = t + int_0^t eps (eps > 0: approaching, compression). In baseband
    y(t) = s(w(t)) exp(j 2 pi fc (w(t) - t))
`warp` applies this; `unwarp` applies the inverse along a given trajectory, which is what both transmit-side
pre-compensation (before sending, with the transmitter's own motion) and receiver compensation (after reception,
with its estimate) do. eps is given per output sample; the difference between evaluating eps at emission or at
reception time is of order eps^2 and is neglected.

Resampling uses a Kaiser-windowed sinc interpolator, accurate for signals oversampled by OVERSAMPLING.
"""
from dataclasses import dataclass, field

import numpy as np

HALF_TAPS = 16  # interpolator half length [samples]
KAISER_BETA = 8.0


# --------------------------------------------------------------------------- resampling and warp
PHASES = 1024  # fractional positions tabulated for the interpolator
_K = np.arange(-HALF_TAPS + 1, HALF_TAPS + 1)
_TABLE = None


def _table():
    """Windowed-sinc taps for PHASES + 1 fractional offsets in [0, 1] (polyphase table, built once)."""
    global _TABLE
    if _TABLE is None:
        x = np.linspace(0, 1, PHASES + 1)[:, None] - _K[None, :]
        tab = np.sinc(x) * _kaiser(x)
        _TABLE = tab / tab.sum(axis=1, keepdims=True)  # exact unit gain at zero frequency for every phase
    return _TABLE


def resample_at(s, pos, chunk=1 << 15):
    """Band-limited interpolation of s at fractional sample positions pos (zeros outside the record). Taps come from
    a polyphase table with linear interpolation between neighbouring phases."""
    s = np.asarray(s)
    pos = np.asarray(pos, float)
    tab = _table()
    sp = np.concatenate([np.zeros(HALF_TAPS + 1, s.dtype), s, np.zeros(HALF_TAPS + 1, s.dtype)])
    out = np.empty(len(pos), complex if np.iscomplexobj(s) else float)
    for a in range(0, len(pos), chunk):
        p = pos[a:a + chunk]
        i0 = np.floor(p).astype(int)
        u = (p - i0) * PHASES
        j = np.minimum(u.astype(int), PHASES - 1)
        w = (u - j)[:, None]
        taps = tab[j] * (1 - w) + tab[j + 1] * w
        idx = np.clip(i0, -HALF_TAPS, len(s) + HALF_TAPS)[:, None] + _K[None, :] + HALF_TAPS + 1
        idx = np.clip(idx, 0, len(sp) - 1)
        out[a:a + chunk] = np.sum(sp[idx] * taps, axis=1)
    return out


def _kaiser(x):
    """Kaiser window evaluated at continuous offsets x (|x| <= HALF_TAPS), zero outside."""
    r = np.clip(x / HALF_TAPS, -1, 1)
    return np.i0(KAISER_BETA * np.sqrt(1 - r ** 2)) / np.i0(KAISER_BETA) * (np.abs(x) <= HALF_TAPS)


def phase_track(eps, fs):
    """phi(t) = int_0^t eps, sampled on the output grid [s] (trapezoidal)."""
    eps = np.asarray(eps, float)
    return np.concatenate([[0.0], np.cumsum(0.5 * (eps[1:] + eps[:-1]))]) / fs


def warp(s, fs, eps, fc):
    """Received baseband for a Doppler-scale trajectory eps (one value per output sample, same length as s)."""
    phi = phase_track(eps, fs)
    n = np.arange(len(eps))
    return resample_at(s, n + phi * fs) * np.exp(2j * np.pi * fc * phi)


def warp_delayed(s, fs, eps, fc, delay):
    """warp of s delayed by `delay` seconds (fractional: band-limited interpolation), i.e. y(n) = s(n + phi(n) fs -
    delay fs) exp(j 2 pi fc phi(n)); the carrier phase of the delay is left to the path gain. Equals warp of a
    zero-padded s when delay fs is an integer."""
    phi = phase_track(eps, fs)
    n = np.arange(len(eps))
    return resample_at(s, n + phi * fs - delay * fs) * np.exp(2j * np.pi * fc * phi)


def unwarp(y, fs, eps, fc):
    """Inverse of `warp` for the trajectory eps: returns z with warp(z, fs, eps, fc) ~ y. Used for transmit-side
    pre-compensation (z is sent) and for receiver compensation (z is demodulated)."""
    phi = phase_track(eps, fs)
    n = np.arange(len(eps))
    w = n + phi * fs  # w(n): position in z of output sample n; invert w by interpolation
    inv = np.interp(n, w, n)  # w^{-1}(n)
    phi_inv = (n - inv) / fs  # w(t) - t at t = w^{-1}(n)
    return resample_at(y, inv) * np.exp(-2j * np.pi * fc * phi_inv)


# --------------------------------------------------------------------------- OFDM
@dataclass
class OFDM:
    """CP-OFDM numerology in complex baseband. Defaults: 16 kHz carrier, 8 kHz band, 1024 subcarriers (7.8 Hz),
    22 ms prefix, as in Cho-Kim 2026, sampled at OVERSAMPLING x the bandwidth."""
    fc: float = 16000.0
    bandwidth: float = 8000.0
    n_sc: int = 1024
    t_cp: float = 0.022
    oversampling: int = 4
    pilot_spacing: int = 4  # every pilot_spacing-th active subcarrier is a known pilot
    guard_sc: int = 16  # unused subcarriers at each band edge
    active: np.ndarray = field(init=False)

    def __post_init__(self):
        self.active = np.zeros(self.n_sc, bool)
        self.active[self.guard_sc:self.n_sc - self.guard_sc] = True

    @property
    def fs(self):
        return self.oversampling * self.bandwidth

    @property
    def df(self):
        return self.bandwidth / self.n_sc

    @property
    def n_fft(self):
        return self.oversampling * self.n_sc

    @property
    def n_cp(self):
        return int(round(self.t_cp * self.fs))

    @property
    def n_symbol(self):
        return self.n_fft + self.n_cp

    def bins(self):
        """FFT bins of the subcarriers (centred band)."""
        return (np.arange(self.n_sc) - self.n_sc // 2) % self.n_fft

    def pilots(self):
        idx = np.where(self.active)[0]
        mask = np.zeros(self.n_sc, bool)
        mask[idx[::self.pilot_spacing]] = True
        return mask

    def modulate(self, X):
        """X: (n_symbols, n_sc) frequency-domain symbols -> baseband samples with cyclic prefixes."""
        X = np.atleast_2d(X)
        F = np.zeros((len(X), self.n_fft), complex)
        F[:, self.bins()] = X
        x = np.fft.ifft(F, axis=1) * self.n_fft / np.sqrt(self.n_sc)
        return np.concatenate([np.hstack([sym[-self.n_cp:], sym]) for sym in x])

    def demodulate(self, y, n_symbols, start=0, backoff=0):
        """Inverse of modulate (prefix removed), symbol by symbol from sample `start`. backoff: the FFT window starts
        `backoff` samples inside the cyclic prefix, so that a signal arriving slightly early does not let the next
        symbol into the window (the resulting linear phase across subcarriers is part of the estimated channel)."""
        out = np.empty((n_symbols, self.n_sc), complex)
        for k in range(n_symbols):
            a = start + k * self.n_symbol + self.n_cp - backoff
            seg = y[a:a + self.n_fft]
            if len(seg) < self.n_fft:
                seg = np.pad(seg, (0, self.n_fft - len(seg)))
            out[k] = np.fft.fft(seg)[self.bins()] * np.sqrt(self.n_sc) / self.n_fft
        return out

    def random_symbols(self, n_symbols, rng, bits=2, mask=None):
        """QPSK (bits=2) or 16-QAM (bits=4) on the active subcarriers (or on `mask`), unit average power."""
        mask = self.active if mask is None else mask
        X = np.zeros((n_symbols, self.n_sc), complex)
        m = int(np.sqrt(2 ** bits))
        lev = 2 * np.arange(m) - (m - 1)
        norm = np.sqrt(2 * np.mean(lev ** 2))
        n = int(mask.sum())
        X[:, mask] = (rng.choice(lev, (n_symbols, n)) + 1j * rng.choice(lev, (n_symbols, n))) / norm
        return X

    def user_masks(self, n_users, guard):
        """Contiguous subbands for n_users within the active band, separated by `guard` empty subcarriers."""
        idx = np.where(self.active)[0]
        width = (len(idx) - (n_users - 1) * guard) // n_users
        masks = []
        for k in range(n_users):
            m = np.zeros(self.n_sc, bool)
            a = k * (width + guard)
            m[idx[a:a + width]] = True
            masks.append(m)
        return masks
