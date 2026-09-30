"""Real ambient-noise recordings: windowed loading, trimming, band-limiting, resampling, scaling.

Plan §6 Phase 1 task 3: read WAV/FLAC, band-pass to the channel band, resample, scale to a
target in-band SNR WITHOUT per-packet normalisation (impulses preserved), one continuous
timeline. Findings applied here (reports/noise_census.md, group_evidence_phase0.md §4):
- duty-cycled SanctSound HI01 files start with ~2.8 s of abnormal (quiet) samples -> `skip_s`;
- a 6-h 48 kHz file is 8 GB as float64 -> always read a window (`start_s`, `duration_s`).
Joining separate files (e.g. HI01's 15-min duty-cycled segments) is allowed (owner decision
2026-09-30, THEORY-B T-6): `load_segments` trims and band-limits each file on its own (no filtering
across a gap) and records the segment boundaries. Default SNR measure: 'robust' (T-6).
Units in names: *_s seconds, *_hz hertz.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from .impulsiveness import robust_sigma

PAD_S = 0.5            # extra read on each side, discarded after zero-phase filtering


@dataclass(frozen=True)
class NoiseWindow:
    samples: np.ndarray          # real passband noise at fs_out_hz (float64)
    fs_out_hz: float
    fs_in_hz: float
    band_hz: tuple[float, float]
    source: str
    start_s: float               # offset of the window within the file, after skip_s
    skip_s: float
    duration_s: float
    clip_frac: float             # raw samples with |x| >= clip_level (before filtering)
    robust_sigma_in_band: float  # of `samples`

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("samples")
        return d


def load_window(path, band_hz, fs_out_hz: float, start_s: float, duration_s: float,
                skip_s: float = 5.0, clip_level: float = 0.999, order: int = 8) -> NoiseWindow:
    """Load `duration_s` seconds starting `skip_s + start_s` into one continuous recording."""
    if isinstance(path, (list, tuple)):
        raise TypeError("load_window reads one file; use load_segments for several")
    path = Path(path)
    lo_hz, hi_hz = band_hz
    if not (duration_s > 0 and start_s >= 0 and skip_s >= 0):
        raise ValueError("need duration_s > 0, start_s >= 0, skip_s >= 0")
    with sf.SoundFile(str(path)) as f:
        fs_in_hz = float(f.samplerate)
        if f.channels != 1:
            raise ValueError(f"{path}: expected mono, got {f.channels} channels")
        if not 0 < lo_hz < hi_hz < min(fs_in_hz, fs_out_hz) / 2:
            raise ValueError(f"band {band_hz} Hz must lie below both Nyquist rates "
                             f"({fs_in_hz / 2}, {fs_out_hz / 2} Hz)")
        first = int(round((skip_s + start_s) * fs_in_hz))
        n_want = int(round(duration_s * fs_in_hz))
        pad = int(round(PAD_S * fs_in_hz))
        a = max(first - pad, 0)
        if first + n_want > f.frames:
            raise ValueError(f"{path}: window ends at {(first + n_want) / fs_in_hz:.1f} s, "
                             f"file has {f.frames / fs_in_hz:.1f} s")
        b = min(first + n_want + pad, f.frames)
        f.seek(a)
        raw = f.read(b - a, dtype="float64")
    clip_frac = float(np.mean(np.abs(raw[first - a:first - a + n_want]) >= clip_level))
    sos = signal.butter(order, band_hz, btype="bandpass", fs=fs_in_hz, output="sos")
    y = signal.sosfiltfilt(sos, raw)[first - a:first - a + n_want]
    frac = Fraction(fs_out_hz / fs_in_hz).limit_denominator(10_000)
    if frac != 1:
        y = signal.resample_poly(y, frac.numerator, frac.denominator)
    return NoiseWindow(samples=y, fs_out_hz=float(fs_out_hz), fs_in_hz=fs_in_hz,
                       band_hz=(float(lo_hz), float(hi_hz)), source=path.name,
                       start_s=float(start_s), skip_s=float(skip_s),
                       duration_s=float(duration_s), clip_frac=clip_frac,
                       robust_sigma_in_band=robust_sigma(y))


@dataclass(frozen=True)
class NoiseTimeline:
    samples: np.ndarray          # concatenated segments at fs_out_hz
    fs_out_hz: float
    boundaries: tuple[int, ...]  # sample index where each segment starts (first is 0)
    segments: tuple[dict, ...]   # NoiseWindow.meta() of each segment


def load_segments(paths, band_hz, fs_out_hz: float, skip_s: float = 5.0,
                  duration_s_per_file: float | None = None, **kw) -> NoiseTimeline:
    """Join duty-cycled segments: each file trimmed and band-limited separately, then concatenated."""
    parts, metas, bounds, n = [], [], [], 0
    for path in paths:
        avail_s = sf.info(str(path)).duration - skip_s
        dur_s = avail_s if duration_s_per_file is None else min(duration_s_per_file, avail_s)
        w = load_window(path, band_hz, fs_out_hz, start_s=0.0, duration_s=dur_s,
                        skip_s=skip_s, **kw)
        bounds.append(n)
        n += w.samples.size
        parts.append(w.samples)
        metas.append(w.meta())
    return NoiseTimeline(samples=np.concatenate(parts), fs_out_hz=float(fs_out_hz),
                         boundaries=tuple(bounds), segments=tuple(metas))


def noise_level(x: np.ndarray, method: str = "robust") -> float:
    """Noise power used for SNR: 'robust' = Gaussian-equivalent (MAD sigma)^2, 'power' = mean square."""
    if method == "robust":
        return robust_sigma(x) ** 2
    if method == "power":
        return float(np.mean(np.asarray(x, float) ** 2))
    raise ValueError(f"method must be 'robust' or 'power', got {method!r}")


def scale_to_snr(x: np.ndarray, signal_power: float, snr_db: float,
                 method: str = "robust") -> np.ndarray:
    """One global gain for the whole timeline so that signal_power / noise_level = snr_db."""
    if not signal_power > 0:
        raise ValueError("signal_power must be > 0")
    level = noise_level(x, method)
    if not level > 0:
        raise ValueError("noise level is zero")
    target = signal_power / 10 ** (snr_db / 10)
    return np.asarray(x, float) * np.sqrt(target / level)
