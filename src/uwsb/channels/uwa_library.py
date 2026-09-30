"""Wrapper around the Underwater Acoustic Channel Library (uwa-channels 0.7.1; THEORY-B T-4).

Why a wrapper (reports/library_notes.md §4, §6):
- `uwa_channels.replay` draws a random start from NumPy's global RNG when `start` is None;
  here `start` is always explicit (seconds -> delay-domain samples), drawn from our Generator.
- `replay` validates with `assert` (skipped under `python -O`) and silently returns zeros for a
  start outside the record (out-of-domain interpolation is filled with 0). The range
  0 <= start < T_max - T - L (delay-domain samples; replay.py lines 81-84) is checked here.
- Records are short (red_1 47.8 s, blue_1 52.3 s): `wrap_start_s` loops a continuous timeline
  through the record so consecutive packets see a continuously evolving channel until the wrap.
Output of `replay` is real passband at the input rate, shape (n_out, n_elements).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

from ..runtime import data_dir

# Measured bands from the library paper (§II, Table I); see reports/library_notes.md §2.
KNOWN_BANDS_HZ = {"red": (20_000.0, 30_000.0), "blue": (10_500.0, 15_500.0)}


def wrap_start_s(t_s: float, phase_s: float, max_start_s: float) -> float:
    """Start time in the record for timeline time t_s: (phase + t) looped over [0, max_start)."""
    if not max_start_s > 0:
        raise ValueError("max_start_s must be > 0 (signal longer than the record?)")
    return float((phase_s + t_s) % max_start_s)


class LibraryChannel:
    """One channel file of the library, e.g. LibraryChannel('red_1')."""

    def __init__(self, name: str, path: Path | None = None):
        import uwa_channels
        self.name = name
        self.path = Path(path) if path else data_dir() / "uwa_library" / f"{name}.mat"
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} (set UWSB_DATA_DIR or download the library file)")
        self._uwa = uwa_channels
        self.ch = uwa_channels.load_channel(str(self.path))
        h = np.asarray(self.ch["h_hat"]["real"])
        self.n_time, self.n_elements, self.n_taps = h.shape
        prm = self.ch["params"]
        self.fc_hz = float(np.asarray(prm["fc"]).ravel()[0])
        self.fs_delay_hz = float(np.asarray(prm["fs_delay"]).ravel()[0])
        self.fs_time_hz = float(np.asarray(prm["fs_time"]).ravel()[0])
        self.record_s = self.n_time / self.fs_time_hz
        self.band_hz = KNOWN_BANDS_HZ.get(name.split("_")[0])

    def _n_delay(self, n_in: int, fs_hz: float) -> int:
        frac = Fraction(self.fs_delay_hz / fs_hz).limit_denominator()
        return int(np.ceil(n_in * frac.numerator / frac.denominator))   # resample_poly length

    def max_start_s(self, n_in: int, fs_hz: float) -> float:
        """Largest valid start (exclusive) for an input of n_in samples at fs_hz."""
        t_max = self.n_time / self.fs_time_hz * self.fs_delay_hz
        return (t_max - self._n_delay(n_in, fs_hz) - self.n_taps) / self.fs_delay_hz

    def draw_start_s(self, rng: np.random.Generator, n_in: int, fs_hz: float) -> float:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator (explicit seeding)")
        return float(rng.uniform(0.0, self.max_start_s(n_in, fs_hz)))

    def replay(self, x, fs_hz: float, elements, start_s: float) -> np.ndarray:
        """Replay real passband x (1-D, fs_hz) through the channel, starting at start_s in the record."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("x must be 1-D")
        elements = list(np.atleast_1d(elements))
        if min(elements) < 0 or max(elements) >= self.n_elements:
            raise ValueError(f"elements must be in [0, {self.n_elements - 1}]")
        max_s = self.max_start_s(x.size, fs_hz)
        if not 0.0 <= start_s < max_s:
            raise ValueError(f"start_s={start_s:.4f} outside [0, {max_s:.4f}) s for this input "
                             f"length (record {self.record_s:.2f} s)")
        start = int(np.floor(start_s * self.fs_delay_hz))
        return np.asarray(self._uwa.replay(x, fs_hz, elements, self.ch, start=start))
