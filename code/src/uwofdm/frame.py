"""
Packet frame for the end-to-end chain (PLAN: Phase 5): preamble, header, OFDM data, postamble.

    | chirp | guard | header symbols | data symbols | guard | chirp |   (each chirp: up-sweep then down-sweep)

The two chirps (linear FM over 90% of the band) give arrival times; the gap between them gives one Doppler scale per
packet (Sharif et al. 2000; the front end of Li et al. 2008 and Lee et al. 2026). Header symbols are ordinary OFDM
symbols that will carry the side information (E1); data symbols carry QPSK with the comb pilots of waveform.OFDM.
All offsets are in samples at o.fs, from the start of the preamble.
"""
from dataclasses import dataclass, field

import numpy as np

from .waveform import OFDM


@dataclass
class Frame:
    o: OFDM = field(default_factory=OFDM)
    n_header: int = 1
    n_data: int = 7
    t_chirp: float = 0.05  # [s]
    t_guard: float = 0.02  # silence after the preamble and before the postamble [s]
    sweep: float = 0.9  # chirp sweep as a fraction of the bandwidth
    null_spacing: int = 16  # every null_spacing-th active subcarrier (offset 2, never a pilot) is left empty

    @property
    def n_chirp(self):
        return int(round(self.t_chirp * self.o.fs))

    @property
    def n_guard(self):
        return int(round(self.t_guard * self.o.fs))

    @property
    def header_start(self):
        return self.n_chirp + self.n_guard

    @property
    def data_start(self):
        return self.header_start + self.n_header * self.o.n_symbol

    @property
    def post_start(self):
        return self.data_start + self.n_data * self.o.n_symbol + self.n_guard

    @property
    def length(self):
        return self.post_start + self.n_chirp

    @property
    def gap(self):
        """Samples from the start of the preamble to the start of the postamble (D in the scale estimate)."""
        return self.post_start

    @property
    def nulls(self):
        """In-band null subcarriers (Li et al. 2008): residual-CFO estimation looks at the energy leaking into them."""
        idx = np.where(self.o.active)[0]
        m = np.zeros(self.o.n_sc, bool)
        m[idx[2::self.null_spacing]] = True
        return m

    @property
    def used(self):
        """Subcarriers that carry header or data symbols (active minus nulls)."""
        return self.o.active & ~self.nulls

    def null_bins(self):
        """FFT bins of the in-band nulls and of the guard subcarriers next to the band."""
        return self.o.bins()[self.nulls | ~self.o.active]

    @property
    def n_half(self):
        return self.n_chirp // 2

    def chirp_up(self):
        """Up-sweep over t_chirp / 2, from -sweep B/2 to +sweep B/2 (unit power)."""
        t = np.arange(self.n_half) / self.o.fs
        k = self.sweep * self.o.bandwidth / (self.n_half / self.o.fs)
        return np.exp(1j * np.pi * (k * t ** 2 - self.sweep * self.o.bandwidth * t))

    def chirp_down(self):
        return np.conj(self.chirp_up())

    def chirp(self):
        """Preamble / postamble: an up-sweep followed by a down-sweep. Each is detected separately (Doppler-tolerant)
        and their positions averaged: a linear FM's arrival time shifts by fc eps / k (range-Doppler coupling), with
        opposite signs for up and down sweeps, so the average is unbiased. With a single up-chirp, a Doppler that
        differs between preamble and postamble biased the packet's mean scale (8.75e-4 vs 8.39e-4 for 0.4 m/s^2
        over 1.3 s). See estimators.detect_chirp."""
        c = np.concatenate([self.chirp_up(), self.chirp_down()])
        return np.pad(c, (0, self.n_chirp - len(c)))

    def build(self, header, data):
        """Baseband frame from header and data symbols (frequency domain, (n, n_sc) each)."""
        s = np.zeros(self.length, complex)
        c = self.chirp()
        s[:self.n_chirp] = c
        header, data = np.array(header, complex), np.array(data, complex)
        header[:, self.nulls] = 0
        data[:, self.nulls] = 0
        s[self.header_start:self.data_start] = self.o.modulate(header)
        s[self.data_start:self.data_start + self.n_data * self.o.n_symbol] = self.o.modulate(data)
        s[self.post_start:] = c
        return s
