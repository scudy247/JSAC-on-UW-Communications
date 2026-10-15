"""
Multi-user uplink (PLAN: Phase 6, candidate A of 14 Oct 2026; the archived compensation plan's C3, reframed for
estimation). K users share the OFDM band on contiguous subbands separated by empty guard subcarriers (OFDMA uplink,
Tu, Duman, Stojanovic, Proakis, Allerton 2011 / IEEE JOE 2013); each user's signal reaches the gateway with its own
Doppler (its motion and the gateway's, projected on its own direction), so one resampling cannot serve all of them.
The gateway must estimate K Doppler trajectories, each from 1/K of the band and in the presence of the other users.

UserFrame: the single-user frame restricted to one user's subband. Data, pilots and in-band nulls are the frame's own
ones inside the band; the preamble / postamble chirps sweep the subband only (same power spectral density as the
full-band chirp, so the time-bandwidth product falls by K); the residual-CFO search watches the user's own nulls and
the empty guard subcarriers next to its band (where the neighbours' Doppler leaks in). Every single-user receiver of
chain.py and the coded data path of receiver.py run unchanged on a UserFrame: a gateway branch per user is the
single-user receiver given that user's frame.

Gateway receivers per user (receive_users):
  common         one resampling for all users (single-branch front end): the mean of the users' preamble / postamble
                 scales, then each user's per-symbol residual CFO
  branch_signal  one branch per user (Tu et al., multiple resampling): the user's own preamble / postamble scale and
                 per-symbol CFO (chain.receive_signal_only)
  branch_kalman  the same branch with the signal-only Kalman smoother (chain.receive_signal_fused), the strongest
                 signal-only receiver of the single-user study
  branch_sensor  branches set from priors: the user's side information (its own IMU share; zero for a static node) +
                 the gateway's IMU projected on the user's direction, then the safe fusion (chain.receive_fused_safe);
                 the header is assumed decoded (Phase 4 measured it)
  branch_pilot   the same branch with closed-loop pilot tracking (chain.receive_pilot_track: per-symbol pilot phase
                 rotations, Kalman smoother, three rounds), the strongest signal-only tracker of the single-user study
  branch_sensor_pilot  pilot tracking started from the sensor prior (chain.receive_pilot_track(prior=...)), with
                 the guard of the safe fusion: the output with the higher null-subcarrier SINR between it and
                 branch_pilot is kept (never much worse than the best signal-only tracker)
  branch_genie   the true Doppler of the user, no estimation (what inter-user interference and noise leave)
"""
from dataclasses import dataclass

import numpy as np

from . import chain, receiver, waveform
from .frame import Frame

SINR_FLOOR_DB = -10.0
RECEIVERS = ("common", "branch_signal", "branch_kalman", "branch_pilot", "branch_sensor", "branch_sensor_pilot",
             "branch_genie")


@dataclass
class UserFrame(Frame):
    band: tuple = (0, 0)  # [lo, hi) subcarrier indices of the user
    quiet: tuple = ()  # empty subcarriers next to the band (guards), extra nulls for the residual-CFO search

    @property
    def band_mask(self):
        m = np.zeros(self.o.n_sc, bool)
        m[self.band[0]:self.band[1]] = True
        return m & self.o.active

    @property
    def nulls(self):
        return Frame.nulls.fget(self) & self.band_mask

    @property
    def used(self):
        return self.band_mask & ~self.nulls

    def null_bins(self):
        q = np.zeros(self.o.n_sc, bool)
        q[list(self.quiet)] = True
        return self.o.bins()[self.nulls | q]

    def _sweep(self):
        """(centre frequency [Hz], swept width [Hz]) of the user's chirps."""
        lo, hi = self.band
        return ((lo + hi - 1) / 2 - self.o.n_sc // 2) * self.o.df, self.sweep * (hi - lo) * self.o.df

    def chirp_up(self):
        f0, W = self._sweep()
        t = np.arange(self.n_half) / self.o.fs
        return np.exp(1j * np.pi * (W / (self.n_half / self.o.fs) * t ** 2 - W * t) + 2j * np.pi * f0 * t)

    def chirp_down(self):
        f0, _ = self._sweep()
        t = np.arange(self.n_half) / self.o.fs
        return np.conj(self.chirp_up()) * np.exp(4j * np.pi * f0 * t)

    def chirp(self):
        """Up- then down-sweep over the subband, scaled to the full-band chirp's power spectral density."""
        lo, hi = self.band
        return Frame.chirp(self) * np.sqrt((hi - lo) / self.o.active.sum())


def contiguous(o, K, guard, n_data=7, edge_quiet=8):
    """K UserFrames on contiguous subbands of the active subcarriers, `guard` empty subcarriers between neighbours."""
    idx = np.where(o.active)[0]
    blocks = np.array_split(idx, K)
    frames = []
    for j, b in enumerate(blocks):
        gl = (guard + 1) // 2 if j > 0 else 0
        gr = guard // 2 if j < K - 1 else 0
        lo, hi = int(b[0]) + gl, int(b[-1]) + 1 - gr
        quiet = list(range(int(b[0]), lo)) + list(range(hi, int(b[-1]) + 1))
        if j > 0:
            quiet += list(range(int(blocks[j - 1][-1]) + 1 - guard // 2, int(b[0])))
        else:
            quiet += list(range(max(0, lo - edge_quiet), lo))
        if j < K - 1:
            quiet += list(range(int(blocks[j + 1][0]), int(blocks[j + 1][0]) + (guard + 1) // 2))
        else:
            quiet += list(range(hi, min(o.n_sc, hi + edge_quiet)))
        frames.append(UserFrame(o=o, n_data=n_data, band=(lo, hi), quiet=tuple(sorted(set(quiet)))))
    return frames


def user_packet(F, rng, mod="qpsk"):
    """(info bits, data symbols X, baseband frame) of one user."""
    info = rng.integers(0, 2, receiver.n_info_bits(F, mod))
    X = receiver.build_data(info, F, mod)
    return info, X, F.build(F.o.random_symbols(F.n_header, rng, mask=F.used), X)


def uplink(xs, eps, fs, fc):
    """Gateway baseband (noise-free): sum of the users' padded signals xs[k] warped by their per-sample Doppler
    eps[k]."""
    n = max(len(x) for x in xs)
    y = np.zeros(n, complex)
    for x, e in zip(xs, eps):
        y[:len(x)] += waveform.warp(x, fs, e[:len(x)], fc)
    return y


def add_noise(y, o, snr_db, rng):
    """Noise at snr_db per subcarrier (unit symbol power: variance per sample n_fft / n_sc / SNR)."""
    sigma = np.sqrt(o.n_fft / o.n_sc / 10 ** (snr_db / 10) / 2)
    return y + sigma * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))


def pilot_grid(F):
    """Known pilot symbols of the user's data symbols on the full subcarrier grid (n_data, n_sc)."""
    P = np.zeros((F.n_data, F.o.n_sc), complex)
    P[:, receiver.masks(F)[1]] = receiver.pilot_symbols(F)
    return P


def receive_users(y, frames, priors, truths, searches, eps_max=3e-3, which=RECEIVERS):
    """Demodulated data symbols per receiver and user: {receiver: [Y_k]}, and per-user info. priors[k] / truths[k]:
    the user's sensor prior / true Doppler per emitted sample of its frame; searches[k]: its preamble search window."""
    out, info = {r: [] for r in which}, {r: [] for r in which}
    need = {"common", "branch_signal"} & set(which)
    sig = [chain.receive_signal_only(y, F, s, eps_max=eps_max) for F, s in zip(frames, searches)] if need else []
    eps_c = float(np.mean([i["eps"] for _, i in sig])) if sig else 0.0
    cache = {}

    def pilot(k, F, s):  # signal-only pilot tracking, shared by branch_pilot and the guard of branch_sensor_pilot
        if k not in cache:
            cache[k] = chain.receive_pilot_track(y, F, pilot_grid(F), s, eps_max=eps_max)
        return cache[k]

    def sensor_pilot(k, F, s):
        Ya, ia = chain.receive_pilot_track(y, F, pilot_grid(F), s, eps_max=eps_max, prior=priors[k])
        Yb, ib = pilot(k, F, s)
        sa, sb = chain.null_sinr(Ya, F), chain.null_sinr(Yb, F)
        return (Ya, dict(ia, choice="sensor")) if sa >= sb else (Yb, dict(ib, choice="signal"))

    for k, (F, s) in enumerate(zip(frames, searches)):
        L = int(F.length * (1 + 2 * eps_max)) + 4096
        run = {"common": lambda: chain.receive_aided(y, F, np.full(L, eps_c), s, prepost=False),
               "branch_signal": lambda: sig[k],
               "branch_kalman": lambda: chain.receive_signal_fused(y, F, s, eps_max=eps_max),
               "branch_pilot": lambda: pilot(k, F, s),
               "branch_sensor_pilot": lambda: sensor_pilot(k, F, s),
               "branch_sensor": lambda: chain.receive_fused_safe(y, F, priors[k], s, eps_max=eps_max),
               "branch_genie": lambda: chain.receive_aided(y, F, truths[k], s, prepost=False, cfo=False)}
        for r in which:
            Y, inf = run[r]()
            out[r].append(Y)
            info[r].append({a: b for a, b in inf.items() if np.ndim(b) == 0})
    return out, info


def score(Y, X, info_bits, F, mod="qpsk"):
    """(SINR [dB] after the receiver's own channel estimate, floored at SINR_FLOOR_DB: a receiver that lost lock
    has no channel estimate (signal power 0) and would put -inf into the averages; packet error)."""
    s, e = receiver.sinr(Y, X, F)
    return max(10 * np.log10(max(s, 1e-30) / max(e, 1e-30)), SINR_FLOOR_DB), \
        bool(np.any(receiver.decode(Y, F, mod) != info_bits))
