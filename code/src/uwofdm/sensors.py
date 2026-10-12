"""
Three-axis motion sensors and per-path Doppler prediction (PLAN: Phase 1 per-path compensation; E1/E2).

IMU3: a node's IMU / INS estimate of its transducer velocity vector at any time: the true fast (+ slow) velocity read
`latency` seconds late, extrapolated with the measured acceleration (if extrapolate), plus the lever-arm velocity
from the gyro (if lever_corrected), plus an independent Gauss-Markov drift (std sigma, correlation tau) per axis.
Side information with two components: the transmitter sends its velocity along the horizontal direction of the link
(u_h) and its vertical velocity over the packet, each as quantised Legendre coefficients (sideinfo.py).
Per-path prediction: eps_p(t) = (v_tx(t) . d_dep,p - v_rx(t + tau) . d_arr,p) / c with d = cos(theta) u_h -
sin(theta) z (multipath._directions), from the decoded side information, the receiver's own IMU and each path's
angles from the link geometry (Bellhop).
"""
import numpy as np

from . import estimates, motion, sideinfo, theory
from .experiment import at_times, spectral_derivative
from .multipath import _directions


class IMU3:
    def __init__(self, track, rng, sigma=0.01, tau=20.0, latency=0.05, extrapolate=True, lever_corrected=True,
                 grade=None, lever_m=None):
        """grade: an imu.ImuGrade -> velocity() comes from the realistic sensor model (imu.RealisticIMU), sigma / tau
        unused; None: the original model (true velocity, latency, Gauss-Markov drift)."""
        self.t, self.fs = track.t, track.fs
        self.real = None
        if grade is not None:
            from .imu import RealisticIMU
            self.real = RealisticIMU(track, grade, rng, latency, extrapolate, lever_corrected, lever_m)
        self.latency, self.extrapolate = latency, extrapolate
        lever = track.velocity_at(track.t, "lever") if lever_corrected else 0.0
        self.v = track.velocity_at(track.t, "fast") + track.velocity_at(track.t, "slow") + lever
        self.dv = np.stack([spectral_derivative(self.v[:, i], self.fs) for i in range(3)], 1)
        self.drift = np.stack([estimates.gauss_markov(len(self.t), self.fs, sigma, tau, rng) for _ in range(3)], 1)

    def velocity(self, times):
        """(len(times), 3) estimated velocity at the given (node-local) times."""
        if self.real is not None:
            return self.real.velocity(times)
        times = np.asarray(times, float)
        tl = times - self.latency
        v = np.stack([at_times(self.v[:, i], self.t, self.fs, tl) for i in range(3)], 1)
        if self.extrapolate:
            v = v + self.latency * np.stack([at_times(self.dv[:, i], self.t, self.fs, tl) for i in range(3)], 1)
        return v + np.stack([np.interp(times, self.t, self.drift[:, i]) for i in range(3)], 1)


def horizontal(u):
    uh = np.array([u[0], u[1], 0.0])
    return uh / max(np.linalg.norm(uh), 1e-12)


class TwoComponentSideInfo:
    """Transmitter side information: velocity along u_h and vertical velocity over the packet, K coefficients each,
    bits allocated by coefficient size per component (sideinfo.allocate), ranges from each component's spectrum."""

    def __init__(self, imu, u, T, bits_h=32, bits_z=32, K=24):
        uh = horizontal(u)
        self.uh, self.T, self.K = uh, T, K
        comps = [imu.v @ uh, imu.v[:, 2]]
        self.std, self.ranges, self.alloc = [], [], []
        for x, b in zip(comps, (bits_h, bits_z)):
            f, S = theory.spectrum(x, imu.fs)
            std = sideinfo.coefficient_std(f, S, T, K)
            self.std.append(std)
            self.ranges.append(sideinfo.LOAD * std)
            self.alloc.append(sideinfo.allocate(std, b))
        self.bits = int(sum(a.sum() for a in self.alloc))

    def encode(self, v_est, t_rel):
        """Quantised coefficients of the transmitter's estimated velocity (n, 3) on packet-relative times t_rel."""
        return [sideinfo.quantise_alloc(sideinfo.project(x, t_rel, self.T, self.K), a, r)
                for x, a, r in zip((v_est @ self.uh, v_est[:, 2]), self.alloc, self.ranges)]

    def decode(self, q, t_rel):
        """(horizontal, vertical) velocity of the transmitter rebuilt at packet-relative times."""
        tt = np.minimum(np.maximum(t_rel, 0.0), self.T)
        return sideinfo.reconstruct(q[0], tt, self.T), sideinfo.reconstruct(q[1], tt, self.T)


def predict_paths(tx_h, tx_z, rx_v, u, angles, c=motion.C):
    """Per-path Doppler trajectories from the transmitter's (horizontal, vertical) velocity, the receiver's velocity
    vector (n, 3) and the paths' (departure, arrival) declinations [deg]."""
    uh = horizontal(u)
    out = []
    for src, rcv in angles:
        dd, da = _directions(u, src), _directions(u, rcv)
        v_tx_on_d = tx_h * (dd @ uh) + tx_z * dd[2]
        out.append((v_tx_on_d - rx_v @ da) / c)
    return out
