"""
Realistic motion-sensor error model (PLAN: Phase 1, sensor model; review of 12 Oct 2026).

The earlier models (experiment.NodeIMU, sensors.IMU3) give the true velocity read `latency` late, extrapolated with
the *true* acceleration, plus a slow Gauss-Markov drift: almost no error at wave frequencies, which is where the
sensor gain comes from. This module builds the velocity estimate the way a motion sensor does, so that its error has
the right spectrum.

Per axis (earth frame north, east, up), on the track's grid:
  measured acceleration  a_m = (1 + k) a + dtheta x f + b(t) + n(t)
      k        scale-factor error, constant per node and axis (std accel_scale)
      dtheta   attitude error (roll, pitch: Gauss-Markov tilt_std / tilt_tau; yaw: yaw_std / tilt_tau); it rotates
               the specific force f = a + g z, so a tilt leaks g dtheta into the horizontal axes (the dominant error
               of strapdown horizontal velocity) and a_h dtheta into the vertical one
      b        bias instability (Gauss-Markov accel_bias / accel_bias_tau); the constant bias is removed by the filter
      n        white noise of density accel_noise [m/s^2/sqrt(Hz)]
  velocity estimate      v_hat = HP(int a_m) + LP(v_aid)         (complementary filter, LP = 1 - HP)
      with aiding (GNSS velocity for a surface buoy, DVL for an AUV): HP a causal 2nd-order Butterworth at hp_hz,
      v_aid = v(t - aid_latency) + white noise of std aid_sigma per sample at aid_rate; aid_latency is the part of the
      aiding latency the filter does not compensate (LP = 1 - HP falls only as 1/f, so it leaks into the wave band).
      Without aiding (aid_sigma None: submerged node, IMU alone) the low band is lost: v_hat = HP(int a_m) with a
      causal 1st-order high-pass at unaided_hp_hz (one integration only); its phase lead distorts swell near the
      corner, as real-time heave filters do (the "5 cm or 5 %" of marine motion reference units).
  transducer             v_hat + (1 + l) v_lever + gyro noise x lever arm      (l: lever-arm length error)
  timing                 the modem reads the estimate with a nominal latency L and an unknown timestamp error dL(t)
                         (Gauss-Markov timing_std / timing_tau, IMU clock vs modem clock), and bridges L with the
                         measured acceleration a_m (not the true one).

Sampling: errors are generated on the track grid (10 Hz OOI, 1.28 Hz CDIP); white-noise terms keep their density, so
on the coarse CDIP grid only their in-band (< 0.64 Hz) part is present, which is the part that integrates into
velocity error.

Grades (GRADES) are representative values of three sensor classes, to be checked against specific datasheets before
they go into the paper; the aided / unaided variants differ only in aid_sigma.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from . import estimates
from .waveform import resample_at

G = 9.81
UG = 1e-6 * G  # 1 micro-g [m/s^2]
DEG = np.pi / 180


@dataclass(frozen=True)
class ImuGrade:
    name: str
    accel_noise: float  # white noise density [m/s^2 / sqrt(Hz)]
    accel_bias: float  # bias instability, Gauss-Markov std [m/s^2]
    accel_bias_tau: float  # [s]
    accel_scale: float  # scale-factor error std (fraction)
    tilt_std: float  # roll / pitch error std [rad]
    yaw_std: float  # heading error std [rad]
    tilt_tau: float  # correlation time of the attitude errors [s]
    gyro_noise: float  # angular random walk density [rad/s / sqrt(Hz)]
    hp_hz: float  # crossover of the complementary filter with aiding (2nd order) [Hz]
    unaided_hp_hz: float = 0.01  # corner of the 1st-order high-pass without aiding [Hz]
    aid_sigma: float = None  # aiding velocity noise per sample [m/s] (GNSS or DVL); None: no aiding
    aid_rate: float = 5.0  # aiding samples per second
    aid_latency: float = 0.0  # uncompensated latency of the aiding samples [s] (the filter time-tags them)
    timing_std: float = 0.0  # IMU-to-modem timestamp error std [s]
    timing_tau: float = 60.0  # [s]
    lever_frac: float = 0.0  # lever-arm length error (fraction)

    def unaided(self):
        return replace(self, name=self.name + "_unaided", aid_sigma=None)


# Representative classes (to verify against datasheets): marine motion reference unit / FOG INS; industrial MEMS
# AHRS; consumer MEMS. Aiding: GNSS velocity for surface nodes.
GRADES = {
    "nav": ImuGrade("nav", accel_noise=20 * UG, accel_bias=10 * UG, accel_bias_tau=300.0, accel_scale=1e-4,
                    tilt_std=0.02 * DEG, yaw_std=0.1 * DEG, tilt_tau=30.0, gyro_noise=0.005 * DEG / 60,
                    hp_hz=0.02, unaided_hp_hz=0.005, aid_sigma=0.02, aid_latency=0.005, timing_std=0.001,
                    lever_frac=0.01),
    "industrial": ImuGrade("industrial", accel_noise=60 * UG, accel_bias=15 * UG, accel_bias_tau=100.0,
                           accel_scale=1e-3, tilt_std=0.25 * DEG, yaw_std=1.0 * DEG, tilt_tau=10.0,
                           gyro_noise=0.2 * DEG / 60, hp_hz=0.03, unaided_hp_hz=0.01, aid_sigma=0.03,
                           aid_latency=0.01, timing_std=0.005, lever_frac=0.02),
    "consumer": ImuGrade("consumer", accel_noise=150 * UG, accel_bias=50 * UG, accel_bias_tau=50.0, accel_scale=5e-3,
                         tilt_std=1.0 * DEG, yaw_std=3.0 * DEG, tilt_tau=5.0, gyro_noise=36 * DEG / 60, hp_hz=0.05,
                         unaided_hp_hz=0.02, aid_sigma=0.05, aid_latency=0.02, timing_std=0.010, lever_frac=0.05),
}
GRADES.update({g.name + "_unaided": g.unaided() for g in list(GRADES.values())})
ZERO = ImuGrade("zero", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, hp_hz=0.02, aid_sigma=0.0, aid_latency=0.0)


def _derivative(x, fs):
    """Time derivative by FFT along axis 0 (the accelerometer's input)."""
    n = len(x)
    X = np.fft.rfft(x - x.mean(0), axis=0)
    return np.fft.irfft(X * (2j * np.pi * np.fft.rfftfreq(n, 1 / fs))[:, None], n, axis=0)


def _highpass(x, fs, fc, order=2):
    """Causal Butterworth high-pass along axis 0, started in steady state on the first sample."""
    sos = butter(order, fc, "highpass", fs=fs, output="sos")
    zi = sosfilt_zi(sos)
    return np.stack([sosfilt(sos, x[:, i], zi=zi * x[0, i])[0] for i in range(x.shape[1])], 1)


def _gm3(n, fs, sigmas, tau, rng):
    return np.stack([estimates.gauss_markov(n, fs, s, tau, rng) if s > 0 else np.zeros(n) for s in sigmas], 1)


def _at(x, t, fs, times):
    """Band-limited interpolation of the columns of x (sampled at t) at arbitrary times, edges held."""
    pad = 64
    pos = (np.asarray(times, float) - t[0]) * fs + pad
    return np.stack([np.real(resample_at(np.pad(x[:, i], pad, mode="edge"), pos)) for i in range(x.shape[1])], 1)


class RealisticIMU:
    """Velocity estimate of a node's transducer (earth frame) from a modelled motion sensor.
    track: motion.Track (its total velocity is the truth; the lever part is the transducer's rotation about the
    sensor); lever_m: lever-arm length [m] for the gyro-noise term (assumed close to vertical, so the gyro noise
    reaches the horizontal axes)."""

    def __init__(self, track, grade, rng, latency=0.05, extrapolate=True, lever_corrected=True, lever_m=None):
        self.grade, self.latency, self.extrapolate = grade, latency, extrapolate
        self.t, self.fs = track.t, track.fs
        n, fs, g = track.n, track.fs, grade
        v = track.vel_fast + track.vel_slow  # motion of the sensor
        a = _derivative(v, fs)
        f = a + np.array([0.0, 0.0, G])  # specific force (earth frame, up positive)
        dth = _gm3(n, fs, (g.tilt_std, g.tilt_std, g.yaw_std), g.tilt_tau, rng)
        scale = g.accel_scale * rng.standard_normal(3)
        white = g.accel_noise * np.sqrt(fs / 2) * rng.standard_normal((n, 3))
        bias = _gm3(n, fs, (g.accel_bias,) * 3, g.accel_bias_tau, rng) + g.accel_bias * rng.standard_normal(3)
        err = np.cross(dth, f) + bias + white
        self.acc = a * (1 + scale) + err  # measured acceleration, earth frame
        # integral of a_m: the true part integrates exactly to v (linear model; a numerical integral of the true
        # acceleration would add a half-sample delay, 0.39 s on the 1.28 Hz CDIP grid); errors by trapezoids
        v_int = v * (1 + scale) + np.concatenate([np.zeros((1, 3)), np.cumsum(0.5 * (err[1:] + err[:-1]), 0)]) / fs
        if g.aid_sigma is None:
            v_hat = _highpass(v_int, fs, g.unaided_hp_hz, order=1)
        else:
            # aiding noise of std aid_sigma per sample at aid_rate, as white noise of the same density on this grid
            sub = np.sqrt(fs / g.aid_rate)
            aid = _at(v, self.t, fs, self.t - g.aid_latency) + g.aid_sigma * sub * rng.standard_normal((n, 3))
            v_hat = _highpass(v_int, fs, g.hp_hz) + aid - _highpass(aid, fs, g.hp_hz)
        if track.vel_lever is not None and lever_corrected:
            arm = 0.0 if lever_m is None else lever_m
            gyro = g.gyro_noise * np.sqrt(fs / 2) * arm * rng.standard_normal((n, 3)) * np.array([1.0, 1.0, 0.0])
            k = 1 + g.lever_frac * rng.standard_normal()
            v_hat = v_hat + k * track.vel_lever + gyro
            self.acc_ext = self.acc + k * _derivative(track.vel_lever, fs)  # bridge the lever part too
        else:
            self.acc_ext = self.acc
        self.v_hat = v_hat
        self.dt = estimates.gauss_markov(n, fs, g.timing_std, g.timing_tau, rng) if g.timing_std > 0 else np.zeros(n)

    def velocity(self, times):
        """(len(times), 3) velocity estimate available at the given node-local times: the estimate L + dL(t) old,
        extrapolated over the nominal L with the measured acceleration."""
        times = np.asarray(times, float)
        tl = times - self.latency - np.interp(times, self.t, self.dt)
        v = _at(self.v_hat, self.t, self.fs, tl)
        if self.extrapolate:
            v = v + self.latency * _at(self.acc_ext, self.t, self.fs, tl)
        return v
