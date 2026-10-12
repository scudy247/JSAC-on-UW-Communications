"""Realistic motion-sensor error model (imu.py) and its use through NodeIMU / IMU3."""
from dataclasses import replace

import numpy as np

from uwofdm import imu, motion, sensors
from uwofdm.experiment import NodeIMU


def _track(n=12000, fs=10.0, lever=False):
    t = np.arange(n) / fs
    v = np.stack([0.3 * np.sin(2 * np.pi * 0.11 * t), 0.2 * np.cos(2 * np.pi * 0.09 * t),
                  0.8 * np.sin(2 * np.pi * 0.13 * t + 1)], 1)
    lev = np.stack([0.1 * np.sin(2 * np.pi * 0.2 * t), np.zeros(n), np.zeros(n)], 1) if lever else None
    return motion.Track(fs, v, np.zeros((n, 3)), np.array([0.0, 0.0, -10.0]), vel_lever=lev)


TT = np.linspace(300, 1100, 2000)


def _err(tr, grade, seed=0, **kw):
    m = imu.RealisticIMU(tr, grade, np.random.default_rng(seed), **kw)
    return m.velocity(TT) - tr.velocity_at(TT, "total")


def test_zero_errors_and_no_latency_give_the_truth():
    """Complementary filter with an exact aid: HP + (1 - HP) = 1, so the estimate is the truth (lever included)."""
    tr = _track(lever=True)
    assert np.max(np.abs(_err(tr, imu.ZERO, latency=0.0))) < 1e-9


def test_latency_bridged_by_measured_acceleration():
    tr = _track()
    e_ext = _err(tr, imu.ZERO, latency=0.05)
    e_raw = _err(tr, imu.ZERO, latency=0.05, extrapolate=False)
    assert np.std(e_ext) < 0.1 * np.std(e_raw)


def test_tilt_leaks_into_horizontal_not_vertical():
    g = replace(imu.ZERO, tilt_std=0.25 * imu.DEG, tilt_tau=10.0, hp_hz=0.03)
    e = _err(_track(), g, latency=0.0)
    assert np.std(e[:, :2]) > 0.01  # g * 4.4e-3 rad integrated
    assert np.std(e[:, 2]) < 0.1 * np.std(e[:, :2])


def test_unaided_filter_keeps_waves_loses_the_low_band():
    """1st-order high-pass at 0.005 Hz: a 0.09-0.13 Hz wave is kept within ~6 % (phase lead ~ fc / f)."""
    g = replace(imu.ZERO, aid_sigma=None, unaided_hp_hz=0.005)
    tr = _track()
    e = _err(tr, g, latency=0.0)
    assert np.std(e, 0).max() < 0.07 * np.std(tr.velocity_at(TT, "total"), 0).max()


def test_grades_ordered_and_reproducible():
    tr = _track()
    s = [np.std(_err(tr, imu.GRADES[k])[:, :2]) for k in ("nav", "industrial", "consumer")]
    assert s[0] < s[1] < s[2]
    assert np.array_equal(_err(tr, imu.GRADES["industrial"], seed=3), _err(tr, imu.GRADES["industrial"], seed=3))


def test_nodeimu_and_imu3_use_the_grade():
    tr = _track()
    u = np.array([1.0, 0.0, 0.0])
    node = NodeIMU(tr, u, +1, 0.0, np.random.default_rng(0), sigmas=(0.01,), latency=0.0, grade=imu.ZERO)
    truth = tr.velocity_at(TT, "total") @ u / motion.C
    assert np.max(np.abs(node.estimate(TT, 0.01) - truth)) < 1e-12
    i3 = sensors.IMU3(tr, np.random.default_rng(0), latency=0.0, grade=imu.ZERO)
    assert np.max(np.abs(i3.velocity(TT) - tr.velocity_at(TT, "total"))) < 1e-9
