"""Motion, link Doppler and estimation input model: sign conventions, split, delay, error statistics, real data."""
import glob

import numpy as np
import pytest

from uwofdm import estimates, motion
from uwofdm.paths import data_dir

C = motion.C


def test_sign_convention():
    """Approaching gives eps > 0 whichever node moves; receding gives eps < 0."""
    fs, n = 10.0, 100
    a = motion.static_track(fs, n, (0, 0, -10))
    b = motion.static_track(fs, n, (1000, 0, -10))
    a_north = motion.auv_leg(fs, n, (0, 0, -10), 1.0, 0.0)  # towards b
    b_north = motion.auv_leg(fs, n, (1000, 0, -10), 1.0, 0.0)  # away from a
    assert np.allclose(motion.link_doppler(a_north, b)["eps"], 1 / C)
    assert np.allclose(motion.link_doppler(a, b_north)["eps"], -1 / C)
    assert np.allclose(motion.link_doppler(a_north, b_north)["eps"], 0)


def test_split_sums_and_geometry():
    fs, n = 10.0, 2000
    rng = np.random.default_rng(0)
    a = motion.Track(fs, rng.standard_normal((n, 3)) * 0.3, np.tile([0.1, 0.0, 0.0], (n, 1)), np.array([0, 0, 0.]))
    b = motion.Track(fs, rng.standard_normal((n, 3)) * 0.3, np.tile([0.0, 0.2, 0.0], (n, 1)), np.array([600, 800, -50.]))
    d = motion.link_doppler(a, b)
    assert np.isclose(d["tau"], np.linalg.norm([600, 800, -50]) / C)
    assert np.isclose(np.linalg.norm(d["u"]), 1)
    assert np.allclose(d["eps"], d["eps_tx"] + d["eps_rx"] + d["eps_slow"])


def test_receiver_motion_taken_at_reception_time():
    """The receiver's velocity enters at t + tau, the transmitter's at t."""
    fs, n = 100.0, 2000  # 20 s
    a = motion.static_track(fs, n, (0, 0, 0))
    b = motion.static_track(fs, n, (1500, 0, 0))  # tau = 1 s
    b.vel_fast[b.t >= 5.0, 0] = 1.0  # receiver starts moving away at t = 5 s
    d = motion.link_doppler(a, b)
    switch = d["times"][np.argmax(d["eps_rx"] < -0.5 / C)]
    assert abs(switch - 4.0) < 0.02  # emitted at 4 s, received at 5 s


def test_estimates():
    rng = np.random.default_rng(1)
    x = np.sin(np.linspace(0, 20, 20000))
    e = estimates.imu(x, 0.01, rng, bias=0.002) - x
    assert abs(e.mean() - 0.002) < 5e-4 and abs(e.std() - 0.01) < 5e-4
    t = np.arange(20000) / 100.0
    old = estimates.aged(t, x, 0.5)
    assert np.allclose(old[100:], x[:-50][50:])  # x(t - 0.5 s)
    assert np.allclose(estimates.link(t, x, 0.0, 0.5, rng), old)


@pytest.mark.skipif(not glob.glob(str(data_dir("buoy", "cdip", "029p1_swell_*.npz"))), reason="CDIP data not downloaded")
def test_cdip_track_matches_phase0():
    name = sorted(glob.glob(str(data_dir("buoy", "cdip", "029p1_swell_*.npz"))))[0].split("/")[-1][:-4]
    tr = motion.buoy_track("cdip", name, (0, 0, 0))
    assert abs(tr.vel_fast[:, 0].std() - 0.174) < 0.005  # north velocity std found in Phase 0
    assert np.allclose(tr.vel_slow, 0)


def test_gauss_markov_and_lever():
    rng = np.random.default_rng(4)
    x = estimates.gauss_markov(200000, 10.0, 0.02, 5.0, rng)
    assert abs(x.std() - 0.02) < 0.002
    lag = 50  # 5 s
    r = np.corrcoef(x[:-lag], x[lag:])[0, 1]
    assert abs(r - np.exp(-1)) < 0.05
    fs, n = 10.0, 100
    a = motion.static_track(fs, n, (0, 0, 0))
    a.vel_lever = np.tile([0.1, 0.0, 0.0], (n, 1))  # transducer swinging north at 0.1 m/s
    b = motion.static_track(fs, n, (1000, 0, 0))
    assert np.allclose(motion.link_doppler(a, b)["eps"], 0.1 / C)


def test_auv_maneuvering():
    rng = np.random.default_rng(6)
    a = motion.auv_maneuvering(10.0, 6000, (0, 0, -50), rng, speed=1.5, speed_amp=0.0, heave_amp=0.0)
    sp = np.linalg.norm(a.vel_slow, axis=1)
    assert np.allclose(sp, 1.5)
    h = np.degrees(np.arctan2(a.vel_slow[:, 1], a.vel_slow[:, 0]))
    assert abs(h[0] - 45.0) < 1e-6 and abs(abs(h[1499] - h[1199]) - 90.0) < 1.0  # one 30 s turn at 3 deg/s by t = 150 s
