"""Three-axis IMU model, two-component side information, per-path Doppler prediction."""
import numpy as np

from uwofdm import motion, multipath, sensors


def _track(n=6000, fs=10.0):
    t = np.arange(n) / fs
    v = np.stack([0.3 * np.sin(2 * np.pi * 0.11 * t), 0.2 * np.cos(2 * np.pi * 0.09 * t),
                  0.8 * np.sin(2 * np.pi * 0.13 * t + 1)], 1)
    return motion.Track(fs, v, np.zeros((n, 3)), np.array([0.0, 0.0, -10.0]))


def test_imu3_exact_without_errors_and_drift_stats():
    tr = _track()
    imu = sensors.IMU3(tr, np.random.default_rng(0), sigma=0.0, latency=0.0)
    tt = np.linspace(100, 500, 300)
    assert np.allclose(imu.velocity(tt), tr.velocity_at(tt, "fast"), atol=1e-6)
    noisy = sensors.IMU3(tr, np.random.default_rng(1), sigma=0.01, latency=0.0)
    d = noisy.velocity(tt) - tr.velocity_at(tt, "fast")
    assert 0.002 < np.std(d) < 0.02


def test_predicted_paths_match_true_path_doppler():
    """With exact sensors and enough side-information bits, the predicted per-path Doppler equals the path Doppler of
    multipath.path_eps (direct path at 0 deg, bottom bounce at +-10 deg)."""
    tx, rx = _track(), _track()
    rx.pos0 = np.array([1000.0, 0.0, -10.0])
    rx.vel_fast = rx.vel_fast[::-1].copy()
    d = motion.link_doppler(tx, rx)
    u, tau = d["u"], d["tau"]
    T = 1.3
    t0 = 200.0
    te = t0 + np.arange(0, 1.3, 1 / 2000)
    itx = sensors.IMU3(tx, np.random.default_rng(0), sigma=0.0, latency=0.0)
    irx = sensors.IMU3(rx, np.random.default_rng(0), sigma=0.0, latency=0.0)
    si = sensors.TwoComponentSideInfo(itx, u, T, bits_h=96, bits_z=96)
    q = si.encode(itx.velocity(te), te - t0)
    h, z = si.decode(q, te - t0)
    angles = [(0.0, 0.0), (10.2, -10.2)]
    pred = sensors.predict_paths(h, z, irx.velocity(te + tau), u, angles)
    for p, (s, r) in zip(pred, angles):
        true = multipath.path_eps(tx, rx, u, tau, te, s, r)
        assert np.std(p - true) < 2e-6 and np.std(true) > 3e-5
