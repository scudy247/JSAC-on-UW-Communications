"""Placement evaluator: node stage as a transfer function, analytic drift, ranking on a synthetic link."""
import numpy as np

from uwofdm import motion, placement, theory


def test_node_filter_matches_extrapolation_kernel():
    fs, f0, L = 10.0, 0.12, 0.3
    t = np.arange(20000) / fs
    x = np.sin(2 * np.pi * f0 * t)
    assert np.allclose(placement.node_filter(x, fs, 0.0, True), x, atol=1e-9)
    r = x - placement.node_filter(x, fs, L, True)
    assert abs(np.mean(r[1000:-1000] ** 2) / 0.5 - theory.kernel_extrapolate(f0, L)) < 1e-3
    r = x - placement.node_filter(x, fs, L, False)
    assert abs(np.mean(r[1000:-1000] ** 2) / 0.5 - theory.kernel_stale(f0, L)) < 1e-3


def test_drift_integral():
    assert abs(placement._drift_var(1.0, 20.0, np.ones_like, 200.0) - 1.0) < 0.01
    # a line over 1 s leaves only the 1/f^2 tail above ~1 Hz: 1 / (pi^2 tau) * (1 / 1 Hz - 1 / f_max)
    k = lambda f: theory.kernel_poly(f, 1.0, 1)
    assert 2e-3 < placement._drift_var(1.0, 20.0, k, 5.0) < 8e-3


def test_ranking_on_synthetic_heave():
    """Transmitter heaving at 0.1 Hz towards a static receiver: an ideal IMU wins without a receiver IMU stage (it
    would only add drift); the receiver's quadratic beats per-symbol estimates over 1 s but not over 8 s."""
    fs, n = 10.0, 30000
    t = np.arange(n) / fs
    v = np.zeros((n, 3))
    v[:, 0] = 0.5 * np.sin(2 * np.pi * 0.1 * t)
    tx = motion.Track(fs, v, np.zeros((n, 3)), np.array([0.0, 0.0, -5.0]))
    rx = motion.static_track(fs, n, (1000.0, 0.0, -5.0))
    s = placement.NodeSensors(sigma=1e-4, latency=0.0)
    fs, comps = placement.link_components(tx, rx, s, s)
    ev = placement.evaluate(fs, comps, placement.ReceiverModel())
    e = {r["name"]: r["e_var"] for r in ev}
    assert ev[0]["tx_imu"] and not ev[0]["rx_imu"]
    assert e["rx_pkt2"] < e["rx_sym"]
    e8 = {r["name"]: r["e_var"] for r in placement.evaluate(fs, comps, placement.ReceiverModel(packet_s=8.0))}
    assert e8["rx_sym"] < e8["rx_pkt2"]
    assert all(r["parts"]["floor"] == 0 for r in ev)


def test_path_floor():
    e = placement.path_floor((0.0, -3.0), (False, True), (0.0, 30.0), 1.0)
    p = 10 ** (-0.3) / (1 + 10 ** (-0.3))
    assert np.isclose(e, p * (2 * 0.5 / motion.C) ** 2)


def test_generic_placements_in_harness():
    """The harness runs any placement name; own[tx]+rx[line2] on a one-sided link equals tx IMU + receiver line, and
    without estimate noise own[tx+rx]+rx[line2] reproduces split exactly."""
    from dataclasses import replace
    from uwofdm.experiment import LinkConfig, parse_placement, run_link
    assert parse_placement("own[tx+rx]+rx[quad]") == (True, True, "quad")
    assert parse_placement("split") is None
    fs, n = 10.0, 6000
    t = np.arange(n) / fs
    v = np.zeros((n, 3))
    v[:, 0] = 0.3 * np.sin(2 * np.pi * 0.1 * t)
    tx = motion.Track(fs, v, np.zeros((n, 3)), np.array([0.0, 0.0, -5.0]))
    rx = motion.static_track(fs, n, (1000.0, 0.0, -50.0))
    cfg = LinkConfig(packet_s=0.6, n_packets=2, sigma_rx=0.0, sigmas_imu=(0.01,),
                     strategies=("split", "own[tx+rx]+rx[line2]", "own[tx]+rx[quad]"))
    res = run_link(tx, rx, cfg, np.random.default_rng(0))
    assert abs(res["split_0.01"]["sir_sim_db"] - res["own[tx+rx]+rx[line2]_0.01"]["sir_sim_db"]) < 1e-6
    assert np.isfinite(res["own[tx]+rx[quad]_0.01"]["sir_theory_db"])
