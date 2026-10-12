"""
Placement rule (PLAN: Phase 3, C2): who should remove which part of the link Doppler.

The link Doppler is a sum of components: the transmitter's own share, the receiver's own share, the slow part (drift,
currents, cruise speed) and, on top, a share specific to each path (surface bounces) that no end can remove. A
placement is a choice of
  - for each node with an IMU / INS: remove its own share live, or not;
  - one receiver stage acting on whatever is left: none, per symbol, constant or line from two estimates per packet,
    quadratic from all per-symbol estimates, or a value fed back from the previous round trip.
Its cost is the residual Doppler-scale variance

    e = sum_c [ int S_c(f) K_rx(f) df + m_c^2 K_rx(0) ] + noise_rx + floor,

where S_c is the spectrum of component c after its own stage (the full share if the node does not compensate it),
m_c its mean, K_rx the receiver stage's kernel (theory.py) and floor the path-specific part. evaluate() enumerates all
placements (at most 2 x 2 x 6) and ranks them; the predicted SIR is theory.ici_sir_db(e).

The node stage is linear, so it is applied as a transfer function to the measured motion rather than as separate
spectra: estimate(t) = G * y(t), with y the node's view of its own share (its velocity projected on the direction it
believes, with or without the lever-arm term) and G = exp(-j w L) (1 + j w L) for a latency L extrapolated with the
measured acceleration (exp(-j w L) if not extrapolated). The residual x - G y keeps the cross terms between latency,
lever arm and direction error that summing their spectra separately (experiment.NodeIMU.residual_spectrum) ignores.
The IMU drift (Gauss-Markov, std sigma, correlation tau) is independent and is integrated analytically.
"""
from dataclasses import dataclass, field
from itertools import product

import numpy as np

from . import motion, theory

STAGES = ("none", "sym", "const2", "line2", "quad", "stale")


@dataclass
class NodeSensors:
    """What a node knows about its own motion. has_imu False: the node cannot compensate."""
    has_imu: bool = True
    sigma: float = 0.01  # velocity drift std [m/s]
    tau: float = 20.0  # drift correlation time [s]
    latency: float = 0.05  # [s]
    extrapolate: bool = True  # bridge the latency with the measured acceleration
    lever_corrected: bool = True  # add omega x r from the gyro
    u_error_deg: float = 0.0  # azimuth error of the path direction it projects on


@dataclass
class ReceiverModel:
    """The receiver's Doppler estimates and the packet. stale_h: age of a fed-back estimate at the packet start
    (None: no feedback)."""
    packet_s: float = 1.0
    t_sym: float = 0.15
    t_fft: float = 0.128
    fc: float = 16e3
    sigma: float = 3e-6  # error of one estimate (Doppler scale)
    stale_h: float = None
    stages: tuple = field(default=STAGES)


def receiver_stage(stage, rx, f):
    """Kernel on f, gain at f = 0 and added noise variance of a receiver stage. The fits use the estimates the receiver
    actually has (theory.kernel_samples): the first and last symbol (const2, line2) or every symbol (quad), over a
    packet of n_sym whole symbols."""
    n_sym = max(1, int(round(rx.packet_s / rx.t_sym)))
    T = n_sym * rx.t_sym
    centres = (np.arange(n_sym) + 0.5) * rx.t_sym
    if stage == "none":
        return np.ones_like(f), 1.0, 0.0
    if stage == "sym":
        return theory.kernel_poly(f, rx.t_sym, 0), 0.0, rx.sigma ** 2
    if stage in ("const2", "line2", "quad"):
        t_s, order = {"const2": (centres[[0, -1]], 0), "line2": (centres[[0, -1]], 1), "quad": (centres, 2)}[stage]
        return (theory.kernel_samples(f, t_s, T, order), 0.0, theory.noise_samples(rx.sigma, t_s, T, order))
    if stage == "stale":
        return theory.kernel_stale_packet(f, rx.stale_h, T), 0.0, rx.sigma ** 2
    raise ValueError(stage)


N_PER_SYMBOL = 75  # grid points per OFDM symbol for the interference kernel (CP 11 points, FFT window 64)


def stage_estimator(stage, rx):
    """The receiver stage as a linear estimator on a grid over the packet: (t_grid, t_samples, W, windows), with W the
    weights from the estimates at t_samples to the compensation trajectory on t_grid (None: no compensation) and
    windows the FFT windows (index ranges). Estimates at symbol centres, as in experiment.run_link."""
    n_sym = max(1, int(round(rx.packet_s / rx.t_sym)))
    n_cp = int(round(N_PER_SYMBOL * (rx.t_sym - rx.t_fft) / rx.t_sym))
    dt = rx.t_sym / N_PER_SYMBOL
    t = np.arange(n_sym * N_PER_SYMBOL + 1) * dt
    windows = [(k * N_PER_SYMBOL + n_cp, (k + 1) * N_PER_SYMBOL + 1) for k in range(n_sym)]
    centres = (np.arange(n_sym) + 0.5) * rx.t_sym
    if stage == "none":
        return t, np.zeros(0), None, windows
    if stage == "sym":
        k = np.minimum((t // rx.t_sym).astype(int), n_sym - 1)
        W = np.zeros((len(t), n_sym))
        W[np.arange(len(t)), k] = 1.0
        return t, centres, W, windows
    if stage in ("const2", "line2", "quad"):
        t_s, order = {"const2": (centres[[0, -1]], 0), "line2": (centres[[0, -1]], 1), "quad": (centres, 2)}[stage]
        scale = t[-1]
        W = np.vander(t / scale, order + 1) @ np.linalg.pinv(np.vander(t_s / scale, order + 1))
        return t, t_s, W, windows
    if stage == "stale":
        return t, np.array([-rx.stale_h]), np.ones((len(t), 1)), windows
    raise ValueError(stage)


_ICI = {}


def ici_kernel(stage, rx, f):
    """theory.kernel_ici of a receiver stage on f (cached: maps reuse each stage and packet length many times)."""
    f = np.asarray(f, float)
    key = (stage, rx.packet_s, rx.t_sym, rx.t_fft, rx.fc, rx.stale_h, len(f), float(f[0]), float(f[-1]))
    if key not in _ICI:
        t, t_s, W, win = stage_estimator(stage, rx)
        _ICI[key] = theory.kernel_ici(f, t, t_s, W, win, rx.fc)
    return _ICI[key]


def ici_noise(stage, rx):
    """Interference from independent estimate errors (std rx.sigma) through the stage's weights."""
    t, _, W, win = stage_estimator(stage, rx)
    if W is None:
        return 0.0
    dt = np.diff(t)[:, None]
    th = 2 * np.pi * rx.fc * np.vstack([np.zeros((1, W.shape[1])), np.cumsum(0.5 * (W[1:] + W[:-1]) * dt, axis=0)])
    v = np.mean([np.sum(np.var(th[i:j], axis=0)) for i, j in win])
    return float(rx.sigma ** 2 * v)


def node_filter(x, fs, latency, extrapolate):
    """G * x: the value `latency` seconds old, extrapolated with its slope if `extrapolate` (FFT, reflected edges)."""
    pad = min(len(x) - 1, 512)
    z = np.pad(np.asarray(x, float), pad, mode="reflect")
    w = 2 * np.pi * np.fft.rfftfreq(len(z), 1 / fs)
    G = np.exp(-1j * w * latency) * ((1 + 1j * w * latency) if extrapolate else 1.0)
    return np.fft.irfft(np.fft.rfft(z) * G, len(z))[pad:len(z) - pad]


def _rotate(u, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1], u[2]])


@dataclass
class Component:
    """One share of the link Doppler on the link grid: the true series, and (if its node can compensate) the series
    left after the node's stage plus the drift std of that stage."""
    name: str
    true: np.ndarray
    after_node: np.ndarray = None
    drift: tuple = None  # (sigma [Doppler scale], tau [s])


def link_components(tx_track, rx_track, tx_sensors, rx_sensors, c=motion.C):
    """Components of the link tx -> rx (motion.link_doppler conventions); returns (fs, [Component])."""
    d = motion.link_doppler(tx_track, rx_track)
    t, u, tau, fs = d["times"], d["u"], d["tau"], tx_track.fs
    comps = []
    for name, tr, tt, sign, sen, true in (("tx", tx_track, t, +1, tx_sensors, d["eps_tx"]),
                                          ("rx", rx_track, t + tau, -1, rx_sensors, d["eps_rx"])):
        comp = Component(name, true)
        if sen is not None and sen.has_imu:
            u_hat = _rotate(u, sen.u_error_deg)
            v = tr.velocity_at(tt, "fast") + (tr.velocity_at(tt, "lever") if sen.lever_corrected else 0.0)
            view = sign * (v @ u_hat) / c
            comp.after_node = true - node_filter(view, fs, sen.latency, sen.extrapolate)
            comp.drift = (sen.sigma / c, sen.tau)
        comps.append(comp)
    comps.append(Component("slow", d["eps_slow"]))
    return fs, comps


def _drift_var(sigma, tau, kernel_fn, f_max):
    """int_0^f_max GM(f) K(f) df on a fine grid (the drift is far narrower than the Welch resolution; f_max: the
    Nyquist frequency of the node's sensor grid, as the first-order model's 1/f^2 tail matters for a fitted line)."""
    f = np.concatenate([np.linspace(0, 0.05, 501), np.geomspace(0.05, f_max, 1500)[1:]])
    S = theory.gauss_markov_spectrum(f, sigma, tau)
    return float(np.trapezoid(S * kernel_fn(f), f))


_SPECTRA = {}


def _series_var(x, fs, kernel_fn, k0):
    key = (id(x), len(x), fs)
    if key not in _SPECTRA:  # spectra are reused across receiver stages and grids (maps evaluate thousands of points)
        _SPECTRA[key] = (x, theory.spectrum(x, fs), float(np.mean(x)))
    _, (f, S), m = _SPECTRA[key]
    return float(np.sum(S * kernel_fn(f)) * (f[1] - f[0]) + m ** 2 * k0)


def path_floor(powers_db, surface, grazing_deg, w_var, c=motion.C):
    """Residual variance from path-specific Doppler: sum_k p_k E[delta_k^2], p normalised, delta_k = 2 sin(g_k) w / c
    on surface-bounced paths (w: vertical velocity of the surface at the bounce point, variance w_var)."""
    p = 10 ** (np.asarray(powers_db, float) / 10)
    p = p / p.sum()
    d2 = np.where(surface, (2 * np.sin(np.radians(grazing_deg)) / c) ** 2 * w_var, 0.0)
    return float(np.sum(p * d2))


def label(tx_on, rx_on, stage):
    """Name of a placement; matches experiment.py strategy names where one exists."""
    known = {(0, 0, "sym"): "rx_sym", (0, 0, "quad"): "rx_pkt2", (0, 0, "stale"): "rx_stale",
             (1, 0, "none"): "tx_own", (0, 1, "none"): "rx_own", (1, 1, "none"): "both",
             (1, 1, "line2"): "split", (1, 1, "const2"): "split0"}
    key = (int(tx_on), int(rx_on), stage)
    if key in known:
        return known[key]
    own = "+".join(n for n, on in (("tx", tx_on), ("rx", rx_on)) if on) or "none"
    return f"own[{own}]+rx[{stage}]"


def evaluate(fs, comps, rx, floor=0.0):
    """All placements, best first (by interference): dicts with tx_imu, rx_imu, stage, name, sir_db (from the
    interference kernel, theory.kernel_ici), ici and parts_ici (per component, receiver noise, floor), and for
    reference e_var / parts (residual Doppler variance) and sir_offset_db (the old offset formula on e_var)."""
    can = {c.name: c.after_node is not None for c in comps}
    out = []
    stages = [s for s in rx.stages if s != "stale" or rx.stale_h is not None]
    offset = (np.pi * rx.fc * rx.t_fft) ** 2 / 3  # interference per unit variance of a constant offset
    for tx_on, rx_on, stage in product((False, True), (False, True), stages):
        if (tx_on and not can["tx"]) or (rx_on and not can["rx"]):
            continue
        kern = lambda f, s=stage: receiver_stage(s, rx, np.asarray(f, float))[0]
        kici = lambda f, s=stage: ici_kernel(s, rx, f)
        _, k0, noise = receiver_stage(stage, rx, np.zeros(1))
        kici0 = float(ici_kernel(stage, rx, np.zeros(1))[0])
        parts, parts_ici = {}, {}
        for comp in comps:
            on = {"tx": tx_on, "rx": rx_on}.get(comp.name, False)
            x = comp.after_node if on else comp.true
            parts[comp.name] = _series_var(x, fs, kern, k0)
            parts_ici[comp.name] = _series_var(x, fs, kici, kici0)
            if on:
                parts[comp.name] += _drift_var(*comp.drift, kern, fs / 2)
                parts_ici[comp.name] += _drift_var(*comp.drift, kici, fs / 2)
        parts["rx_noise"], parts["floor"] = noise, floor
        parts_ici["rx_noise"], parts_ici["floor"] = ici_noise(stage, rx), floor * offset
        e, ici = sum(parts.values()), sum(parts_ici.values())
        out.append(dict(tx_imu=tx_on, rx_imu=rx_on, stage=stage, name=label(tx_on, rx_on, stage),
                        sir_db=float(-10 * np.log10(ici)), ici=ici, parts_ici=parts_ici,
                        e_var=e, parts=parts, sir_offset_db=theory.ici_sir_db(e, rx.fc, rx.t_fft)))
    return sorted(out, key=lambda r: r["ici"])
