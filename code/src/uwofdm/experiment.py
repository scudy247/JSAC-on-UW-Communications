"""
Shared harness for the C1/C2 link experiments (PLAN: Phase 3, `experiment.py`). Moved and generalised from
scripts/c2_placement.py, c2_buoy_buoy.py and c2_auv.py (9 Oct 2026) so that scenarios and sweeps use one
implementation.

A link is two tracks (transmitter, receiver). For each packet the harness builds a padded CP-OFDM packet, applies the
true wideband Doppler, runs each requested strategy and measures the power-averaged signal-to-interference ratio
between subcarriers after a genie that removes per-symbol common phase and timing; it also returns the closed-form
prediction of each strategy from the spectra (theory.py).

Strategies (names as in RESULTS.md):
  rx_sym          receiver only, piecewise constant, one estimate per symbol
  rx_pkt2         receiver only, one quadratic per packet from per-symbol estimates
  rx_stale        receiver only, one value `stale_h` seconds before the packet, held
  tx_own          transmitter removes its own motion (IMU / INS) while emitting
  rx_own          receiver removes its own motion while receiving
  both            tx_own + rx_own
  split, split0   both + receiver line / constant from two estimates per packet (first and last symbol)
  si<B>+own[A]+rx[S]  E1 side information: the transmitter's IMU share over the packet sent as B bits (Legendre
                  coefficients, stage-aware allocation, sideinfo.py); the receiver rebuilds it, adds its own IMU share
                  if A = rx, fits the remainder with stage S and resamples the received signal (no theory column)
  own[A]+rx[S]    any placement of placement.py: A in none / tx / rx / tx+rx (node IMU stages), S in none / sym /
                  const2 / line2 / quad (receiver stage on the remainder: per symbol, constant or line from the first
                  and last symbol, quadratic from every symbol); predicted by placement.evaluate (interference kernel)
"""
import re
from dataclasses import dataclass, field

import numpy as np

from . import channel, estimates, motion, placement, sideinfo, theory, waveform
from .compensate import compensate, fit_trajectory

PAD = 4096  # zeros appended to each packet (128 ms at 32 kHz): stretching must never cut a packet


def at_times(x, t_day, fs, times):
    """Band-limited interpolation of a series sampled at fs, starting at t_day[0], at arbitrary times."""
    return np.real(waveform.resample_at(np.asarray(x, float), (np.asarray(times, float) - t_day[0]) * fs))


def spectral_derivative(x, fs):
    """Time derivative by FFT (an IMU measures acceleration directly)."""
    n = len(x)
    X = np.fft.rfft(x - x.mean())
    return np.fft.irfft(X * 2j * np.pi * np.fft.rfftfreq(n, 1 / fs), n)


MAX_DELAY = 0.020  # delay span assumed by the channel estimator [s] (longest path of Multipath: 13.5 ms)
MIN_DELAY = -0.005  # taps before the first arrival: residual timing drift after compensation can be either sign
BACKOFF = 0.002  # pilot-based receiver: FFT window starts 2 ms inside the cyclic prefix (tolerates early arrival)
_EST = {}


def _estimator(o):
    """Delay-domain least-squares channel estimator from the pilots (cached per numerology): the channel is modelled
    as taps on a grid of spacing 1 / (2 N_pilots * pilot spacing [Hz]) (twice oversampled, so that arrivals between
    grid points are fitted), regularised (singular values below 1e-2 of the largest dropped: without it the
    oversampled fit amplifies interference on the pilots by up to 50 dB; with it, -42.5 dB error on a static 3-path
    channel and no amplification, i.e. a ceiling of about 42 dB on measured SIR) from MIN_DELAY (negative: a
    residual timing drift after compensation can make the signal arrive early) to MAX_DELAY, fitted on the
    pilots and evaluated on the data subcarriers. Pilots every 4 subcarriers (31 Hz) see delays up to 32 ms without
    aliasing."""
    key = (o.n_sc, o.pilot_spacing, o.guard_sc, o.bandwidth)
    if key not in _EST:
        act = np.where(o.active)[0]
        pil, dat = act[o.pilots()[act]], act[~o.pilots()[act]]
        f_p, f_d = (pil - pil[0]) * o.df, (dat - pil[0]) * o.df
        dtau = 0.5 / (len(pil) * o.pilot_spacing * o.df)
        taus = np.arange(int(np.floor(MIN_DELAY / dtau)), int(np.ceil(MAX_DELAY / dtau))) * dtau
        A_p, A_d = np.exp(-2j * np.pi * np.outer(f_p, taus)), np.exp(-2j * np.pi * np.outer(f_d, taus))
        _EST[key] = (pil, dat, A_d @ np.linalg.pinv(A_p, rcond=1e-2))
    return _EST[key]


def pilot_powers(Y, X, o):
    """Signal and interference power on the data subcarriers after per-symbol channel estimation from the pilots
    (every o.pilot_spacing-th active subcarrier; delay-domain least squares, see _estimator): a realistic receiver
    for frequency-selective (multipath) channels."""
    pil, dat, M = _estimator(o)
    sig = err = 0.0
    for y, x in zip(Y, X):
        h = M @ (y[pil] / x[pil])
        e = y[dat] - h * x[dat]
        sig += np.sum(np.abs(h * x[dat]) ** 2)
        err += np.sum(np.abs(e) ** 2)
    return sig, err


@dataclass
class Multipath:
    """Simple shallow-water multipath for a ~1 km link: direct, bottom and surface-bounced arrivals. Surface-bounced
    paths add their own Doppler 2 sin(grazing) w(t) / c, with w the vertical velocity of the sea surface at the bounce
    point, taken from independent stretches (shifted by shift_s per path) of a recorded buoy's vertical velocity."""
    delays_ms: tuple = (0.0, 2.1, 4.7, 8.9, 13.5)
    powers_db: tuple = (0.0, -2.0, -4.0, -7.0, -10.0)
    surface: tuple = (False, True, False, True, True)
    grazing_deg: tuple = (0.0, 8.0, 10.0, 14.0, 18.0)
    path_doppler: bool = True
    shift_s: float = 97.0
    seed: int = 123


def make_paths(mp, surface_track, t_day, tg, fs_track):
    """channel.Path list for one packet (emission grid tg); gains fixed by mp.seed, surface Doppler time-varying."""
    rng = np.random.default_rng(mp.seed)
    w = surface_track.vel_fast[:, 2]
    paths = []
    for k, (d, p, surf, g) in enumerate(zip(mp.delays_ms, mp.powers_db, mp.surface, mp.grazing_deg)):
        gain = 10 ** (p / 20) * np.exp(2j * np.pi * rng.uniform())
        delta = 0.0
        if surf and mp.path_doppler:
            wk = np.roll(w, -int(k * mp.shift_s * fs_track))
            delta = 2 * np.sin(np.radians(g)) * at_times(wk, t_day, fs_track, tg) / motion.C
        paths.append(channel.Path(d * 1e-3, gain, delta))
    norm = np.sqrt(sum(abs(p.gain) ** 2 for p in paths))
    for p in paths:
        p.gain /= norm
    return paths


def rotate_horizontal(u, deg):
    """u rotated by deg about the vertical axis (an azimuth error in the path direction a node uses)."""
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([c * u[0] - s * u[1], s * u[0] + c * u[1], u[2]])


def _ramp(h, over=16):
    """Linear phase per subcarrier b maximising |sum_k h_k exp(-j b k)|."""
    n = over * len(h)
    F = np.abs(np.fft.fft(h, n))
    i = int(np.argmax(F))
    l, c, r = F[i - 1], F[i], F[(i + 1) % n]
    d = 0.5 * (l - r) / (l - 2 * c + r) if (l - 2 * c + r) != 0 else 0.0
    return float(np.angle(np.exp(2j * np.pi * (i + d) / n)))


def genie_powers(Y, X, active):
    """Signal and interference power over a packet after removing, per symbol, a common complex gain and a linear
    phase ramp across subcarriers (what channel estimation removes); powers are accumulated, not ratios.
    The ramp is the maximum-likelihood one (peak of |sum h exp(-j b k)|, zero-padded FFT refined by a parabola): the
    adjacent-product estimate is biased by inter-carrier interference (~3e-3 rad per subcarrier at a 0.2-subcarrier
    offset, i.e. 3 rad across the band), which made strong-interference SIRs come out 5-20 dB too low."""
    k = np.where(active)[0]
    sig = err = 0.0
    for y, x in zip(Y[:, k], X[:, k]):
        h = y / x
        b = _ramp(h)
        a = np.mean(h * np.exp(-1j * b * np.arange(len(h))))
        e = y - a * np.exp(1j * b * np.arange(len(h))) * x
        sig += np.sum(np.abs(a * x) ** 2)
        err += np.sum(np.abs(e) ** 2)
    return sig, err


class NodeIMU:
    """What a node's IMU (or INS) tells it about its own share of the link Doppler, on the emission-time grid.
    sign = +1 for the transmitter (eps_tx = v.u/c), -1 for the receiver (eps_rx = -v(t + tau).u/c, read tau later).
    The estimate: own share LATENCY seconds old, extrapolated with the measured acceleration (if extrapolate), plus
    the lever-arm share from the gyro (if lever_corrected), plus a Gauss-Markov drift of std sigma."""

    def __init__(self, track, u, sign, shift, rng, sigmas, tau_imu=20.0, latency=0.05, extrapolate=True,
                 lever_corrected=True, with_slow=False, u_hat=None, grade=None, lever_m=None):
        """u_hat: the path direction the node believes (default: the true u); its own velocity is projected on it.
        grade: an imu.ImuGrade -> the estimate comes from the realistic sensor model (imu.RealisticIMU: accelerometer
        noise, bias, scale, attitude leakage, complementary filter with or without aiding, timing error; it measures
        the node's total motion) and `sigma` in estimate() is ignored; None: the original model below."""
        self.t, self.fs, self.shift = track.t, track.fs, shift
        self.latency, self.extrapolate, self.lever_corrected = latency, extrapolate, lever_corrected
        self.tau_imu = tau_imu
        u_hat = u if u_hat is None else u_hat
        v = track.velocity_at(track.t, "fast") + (track.velocity_at(track.t, "slow") if with_slow else 0.0)
        vl = track.velocity_at(track.t, "lever")
        self.own, self.lev = sign * (v @ u_hat) / motion.C, sign * (vl @ u_hat) / motion.C
        self.proj_err = sign * ((v + vl) @ (u - u_hat)) / motion.C  # what a wrong direction leaves
        self.d_own, self.d_lev = spectral_derivative(self.own, self.fs), spectral_derivative(self.lev, self.fs)
        self.drift = {s: estimates.gauss_markov(len(self.t), self.fs, s, tau_imu, rng) / motion.C for s in sigmas}
        self.sign, self.u_hat, self.real = sign, u_hat, None
        if grade is not None:
            from .imu import RealisticIMU
            self.real = RealisticIMU(track, grade, rng, latency, extrapolate, lever_corrected, lever_m)

    def estimate(self, times, sigma):
        times = np.asarray(times, float)
        if self.real is not None:
            return self.sign * (self.real.velocity(times + self.shift) @ self.u_hat) / motion.C
        tl = times + self.shift - self.latency
        x, dx = self.own, self.d_own
        if self.lever_corrected:
            x, dx = x + self.lev, dx + self.d_lev
        est = at_times(x, self.t, self.fs, tl)
        if self.extrapolate:
            est = est + self.latency * at_times(dx, self.t, self.fs, tl)
        return est + np.interp(times + self.shift, self.t, self.drift[sigma])

    def residual_spectrum(self, f, sigma):
        """Spectrum of (true share - estimate), independent parts summed (cross terms ignored)."""
        _, S_own = theory.spectrum(self.own * motion.C, self.fs)
        _, S_lev = theory.spectrum(self.lev * motion.C, self.fs)
        k = theory.kernel_extrapolate(f, self.latency) if self.extrapolate else theory.kernel_stale(f, self.latency)
        lev = S_lev * k if self.lever_corrected else S_lev
        _, S_dir = theory.spectrum(self.proj_err * motion.C, self.fs)
        return theory.gauss_markov_spectrum(f, sigma, self.tau_imu) + S_own * k + lev + S_dir

    def full_spectrum(self):
        return theory.spectrum((self.own + self.lev + self.proj_err) * motion.C, self.fs)[1]


@dataclass
class LinkConfig:
    packet_s: float = 1.0
    n_packets: int = 10
    sigma_rx: float = 3e-6
    sigmas_imu: tuple = (0.01,)
    strategies: tuple = ("rx_sym", "rx_pkt2", "tx_own", "split")
    stale_h: float = 1.0
    tau_imu: float = 20.0
    latency: float = 0.05
    extrapolate: bool = True
    lever_corrected: bool = True
    tx_with_slow: bool = False
    rx_with_slow: bool = False
    u_error_deg: float = 0.0  # azimuth error of the path direction used by both nodes' IMU projection
    multipath: object = None  # None (single path) or a Multipath
    metric: str = "genie"  # 'genie' (per-symbol phase and timing removed) or 'pilot' (channel from the pilots)
    window: tuple = field(default=(30.0, 5.0))  # margins at the start and end of the record [s]


GENERIC = re.compile(r"own\[(none|tx|rx|tx\+rx)\]\+rx\[(none|sym|const2|line2|quad)\]$")
STAGE_MODEL = {"sym": "piece", "quad": 2, "const2": 0, "line2": 1}
SIDEINFO = re.compile(r"si(\d+)\+own\[(none|rx)\]\+rx\[(none|sym|const2|line2|quad)\]$")
SI_KMAX = 24


def parse_sideinfo(name):
    """(bits, rx_on, stage) of a side-information strategy name, or None."""
    m = SIDEINFO.match(name)
    return None if not m else (int(m.group(1)), m.group(2) == "rx", m.group(3))


def parse_placement(name):
    """(tx_on, rx_on, stage) of a generic placement name, or None."""
    m = GENERIC.match(name)
    if not m:
        return None
    return "tx" in m.group(1).split("+"), "rx" in m.group(1).split("+"), m.group(2)


def run_link(tx_track, rx_track, cfg, rng, o=None):
    """Simulate and predict every strategy in cfg for the link tx -> rx. Returns {strategy key: {sir_sim_db,
    sir_sim_p10_db, sir_theory_db}}; keys of IMU-based strategies carry the IMU sigma (e.g. 'split_0.01')."""
    o = o or waveform.OFDM()
    t_sym, t_fft = o.n_symbol / o.fs, o.n_fft / o.fs
    d = motion.link_doppler(tx_track, rx_track)
    t_day, eps, u, tau = d["times"], d["eps"], d["u"], d["tau"]
    kw = dict(sigmas=cfg.sigmas_imu, tau_imu=cfg.tau_imu, latency=cfg.latency, extrapolate=cfg.extrapolate,
              lever_corrected=cfg.lever_corrected)
    u_hat = rotate_horizontal(u, cfg.u_error_deg)
    tx = NodeIMU(tx_track, u, +1, 0.0, rng, with_slow=cfg.tx_with_slow, u_hat=u_hat, **kw)
    rx = NodeIMU(rx_track, u, -1, tau, rng, with_slow=cfg.rx_with_slow, u_hat=u_hat, **kw)
    fs = tx_track.fs
    f, S = theory.spectrum(eps * motion.C, fs)
    T, n_sym = cfg.packet_s, int(round(cfg.packet_s / t_sym))
    starts = np.linspace(t_day[0] + cfg.window[0] + cfg.stale_h, t_day[-1] - T - cfg.window[1], cfg.n_packets)
    st = set(cfg.strategies)
    generic = [(n,) + parse_placement(n) for n in cfg.strategies if parse_placement(n)]
    side = [(n,) + parse_sideinfo(n) for n in cfg.strategies if parse_sideinfo(n)]
    if side:  # quantiser ranges and stage-aware allocation, fixed for the run (they depend on the sea, not the packet)
        T_pk = n_sym * t_sym
        rxm = placement.ReceiverModel(packet_s=T, t_sym=t_sym, t_fft=t_fft, fc=o.fc)
        f_s, S_s = theory.spectrum(tx.own + (tx.lev if cfg.lever_corrected else 0.0), fs)
        si_std = sideinfo.coefficient_std(f_s, S_s, T_pk, SI_KMAX)
        si_ranges = sideinfo.LOAD * si_std
        si_alloc = {}
        for name, bits, _, stage in side:
            tt, ts_, W_, win = placement.stage_estimator(stage, rxm)
            si_alloc[name] = sideinfo.allocate_weighted(si_std, sideinfo.stage_weights(tt, ts_, W_, win, o.fc,
                                                                                       SI_KMAX), bits)
    sims = {}
    for t0 in starts:
        X = o.random_symbols(n_sym, rng)
        s = np.pad(o.modulate(X), (0, PAD))
        tg = t0 + np.arange(len(s)) / o.fs
        e_true = at_times(eps, t_day, fs, tg)
        t_est = t0 + (np.arange(n_sym) + 0.5) * t_sym
        e_est = at_times(eps, t_day, fs, t_est) + cfg.sigma_rx * rng.standard_normal(n_sym)
        if cfg.multipath is not None:
            paths = make_paths(cfg.multipath, tx_track, t_day, tg, fs)
            through = lambda x: channel.apply(x, o.fs, o.fc, e_true, paths)
        else:
            through = lambda x: waveform.warp(x, o.fs, e_true, o.fc)
        y = through(s)

        def run(key, z):
            if cfg.metric == "pilot":
                pw = pilot_powers(o.demodulate(z, n_sym, backoff=int(BACKOFF * o.fs)), X, o)
            else:
                pw = genie_powers(o.demodulate(z, n_sym), X, o.active)
            sims.setdefault(key, []).append(pw)

        if "rx_sym" in st:
            run("rx_sym", compensate(y, o.fs, o.fc, fit_trajectory(t_est, e_est, tg, "piece")))
        if "rx_pkt2" in st:
            run("rx_pkt2", compensate(y, o.fs, o.fc, fit_trajectory(t_est, e_est, tg, 2)))
        if "rx_stale" in st:
            old = at_times(eps, t_day, fs, [t0 - cfg.stale_h])[0] + cfg.sigma_rx * rng.standard_normal()
            run("rx_stale", compensate(y, o.fs, o.fc, np.full(len(tg), old)))
        for sg in cfg.sigmas_imu:
            need_tx = st & {"tx_own", "both", "split", "split0"} or any(g[1] for g in generic)
            y_tx = through(waveform.unwarp(s, o.fs, tx.estimate(tg, sg), o.fc)) if need_tx else None
            need_rx = st & {"rx_own", "both", "split", "split0"} or any(g[2] for g in generic) \
                or any(g[2] for g in side)
            rx_own = rx.estimate(tg, sg) if need_rx else None
            if "tx_own" in st:
                run(f"tx_own_{sg:g}", y_tx)
            if "rx_own" in st:
                run(f"rx_own_{sg:g}", compensate(y, o.fs, o.fc, rx_own))
            if "both" in st:
                run(f"both_{sg:g}", compensate(y_tx, o.fs, o.fc, rx_own))
            if st & {"split", "split0"}:
                t2 = t_est[[0, -1]]
                r2 = at_times(eps, t_day, fs, t2) - tx.estimate(t2, sg) - rx.estimate(t2, sg) \
                    + cfg.sigma_rx * rng.standard_normal(2)
                for key, order in (("split", 1), ("split0", 0)):
                    if key in st:
                        run(f"{key}_{sg:g}", compensate(y_tx, o.fs, o.fc, rx_own + fit_trajectory(t2, r2, tg, order)))
            for name, ton, ron, stage in generic:
                z, traj = (y_tx if ton else y), (rx_own if ron else np.zeros(len(tg)))
                if stage != "none":
                    ts = t_est if stage in ("sym", "quad") else t_est[[0, -1]]
                    r = at_times(eps, t_day, fs, ts) + cfg.sigma_rx * rng.standard_normal(len(ts))
                    r = r - (tx.estimate(ts, sg) if ton else 0.0) - (rx.estimate(ts, sg) if ron else 0.0)
                    traj = traj + fit_trajectory(ts, r, tg, STAGE_MODEL[stage])
                run(f"{name}_{sg:g}", compensate(z, o.fs, o.fc, traj) if (ron or stage != "none") else z)
            if side:
                tx_hat = tx.estimate(tg, sg)
                pk = tg < t0 + T_pk
                c = sideinfo.project(tx_hat[pk], tg[pk] - t0, T_pk, SI_KMAX)
                for name, bits, ron, stage in side:
                    q = sideinfo.quantise_alloc(c, si_alloc[name], si_ranges)
                    traj = sideinfo.reconstruct(q, np.minimum(tg - t0, T_pk), T_pk)  # held after the packet
                    if ron:
                        traj = traj + rx_own
                    if stage != "none":
                        ts = t_est if stage in ("sym", "quad") else t_est[[0, -1]]
                        r = at_times(eps, t_day, fs, ts) + cfg.sigma_rx * rng.standard_normal(len(ts))
                        r = r - sideinfo.reconstruct(q, ts - t0, T_pk) - (rx.estimate(ts, sg) if ron else 0.0)
                        traj = traj + fit_trajectory(ts, r, tg, STAGE_MODEL[stage])
                    run(f"{name}_{sg:g}", compensate(y, o.fs, o.fc, traj))
    noise = cfg.sigma_rx ** 2
    th = {"rx_sym": theory.residual_eps_var(f, S, theory.kernel_poly(f, t_sym, 0)) + noise,
          "rx_pkt2": theory.residual_eps_var(f, S, theory.kernel_poly(f, T, 2)) + theory.noise_term(cfg.sigma_rx, 2, n_sym),
          "rx_stale": theory.residual_eps_var(f, S, theory.kernel_stale_packet(f, cfg.stale_h, T)) + noise}
    for sg in cfg.sigmas_imu:
        R_tx, R_rx = tx.residual_spectrum(f, sg), rx.residual_spectrum(f, sg)
        one = np.ones_like(f)
        th[f"tx_own_{sg:g}"] = theory.residual_eps_var(f, R_tx + rx.full_spectrum(), one)
        th[f"rx_own_{sg:g}"] = theory.residual_eps_var(f, R_rx + tx.full_spectrum(), one)
        th[f"both_{sg:g}"] = theory.residual_eps_var(f, R_tx + R_rx, one)
        th[f"split_{sg:g}"] = theory.residual_eps_var(f, R_tx + R_rx, theory.kernel_poly(f, T, 1)) + noise
        th[f"split0_{sg:g}"] = theory.residual_eps_var(f, R_tx + R_rx, theory.kernel_poly(f, T, 0)) + noise / 2
    th_db = {}
    if generic:
        rxm = placement.ReceiverModel(packet_s=T, t_sym=t_sym, t_fft=t_fft, fc=o.fc, sigma=cfg.sigma_rx,
                                      stages=tuple({g[3] for g in generic}))
        for sg in cfg.sigmas_imu:
            sen = placement.NodeSensors(sigma=sg, tau=cfg.tau_imu, latency=cfg.latency, extrapolate=cfg.extrapolate,
                                        lever_corrected=cfg.lever_corrected, u_error_deg=cfg.u_error_deg)
            ev = {r["name"]: r["sir_db"] for r in placement.evaluate(*placement.link_components(tx_track, rx_track, sen, sen), rxm)}
            for name, ton, ron, stage in generic:
                th_db[f"{name}_{sg:g}"] = ev.get(placement.label(ton, ron, stage), np.nan)
    for name, *_ in side:
        for sg in cfg.sigmas_imu:
            th_db[f"{name}_{sg:g}"] = float("nan")
    out = {}
    for key, pw in sims.items():
        pw = np.array(pw)
        per = 10 * np.log10(pw[:, 0] / pw[:, 1])
        out[key] = {"sir_sim_db": float(10 * np.log10(pw[:, 0].sum() / pw[:, 1].sum())),
                    "sir_sim_p10_db": float(np.percentile(per, 10)),
                    "sir_theory_db": th_db[key] if key in th_db else theory.ici_sir_db(th[key], o.fc, t_fft)}
    return out
