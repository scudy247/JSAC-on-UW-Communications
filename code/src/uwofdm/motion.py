"""
Node motion and link Doppler for the compensation study (README: Model; code/PLAN.md: Phase 1).

A Track is one transducer's velocity on a regular time grid, earth frame (north, east, up), split into a *fast* part
(wave-driven motion, measured live by the node's IMU) and a *slow* part (drift, cruise; predictable and shared over
the link). Positions follow from the mean position and the integrated velocity.

Link Doppler. For a signal emitted by node A at time t and received by node B at t + tau, with u the unit vector from
A to B,
    eps(t) = (v_A(t) - v_B(t + tau)) . u / c        (eps > 0: approaching, received signal compressed)
so r(t') ~ s(t' + int eps) as in the deck. It splits into
    eps_tx(t)   =  v_A,fast(t) . u / c              known at A while it emits
    eps_rx(t)   = -v_B,fast(t + tau) . u / c        known at B while it receives
    eps_slow(t) = (v_A,slow(t) - v_B,slow(t + tau)) . u / c
and eps = eps_tx + eps_rx + eps_slow. The direction u and the delay tau are taken from the mean positions (ranges of
hundreds of metres to kilometres against wave displacements of metres); their slow change over a packet is neglected.
"""
from dataclasses import dataclass
import glob

import numpy as np

from . import buoy
from .waveform import resample_at
from .paths import data_dir

C = 1500.0  # sound speed [m/s]


@dataclass
class Track:
    """Velocity of one transducer on t = t0 + k / fs; vel_fast, vel_slow: (N, 3) [m/s], (north, east, up)."""
    fs: float
    vel_fast: np.ndarray
    vel_slow: np.ndarray
    pos0: np.ndarray  # mean position (north, east, up) [m]; up < 0 below the surface
    t0: float = 0.0
    vel_lever: np.ndarray = None  # (N, 3) transducer velocity from rotation about the IMU (omega x r), or None

    @property
    def n(self):
        return len(self.vel_fast)

    @property
    def t(self):
        return self.t0 + np.arange(self.n) / self.fs

    @property
    def vel(self):
        return self.vel_fast + self.vel_slow

    def velocity_at(self, times, part="total"):
        """Velocity at arbitrary times, band-limited interpolation (straight lines between samples would put kinks in
        the motion, e.g. every 0.78 s for CDIP records); part: 'total', 'fast', 'slow' or 'lever'."""
        lever = np.zeros_like(self.vel_fast) if self.vel_lever is None else self.vel_lever
        v = {"total": self.vel + lever, "fast": self.vel_fast, "slow": self.vel_slow, "lever": lever}[part]
        pad = 64  # edge values held beyond the record, so the interpolator does not see zeros there
        pos = (np.asarray(times, float) - self.t0) * self.fs + pad
        return np.stack([np.real(resample_at(np.pad(v[:, i], pad, mode="edge"), pos)) for i in range(3)], 1)


# --------------------------------------------------------------------------- tracks
def static_track(fs, n, pos):
    """A node that does not move (seabed, bottom-mounted)."""
    z = np.zeros((n, 3))
    return Track(fs, z, z.copy(), np.asarray(pos, float))


def auv_leg(fs, n, pos, speed, heading_deg, climb=0.0):
    """Constant-velocity leg: speed [m/s] along heading [deg from north, clockwise], vertical rate climb [m/s]. All
    slow motion (predictable); add a fast part separately if needed."""
    h = np.radians(heading_deg)
    v = np.tile([speed * np.cos(h), speed * np.sin(h), climb], (n, 1))
    return Track(fs, np.zeros((n, 3)), v, np.asarray(pos, float))


def auv_maneuvering(fs, n, pos, rng, speed=1.5, heading_deg=45.0, turn_every=240.0, turn_rate_deg=3.0,
                    turn_s=30.0, speed_amp=0.2, speed_period=60.0, heave_amp=0.02, heave_period=15.0):
    """Modelled AUV (no logs available): cruise at `speed` on `heading_deg`, a turn of +-turn_rate_deg/s lasting turn_s
    every turn_every seconds (random direction), sinusoidal speed changes (speed_amp, speed_period) and a small
    control-induced heave oscillation (heave_amp, heave_period) as the fast part. Below the wave-affected layer, so no
    wave orbital motion. Cruise, turns and speed changes are the slow part; the AUV's navigation system (INS + DVL)
    measures all of it."""
    t = np.arange(n) / fs
    rate = np.zeros(n)
    for k, t_turn in enumerate(np.arange(turn_every / 2, t[-1], turn_every)):
        rate[(t >= t_turn) & (t < t_turn + turn_s)] = rng.choice([-1.0, 1.0]) * np.radians(turn_rate_deg)
    heading = np.radians(heading_deg) + np.cumsum(rate) / fs
    sp = speed + speed_amp * np.sin(2 * np.pi * t / speed_period + rng.uniform(0, 2 * np.pi))
    slow = np.stack([sp * np.cos(heading), sp * np.sin(heading), np.zeros(n)], 1)
    fast = np.zeros((n, 3))
    fast[:, 2] = heave_amp * np.sin(2 * np.pi * t / heave_period + rng.uniform(0, 2 * np.pi))
    return Track(fs, fast, slow, np.asarray(pos, float))


def buoy_track(source, name, pos, segment=0, drift=(0.0, 0.0, 0.0), lever=None):
    """Recorded buoy motion as the fast part (Phase 0 data), plus an optional constant drift as the slow part.
    source 'cdip': name is a day file stem such as '029p1_swell_2022-01-27' (one continuous day, 1.28 Hz).
    source 'ooi': name is 'MOORING/YYYYMMDD', segment selects the 20-minute burst (10 Hz). For OOI, lever = (x, y, z)
    [m, buoy frame north-east-down] places the transducer relative to the motion sensor; its rotation velocity
    omega x lever (small-tilt approximation: body ~ earth axes) is stored in vel_lever (earth north, east, up)."""
    if source == "cdip":
        path = glob.glob(str(data_dir("buoy", "cdip", f"{name}.npz")))
        if not path:
            raise FileNotFoundError(f"CDIP day {name}: run scripts/fetch_cdip.py")
        fs, vel, _ = buoy.cdip_day(path[0])
    elif source == "ooi":
        mooring, day = name.split("/")
        files = sorted(data_dir("buoy", "ooi", mooring, "mopak", day).glob("*.mopak.log"))
        if not files:
            raise FileNotFoundError(f"OOI day {name}: run scripts/fetch_ooi.py")
        fs, vel, _, w = buoy.ooi_burst(files[segment], rates=True)
        if lever is not None:
            v_ned = np.cross(w, np.asarray(lever, float))
            lev = v_ned * np.array([1.0, 1.0, -1.0])  # north, east, down -> north, east, up
            return Track(fs, vel, np.tile(np.asarray(drift, float), (len(vel), 1)), np.asarray(pos, float),
                         vel_lever=lev)
    else:
        raise ValueError(source)
    return Track(fs, vel, np.tile(np.asarray(drift, float), (len(vel), 1)), np.asarray(pos, float))


# --------------------------------------------------------------------------- link Doppler
def geometry(tx, rx, c=C):
    """Unit vector u from tx to rx and propagation delay tau [s], from the mean positions."""
    d = rx.pos0 - tx.pos0
    r = float(np.linalg.norm(d))
    return d / r, r / c


def link_doppler(tx, rx, times=None, lever_tx=None, lever_rx=None, c=C):
    """Doppler of the link tx -> rx at emission times `times` (default: tx's grid, limited to times for which the
    reception time lies within rx's record). Returns a dict with times, tau, u and eps, eps_tx, eps_rx, eps_slow
    (eps = eps_tx + eps_rx + eps_slow). lever_tx / lever_rx: optional (N, 3) velocity of the transducer relative to the
    IMU (rotation times lever arm) on the node's own grid, added to its fast part. A track's own vel_lever
    is always included: the true Doppler is the transducer's."""
    u, tau = geometry(tx, rx, c)
    if times is None:
        times = tx.t
        times = times[(times + tau >= rx.t[0]) & (times + tau <= rx.t[-1])]
    vf_tx = tx.velocity_at(times, "fast") + tx.velocity_at(times, "lever")
    vf_rx = rx.velocity_at(times + tau, "fast") + rx.velocity_at(times + tau, "lever")
    vs_tx, vs_rx = tx.velocity_at(times, "slow"), rx.velocity_at(times + tau, "slow")
    if lever_tx is not None:
        vf_tx = vf_tx + np.stack([np.interp(times, tx.t, lever_tx[:, i]) for i in range(3)], 1)
    if lever_rx is not None:
        vf_rx = vf_rx + np.stack([np.interp(times + tau, rx.t, lever_rx[:, i]) for i in range(3)], 1)
    eps_tx = vf_tx @ u / c
    eps_rx = -(vf_rx @ u) / c
    eps_slow = (vs_tx - vs_rx) @ u / c
    return {"times": times, "tau": tau, "u": u, "eps": eps_tx + eps_rx + eps_slow,
            "eps_tx": eps_tx, "eps_rx": eps_rx, "eps_slow": eps_slow}
