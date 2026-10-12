"""
Bellhop multipath for the end-to-end chain (PLAN: Phase 1, Bellhop paths in the chain).

Coherent paths: Bellhop arrivals for the link geometry with the measured rough-surface loss (bellhop.py), merged.
Each path has its own Doppler from its departure and arrival angles: the transmitter's velocity projected on the
departure direction minus the receiver's on the arrival direction,
    eps_p(t) = (v_tx(t) . d_dep - v_rx(t + tau) . d_arr) / c,   d = (cos theta) u_h + (-sin theta) z,
with theta the declination (positive downward, Bellhop's convention), u_h the horizontal direction of the link and z
up. A steep bottom bounce sees the heave of a buoy differently from the direct path: this multi-scale Doppler is not
removed by one resampling.
Diffuse surface scattering (optional, level rho relative to the coherent channel): n_scatter weak paths with random
phases, delays around the surface-bounced arrivals of the smooth-surface Bellhop run (power-weighted), each with the
Doppler of that path geometry plus its surface bounces' own term 2 sin(g) w_k(t) / c per bounce, w_k the vertical
velocity of independent surface patches (stretches of the buoy's heave record).
The bulk Doppler (what sensors and resampling address) is that of the strongest coherent path; every path's
delta_eps is relative to it.
"""
import numpy as np

from . import bellhop, motion
from .channel import Path
from .experiment import at_times


# Downward-refracting summer profile for a 100 m shelf (warm mixed layer over a thermocline), [(depth m, c m/s)]
SUMMER_SSP = [(0.0, 1515.0), (15.0, 1514.0), (30.0, 1500.0), (50.0, 1494.0), (100.0, 1492.0)]


def ssp_to_depth(ssp, depth):
    """The profile cut or extended (last speed held) to end exactly at the water depth (Bellhop requires it)."""
    if ssp is None:
        return None
    z = np.array([p[0] for p in ssp], float)
    c = np.array([p[1] for p in ssp], float)
    keep = [(float(zi), float(ci)) for zi, ci in zip(z, c) if zi < depth]
    return keep + [(float(depth), float(np.interp(depth, z, c)))]


def nominal_angles(geometry, ssp=None, min_rel_db=-20.0, surface_rms=None):
    """Path (departure, arrival) angles and relative delays a receiver would predict from its own Bellhop run on a
    nominal geometry / profile (which may differ from the true one): list of (src_deg, rcv_deg, rel_delay_s),
    strongest paths within min_rel_db, sorted by delay."""
    kw = {} if ssp is None else dict(ssp=ssp_to_depth(ssp, geometry[3]))
    arr = bellhop.merge(bellhop.arrivals(*geometry, surface_rms=surface_rms, **kw))
    p = np.array([abs(a.amp) ** 2 for a in arr])
    keep = sorted([a for a, pp in zip(arr, p) if 10 * np.log10(pp / p.max()) >= min_rel_db], key=lambda a: a.delay)
    t0 = keep[0].delay
    return [(a.src_angle_deg, a.rcv_angle_deg, a.delay - t0) for a in keep]


def _directions(u, theta_deg):
    """Unit propagation direction for a declination theta (positive down) along the horizontal direction of u."""
    uh = np.array([u[0], u[1], 0.0])
    uh = uh / max(np.linalg.norm(uh), 1e-12)
    th = np.radians(theta_deg)
    return np.cos(th) * uh + np.array([0.0, 0.0, -np.sin(th)])


def path_eps(tx, rx, u, tau, tg, src_deg, rcv_deg, c=motion.C):
    """Doppler scale of one path on the emission grid tg."""
    v_tx = tx.velocity_at(tg, "total")
    v_rx = rx.velocity_at(tg + tau, "total")
    return (v_tx @ _directions(u, src_deg) - v_rx @ _directions(u, rcv_deg)) / c


class LinkMultipath:
    """Bellhop paths of one link, fixed for a run; build(tg, rng) gives the bulk Doppler and the paths of one packet."""

    def __init__(self, tx, rx, geometry, surface_rms, w_record, fs_record, rho_db=None, n_scatter=24,
                 min_rel_db=-20.0, seed=0, ssp=None, cluster=None):
        """geometry: (range [m], tx depth, rx depth, water depth) for Bellhop; w_record: vertical velocity of the sea
        surface (e.g. the buoy's heave velocity) at fs_record, for the scattered paths. ssp: sound-speed profile
        [(z, c)] (None: isovelocity; SUMMER_SSP: downward refracting). cluster: (n_sub, angle_spread_deg,
        delay_spread_s) splits every coherent path into n_sub sub-paths with random angle and delay offsets and
        complex Gaussian gains drawn per packet: each path then fades and has its own Doppler spread within the
        packet (micro-multipath from small-scale roughness and bathymetry); None: one frozen ray per path."""
        self.tx, self.rx = tx, rx
        d = motion.link_doppler(tx, rx)
        self.u, self.tau = d["u"], d["tau"]
        kw = {} if ssp is None else dict(ssp=ssp_to_depth(ssp, geometry[3]))
        rough = bellhop.merge(bellhop.arrivals(*geometry, surface_rms=surface_rms, **kw))
        p = np.array([abs(a.amp) ** 2 for a in rough])
        self.coh = [a for a, pp in zip(rough, p) if 10 * np.log10(pp / p.max()) >= min_rel_db]
        self.cluster = cluster
        smooth = bellhop.merge(bellhop.arrivals(*geometry, **kw))
        self.surf = [a for a in smooth if a.n_top > 0]
        self.rho_db, self.n_scatter = rho_db, n_scatter
        self.w, self.fs_w = np.asarray(w_record, float), fs_record
        self.rng = np.random.default_rng(seed)

    def build(self, tg, rng):
        t0 = min(a.delay for a in self.coh)
        eps = [path_eps(self.tx, self.rx, self.u, self.tau, tg, a.src_angle_deg, a.rcv_angle_deg) for a in self.coh]
        g = np.array([a.amp for a in self.coh], complex)
        g = g / np.sqrt(np.sum(np.abs(g) ** 2))
        ref = int(np.argmax(np.abs(g)))
        bulk = eps[ref]
        if self.cluster is None:
            paths = [Path(a.delay - t0, gk, e - bulk) for a, gk, e in zip(self.coh, g, eps)]
        else:
            n_sub, ang, dly = self.cluster
            paths = []
            for a, gk in zip(self.coh, g):
                c = (rng.standard_normal(n_sub) + 1j * rng.standard_normal(n_sub)) / np.sqrt(2 * n_sub)
                for j in range(n_sub):
                    e = path_eps(self.tx, self.rx, self.u, self.tau, tg, a.src_angle_deg + ang * rng.standard_normal(),
                                 a.rcv_angle_deg + ang * rng.standard_normal())
                    paths.append(Path(max(0.0, a.delay - t0 + dly * rng.standard_normal()), gk * c[j], e - bulk))
        if self.rho_db is not None and self.surf:
            rho = 10 ** (self.rho_db / 10)
            ps = np.array([abs(a.amp) ** 2 for a in self.surf])
            pick = rng.choice(len(self.surf), self.n_scatter, p=ps / ps.sum())
            t_w = np.arange(len(self.w)) / self.fs_w
            for k in pick:
                a = self.surf[k]
                e = path_eps(self.tx, self.rx, self.u, self.tau, tg, a.src_angle_deg, a.rcv_angle_deg)
                for _ in range(a.n_top):  # independent surface patches, one per bounce
                    off = rng.uniform(60, t_w[-1] - (tg[-1] - tg[0]) - 60)
                    w = at_times(self.w, t_w, self.fs_w, off + (tg - tg[0]))
                    e = e + 2 * np.sin(np.radians(abs(a.src_angle_deg))) * w / motion.C
                delay = max(0.0, a.delay - t0 + rng.uniform(-2e-3, 2e-3))
                gain = np.sqrt(rho / self.n_scatter) * np.exp(2j * np.pi * rng.uniform())
                paths.append(Path(delay, gain, e - bulk))
        return bulk, paths

    def summary(self):
        p = np.array([abs(a.amp) ** 2 for a in self.coh])
        return [dict(delay_ms=1e3 * (a.delay - min(b.delay for b in self.coh)), power_db=10 * np.log10(pp / p.max()),
                     src_deg=a.src_angle_deg, rcv_deg=a.rcv_angle_deg, n_top=a.n_top, n_bot=a.n_bot)
                for a, pp in zip(self.coh, p)]
