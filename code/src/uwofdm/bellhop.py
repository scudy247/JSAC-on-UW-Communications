"""
Multipath from Bellhop (PLAN: Phase 1, Bellhop paths): arrivals for a link geometry, merged into paths with delays,
gains, angles and bounce counts, and the path-specific Doppler floor they imply.

Bellhop (Acoustics Toolbox; the binary of the DESERT build, or $BELLHOP_EXE) is run in arrivals mode ('A') on a
range-independent waveguide: isovelocity or a given sound-speed profile, vacuum surface, acoustic half-space bottom.
Each arrival carries amplitude, phase, delay, departure and arrival angles and the numbers of surface (top) and
bottom bounces. Neighbouring beams of the same eigenray come out as near-duplicates; they are merged coherently.

Path-specific Doppler: each surface bounce adds 2 sin(g) w / c, with g the grazing angle at the surface and w the
vertical velocity of the surface at the bounce point; bounce points are treated as independent, so a path with n
surface bounces has variance n (2 sin g / c)^2 var(w). The floor is the power-weighted sum (as placement.path_floor).
Not modelled yet: the different projections of the nodes' own motion on each path's departure / arrival direction.

Surface loss (optional, surface_rms): the surface becomes a tabulated reflection coefficient (Bellhop top option 'F',
file .trc) with the coherent (Kirchhoff / Rayleigh) loss of a rough pressure-release surface,
|R| = exp(-2 (k sigma sin g)^2), phase 180 deg, sigma the rms elevation (Hs / 4). It removes the energy scattered away
from the specular direction; that energy is not tracked. Floors from coherent paths only are therefore a lower bound
on the true floor (scattered energy lost), the smooth surface an upper bound (all energy arriving coherently).
"""
import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np

BELLHOP_EXE = os.environ.get(
    "BELLHOP_EXE", "/home/dario/Underwater/DESERT_Underwater/DESERT_buildCopy_LOCAL/bin/bellhop.exe")
C = 1500.0


@dataclass
class Arrival:
    delay: float  # [s]
    amp: complex
    src_angle_deg: float  # departure angle (positive downward)
    rcv_angle_deg: float
    n_top: int  # surface bounces
    n_bot: int  # bottom bounces


def env_text(range_m, z_tx, z_rx, depth, fc, ssp=None, bottom=(1600.0, 1.8, 0.8), angles=(-80.0, 80.0),
             rough_top=False):
    """Bellhop .env for one source depth, one receiver depth and range, arrivals in ASCII. ssp: [(z, c)] or
    isovelocity C; bottom: (sound speed, density, attenuation dB/wavelength); rough_top: surface from a .trc table."""
    ssp = ssp or [(0.0, C), (depth, C)]
    top = "'SFW'" if rough_top else "'SVW'"
    lines = ["'uwofdm link'", f"{fc:.1f}", "1", top, f"0 0.0 {depth:.2f}"]
    lines += [f"{z:.2f} {c:.2f} /" for z, c in ssp]
    cb, rho, att = bottom
    lines += ["'A' 0.0", f"{depth:.2f} {cb:.2f} 0.0 {rho:.2f} {att:.2f} /",
              "1", f"{z_tx:.2f} /", "1", f"{z_rx:.2f} /", "1", f"{range_m / 1000:.5f} /",
              "'A'", "0", f"{angles[0]:.1f} {angles[1]:.1f} /", f"0.0 {depth * 1.1:.2f} {range_m / 1000 * 1.1:.4f}"]
    return "\n".join(lines) + "\n"


def read_arr(path):
    """Arrivals of a 2D ASCII .arr file with one source, one receiver depth and one range."""
    tok = open(path).read().split()
    if tok[0] != "'2D'":
        raise ValueError("not a 2D ASCII arrivals file")
    i = 2  # skip flag and frequency
    for _ in range(3):  # source depths, receiver depths, ranges: count then values
        n = int(tok[i])
        i += 1 + n
    i += 1  # maximum number of arrivals for this source
    n_arr = int(tok[i])
    i += 1
    out = []
    for _ in range(n_arr):
        a, ph, dr, di, sa, ra, nt, nb = tok[i:i + 8]
        i += 8
        out.append(Arrival(float(dr), float(a) * np.exp(1j * np.radians(float(ph))), float(sa), float(ra),
                           int(nt), int(nb)))
    return out


def rough_surface_coefficient(grazing_deg, fc, sigma, c=C):
    """Coherent reflection coefficient magnitude of a rough pressure-release surface (Kirchhoff / Rayleigh)."""
    k = 2 * np.pi * fc / c
    return np.exp(-2 * (k * sigma * np.sin(np.radians(grazing_deg))) ** 2)


def trc_text(fc, sigma, c=C, n=181):
    """Top reflection coefficient table (.trc): count, then angle [deg], |R|, phase [deg]."""
    th = np.linspace(0.0, 90.0, n)
    r = rough_surface_coefficient(th, fc, sigma, c)
    return f"{n}\n" + "\n".join(f"{t:.3f} {x:.6e} 180.0" for t, x in zip(th, r)) + "\n"


def arrivals(range_m, z_tx, z_rx, depth, fc=16e3, exe=BELLHOP_EXE, surface_rms=None, **kw):
    """Run Bellhop for one link and return its arrivals (a temporary directory, removed afterwards). surface_rms:
    rms surface elevation [m] for the coherent surface loss (None: smooth surface)."""
    with tempfile.TemporaryDirectory(prefix="bellhop_") as d:
        with open(os.path.join(d, "link.env"), "w") as fh:
            fh.write(env_text(range_m, z_tx, z_rx, depth, fc, rough_top=surface_rms is not None, **kw))
        if surface_rms is not None:
            with open(os.path.join(d, "link.trc"), "w") as fh:
                fh.write(trc_text(fc, surface_rms))
        subprocess.run([exe, "link"], cwd=d, check=True, capture_output=True, timeout=300,
                       env=dict(os.environ, OMP_NUM_THREADS="1"))
        return read_arr(os.path.join(d, "link.arr"))


def merge(arrs, tol=1e-4):
    """Merge near-duplicate arrivals (same bounce counts, delays within tol seconds): amplitudes summed coherently,
    delay and angles power-weighted."""
    groups = []
    for a in sorted(arrs, key=lambda a: a.delay):
        g = next((g for g in groups if g[0].n_top == a.n_top and g[0].n_bot == a.n_bot
                  and abs(g[-1].delay - a.delay) < tol and np.sign(g[0].rcv_angle_deg) == np.sign(a.rcv_angle_deg)), None)
        (g.append(a) if g is not None else groups.append([a]))
    out = []
    for g in groups:
        w = np.array([abs(a.amp) ** 2 for a in g])
        w = w / w.sum()
        out.append(Arrival(float(np.dot(w, [a.delay for a in g])), complex(sum(a.amp for a in g)),
                           float(np.dot(w, [a.src_angle_deg for a in g])), float(np.dot(w, [a.rcv_angle_deg for a in g])),
                           g[0].n_top, g[0].n_bot))
    return sorted(out, key=lambda a: a.delay)


def profile(arrs, min_rel_db=-30.0):
    """Paths relative to the first arrival and the strongest path: delay [ms], power [dB], surface bounces, grazing
    angle at the surface [deg] (isovelocity: the departure angle). Paths weaker than min_rel_db are dropped."""
    p = np.array([abs(a.amp) ** 2 for a in arrs])
    keep = 10 * np.log10(p / p.max()) >= min_rel_db
    sel = [a for a, k in zip(arrs, keep) if k]
    t0, pmax = min(a.delay for a in sel), p.max()
    return [dict(delay_ms=1e3 * (a.delay - t0), power_db=10 * np.log10(abs(a.amp) ** 2 / pmax), n_top=a.n_top,
                 n_bot=a.n_bot, grazing_deg=abs(a.src_angle_deg)) for a in sel]


def floor(prof, w_var, c=C):
    """Residual Doppler variance from the surface bounces of a profile (independent bounce points)."""
    p = np.array([10 ** (r["power_db"] / 10) for r in prof])
    p = p / p.sum()
    d2 = np.array([r["n_top"] * (2 * np.sin(np.radians(r["grazing_deg"])) / c) ** 2 * w_var for r in prof])
    return float(np.sum(p * d2))


def diffuse_doppler_factor(prof_smooth, c=C):
    """D^2 [per unit var(w)]: Doppler-scale variance of surface-scattered energy, taken from the surface-bounced paths
    of the smooth-surface profile (power-weighted n_top (2 sin g / c)^2). The scattered energy is assumed to leave
    the surface around the specular directions of those paths, with independent bounce points."""
    surf = [r for r in prof_smooth if r["n_top"] > 0]
    if not surf:
        return 0.0
    p = np.array([10 ** (r["power_db"] / 10) for r in surf])
    d2 = np.array([r["n_top"] * (2 * np.sin(np.radians(r["grazing_deg"])) / c) ** 2 for r in surf])
    return float(np.sum(p * d2) / p.sum())


def diffuse_floor(rho_db, d2, w_var):
    """Residual Doppler variance from diffusely scattered surface power rho (relative to the coherent channel, dB):
    the scattered share rho / (1 + rho) of the received power carries Doppler offsets of variance d2 var(w)."""
    rho = 10 ** (np.asarray(rho_db, float) / 10)
    return rho / (1 + rho) * d2 * w_var
