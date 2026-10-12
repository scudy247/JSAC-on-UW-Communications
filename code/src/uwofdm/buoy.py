"""
Buoy motion records (Phase 0 data) as earth-frame velocity tracks.

CDIP Waverider (scripts/fetch_cdip.py): displacement north / west / vertical at 1.28 Hz -> velocity and acceleration
by spectral differentiation (detrended, tapered ends, EDGE samples dropped).
OOI surface buoys (scripts/fetch_ooi.py): MicroStrain 3DM-GX3-25 records (0xCB, 43 bytes, 62 500 ticks/s, buoy frame
north-east-down) -> attitude by a complementary filter (gyro offset removed, 30 s correction time constant) -> earth-
frame acceleration -> velocity by spectral integration keeping f >= F_HP; TRIM samples dropped at each end of a burst.

Earth frame of the tracks: (north, east, up), velocities in m/s. Moved here from scripts/motion_predictability.py and
scripts/ooi_motion.py (9 Oct 2026) so that simulations and analyses share one implementation.
"""
import re

import numpy as np

from .paths import data_dir

G = 9.80665
OOI_FS = 10.0
TAU_ATT = 30.0  # attitude correction time constant [s]
F_HP = 0.03  # lower edge kept by the integration [Hz]
TRIM = 300  # samples dropped at each end of an OOI burst (30 s)
EDGE = 300  # samples dropped at each end of a CDIP day after differentiation (FFT edge effects)


def derivatives(x, fs):
    """Velocity and acceleration of a displacement record by spectral differentiation (detrended, tapered ends)."""
    n = len(x)
    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    w = np.ones(n)
    m = EDGE // 2
    w[:m] = w[-m:][::-1] = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
    X = np.fft.rfft(x * w)
    jw = 2j * np.pi * np.fft.rfftfreq(n, 1 / fs)
    v, a = np.fft.irfft(X * jw, n), np.fft.irfft(X * jw ** 2, n)
    return v[EDGE:-EDGE], a[EDGE:-EDGE]


def decode(path):
    """Records of one MOPAK burst: t [s], f (N,3) specific force [m/s^2], w (N,3) [rad/s], m (N,3) [gauss]."""
    b = open(path, "rb").read()
    out, i = [], 0
    while i + 43 <= len(b):
        rec = b[i:i + 43]
        if rec[0] == 0xCB and (sum(rec[:41]) & 0xFFFF) == int.from_bytes(rec[41:43], "big"):
            out.append(rec)
            i += 43
        else:  # resynchronise on the next header
            j = b.find(b"\xcb", i + 1)
            if j < 0:
                break
            i = j
    a = np.frombuffer(b"".join(out), dtype=np.dtype([("h", "u1"), ("v", ">f4", 9), ("t", ">u4"), ("c", ">u2")]))
    t = np.unwrap(a["t"].astype(float), period=2 ** 32) / 62500.0
    v = a["v"].astype(float)
    return t - t[0], v[:, 0:3] * G, v[:, 3:6], v[:, 6:9]


def qmul(p, q):
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rot(q):
    """Rotation matrix body -> earth for the unit quaternion q = (w, x, y, z)."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def triad(f, m):
    """Initial attitude from mean specific force and magnetic field (earth frame: magnetic north, east, down)."""
    down = -f / np.linalg.norm(f)
    east = np.cross(down, m)
    east /= np.linalg.norm(east)
    north = np.cross(east, down)
    R = np.vstack([north, east, down])  # rows: earth axes in body coordinates -> R maps body to earth
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = np.array([s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        k = int(np.argmax(np.diag(R)))
        i, j = (k + 1) % 3, (k + 2) % 3
        s = 2 * np.sqrt(1 + R[k, k] - R[i, i] - R[j, j])
        q = np.zeros(4)
        q[0] = (R[j, i] - R[i, j]) / s
        q[1 + k] = s / 4
        q[1 + i] = (R[i, k] + R[k, i]) / s
        q[1 + j] = (R[j, k] + R[k, j]) / s
    return q / np.linalg.norm(q)


def earth_acceleration(t, f, w, m):
    """Wave acceleration (north, east, down) [m/s^2] through a complementary attitude filter."""
    n0 = int(30 * OOI_FS)
    q = triad(f[:n0].mean(0), m[:n0].mean(0))
    w = w - w.mean(0)  # gyro offset: a moored buoy has no net rotation over a burst; left in, the filter would turn
    #                    it into a tilt error of offset * TAU_ATT (about 10 degrees for 0.005 rad/s)
    kp = 1.0 / TAU_ATT
    out = np.empty_like(f)
    dts = np.diff(t, prepend=t[0] - 1 / OOI_FS)
    for i in range(len(t)):
        R = rot(q)
        d_est = R[2]  # earth "down" expressed in the body frame
        fi = f[i]
        e = np.cross(-fi / np.linalg.norm(fi), d_est)  # tilt error
        mb = m[i] / np.linalg.norm(m[i])
        me = R @ mb
        ref = np.array([np.hypot(me[0], me[1]), 0.0, me[2]])
        e_m = np.cross(mb, R.T @ ref)
        e = e + np.dot(e_m, d_est) * d_est  # heading correction about the vertical only
        wc = w[i] + kp * e
        q = q + 0.5 * qmul(q, np.r_[0.0, wc]) * dts[i]
        q /= np.linalg.norm(q)
        out[i] = rot(q) @ fi + np.array([0.0, 0.0, G])
    return out


def integrate(a, order):
    """Spectral integration (order 1: velocity, 2: displacement), keeping f >= F_HP with a cosine ramp."""
    n = len(a)
    F = np.fft.rfftfreq(n, 1 / OOI_FS)
    A = np.fft.rfft(a - a.mean())
    ramp = np.clip((F - 0.5 * F_HP) / (0.5 * F_HP), 0, 1)
    ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)
    jw = 2j * np.pi * np.where(F > 0, F, 1.0)
    return np.fft.irfft(A * ramp / jw ** order, n), np.fft.irfft(A * ramp, n)


def cdip_day(path):
    """CDIP day file -> (fs, vel (N, 3), acc (N, 3)) in (north, east, up); EDGE samples dropped at each end."""
    d = np.load(path)
    fs = float(d["fs"])
    vel, acc = [], []
    for x, sign in ((d["north"], 1.0), (d["west"], -1.0), (d["z"], 1.0)):
        v, a = derivatives(sign * x.astype(float), fs)
        vel.append(v)
        acc.append(a)
    return fs, np.stack(vel, 1), np.stack(acc, 1)


def ooi_burst(path, rates=False):
    """OOI MOPAK burst -> (fs, vel (N, 3), acc (N, 3)) in (north, east, up); TRIM samples dropped at each end. With
    rates=True also the body angular rates (N, 3) [rad/s] (gyro offset removed), for lever-arm velocities."""
    t, f, w, m = decode(path)
    ae = earth_acceleration(t, f, w, m)  # north, east, down
    vel, acc = [], []
    for comp, sign in ((0, 1.0), (1, 1.0), (2, -1.0)):
        v, a = integrate(sign * ae[:, comp], 1)
        vel.append(v[TRIM:-TRIM])
        acc.append(a[TRIM:-TRIM])
    out = (OOI_FS, np.stack(vel, 1), np.stack(acc, 1))
    return out + ((w - w.mean(0))[TRIM:-TRIM],) if rates else out


def ooi_day(mooring, day):
    """All bursts of one OOI day: list of (hour, fs, vel, acc); bursts shorter than 10 min are skipped."""
    out = []
    for p in sorted(data_dir("buoy", "ooi", mooring, "mopak", day).glob("*.mopak.log")):
        if p.stat().st_size < 6000 * 43:
            continue
        fs, vel, acc = ooi_burst(p)
        out.append((int(re.match(r"\d{8}_(\d{2})", p.name).group(1)), fs, vel, acc))
    return out
