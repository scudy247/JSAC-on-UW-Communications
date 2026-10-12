"""
Fetch CDIP wave-buoy data for the Phase 0 predictability check (see code/PLAN.md).

Two steps, each a subcommand:
  python scripts/fetch_cdip.py summary            wave summary (time, Hs, Tp) + displacement metadata per deployment
  python scripts/fetch_cdip.py days               3 days of raw displacement per station, chosen from the summary
                                                  (swell-dominated, mixed, wind sea)

Data come from the CDIP THREDDS server through OPeNDAP (DAP2 binary), one variable per request, sliced on the server,
so only the requested samples are transferred. Output goes to <data>/buoy/cdip/ (paths.data_dir); nothing else is
written. Only the standard library and NumPy are needed.
"""
import argparse
import json
import re
import sys
import urllib.request

import numpy as np

from uwofdm.paths import data_dir

SERVER = "https://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive"
# station -> deployment: the most recent long deployment in the archive (checked 9 Oct 2026)
DEPLOYMENTS = {"029p1": "d25", "067p1": "d30", "100p1": "d19"}
NAMES = {"029p1": "Point Reyes", "067p1": "San Nicolas Island", "100p1": "Torrey Pines Outer"}
DTYPES = {"Float32": ">f4", "Float64": ">f8", "Int32": ">i4", "Byte": "u1"}


def out_dir():
    d = data_dir("buoy", "cdip")
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch(station, var, sl=""):
    """One variable through OPeNDAP .dods; sl is an optional slice such as '[0:1:99]'. Returns a NumPy array."""
    url = f"{SERVER}/{station}/{station}_{DEPLOYMENTS[station]}.nc.dods?{var}{sl}"
    with urllib.request.urlopen(url, timeout=300) as r:
        raw = r.read()
    head, _, body = raw.partition(b"\nData:\n")
    m = re.search(rb"(Float32|Float64|Int32|Byte)\s+" + var.encode() + rb"(\[.*?\])?;", head)
    if not m:
        raise RuntimeError(f"unexpected DDS for {var}: {head[:300]!r}")
    dt = DTYPES[m.group(1).decode()]
    if m.group(2) is None:  # scalar: just the value (bytes padded to 4)
        return np.frombuffer(body[:4], dtype=">f4" if dt == "u1" else dt)[:1].astype(dt)
    n = int(np.frombuffer(body[:4], ">u4")[0])  # array: length written twice, then the values
    return np.frombuffer(body[8:8 + n * np.dtype(dt).itemsize], dt).copy()


def summary(args):
    meta = {}
    for st in DEPLOYMENTS:
        t, hs, tp = fetch(st, "waveTime"), fetch(st, "waveHs"), fetch(st, "waveTp")
        m = {k: float(fetch(st, k)[0]) for k in ("xyzStartTime", "xyzSampleRate", "xyzFilterDelay",
                                                   "metaDeployLatitude", "metaDeployLongitude", "metaWaterDepth")}
        m.update(station=st, name=NAMES[st], deployment=DEPLOYMENTS[st], n_wave=int(len(t)))
        np.savez(out_dir() / f"{st}_summary.npz", waveTime=t, waveHs=hs, waveTp=tp)
        meta[st] = m
        print(f"{st} {NAMES[st]}: {len(t)} wave records, xyz at {m['xyzSampleRate']} Hz, depth {m['metaWaterDepth']} m")
    (out_dir() / "meta.json").write_text(json.dumps(meta, indent=2))


def pick_days(t, hs, tp):
    """Three UTC days: swell (median Tp nearest 14 s with Hs >= 1 m; typical rather than extreme swell), mixed (median
    Tp nearest 9.5 s) and wind sea (shortest median Tp with Hs >= 0.5 m). Days need at least 40 of 48 half-hourly
    records."""
    day = (t // 86400).astype(int)
    stats = []
    for d in np.unique(day):
        sel = (day == d) & (hs > 0) & (tp > 0)
        if sel.sum() >= 40:
            stats.append((d, float(np.median(tp[sel])), float(np.median(hs[sel]))))
    s = np.array(stats)
    swell = s[s[:, 2] >= 1.0][np.argmin(np.abs(s[s[:, 2] >= 1.0][:, 1] - 14.0))]
    mixed = s[np.argmin(np.abs(s[:, 1] - 9.5))]
    wind = s[s[:, 2] >= 0.5][np.argmin(s[s[:, 2] >= 0.5][:, 1])]
    return {"swell": swell, "mixed": mixed, "wind": wind}


def days(args):
    meta = json.loads((out_dir() / "meta.json").read_text())
    for st, m in meta.items():
        z = np.load(out_dir() / f"{st}_summary.npz")
        fs, t0, delay = m["xyzSampleRate"], m["xyzStartTime"], m["xyzFilterDelay"]
        for kind, (d, tp_med, hs_med) in pick_days(z["waveTime"], z["waveHs"], z["waveTp"]).items():
            i0 = int(round((d * 86400 - t0 + delay) * fs))
            i1 = i0 + int(86400 * fs) - 1
            sl = f"[{i0}:1:{i1}]"
            arr = {v: fetch(st, v, sl) for v in ("xyzZDisplacement", "xyzXDisplacement", "xyzYDisplacement",
                                                 "xyzFlagPrimary")}
            f = out_dir() / f"{st}_{kind}_{np.datetime64(int(d), 'D')}.npz"
            np.savez(f, z=arr["xyzZDisplacement"], north=arr["xyzXDisplacement"], west=arr["xyzYDisplacement"],
                     flag=arr["xyzFlagPrimary"], fs=fs, t_start=t0 + i0 / fs - delay, tp_median=tp_med,
                     hs_median=hs_med, station=st)
            # flags: 1 good, 2 not evaluated (CDIP's usual value for raw xyz), 3 questionable, 4 bad, 9 missing
            usable = float(np.mean(np.isin(arr["xyzFlagPrimary"], (1, 2))))
            print(f"{st} {kind:5s} {np.datetime64(int(d), 'D')}: Tp {tp_med:.1f} s, Hs {hs_med:.2f} m, "
                  f"{len(arr['xyzZDisplacement'])} samples, {usable:.0%} good or not evaluated -> {f.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["summary", "days"])
    a = ap.parse_args()
    {"summary": summary, "days": days}[a.step](a)
    sys.exit(0)
