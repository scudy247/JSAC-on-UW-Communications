"""
Fetch OOI surface-mooring motion data for the Phase 0 large-platform check (see code/PLAN.md).

OOI coastal and global surface buoys are about 3.3 m in diameter and about 4 t, against 0.9 m for the CDIP Waverider.
Each carries a MicroStrain 3DM-GX3-25 motion package (MOPAK: acceleration, angular rate and magnetic field in the buoy
frame, 10 Hz, one 20-minute burst per hour) and an AXYS TriAxys wave sensor (WAVSS: wave statistics every hour).

Steps (subcommands):
  python scripts/fetch_ooi.py wavss     daily WAVSS logs, one day in STEP_DAYS, for choosing sea states
  python scripts/fetch_ooi.py mopak     all MOPAK bursts of three days per mooring (swell, mixed, wind sea), chosen
                                        from the WAVSS statistics with the same rule as fetch_cdip.py

Source: https://rawdata.oceanobservatories.org/files/ (public). Output: <data>/buoy/ooi/<mooring>/{wavss,mopak}/.
Files already present are not downloaded again.
"""
import argparse
import json
import re
import urllib.request

import numpy as np

from uwofdm.paths import data_dir

SERVER = "https://rawdata.oceanobservatories.org/files"
# mooring -> recovered deployment (checked 9 Oct 2026)
MOORINGS = {"CE04OSSM": "R00015", "GI01SUMO": "R00010"}
NAMES = {"CE04OSSM": "Oregon offshore, 588 m", "GI01SUMO": "Irminger Sea, 2800 m"}
STEP_DAYS = 4


def url(mooring, sub):
    return f"{SERVER}/{mooring}/{MOORINGS[mooring]}/cg_data/{sub}"


def listing(u):
    with urllib.request.urlopen(u, timeout=120) as r:
        html = r.read().decode("utf-8", "replace")
    return re.findall(r'href="([^"/?][^"?]*)"', html)


def get(u, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(u, timeout=300) as r:
        data = r.read()
    dest.write_bytes(data)
    return True


def wavss(args):
    for m in MOORINGS:
        files = sorted(f for f in listing(url(m, "dcl12/wavss/")) if re.fullmatch(r"\d{8}\.wavss\.log", f))
        pick = files[::STEP_DAYS]
        new = sum(get(url(m, f"dcl12/wavss/{f}"), data_dir("buoy", "ooi", m, "wavss", f)) for f in pick)
        print(f"{m} ({NAMES[m]}): {len(files)} daily WAVSS logs, {len(pick)} kept, {new} downloaded")


def wave_stats(path):
    """Hourly (Hs, Tp) from the $TSPWA sentences of one WAVSS log (HMO as Hs, TP as Tp; see datafmt.txt)."""
    hs, tp = [], []
    for line in path.read_text(errors="replace").splitlines():
        i = line.find("$TSPWA")
        if i < 0:
            continue
        f = line[i:].split("*")[0].split(",")
        try:
            h, t = float(f[18]), float(f[16])
        except (IndexError, ValueError):
            continue
        if h > 0 and t > 0:
            hs.append(h)
            tp.append(t)
    return np.array(hs), np.array(tp)


def mopak(args):
    chosen = {}
    for m in MOORINGS:
        bursts = [f for f in listing(url(m, "dcl11/mopak/")) if re.fullmatch(r"\d{8}_\d{6}\.mopak\.log", f)]
        per_day = {}
        for f in bursts:
            per_day.setdefault(f[:8], []).append(f)
        stats = []
        for p in sorted(data_dir("buoy", "ooi", m, "wavss").glob("*.wavss.log")):
            day = p.name[:8]
            hs, tp = wave_stats(p)
            if len(hs) >= 20 and len(per_day.get(day, [])) >= 20:  # nearly full day of waves and of bursts
                stats.append((day, float(np.median(tp)), float(np.median(hs))))
        if not stats:
            raise SystemExit(f"{m}: no day with both WAVSS statistics and MOPAK bursts")
        days = np.array([s[0] for s in stats])
        tp, hs = np.array([s[1] for s in stats]), np.array([s[2] for s in stats])
        big, mid = hs >= 1.0, hs >= 0.5
        pick = {"swell": days[big][np.argmin(np.abs(tp[big] - 14.0))] if big.any() else None,
                "mixed": days[np.argmin(np.abs(tp - 9.5))],
                "wind": days[mid][np.argmin(tp[mid])] if mid.any() else None}
        chosen[m] = {}
        for kind, day in pick.items():
            if day is None:
                print(f"{m} {kind}: no suitable day")
                continue
            j = int(np.where(days == day)[0][0])
            files = sorted(per_day[day])
            new = sum(get(url(m, f"dcl11/mopak/{f}"), data_dir("buoy", "ooi", m, "mopak", day, f)) for f in files)
            chosen[m][kind] = {"day": str(day), "tp_median": tp[j], "hs_median": hs[j], "bursts": len(files)}
            print(f"{m} {kind:5s} {day}: Tp {tp[j]:.1f} s, Hs {hs[j]:.2f} m, {len(files)} bursts, {new} downloaded")
    (data_dir("buoy", "ooi") / "chosen_days.json").write_text(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["wavss", "mopak"])
    a = ap.parse_args()
    {"wavss": wavss, "mopak": mopak}[a.step](a)
