"""
Phase 0, large platform: predictability of the motion of OOI surface buoys (about 3.3 m, 4 t), to set against the
0.9 m CDIP Waverider of scripts/motion_predictability.py.

Input: MOPAK bursts from scripts/fetch_ooi.py (MicroStrain 3DM-GX3-25, 10 Hz, 20 min per hour, buoy frame,
north-east-down axes; acceleration in g, angular rate in rad/s, magnetic field in gauss) and the WAVSS logs of the
same buoy (for checking).

Processing per burst:
  1. decode the 0xCB records (43 bytes, checksum verified; 62 500 timer ticks per second);
  2. attitude by a complementary (Mahony-type) filter: gyro integration (burst-mean rate removed as gyro offset),
     corrected towards gravity (tilt) and the magnetic field (heading) with a 30 s time constant, so that wave
     accelerations average out;
  3. acceleration in the earth frame (magnetic north, east, down) minus gravity;
  4. velocity and displacement by spectral integration, keeping periods shorter than about 33 s (as wave buoys do);
     30 s are dropped at each end of the burst.
Check: Hs = 4 std(vertical displacement) against the WAVSS HMO of the same hour.

Prediction: as in motion_predictability.py, with noisy measurements every 0.8, 1.6, 2.4 or 4.0 s and a best linear
predictor on the last 60 s of velocity and acceleration, fitted on the first 12 bursts of the day and tested on the
others (windows never cross a burst boundary).

Outputs (results/ooi_motion/): metrics.json and printed tables. Usage:
  python scripts/ooi_motion.py [--noise-v 0.02] [--noise-a 0.02]
"""
import argparse
import json
import os
import re

import numpy as np

from uwofdm.buoy import TRIM, decode, earth_acceleration, integrate
from uwofdm.paths import data_dir, output_dir

FS = 10.0
STRIDES = (8, 16, 24, 40)  # measurement spacing in samples: 0.8, 1.6, 2.4, 4.0 s
H_SAMPLES = np.arange(8, 104, 8)  # horizons 0.8 ... 10.4 s
MEMORY_S = 60.0
N_TRAIN_BURSTS = 12


# --------------------------------------------------------------------------- decoding and attitude

# --------------------------------------------------------------------------- data per day
def wavss_hmo(mooring, day):
    """Hourly HMO from the WAVSS log of the day (if downloaded), keyed by hour."""
    p = data_dir("buoy", "ooi", mooring, "wavss", f"{day}.wavss.log")
    out = {}
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            i = line.find("$TSPWA")
            if i >= 0:
                f = line[i:].split("*")[0].split(",")
                try:
                    out[int(f[2][:2])] = float(f[18])
                except (IndexError, ValueError):
                    pass
    return out


def load_day(mooring, day):
    """Per burst: hour, vertical (up) and north velocity and acceleration, Hs from the motion."""
    bursts = []
    for p in sorted(data_dir("buoy", "ooi", mooring, "mopak", day).glob("*.mopak.log")):
        t, f, w, m = decode(p)
        if len(t) < 6000:
            continue
        ae = earth_acceleration(t, f, w, m)
        rec = {"hour": int(re.match(r"\d{8}_(\d{2})", p.name).group(1))}
        for name, comp, sign in (("vertical", 2, -1.0), ("north", 0, 1.0)):
            v, a = integrate(sign * ae[:, comp], 1)
            rec[name] = (v[TRIM:-TRIM], a[TRIM:-TRIM])
        z, _ = integrate(-ae[:, 2], 2)
        rec["hs"] = 4 * float(z[TRIM:-TRIM].std())
        bursts.append(rec)
    return bursts


# --------------------------------------------------------------------------- prediction
def design(v, a, k, h, p, nv, na, rng):
    """Rows of one burst: features = last p noisy (v, a) measurements spaced k, target = v and a at +h."""
    vm, am = v + nv * rng.standard_normal(len(v)), a + na * rng.standard_normal(len(a))
    idx = np.arange((p - 1) * k, len(v) - h, k)
    Xv = np.stack([vm[idx - j * k] for j in range(p)], 1)
    Xa = np.stack([am[idx - j * k] for j in range(p)], 1)
    return Xv, Xa, v[idx + h], a[idx + h]


def evaluate(bursts, axis, k, nv, na, rng):
    p = max(2, int(round(MEMORY_S * FS / k)))
    out = {"hold_v": [], "ca_v": [], "lp_v": [], "hold_a": [], "lp_a": []}
    for h in H_SAMPLES:
        parts = [design(*b[axis], k, h, p, nv, na, rng) for b in bursts]
        tr, te = parts[:N_TRAIN_BURSTS], parts[N_TRAIN_BURSTS:]
        cat = lambda ps, i: np.concatenate([x[i] for x in ps])
        Xv_tr, Xa_tr, yv_tr, ya_tr = (cat(tr, i) for i in range(4))
        Xv_te, Xa_te, yv_te, ya_te = (cat(te, i) for i in range(4))
        rmse = lambda pred, y: float(np.sqrt(np.mean((pred - y) ** 2)))
        out["hold_v"].append(rmse(Xv_te[:, 0], yv_te))
        out["ca_v"].append(rmse(Xv_te[:, 0] + Xa_te[:, 0] * h / FS, yv_te))
        out["hold_a"].append(rmse(Xa_te[:, 0], ya_te))
        A_tr = np.hstack([Xv_tr, Xa_tr, np.ones((len(Xv_tr), 1))])
        A_te = np.hstack([Xv_te, Xa_te, np.ones((len(Xv_te), 1))])
        for name, y_tr, y_te in (("lp_v", yv_tr, yv_te), ("lp_a", ya_tr, ya_te)):
            coef = np.linalg.lstsq(A_tr, y_tr, rcond=None)[0]
            out[name].append(rmse(A_te @ coef, y_te))
    return out


def band_shares(bursts, axis):
    """Share of velocity variance below 0.1 Hz, 0.1-0.2 Hz, above 0.2 Hz (averaged over bursts)."""
    s = []
    for b in bursts:
        v = b[axis][0]
        F = np.fft.rfftfreq(len(v), 1 / FS)
        P = np.abs(np.fft.rfft(v)) ** 2
        s.append([P[F < 0.1].sum(), P[(F >= 0.1) & (F < 0.2)].sum(), P[F >= 0.2].sum()] / P.sum())
    return np.mean(s, 0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-v", type=float, default=0.02)
    ap.add_argument("--noise-a", type=float, default=0.02)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    chosen = json.loads(data_dir("buoy", "ooi", "chosen_days.json").read_text())
    res = {"noise_v": args.noise_v, "noise_a": args.noise_a, "strides_s": [k / FS for k in STRIDES],
           "horizons_s": (H_SAMPLES / FS).tolist(), "days": {}}
    for mooring, kinds in chosen.items():
        for kind, info in kinds.items():
            bursts = load_day(mooring, info["day"])
            hmo = wavss_hmo(mooring, info["day"])
            pairs = [(b["hs"], hmo[b["hour"]]) for b in bursts if b["hour"] in hmo]
            name = f"{mooring}_{kind}_{info['day']}"
            day = {"tp_median": info["tp_median"], "hs_median": info["hs_median"], "bursts": len(bursts),
                   "hs_motion_vs_wavss": pairs, "axes": {}}
            for axis in ("vertical", "north"):
                std_v = float(np.concatenate([b[axis][0] for b in bursts[N_TRAIN_BURSTS:]]).std())
                std_a = float(np.concatenate([b[axis][1] for b in bursts[N_TRAIN_BURSTS:]]).std())
                day["axes"][axis] = {"std_v": std_v, "std_a": std_a, "band_shares": band_shares(bursts, axis),
                                     "rmse": {str(k): evaluate(bursts, axis, k, args.noise_v, args.noise_a, rng)
                                              for k in STRIDES}}
            res["days"][name] = day
            hs_m = np.mean([p[0] for p in pairs]) if pairs else float("nan")
            hs_w = np.mean([p[1] for p in pairs]) if pairs else float("nan")
            print(f"{name}: {len(bursts)} bursts; Hs from motion {hs_m:.2f} m vs WAVSS {hs_w:.2f} m")
    out = output_dir("ooi_motion")
    with open(os.path.join(out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    report(res)


def report(res):
    hs = np.array(res["horizons_s"])
    k = str(STRIDES[2])
    print(f"\nNRMSE of velocity prediction (vertical | north), noisy measurements every {STRIDES[2] / FS:.1f} s:")
    print(f"{'day':28s} {'H':>4s}  {'hold':>11s} {'ca':>11s} {'lp':>11s}   lp RMSE north [m/s]   var. share <0.1|0.1-0.2|>0.2 Hz (north)")
    for name, day in res["days"].items():
        for j in (2, 3):  # H = 2.4 s, 3.2 s
            cells = []
            for m in ("hold_v", "ca_v", "lp_v"):
                cells.append("|".join(f"{day['axes'][ax]['rmse'][k][m][j] / day['axes'][ax]['std_v']:5.2f}"
                                      for ax in ("vertical", "north")))
            sh = day["axes"]["north"]["band_shares"]
            print(f"{name:28s} {hs[j]:4.1f}  " + " ".join(cells)
                  + f"   {day['axes']['north']['rmse'][k]['lp_v'][j]:.3f}" + (f"                 {sh[0]:.2f}|{sh[1]:.2f}|{sh[2]:.2f}" if j == 2 else ""))


if __name__ == "__main__":
    main()
