"""
Phase 0, step 1 (code/PLAN.md): how far ahead real wave-driven buoy motion can be predicted.

Input: CDIP displacement days in <data>/buoy/cdip/ (scripts/fetch_cdip.py). For each day and axis (vertical, north),
velocity v and acceleration a are obtained by spectral differentiation of the displacement. These are accurate values
of the true motion at each sample time, so using past values as predictor inputs leaks no future information.

Question: given measurements up to time t, how well can v(t + H) (what transmitter pre-compensation needs) and
a(t + H) (what the receiver needs) be predicted, as a function of the horizon H?

Measurements: every k-th sample (k = 1, 2, 3, 5 -> spacing 0.78, 1.56, 2.34, 3.91 s, i.e. dense to one-way exchange
at about 2 km), optionally with white noise (sigma_v on velocity, sigma_a on acceleration).
Predictors of v(t+H):
  hold      last measured value
  ca        per-packet kinematic extrapolation v(t) + a(t) H (a measured too: the best a per-packet receiver can do)
  lp        best linear predictor: least squares on the last p measurements (covering 60 s), fitted on the first
            6 h of the day and tested on the rest; a stand-in for the state-space forecaster still to be written
Same for a(t+H) (hold and lp; ca would need the jerk).
Metric: RMSE on the test part, also normalised by the standard deviation of the target (NRMSE; 1 = no skill
beyond the mean, about 1.41 for a badly-timed hold).

Platform: by default the motion of the buoy itself (a 0.9 m Datawell Waverider, which follows the waves). With
--platform-lowpass F the displacement is low-pass filtered at F Hz (2nd-order Butterworth, zero phase) first: a crude
model of a larger platform (big buoy, spar, ship) that does not follow short waves.

Outputs (results/motion_predictability/): metrics*.json, predictability*.png. Usage:
  python scripts/motion_predictability.py [--noise-v 0.02] [--noise-a 0.02] [--platform-lowpass 0.1]
"""
import argparse
import glob
import json
import os

import numpy as np
import scipy.signal as sg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uwofdm.buoy import derivatives
from uwofdm.paths import data_dir, output_dir

STRIDES = (1, 2, 3, 5)  # measurement every k samples
H_SAMPLES = np.arange(1, 14)  # horizons in samples (0.78 ... 10.2 s at 1.28 Hz)
TRAIN_HOURS = 6.0
MEMORY_S = 60.0
C = 1500.0



def lags(y, k, p):
    """Rows [y[i], y[i-k], ..., y[i-(p-1)k]] for every index i with a full history."""
    idx = np.arange((p - 1) * k, len(y))
    return idx, np.stack([y[idx - j * k] for j in range(p)], axis=1)


def evaluate(v, a, fs, k, noise_v, noise_a, rng):
    """RMSE of each predictor, per horizon, on the test part of the day."""
    vm = v + noise_v * rng.standard_normal(len(v))
    am = a + noise_a * rng.standard_normal(len(a))
    p = max(2, int(round(MEMORY_S * fs / k)))
    n_train = int(TRAIN_HOURS * 3600 * fs)
    idx, Xv = lags(vm, k, p)
    _, Xa = lags(am, k, p)
    idx = idx[::k]  # measurement instants only
    Xv, Xa = Xv[::k], Xa[::k]
    out = {"hold_v": [], "ca_v": [], "lp_v": [], "hold_a": [], "lp_a": []}
    for h in H_SAMPLES:
        ok = idx + h < len(v)
        i, xv, xa = idx[ok], Xv[ok], Xa[ok]
        tv, ta = v[i + h], a[i + h]
        tr, te = i < n_train - h, i >= n_train
        rmse = lambda pred, true: float(np.sqrt(np.mean((pred[te] - true[te]) ** 2)))
        out["hold_v"].append(rmse(xv[:, 0], tv))
        out["ca_v"].append(rmse(xv[:, 0] + xa[:, 0] * h / fs, tv))
        out["hold_a"].append(rmse(xa[:, 0], ta))
        for name, X, true in (("lp_v", np.hstack([xv, xa]), tv), ("lp_a", np.hstack([xv, xa]), ta)):
            A = np.hstack([X, np.ones((len(X), 1))])
            coef = np.linalg.lstsq(A[tr], true[tr], rcond=None)[0]
            out[name].append(rmse(A @ coef, true))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-v", type=float, default=0.0, help="measurement noise on velocity [m/s]")
    ap.add_argument("--noise-a", type=float, default=0.0, help="measurement noise on acceleration [m/s^2]")
    ap.add_argument("--platform-lowpass", type=float, default=0.0, help="low-pass the motion at this frequency [Hz]")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    files = sorted(glob.glob(str(data_dir("buoy", "cdip", "*_*_20*.npz"))))
    if not files:
        raise SystemExit("no CDIP days found: run scripts/fetch_cdip.py summary, then days")
    res = {"noise_v": args.noise_v, "noise_a": args.noise_a, "platform_lowpass_hz": args.platform_lowpass,
           "strides": STRIDES, "train_hours": TRAIN_HOURS,
           "memory_s": MEMORY_S, "days": {}}
    for f in files:
        d = np.load(f)
        fs = float(d["fs"])
        name = os.path.basename(f)[:-4]
        res["days"][name] = {"fs": fs, "tp_median": float(d["tp_median"]), "hs_median": float(d["hs_median"]),
                             "horizons_s": (H_SAMPLES / fs).tolist(), "axes": {}}
        for axis, x in (("vertical", d["z"]), ("north", d["north"])):
            x = x.astype(float)
            if args.platform_lowpass:
                x = sg.filtfilt(*sg.butter(2, args.platform_lowpass / (fs / 2)), x)
            v, a = derivatives(x, fs)
            r = {"std_v": float(v.std()), "std_a": float(a.std()), "rmse": {}}
            for k in STRIDES:
                r["rmse"][str(k)] = evaluate(v, a, fs, k, args.noise_v, args.noise_a, rng)
            res["days"][name]["axes"][axis] = r
        print(f"{name}: Tp {res['days'][name]['tp_median']:.1f} s, Hs {res['days'][name]['hs_median']:.2f} m")
    out = output_dir("motion_predictability")
    tag = (f"_nv{args.noise_v:g}_na{args.noise_a:g}" if (args.noise_v or args.noise_a) else "") \
        + (f"_lp{args.platform_lowpass:g}" if args.platform_lowpass else "")
    with open(os.path.join(out, f"metrics{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    report(res, out, tag)


def report(res, out, tag):
    """Summary table at H around 2.3 s and 3.1 s (2-3 s ahead, as at about 2 km) and the NRMSE figure."""
    print("\nNRMSE of velocity prediction (vertical | north), measurements every 2.34 s (k = 3):")
    print(f"{'day':32s} {'H':>5s}  {'hold':>11s} {'ca':>11s} {'lp':>11s}   lp RMSE v [m/s] | eps")
    for name, day in res["days"].items():
        hs = np.array(day["horizons_s"])
        for j in (2, 3):  # H = 2.34 s, 3.13 s
            cells = {m: [] for m in ("hold_v", "ca_v", "lp_v")}
            for axis in ("vertical", "north"):
                ax = day["axes"][axis]
                for m in cells:
                    cells[m].append(ax["rmse"]["3"][m][j] / ax["std_v"])
            lp_abs = day["axes"]["north"]["rmse"]["3"]["lp_v"][j]
            print(f"{name:32s} {hs[j]:5.2f}  " + " ".join(f"{c[0]:5.2f}|{c[1]:5.2f}" for c in cells.values())
                  + f"   {lp_abs:.3f} | {lp_abs / C:.1e}")
    names = list(res["days"])
    stations = sorted({n.split("_")[0] for n in names})
    kinds = ("swell", "mixed", "wind")
    fig, axs = plt.subplots(len(kinds), len(stations), figsize=(5 * len(stations), 3.4 * len(kinds)), squeeze=False)
    cols = {"hold_v": "tab:gray", "ca_v": "tab:orange", "lp_v": "tab:blue"}
    for name in names:
        st, kind = name.split("_")[:2]
        if kind not in kinds:
            continue
        r, cc = kinds.index(kind), stations.index(st)
        day = res["days"][name]
        hs = np.array(day["horizons_s"])
        ax = axs[r, cc]
        axd = day["axes"]["north"]
        for m, col in cols.items():
            for k, ls in (("1", ":"), ("3", "-")):
                ax.plot(hs, np.array(axd["rmse"][k][m]) / axd["std_v"], ls, color=col,
                        label=f"{m[:-2]}, every {int(k) / day['fs']:.2f} s")
        ax.set(title=f"{name} (north velocity)", xlabel="horizon H [s]", ylabel="NRMSE", ylim=(0, 1.5))
        ax.axvspan(2, 3, color="tab:green", alpha=0.08)
        ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"predictability{tag}.png"), dpi=120)


if __name__ == "__main__":
    main()
