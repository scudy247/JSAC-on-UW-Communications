"""
Side-information quantiser ranges per sea-state class (PLAN: Phase 4). Until now both ends computed the ranges and the
bit allocation from the tested record's own spectrum (in-sample). A deployed link can only agree on ranges per
sea-state class, known in advance or signalled rarely. Classes by significant wave height: calm (Hs < 1.6 m),
moderate (1.6-3 m), rough (> 3 m), over the 15 downloaded days (buoy -> seabed, transmitter = the buoy, GNSS-aided
industrial MEMS, imu.py).

For each day, the transmitter's share is quantised with ranges and allocation from (a) its own spectrum (in-sample,
as before), (b) the mean coefficient spread of the other days of its class (leave-one-out), (c) their maximum
(conservative against overload). Interference of the quantisation, sum_k w_k D_k (T2 structure, validated against
simulation), with D_k from Monte Carlo on Gaussian coefficients (overload included); receiver stages none and per
symbol; 32 bits on 1 s packets, 96 bits on 4 s packets. Output: results/b11_sea_classes/points.json.
"""
import json
import os

import numpy as np

from uwofdm import motion, placement as P, sideinfo, theory
from uwofdm.paths import output_dir
from b3_sea_states import days, hs
from c2_sweeps import seabed_link
from e2_imu_grades import node

CLASSES = (("calm", 0.0, 1.6), ("moderate", 1.6, 3.0), ("rough", 3.0, 99.0))
K = 24
BUDGET = {1.0: 32, 4.0: 96}
FC = 16e3


def mc_distortion(std_true, std_q, bits, rng, n=20000):
    """MSE per coefficient: true N(0, std_true^2) quantised with range LOAD * std_q and `bits` (0 = not sent)."""
    out = np.empty(len(std_true))
    for k, (s, q, b) in enumerate(zip(std_true, std_q, bits)):
        x = s * rng.standard_normal(n)
        out[k] = np.mean((x - sideinfo.quantise(x, int(b), np.full(n, sideinfo.LOAD * q))) ** 2) if b > 0 else s ** 2
    return out


def main():
    rng = np.random.default_rng(3)
    info = []
    for i, (source, name, lever) in enumerate(days()):
        tx, rx = seabed_link(source, name, lever)
        d = motion.link_doppler(tx, rx)
        imu = node(tx, d["u"], +1, 0.0, "industrial", 2000 + i)
        share = imu.estimate(tx.t[int(240 * tx.fs):-int(20 * tx.fs)], 0.01)
        f, S = theory.spectrum(share, tx.fs)
        h = hs(tx)
        cls = next(c for c, lo, hi in CLASSES if lo <= h < hi)
        info.append(dict(day=f"{source}:{name}", hs=h, cls=cls, f=f, S=S))
        print(f"{source}:{name:24s} Hs {h:4.2f} m -> {cls}", flush=True)
    rows = []
    for T, B in BUDGET.items():
        rxm = P.ReceiverModel(packet_s=T, stages=("none", "sym"))
        est = {st: P.stage_estimator(st, rxm) for st in ("none", "sym")}
        tg, windows = est["none"][0], est["none"][3]
        std = {x["day"]: sideinfo.coefficient_std(x["f"], x["S"], tg[-1], K) for x in info}
        w = {st: sideinfo.stage_weights(tg, est[st][1], est[st][2], windows, FC, K) for st in est}
        for x in info:
            others = [std[y["day"]] for y in info if y["cls"] == x["cls"] and y["day"] != x["day"]]
            ref = {"own": std[x["day"]], "class_mean": np.sqrt(np.mean(np.square(others), 0)),
                   "class_max": np.max(others, 0)}
            for st in est:
                res = {}
                for kind, sq in ref.items():
                    bits = sideinfo.allocate_weighted(sq, w[st], B)
                    I = float(np.sum(w[st] * mc_distortion(std[x["day"]], sq, bits, rng)))
                    res[kind] = -10 * np.log10(I)
                rows.append(dict(day=x["day"], hs_m=x["hs"], sea_class=x["cls"], packet_s=T, bits=B, stage=st,
                                 **{f"sir_quant_{k}_db": v for k, v in res.items()}))
                print(f"{x['day']:30s} {x['cls']:8s} T {T:.0f} s {st:4s}: quantisation SIR own {res['own']:5.1f} | "
                      f"class mean {res['class_mean']:5.1f} | class max {res['class_max']:5.1f} dB", flush=True)
    for T in BUDGET:
        for st in ("none", "sym"):
            sel = [r for r in rows if r["packet_s"] == T and r["stage"] == st]
            for kind in ("class_mean", "class_max"):
                loss = np.array([r["sir_quant_own_db"] - r[f"sir_quant_{kind}_db"] for r in sel])
                print(f"T {T:.0f} s {st:4s} {kind:10s}: loss vs in-sample ranges median {np.median(loss):.1f} dB, "
                      f"max {np.max(loss):.1f} dB", flush=True)
    with open(os.path.join(output_dir("b11_sea_classes"), "points.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
