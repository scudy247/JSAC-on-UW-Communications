"""
Placement rule, validation (PLAN: Phase 3): the evaluator (uwofdm/placement.py) against the simulated sweep points.

For every point of results/c2_sweeps/*.json and results/c2_robustness/direction.json the same link and settings are
rebuilt, all placements are evaluated, and for the simulated strategies the predicted SIR is set beside the simulated
one. Reported: whether the predicted best simulated strategy is the simulated best (and the loss if not), the
prediction error, and the best placement overall (which may be one the harness has not simulated yet).
Output: results/c2_rule/validation.json; printed summary. Closed forms only, single thread (a few minutes: the band-limited
interpolation of day-long records dominates). Usage: python scripts/c2_rule.py [sweep ...] (default: all).
"""
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

from uwofdm import motion, placement as P
from uwofdm.paths import output_dir
from c2_sweeps import CDIP_DAY, OOI_DAY, buoy_buoy_link, seabed_link

LINKS = {"seabed_cdip": lambda: seabed_link("cdip", CDIP_DAY),
         "seabed_ooi": lambda: seabed_link("ooi", OOI_DAY, (0.0, 0.0, 2.0)),
         "buoy_buoy_ooi": buoy_buoy_link}
_cache = {}


def link(key, make):
    if key not in _cache:
        _cache[key] = make()
    return _cache[key]


def point_setup(sweep, p):
    """(tx, rx, NodeSensors kwargs, ReceiverModel kwargs) of one sweep point (see c2_sweeps.py, c2_robustness.py)."""
    sen, rxm = {}, {}
    if sweep in ("imu", "direction"):
        tx, rx = link(p["scenario"], LINKS[p["scenario"]])
        sen["u_error_deg"] = float(p.get("u_error_deg", 0.0))
    elif sweep == "latency":
        tx, rx = link("seabed_ooi", LINKS["seabed_ooi"])
        sen.update(latency=float(p["latency_s"]), extrapolate=bool(p["extrapolate"]))
    elif sweep == "lever":
        arm = float(p["lever_m"])
        tx, rx = link(("lever", arm), lambda: seabed_link("ooi", OOI_DAY, (0.0, 0.0, arm)))
        sen["lever_corrected"] = bool(p["corrected"])
    elif sweep == "packet":
        tx, rx = link("buoy_buoy_ooi", buoy_buoy_link)
        rxm["packet_s"] = float(p["packet_s"])
    elif sweep == "range":
        r = float(p["range_m"])
        tx, rx = link(("range", r), lambda: buoy_buoy_link(range_m=r))
        rxm["stale_h"] = float(p["stale_h"])
    elif sweep == "sea":
        source, name = p["day"].split(":", 1)
        lever = (0.0, 0.0, 2.0) if source == "ooi" else None
        tx, rx = link(("sea", p["day"]), lambda: seabed_link(source, name, lever))
        rxm["packet_s"] = float(p["packet_s"])
    else:
        raise ValueError(sweep)
    return tx, rx, sen, rxm


def strategy_sigma(name):
    m = re.match(r"(.+)_([0-9.e-]+)$", name)
    return (m.group(1), float(m.group(2))) if m else (name, None)


def load_points(only=()):
    files = [(os.path.basename(f)[:-5], f) for f in sorted(output_dir("c2_sweeps").glob("*.json"))]
    files.append(("direction", str(output_dir("c2_robustness") / "direction.json")))
    for sweep, path in files:
        if only and sweep not in only:
            continue
        groups = defaultdict(list)
        for row in json.load(open(path)):
            base, sg = strategy_sigma(row["strategy"])
            point = {k: v for k, v in row.items() if k not in ("strategy", "sir_sim_db", "sir_sim_p10_db", "sir_theory_db")}
            groups[(json.dumps(point, sort_keys=True), sg)].append(dict(row, base=base))
        # strategies without a sigma (receiver only) belong to every sigma of the same point
        for (pt, sg), rows in list(groups.items()):
            if sg is not None and (pt, None) in groups:
                rows += groups[(pt, None)]
        for (pt, sg), rows in groups.items():
            if sg is None and any(k[0] == pt and k[1] is not None for k in groups):
                continue
            yield sweep, json.loads(pt), sg, rows


def main():
    out, n, same, loss, errs = [], 0, 0, [], defaultdict(list)
    only = sys.argv[1:]
    for sweep, point, sg, rows in load_points(only):
        tx, rx, sen, rxm = point_setup(sweep, point)
        s = P.NodeSensors(sigma=sg if sg is not None else 0.01, **sen)
        fs, comps = P.link_components(tx, rx, s, s)
        ev = P.evaluate(fs, comps, P.ReceiverModel(**rxm))
        pred = {r["name"]: r["sir_db"] for r in ev}
        sim = {r["base"]: r["sir_sim_db"] for r in rows}
        common = [k for k in sim if k in pred]
        best_sim = max(common, key=sim.get)
        best_pred = max(common, key=pred.get)
        n += 1
        same += best_sim == best_pred
        loss.append(sim[best_sim] - sim[best_pred])
        for k in common:
            errs[k].append(pred[k] - sim[k])
        out.append(dict(sweep=sweep, point=point, sigma_imu=sg, sim=sim, pred={k: pred[k] for k in common},
                        best_sim=best_sim, best_pred=best_pred, best_overall=ev[0]["name"],
                        best_overall_sir_db=ev[0]["sir_db"], ranking=[(r["name"], r["sir_db"]) for r in ev[:5]]))
        print(f"{sweep:9s} {json.dumps(point):60s} sim best {best_sim:8s} {sim[best_sim]:5.1f}  pred best {best_pred:8s}"
              f"  overall {ev[0]['name']:22s} {ev[0]['sir_db']:5.1f}", flush=True)
    print(f"\npoints {n}: same winner {same}, loss when not: max {max(loss):.1f} dB")
    for k, e in sorted(errs.items()):
        e = np.array(e)
        print(f"  {k:9s} prediction - simulation: median {np.median(e):5.1f} dB, |.| p90 {np.percentile(abs(e), 90):4.1f} dB (n={len(e)})")
    with open(os.path.join(output_dir("c2_rule"), "validation" + ("_" + "_".join(only) if only else "") + ".json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)  # overwritten by each run


if __name__ == "__main__":
    main()
