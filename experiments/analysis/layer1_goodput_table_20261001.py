"""Layer-1 goodput grid (results/b_l1gp_*): final regret (goodput units, bits/symbol x slots) per agent, the best
agent per point, and NIR-UCB v1-goodput vs ACK TS and the AWGN-calibrated structured learner (paired per seed).
No randomness; reads summary.json files only."""
import glob
import json

import numpy as np

from experiments.make_b_layer1_goodput import SNRS
from experiments.make_b_layer1_sweep import ALPHAS

SHORT = {"ACK TS (per arm)": "ackTS", "ACK UCB1 (per arm)": "ackUCB", "Structured UCB (AWGN-calibrated)": "strAWGN",
         "Structured UCB (calibrated at alpha=1.2)": "str1.2", "NIR-UCB v1-goodput (gauss)": "v1g",
         "NIR-UCB v1-goodput (bernstein)": "v1b", "Threshold AMC (median W=16)": "AMC"}
COLS = list(SHORT.values())


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(d.size)


if __name__ == "__main__":
    rows = {}
    for d in glob.glob("results/b_l1gp_*"):
        cfg = json.load(open(d + "/config.json"))
        s = json.load(open(d + "/summary.json"))
        rows[cfg["name"]] = ({SHORT[k]: v for k, v in s["agents"].items()}, max(s["true_means"]))
    print("final regret (mean ± 95% CI half), T = 1e4, 50 seeds; best = lowest mean; paired = v1g minus other")
    print(f"{'point':11s}" + "".join(f"{c:>12s}" for c in COLS) + "   best    v1g-ackTS      v1g-strAWGN")
    for a in ALPHAS:
        for snr in SNRS:
            v, gmax = rows[f"b_l1gp_a{a:g}_s{snr:g}"]
            best = min(COLS, key=lambda c: v[c]["final_mean"])
            pa = paired(v["v1g"]["final_per_seed"], v["ackTS"]["final_per_seed"])
            pw = paired(v["v1g"]["final_per_seed"], v["strAWGN"]["final_per_seed"])
            print(f"a{a:<4g} s{snr:<4g}" + "".join(f"{v[c]['final_mean']:7.0f}±{v[c]['final_ci95_half']:<4.0f}" for c in COLS)
                  + f"   {best:6s} {pa[0]:6.0f}±{pa[1]:<5.0f} {pw[0]:7.0f}±{pw[1]:<5.0f}  (best goodput {gmax:.2f})")
