"""Final-regret tables of the Layer-1 alpha x gap sweep and the T = 1e5 runs (results/b_l1*).
No randomness; reads summary.json files only."""
import glob
import json

SHORT = {"UCB (naive, oracle sigma)": "nUCB", "Gaussian TS (naive)": "nTS", "ACK UCB1": "aUCB",
         "ACK Bernoulli TS": "aTS", "Trunc-mean UCB (oracle)": "Trunc", "Trunc-mean UCB (oracle, eps=1)": "Trunc",
         "MoM UCB (oracle)": "MoM", "MoM UCB (oracle, eps=1)": "MoM", "AdaR-UCB": "AdaR",
         "NIR-UCB v0": "NIR", "NIR-UCB v0 (misspecified)": "NIR"}
COLS = ["nUCB", "nTS", "aUCB", "aTS", "Trunc", "MoM", "AdaR", "NIR"]

if __name__ == "__main__":
    rows = {}
    for d in sorted(glob.glob("results/b_l1*")):
        cfg = json.load(open(d + "/config.json"))
        rows[cfg["name"]] = json.load(open(d + "/summary.json"))
    for br in "SE":
        print(f"\nBranch {br}: final regret (mean ± 95% CI half), T=1e4, 50 seeds")
        print(f"{'point':10s}" + "".join(f"{c:>13s}" for c in COLS) + "   best")
        for a in [1.2, 1.4, 1.6, 1.8, 1.95]:
            for g in [0.5, 1, 2]:
                s = rows[f"b_l1sw_{br}_a{a:g}_g{g:g}"]
                v = {SHORT[k]: (x["final_mean"], x["final_ci95_half"]) for k, x in s["agents"].items()}
                best = min(v, key=lambda k: v[k][0])
                print(f"a{a:<4g} g{g:<3g}" + "".join(f"{v[c][0]:7.0f}±{v[c][1]:<5.0f}" for c in COLS) + f"   {best}")
        s = rows[f"b_l1long_{br}_a1.6"]
        ck = s["checkpoints"]
        i = min(range(len(ck)), key=lambda j: abs(ck[j] - 10000))
        print("LONG a1.6 g1, T=1e5, 20 seeds:")
        for k, x in s["agents"].items():
            m = x["mean"]
            print(f"   {SHORT[k]:6s} R(1e4)={m[i]:8.0f}  R(1e5)={x['final_mean']:8.0f} ± {x['final_ci95_half']:<6.0f} "
                  f"growth x{x['final_mean'] / max(m[i], 1e-9):.1f}  ({x['wall_s_mean']:.0f} s/seed)")
