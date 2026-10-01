"""NIR-UCB v1 checkpoint grid (results/b_l1v1_*): final regret per agent, and v1 relative to the references.
No randomness; reads summary.json files only."""
import glob
import json

from experiments.make_b_layer1_sweep import ALPHAS, GAP_SCALES

SHORT = {"UCB (naive, oracle sigma)": "oracle", "UCB (naive, calibrated at alpha=2)": "cal2",
         "UCB (naive, calibrated at alpha=1.2)": "cal1.2", "ACK Bernoulli TS": "ackTS",
         "NIR-UCB v1 (gauss)": "v1g", "NIR-UCB v1 (bernstein)": "v1b"}

if __name__ == "__main__":
    rows = {}
    for d in glob.glob("results/b_l1v1_*"):
        cfg = json.load(open(d + "/config.json"))
        rows[cfg["name"]] = {SHORT[k]: (v["final_mean"], v["final_ci95_half"])
                             for k, v in json.load(open(d + "/summary.json"))["agents"].items()}
    for br, cols in (("E", ["oracle", "cal2", "cal1.2", "ackTS", "v1g", "v1b"]), ("S", ["oracle", "ackTS", "v1g"])):
        print(f"\nBranch {br}: final regret (mean ± 95% CI half), T = 1e4, 50 seeds")
        print(f"{'point':11s}" + "".join(f"{c:>14s}" for c in cols) + "   v1g/oracle  v1g vs ackTS  v1g vs best fixed cal")
        wins = {"match": 0, "beat_ack": 0, "beat_cal": 0, "n": 0}
        for a in ALPHAS:
            for g in GAP_SCALES:
                v = rows[f"b_l1v1_{br}_a{a:g}_g{g:g}"]
                line = f"a{a:<4g} g{g:<4g}" + "".join(f"{v[c][0]:8.0f}±{v[c][1]:<5.0f}" for c in cols)
                r_or = v["v1g"][0] / v["oracle"][0]
                ack = "better" if v["v1g"][0] + v["v1g"][1] < v["ackTS"][0] - v["ackTS"][1] else (
                      "worse" if v["v1g"][0] - v["v1g"][1] > v["ackTS"][0] + v["ackTS"][1] else "tie")
                line += f"   {r_or:6.2f}      {ack:7s}"
                wins["n"] += 1
                wins["match"] += r_or <= 1.2
                wins["beat_ack"] += ack == "better"
                if br == "E":
                    bc = min(v["cal2"][0], v["cal1.2"][0])
                    line += f"   {v['v1g'][0] / bc:5.2f}"
                    wins["beat_cal"] += v["v1g"][0] < bc
                print(line)
        print(f"  summary {br}: v1g within 20 % of oracle in {wins['match']}/{wins['n']}; better than ACK TS (CIs "
              f"disjoint) in {wins['beat_ack']}/{wins['n']}" + (f"; below the better fixed calibration in "
              f"{wins['beat_cal']}/{wins['n']}" if br == "E" else ""))
