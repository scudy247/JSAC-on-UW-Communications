"""
T2: closed form for the side-information bits (PLAN: Phase 4, theory), checked against the E1 simulations
(results/e1_sideinfo/stage_allocation.json: time-domain scoring on 200 real-motion packets, stage-aware allocation).

Closed form: interference I(B) = I_floor + sum_k w_k D_k(b_k) (sideinfo.predicted_interference), with
  std_k   Legendre-coefficient std of the transmitter's IMU share from its spectrum (sideinfo.coefficient_std)
  w_k     interference of a unit error in coefficient k after the receiver's stage (sideinfo.stage_weights)
  D_k     std_k^2 if not sent, else the uniform quantiser's (2 LOAD std_k)^2 / 12 / 4^b_k
  I_floor the simulated "transmitter applies its estimate" reference (sensor errors, receiver stage)
For every simulated allocation the predicted SIR is set against the simulated one. Rate-distortion limit
(sideinfo.rd_bits): minimum bits to come within 1 dB of the floor with ideal quantisation, vs the bits the uniform
quantiser needs (closed form, greedy allocation) and the simulated count.
Usage: python scripts/b10_t2_bits.py. Output: results/b10_t2_bits/points.json.
"""
import json
import os

import numpy as np

from uwofdm import placement as P, sideinfo, theory
from uwofdm.paths import output_dir, results_dir
from e1_allocation import BUDGETS, K_MAX
from e1_sideinfo import FC, SCENARIOS, STAGES, series

K_TAIL = 40  # coefficients used for the truncation tail beyond K_MAX


def main():
    sim = json.load(open(os.path.join(results_dir(), "e1_sideinfo", "stage_allocation.json")))
    rng = np.random.default_rng(1)
    rows, errs = [], []
    for label, (make, rx_moves) in SCENARIOS.items():
        s = series(*make(), rx_moves, rng)
        for T in (1.0, 4.0):
            rxm = P.ReceiverModel(packet_s=T, stages=STAGES)
            est = {st: P.stage_estimator(st, rxm) for st in STAGES}
            tg, windows = est["none"][0], est["none"][3]
            f, S = theory.spectrum(s["tx_est"], s["fs"])
            std = sideinfo.coefficient_std(f, S, tg[-1], K_TAIL)
            for st in STAGES:
                w = sideinfo.stage_weights(tg, est[st][1], est[st][2], windows, FC, K_TAIL)
                sel = [r for r in sim if r["scenario"] == label and r["packet_s"] == T and r["stage"] == st]
                ref = {r["reference"]: r["sir_db"] for r in sel if "reference" in r}
                floor = 10 ** (-ref["applied"] / 10)
                line = []
                for r in sorted((r for r in sel if "bits" in r), key=lambda r: r["bits"]):
                    bits = np.concatenate([r["allocation"], np.zeros(K_TAIL - K_MAX, int)])
                    pred = -10 * np.log10(sideinfo.predicted_interference(std, w, bits, floor))
                    rows.append(dict(scenario=label, packet_s=T, stage=st, bits=r["bits"], sir_sim_db=r["sir_db"],
                                     sir_pred_db=float(pred)))
                    errs.append(pred - r["sir_db"])
                    line.append(f"{r['bits']}: {r['sir_db']:.1f}/{pred:.1f}")
                target = floor * (10 ** 0.1 - 1)  # within 1 dB of the floor
                b_rd, _ = sideinfo.rd_bits(std, w, target)
                b_unif = next((b for b in range(1, 400) if sideinfo.predicted_interference(
                    std, w, sideinfo.allocate_weighted(std, w, b), 0.0) <= target), None)
                b_sim = next((r["bits"] for r in sorted((r for r in sel if "bits" in r), key=lambda r: r["bits"])
                              if r["sir_db"] >= ref["applied"] - 1.0), None)
                rows.append(dict(scenario=label, packet_s=T, stage=st, applied_db=ref["applied"],
                                 bits_rd=b_rd, bits_uniform_closed_form=b_unif, bits_simulated=b_sim))
                print(f"{label:30s} {T:.0f} s {st:5s}: bits within 1 dB of applying: rate-distortion {b_rd:5.1f}, "
                      f"uniform quantiser (closed form) {b_unif}, simulated {b_sim} | SIR sim/pred by bits: "
                      + "  ".join(line[::2]), flush=True)
    e = np.array(errs)
    print(f"prediction - simulation over {len(e)} points: median {np.median(e):+.2f} dB, |.| p90 "
          f"{np.percentile(np.abs(e), 90):.2f} dB, max |.| {np.max(np.abs(e)):.2f} dB")
    with open(os.path.join(output_dir("b10_t2_bits"), "points.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
