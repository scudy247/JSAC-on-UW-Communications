"""
Diffusely scattered surface power as a swept parameter (PLAN: Phase 1, Bellhop paths; robustness of C2).

At 16 kHz with the measured sea states the coherent surface reflection is gone (bellhop.py, rough surface), but part
of the scattered energy reaches the receiver, spread in delay and Doppler. Its level depends on the bistatic surface
scattering strength, which is not modelled; it is therefore swept: rho = scattered surface power relative to the
coherent channel, -40 ... 0 dB. Its Doppler variance D^2 var(w) comes from the smooth-surface Bellhop profile of the
same geometry (bellhop.diffuse_doppler_factor) and the surface velocity of the buoy record.
For each scenario and placement: SIR = -10 log10(I_endpoint + I_scatter), I_endpoint from the evaluator
(placement.evaluate, interference kernel), I_scatter = diffuse_floor * (pi fc T)^2 / 3.
Reported: SIR vs rho per placement, and the rho at which split's margin over the per-symbol receiver falls below
1 dB (placement stops mattering).
Output: results/surface_scatter/sweep.json, sweep.png. Closed forms + Bellhop: a few minutes, single thread.
"""
import json
import os

import numpy as np

from uwofdm import bellhop, placement as P
from uwofdm.paths import output_dir
from c2_sweeps import CDIP_DAY, OOI_DAY, buoy_buoy_link, seabed_link

RHO_DB = np.arange(-40.0, 0.1, 1.0)
FC, T_FFT = 16e3, 0.128
SCENARIOS = {  # label: (link, Bellhop geometry: range, z_tx, z_rx, depth; placements shown)
    "buoy -> seabed, CDIP swell": (lambda: seabed_link("cdip", CDIP_DAY), (1000.0, 2.0, 99.5, 100.0),
                                   ("rx_sym", "rx_pkt2", "split", "own[tx]+rx[quad]")),
    "buoy -> seabed, OOI Irminger": (lambda: seabed_link("ooi", OOI_DAY, (0.0, 0.0, 2.0)), (1000.0, 2.0, 99.5, 100.0),
                                     ("rx_sym", "rx_pkt2", "split", "own[tx]+rx[quad]")),
    "buoy - buoy, OOI Irminger": (buoy_buoy_link, (1000.0, 10.0, 10.0, 100.0),
                                  ("rx_sym", "rx_pkt2", "split", "own[tx+rx]+rx[quad]")),
}
NAMES = {"rx_sym": "receiver per symbol", "rx_pkt2": "receiver quadratic (all symbols)",
         "split": "split (IMUs + receiver line, 2 est.)", "own[tx]+rx[quad]": "tx IMU + receiver quadratic (all)",
         "own[tx+rx]+rx[quad]": "IMUs + receiver quadratic (all)"}
# colour and marker follow the placement, never its position in a panel (dataviz default slots, fixed order)
STYLE = {"rx_sym": ("#2a78d6", "o"), "rx_pkt2": ("#eb6834", "s"), "split": ("#1baf7a", "^"),
         "own[tx]+rx[quad]": ("#eda100", "D"), "own[tx+rx]+rx[quad]": ("#e87ba4", "v")}


def main():
    out, res = output_dir("surface_scatter"), {}
    offset = (np.pi * FC * T_FFT) ** 2 / 3
    for label, (make, geo, shown) in SCENARIOS.items():
        tx, rx = make()
        w_var = float(tx.vel_fast[:, 2].var())
        d2 = bellhop.diffuse_doppler_factor(bellhop.profile(bellhop.merge(bellhop.arrivals(*geo)), -20))
        s = P.NodeSensors()
        ev = {r["name"]: r["ici"] for r in P.evaluate(*P.link_components(tx, rx, s, s), P.ReceiverModel(packet_s=1.0))}
        i_sc = offset * bellhop.diffuse_floor(RHO_DB, d2, w_var)
        sir = {k: (-10 * np.log10(ev[k] + i_sc)).tolist() for k in shown}
        margin = np.array(sir["split"]) - np.array(sir["rx_sym"])
        stop = float(RHO_DB[np.argmax(margin < 1.0)]) if np.any(margin < 1.0) else None
        res[label] = dict(w_var=w_var, d2=d2, rho_db=RHO_DB.tolist(), sir_db=sir,
                          endpoint_sir_db={k: float(-10 * np.log10(ev[k])) for k in shown}, rho_stop_db=stop)
        print(f"{label}: endpoint " + ", ".join(f"{k} {-10 * np.log10(ev[k]):.1f}" for k in shown)
              + f" dB; split margin < 1 dB from rho = {stop} dB", flush=True)
    with open(os.path.join(out, "sweep.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    plot(res, os.path.join(out, "sweep.png"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(res), figsize=(4.2 * len(res), 3.8), sharey=True)
    for ax, (label, r) in zip(axes, res.items()):
        for k, y in r["sir_db"].items():
            ax.plot(r["rho_db"], y, color=STYLE[k][0], lw=2, marker=STYLE[k][1], markevery=8, ms=6, label=NAMES[k])
        if r["rho_stop_db"] is not None:
            ax.axvline(r["rho_stop_db"], color="#8a8a8a", lw=1, ls="--")
            ax.text(r["rho_stop_db"] + 0.5, 3, "placement\nstops mattering", fontsize=7, color="#555555")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("scattered surface power / coherent channel [dB]", fontsize=8)
        ax.grid(color="#e6e6e6", lw=0.6)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("SIR between subcarriers [dB]", fontsize=8)
    axes[0].set_ylim(0, 48)
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
