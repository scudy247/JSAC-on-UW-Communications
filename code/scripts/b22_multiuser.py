"""
A: multi-user uplink (PLAN: Phase 6; candidate A of 14 Oct 2026). K users on contiguous subbands with guard
subcarriers (uwofdm.multiuser), real buoy motion, one gateway buoy. Each user's Doppler must be estimated from its
own subband (1/K of the band, chirps with 1/K of the time-bandwidth product) with the other users present; the
sensor-aided gateway sets each branch from the user's side information and its own IMU projected on the user's
direction, then refines with the signal (safe fusion).
Scenarios (OOI Irminger, 19 Sep 2023, industrial MEMS at every moving node; gateway: buoy, transducer 2 m below the
motion sensor, all users at 1000 m on a ring at different azimuths):
  seabed users   static nodes 100 m deep: each user's Doppler is the gateway's motion projected on its direction;
                 the sensor prior needs no side information at all
  buoy users     moored buoys 10 m deep, each with its own motion (independent 20-min bursts of the same day, 2 h
                 apart)
Asynchrony: each user's frame starts up to MAX_OFF later than the first (within the cyclic prefix).
Receivers (multiuser.RECEIVERS) plus genie_alone: the genie branch with the user alone in the band (noise only), to
split the inter-user loss from the estimation loss. Metrics per user: SINR after the receiver's own channel estimate,
packet error (coded QPSK); per configuration: mean and worst-user SINR, PER, and goodput (info bits of the decoded
packets per second of frame, summed over the users).
Usage: python scripts/b22_multiuser.py [n_packets] [--mod qpsk|16qam] [--cases ...] [--ks 2,4,8] [--guards 0,8,32]
[--snrs 20,30] [--no-alone] [--out name]. Output: results/b22_multiuser/<name>.json (95 % confidence intervals,
paired gains of ours over the signal-only baselines).
"""
import argparse
import json
import os
from functools import lru_cache

import numpy as np

from uwofdm import motion, multiuser as mu, waveform
from uwofdm.experiment import at_times
from uwofdm.paths import output_dir
from uwofdm import receiver
from uwofdm.receiver import n_info_bits
from uwofdm.stats import ci95, seed_of, wilson
from c2_sweeps import OOI_DAY
from e2_chain import SIGMA_IMU
from e2_imu_grades import _static
from uwofdm import imu
from uwofdm.experiment import NodeIMU

CASES = ("seabed users", "buoy users")
KS = (2, 4, 8)
GUARDS = (0, 8, 32)  # empty subcarriers between neighbouring users
SNRS = (20, 30)
R, LEVER, GRADE, GW_SEED = 1000.0, (0.0, 0.0, 2.0), "industrial", 7
MAX_OFF = 0.005  # [s] asynchrony between users' frame starts


@lru_cache(maxsize=None)
def tracks(case, K):
    """(gateway track, user tracks) on a common grid length (cached: loading the OOI bursts is the slow part; the
    tracks are not modified afterwards)."""
    gw = motion.buoy_track("ooi", OOI_DAY, (0.0, 0.0, -10.0), segment=3, lever=LEVER)
    az = 2 * np.pi * np.arange(K) / K + 0.3
    if case == "seabed users":
        users = [motion.static_track(gw.fs, gw.n, (R * np.cos(a), R * np.sin(a), -100.0)) for a in az]
    else:
        users = [motion.buoy_track("ooi", OOI_DAY, (R * np.cos(a), R * np.sin(a), -10.0), segment=5 + 2 * k,
                                   lever=LEVER) for k, a in enumerate(az)]
    n = min(t.n for t in [gw] + users)
    for tr in [gw] + users:
        tr.vel_fast, tr.vel_slow = tr.vel_fast[:n], tr.vel_slow[:n]
        tr.vel_lever = None if tr.vel_lever is None else tr.vel_lever[:n]
    return gw, users


def rotate(u, deg):
    """Direction u turned by deg in azimuth (the horizontal part rotated, the vertical kept)."""
    a = np.deg2rad(deg)
    return np.array([u[0] * np.cos(a) - u[1] * np.sin(a), u[0] * np.sin(a) + u[1] * np.cos(a), u[2]])


def node_hat(track, u, sign, shift, grade, seed, u_hat):
    """e2_imu_grades.node with the node projecting its velocity on a believed direction u_hat."""
    kw = dict(sigmas=(SIGMA_IMU,), tau_imu=20.0, latency=0.05, extrapolate=True, lever_corrected=True, u_hat=u_hat)
    if _static(track):
        return NodeIMU(track, u, sign, shift, np.random.default_rng(seed), grade=imu.ZERO, **dict(kw, latency=0.0))
    lever_m = 2.0 if track.vel_lever is not None else None
    return NodeIMU(track, u, sign, shift, np.random.default_rng(seed), grade=imu.GRADES[grade], lever_m=lever_m, **kw)


def run(case, K, guard, snr_db, n, rng, o=None, mod="qpsk", alone=True, grade=GRADE, dir_err_deg=0.0):
    o = o or waveform.OFDM()
    frames = mu.contiguous(o, K, guard)
    gw, users = tracks(case, K)
    links = [motion.link_doppler(u, gw) for u in users]
    # one gateway sensor: same seed, projected per user; dir_err_deg: both ends believe a direction turned by a
    # random azimuth error (std dir_err_deg, fixed per user), e.g. from stale or coarse position knowledge
    errs = np.random.default_rng(77 + K).normal(0.0, dir_err_deg, K)
    nodes = [(node_hat(u, d["u"], +1, 0.0, grade, 100 + k, rotate(d["u"], e)),
              node_hat(gw, d["u"], -1, d["tau"], grade, GW_SEED, rotate(d["u"], e)))
             for k, (u, d, e) in enumerate(zip(users, links, errs))]
    T = frames[0].length / o.fs
    lo = max(d["times"][0] for d in links) + 240
    hi = min(d["times"][-1] for d in links) - T - 20
    names = mu.RECEIVERS + (("genie_alone",) if alone else ())
    rec = {r: dict(sinr=[], err=[]) for r in names}
    for t0 in rng.uniform(lo, hi, n):
        rp = np.random.default_rng(rng.integers(1 << 30))
        lead, tail, m_off = int(rp.integers(500, 3000)), 8000, int(MAX_OFF * o.fs)
        offs = [0] + list(rp.integers(0, m_off, K - 1))
        tg = t0 - lead / o.fs + np.arange(lead + m_off + frames[0].length + tail) / o.fs
        pk, xs, eps, priors, truths = [], [], [], [], []
        for k, F in enumerate(frames):
            info, X, s = mu.user_packet(F, rp, mod)
            x = np.zeros(len(tg), complex)
            x[lead + offs[k]:lead + offs[k] + len(s)] = s
            d = links[k]
            te = t0 + offs[k] / o.fs + np.arange(F.length + tail // 2) / o.fs
            pk.append((info, X))
            xs.append(x)
            eps.append(at_times(d["eps"], d["times"], users[k].fs, tg))
            truths.append(at_times(d["eps"], d["times"], users[k].fs, te))
            priors.append(nodes[k][0].estimate(te, SIGMA_IMU) + nodes[k][1].estimate(te, SIGMA_IMU))
        y = mu.add_noise(mu.uplink(xs, eps, o.fs, o.fc), o, snr_db, rp)
        searches = [(0, lead + m_off + 1500)] * K
        Ys, _ = mu.receive_users(y, frames, priors, truths, searches)
        for r, by_user in Ys.items():
            for k, Y in enumerate(by_user):
                s_db, e = mu.score(Y, pk[k][1], pk[k][0], frames[k], mod)
                rec[r]["sinr"].append(s_db)
                rec[r]["err"].append(e)
        for k, F in enumerate(frames if alone else ()):  # the user alone in the band, same noise level
            ya = mu.add_noise(mu.uplink([xs[k]], [eps[k]], o.fs, o.fc), o, snr_db, rp)
            Y = mu.receive_users(ya, [F], [priors[k]], [truths[k]], [searches[k]], which=("branch_genie",))[0][
                "branch_genie"][0]
            s_db, e = mu.score(Y, pk[k][1], pk[k][0], F, mod)
            rec["genie_alone"]["sinr"].append(s_db)
            rec["genie_alone"]["err"].append(e)
    bits = sum(n_info_bits(F, mod) for F in frames) / K
    out = {}
    for r, v in rec.items():
        sinr = np.array(v["sinr"]).reshape(-1, K)
        err = np.array(v["err"], float)
        out[r] = dict(mean_sinr_db=float(np.mean(sinr)), mean_sinr_ci95=ci95(sinr.mean(1)),
                      worst_user_sinr_db=float(np.mean(sinr.min(1))), worst_user_ci95=ci95(sinr.min(1)),
                      p10_sinr_db=float(np.percentile(sinr, 10)), per=float(err.mean()),
                      per_ci95=wilson(int(err.sum()), len(err)), goodput_bps=float(K * bits * (1 - err.mean()) / T),
                      per_user_packet_sinr_db=[round(float(x), 2) for x in sinr.ravel()],
                      per_user_packet_err=[bool(x) for x in err])
    for ours in ("branch_sensor", "branch_sensor_pilot"):  # paired gain of ours over each signal-only baseline
        for base in ("branch_signal", "branch_pilot"):
            d = np.array(rec[ours]["sinr"]) - np.array(rec[base]["sinr"])
            out[ours]["gain_over_" + base] = dict(mean_db=float(d.mean()), ci95=ci95(d.reshape(-1, K).mean(1)),
                                                  fraction_positive=float(np.mean(d > 0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int, nargs="?", default=8, help="packets per point")
    ap.add_argument("--mod", default="qpsk")
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--guards", default=",".join(map(str, GUARDS)))
    ap.add_argument("--snrs", default=",".join(map(str, SNRS)))
    ap.add_argument("--no-alone", action="store_true", help="skip the genie with the user alone")
    ap.add_argument("--grade", default=GRADE, help="motion-sensor grade of every moving node (imu.GRADES)")
    ap.add_argument("--dir-err-deg", type=float, default=0.0, help="std of the azimuth error of the believed directions")
    ap.add_argument("--noise-source", default="nulls", choices=("nulls", "max"),
                    help="decoder noise: in-band nulls only, or also the pilot residual (receiver.NOISE_SOURCE)")
    ap.add_argument("--out", default="points", help="output file name (results/b22_multiuser/<out>.json)")
    a = ap.parse_args()
    receiver.NOISE_SOURCE = a.noise_source
    ints = lambda t: [int(v) for v in t.split(",")]
    rows = []
    for case in a.cases.split(","):
        for K in ints(a.ks):
            for guard in ints(a.guards):
                for snr in ints(a.snrs):
                    # per-configuration seed (stats.seed_of): the same configuration draws the same packets whatever
                    # the order or the split of the jobs; receiver and sensor settings (noise source, grade,
                    # direction errors) are not part of it, so a variant sees exactly the packets and noise of the
                    # reference run (paired comparisons)
                    rng = np.random.default_rng(seed_of(case, K, guard, snr, a.mod))
                    res = run(case, K, guard, snr, a.n, rng, mod=a.mod, alone=not a.no_alone, grade=a.grade,
                              dir_err_deg=a.dir_err_deg)
                    rows.append(dict(scenario=case, K=K, guard=guard, snr_db=snr, n_packets=a.n, mod=a.mod,
                                     grade=a.grade, dir_err_deg=a.dir_err_deg, noise_source=a.noise_source, **res))
                    g, gp = res["branch_sensor"], res["branch_sensor_pilot"]
                    print(f"{case:13s} K {K} guard {guard:2d} SNR {snr} {a.mod} | mean SINR / PER: " + "  ".join(
                        f"{r} {v['mean_sinr_db']:5.1f}/{v['per']:.2f}" for r, v in res.items()) +
                        f" | ours - Tu {g['gain_over_branch_signal']['mean_db']:+.1f}"
                        f"+-{g['gain_over_branch_signal']['ci95']:.1f}, ours - pilot "
                        f"{g['gain_over_branch_pilot']['mean_db']:+.1f}+-{g['gain_over_branch_pilot']['ci95']:.1f}"
                        f", ours+pilot - pilot {gp['gain_over_branch_pilot']['mean_db']:+.1f}"
                        f"+-{gp['gain_over_branch_pilot']['ci95']:.1f}",
                        flush=True)
                    with open(os.path.join(output_dir("b22_multiuser"), a.out + ".json"), "w", encoding="utf-8") as fh:
                        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
