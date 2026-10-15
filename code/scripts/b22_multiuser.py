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
Usage: python scripts/b22_multiuser.py [n_packets]. Output: results/b22_multiuser/points.json.
"""
import json
import os
import sys
from functools import lru_cache

import numpy as np

from uwofdm import motion, multiuser as mu, waveform
from uwofdm.experiment import at_times
from uwofdm.paths import output_dir
from uwofdm.receiver import n_info_bits
from c2_sweeps import OOI_DAY
from e2_chain import SIGMA_IMU
from e2_imu_grades import node

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


def run(case, K, guard, snr_db, n, rng, o=None):
    o = o or waveform.OFDM()
    frames = mu.contiguous(o, K, guard)
    gw, users = tracks(case, K)
    links = [motion.link_doppler(u, gw) for u in users]
    nodes = [(node(u, d["u"], +1, 0.0, GRADE, 100 + k), node(gw, d["u"], -1, d["tau"], GRADE, GW_SEED))
             for k, (u, d) in enumerate(zip(users, links))]  # one gateway sensor: same seed, projected per user
    T = frames[0].length / o.fs
    lo = max(d["times"][0] for d in links) + 240
    hi = min(d["times"][-1] for d in links) - T - 20
    rec = {r: dict(sinr=[], err=[]) for r in mu.RECEIVERS + ("genie_alone",)}
    for t0 in rng.uniform(lo, hi, n):
        rp = np.random.default_rng(rng.integers(1 << 30))
        lead, tail, m_off = int(rp.integers(500, 3000)), 8000, int(MAX_OFF * o.fs)
        offs = [0] + list(rp.integers(0, m_off, K - 1))
        tg = t0 - lead / o.fs + np.arange(lead + m_off + frames[0].length + tail) / o.fs
        pk, xs, eps, priors, truths = [], [], [], [], []
        for k, F in enumerate(frames):
            info, X, s = mu.user_packet(F, rp)
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
                s_db, e = mu.score(Y, pk[k][1], pk[k][0], frames[k])
                rec[r]["sinr"].append(s_db)
                rec[r]["err"].append(e)
        for k, F in enumerate(frames):  # the user alone in the band, same noise level
            ya = mu.add_noise(mu.uplink([xs[k]], [eps[k]], o.fs, o.fc), o, snr_db, rp)
            Y = mu.receive_users(ya, [F], [priors[k]], [truths[k]], [searches[k]], which=("branch_genie",))[0][
                "branch_genie"][0]
            s_db, e = mu.score(Y, pk[k][1], pk[k][0], F)
            rec["genie_alone"]["sinr"].append(s_db)
            rec["genie_alone"]["err"].append(e)
    bits = sum(n_info_bits(F) for F in frames) / K
    out = {}
    for r, v in rec.items():
        sinr = np.array(v["sinr"]).reshape(-1, K)
        err = np.array(v["err"], float)
        out[r] = dict(mean_sinr_db=float(np.mean(sinr)), worst_user_sinr_db=float(np.mean(sinr.min(1))),
                      per=float(err.mean()), goodput_bps=float(K * bits * (1 - err.mean()) / T))
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    rng = np.random.default_rng(13)
    rows = []
    for case in CASES:
        for K in KS:
            for guard in GUARDS:
                for snr in SNRS:
                    res = run(case, K, guard, snr, n, rng)
                    rows.append(dict(scenario=case, K=K, guard=guard, snr_db=snr, n_packets=n, **res))
                    print(f"{case:13s} K {K} guard {guard:2d} SNR {snr} | mean SINR / PER: " + "  ".join(
                        f"{r} {v['mean_sinr_db']:5.1f}/{v['per']:.2f}" for r, v in res.items()), flush=True)
                    with open(os.path.join(output_dir("b22_multiuser"), "points.json"), "w", encoding="utf-8") as fh:
                        json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
