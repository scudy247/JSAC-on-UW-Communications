"""
Job list of the campaign of 17 Oct 2026 (PLAN: Phases 2, 5, 6; run by scripts/campaign.py on 25 cores).
Each job: name, script, args, est_min (single-core minutes, from the rates measured on 14-16 Oct), after (jobs that
must be done first). Rates: multi-user ~2.8 s per user-packet (all receivers incl. the sensor-seeded pilot
tracker); multipath ~2.4 min per 1 s packet and ~8 min per 4 s packet (all SNRs of the configuration); B ~20 min per
(case, carrier) at 60 packets (Viterbi tracker ~1.3x); single-path PER / sea states / numerology: guesses.
Groups (name prefixes):
  mu_main  multi-user grid: seabed and buoy users, K 2/4/8, guards 0/8/32, SNR 15/20/25/30 dB, 16-QAM, 100 packets
  mu_var   multi-user robustness on the reference packets (same seeds): consumer sensors, 2 and 5 deg direction
           errors, 10 ppm modem clocks, a moderate (Hs 2.3 m) and a calm (Hs 0.9 m) OOI day (8 guards);
           pilot-residual decoder noise (no guard); buoy users, K 4/8, 20 and 30 dB
  mu_mp    multi-user with each user through its own Bellhop channel (b22 --multipath): seabed and buoy users,
           K 2/4/8, 8 guards, 20 and 30 dB, 100 packets (receivers unchanged: single branch per user)
  mu_sic   the same for buoy users with per-user path cancellation (--sic): K 2/4 (60 packets), 8 (40)
  mp       multipath (b13): 1 s QPSK / 16-QAM / rho -20 dB (100 packets), 4 s QPSK (60); 1 s 16-QAM with 10 ppm
           clocks, on the moderate and on the calm day (100, paired with the reference); chunks merged
  B        long integration (b21): per-segment ML and Viterbi trackers, industrial; Viterbi with consumer sensors;
           Viterbi with 10 ppm clocks
  sp       single path, held high-statistics reruns: coded PER curves, sea states, numerology
"""
CASES = ("buoy users", "seabed users")
USER_PACKET_S = 2.8
JOBS = []


def add(name, script, args, est_min, after=()):
    JOBS.append(dict(name=name, script=script, args=list(args), est_min=round(est_min), after=list(after)))


def q(s):
    return "'" + s + "'"  # one shell argument


# ---------------------------------------------------------------- multi-user, main grid
mu_parts = []
for case in CASES:
    tag = case.split()[0]
    for K in (2, 4, 8):
        for g in (0, 8, 32):
            for snr in (15, 20, 25, 30):
                name = f"mu_main_{tag}_K{K}_g{g}_s{snr}"
                add(name, "b22_multiuser.py", ["100", "--mod", "16qam", "--cases", q(case), "--ks", str(K),
                                               "--guards", str(g), "--snrs", str(snr), "--no-alone",
                                               "--out", "camp_" + name], K * 100 * USER_PACKET_S / 60)
                mu_parts.append(name)
add("mu_main_merge", "campaign_merge.py", ["b22_multiuser", "camp_mu_main"] + ["camp_" + p for p in mu_parts], 1,
    after=mu_parts)

# ---------------------------------------------------------------- multi-user, robustness (paired with mu_main)
var_parts = []
VARIANTS = {"consumer": (["--grade", "consumer"], 8), "direrr2": (["--dir-err-deg", "2"], 8),
            "direrr5": (["--dir-err-deg", "5"], 8), "noisemax": (["--noise-source", "max"], 0),
            "clk10": (["--clock-ppm", "10"], 8), "moderate": (["--day", "GI01SUMO/20230915"], 8),
            "calm": (["--day", "CE04OSSM/20230731"], 8)}
for v, (extra, g) in VARIANTS.items():
    for K in (4, 8):
        for snr in (20, 30):
            name = f"mu_var_{v}_K{K}_s{snr}"
            add(name, "b22_multiuser.py", ["100", "--mod", "16qam", "--cases", q("buoy users"), "--ks", str(K),
                                           "--guards", str(g), "--snrs", str(snr), "--no-alone"] + extra +
                ["--out", "camp_" + name], K * 100 * USER_PACKET_S / 60)
            var_parts.append(name)
add("mu_var_merge", "campaign_merge.py", ["b22_multiuser", "camp_mu_var"] + ["camp_" + p for p in var_parts], 1,
    after=var_parts)

# ---------------------------------------------------------------- multi-user over per-user multipath (16 Oct 2026)
mp_parts = []
for case in CASES:
    tag = case.split()[0]
    for K in (2, 4, 8):
        for snr in (20, 30):
            name = f"mu_mp_{tag}_K{K}_s{snr}"
            add(name, "b22_multiuser.py", ["100", "--mod", "16qam", "--cases", q(case), "--ks", str(K), "--guards", "8",
                                           "--snrs", str(snr), "--no-alone", "--multipath", "--out", "camp_" + name],
                K * 100 * 3.2 / 60)
            mp_parts.append(name)
add("mu_mp_merge", "campaign_merge.py", ["b22_multiuser", "camp_mu_mp"] + ["camp_" + p for p in mp_parts], 1,
    after=mp_parts)
sic_parts = []  # the same with per-user path cancellation (signal-only / sensor-aided / ideal), buoy users, ~40 s/user-pkt
for K, n in ((2, 60), (4, 60), (8, 40)):
    for snr in (20, 30):
        name = f"mu_sic_buoy_K{K}_s{snr}"
        add(name, "b22_multiuser.py", [str(n), "--mod", "16qam", "--cases", q("buoy users"), "--ks", str(K), "--guards",
                                       "8", "--snrs", str(snr), "--no-alone", "--multipath", "--sic",
                                       "--out", "camp_" + name], K * n * 40 / 60)
        sic_parts.append(name)
add("mu_sic_merge", "campaign_merge.py", ["b22_multiuser", "camp_mu_sic"] + ["camp_" + p for p in sic_parts], 1,
    after=sic_parts)

# ---------------------------------------------------------------- multipath, chunks of one long run per configuration
MP = {"bb_iso_1s": (100, 20, 2.4), "bb_iso_1s_16qam": (100, 20, 2.4), "bb_iso_1s_rho20": (100, 20, 2.4),
      "bb_iso_4s": (60, 10, 8.0), "bb_iso_1s_16qam_clk10": (100, 20, 2.4),
      "bb_iso_1s_16qam_moderate": (100, 20, 2.4), "bb_iso_1s_16qam_calm": (100, 20, 2.4)}
for cfg, (n, chunk, rate) in MP.items():
    chunks = []
    for c, start in enumerate(range(0, n, chunk)):
        name = f"mp_{cfg}_c{c}"
        add(name, "b13_multipath.py", [cfg, str(chunk), f"_camp_c{c}", str(start)], chunk * rate)
        chunks.append(name)
    add(f"mp_{cfg}_merge", "b13_multipath.py", ["merge", cfg, "_camp"] + [f"_camp_c{c}" for c in range(len(chunks))],
        1, after=chunks)

# ---------------------------------------------------------------- B, long integration
for variant, (tracker, grade, rate, extra) in {"ml": ("ml", "industrial", 20, []),
                                               "hmm": ("hmm", "industrial", 26, []),
                                               "hmm_consumer": ("hmm", "consumer", 26, []),
                                               "hmm_clk10": ("hmm", "industrial", 26, ["--clock-ppm", "10"])}.items():
    parts = []
    for ci in range(3):
        for fc in (16000, 4000):
            name = f"B_{variant}_c{ci}_f{fc // 1000}"
            add(name, "b21_waveform_b.py", ["60", tracker, "--cases", str(ci), "--fcs", str(fc), "--grade", grade]
                + extra + ["--out", "camp_" + name], rate)
            parts.append(name)
    add(f"B_{variant}_merge", "campaign_merge.py", ["b21_waveform_b", f"camp_B_{variant}"] + ["camp_" + p for p in parts],
        1, after=parts)

# ---------------------------------------------------------------- single path, held reruns
for key, est in (("qpsk_7", 180), ("16qam_7", 180), ("16qam_27", 200), ("qpsk_27", 240)):
    add(f"sp_per_{key}", "paper_stats.py", ["per", key, f"_{key}_camp"], est)
add("sp_sea_states", "b3_sea_states.py", ["40", "points_camp"], 180)
add("sp_numerology", "b4_numerology.py", ["40", "numerology_camp"], 120)
