"""
Smoke stage of the campaign (PLAN: Phase 2; 16 Oct 2026): one representative of every job type of
campaign_jobs.JOBS with 1-2 packets and its own output names (smoke_*), run first on the same 25 cores to catch
crashes in minutes. Run with
  CAMPAIGN_JOBS=campaign_jobs_smoke CAMPAIGN_STATE=results/campaign_smoke python scripts/campaign.py run
(launch / check / status the same way). Merges are exercised on two smoke chunks of one multipath configuration.
"""
from campaign_jobs import JOBS as FULL

JOBS = []
seen = set()


def kind(j):
    """Job type: the script plus the options that change the code path (not K / guard / SNR / chunk index)."""
    a = j["args"]
    if j["script"] == "b22_multiuser.py":
        opts = tuple(x for x in a if x.startswith("--") and x not in ("--ks", "--guards", "--snrs", "--out", "--mod",
                                                                       "--cases", "--no-alone"))
        day = a[a.index("--day") + 1] if "--day" in a else None  # another mooring's files: its own smoke test
        return (j["script"], a[a.index("--cases") + 1], opts, day)
    if j["script"] == "b13_multipath.py":
        return (j["script"], a[0]) if a[0] != "merge" else None
    if j["script"] == "b21_waveform_b.py":
        return (j["script"], a[1], tuple(x for x in a if x.startswith("--grade") or x.startswith("--clock")),
                a[a.index("--grade") + 1])
    if j["script"] == "campaign_merge.py":
        return None
    return (j["script"], tuple(a[:2]))


for j in FULL:
    k = kind(j)
    if k is None or k in seen:
        continue
    seen.add(k)
    a = list(j["args"])
    if j["script"] == "b22_multiuser.py":
        a[0] = "2"
        a[a.index("--ks") + 1] = "8"  # the largest case exercises the most code
        a[a.index("--out") + 1] = "smoke_" + j["name"]
    elif j["script"] == "b13_multipath.py":
        a = [a[0], "1", f"_smoke_c0", "0"]
    elif j["script"] == "b21_waveform_b.py":
        a[0] = "2"
        a[a.index("--out") + 1] = "smoke_" + j["name"]
        a += ["--cnrs", "10"]
    elif j["script"] == "paper_stats.py":
        a = ["per", a[1], f"_{a[1]}_smoke", "1"]
    else:  # b3 / b4: one packet per point, own output name
        a = ["1", "smoke"]
    JOBS.append(dict(name="smoke_" + j["name"], script=j["script"], args=a, est_min=5, after=[]))

# merge path of the multipath chunks: a second one-packet chunk, then the merge of the two
JOBS.append(dict(name="smoke_mp_chunk1", script="b13_multipath.py", args=["bb_iso_1s", "1", "_smoke_c1", "1"],
                 est_min=3, after=[]))
JOBS.append(dict(name="smoke_mp_merge", script="b13_multipath.py",
                 args=["merge", "bb_iso_1s", "_smoke", "_smoke_c0", "_smoke_c1"], est_min=1,
                 after=["smoke_mp_bb_iso_1s_c0", "smoke_mp_chunk1"]))
