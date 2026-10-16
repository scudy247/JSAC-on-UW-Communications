"""
Campaign runner for the shared machine (PLAN: Phase 2 / 8; set up 16 Oct 2026 for the campaign of 17 Oct).

Jobs come from scripts/campaign_jobs.py (JOBS: name, script + arguments, estimated minutes, dependencies). Each job
runs as its own process: single-thread BLAS, nice 19, an address-space cap, its own log in
results/campaign/logs/<name>.log; exit 0 leaves results/campaign/<name>.done (exit != 0: <name>.failed). A finished
job is never rerun (resumable: run the same command again after a crash). Longest jobs start first.

Rule of the shared machine (user, 16 Oct 2026): before starting anything, check the load; if WORKERS cores are not
free (nproc - load average) or memory is short, start NOTHING. During the run, no new job starts while the machine is
saturated (someone else's load came back); running jobs are never touched.

  python scripts/campaign.py check  [--workers 25]          free cores / memory, exit 1 if not enough
  python scripts/campaign.py launch [--workers 25] [--only P] check, then start the scheduler in tmux "doppler"
  python scripts/campaign.py run    [--workers 25] [--only P] the scheduler itself (what launch starts)
  python scripts/campaign.py status [--only P]                done / running / failed / pending, time estimate
  python scripts/campaign.py list   [--only P]                the jobs and their commands
  python scripts/campaign.py retry  [--only P]                re-queue failed jobs (their .failed markers removed)
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # code/
PY = "/home/dario/Underwater/dario/bin/python"
STATE = os.environ.get("CAMPAIGN_STATE", os.path.join(HERE, "results", "campaign"))
LOGS = os.path.join(STATE, "logs")
MEM_CAP_GB = 4  # address-space cap per job (the jobs use < 0.5 GB)
MEM_NEED_GB = 1.0  # available memory required per worker at start
POLL_S = float(os.environ.get("CAMPAIGN_POLL_S", 20))


def load():
    with open("/proc/loadavg") as fh:
        l1, l5 = (float(x) for x in fh.read().split()[:2])
    return l1, l5


def mem_available_gb():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20
    return 0.0


def check(workers, verbose=True):
    """True if `workers` cores are free now (nproc minus the larger of the 1- and 5-minute load) and memory is
    enough. Our own running campaign jobs do not count against the budget."""
    n = os.cpu_count()
    l1, l5 = load()
    ours = len(running_pids())
    free = n - max(l1, l5) + ours
    mem = mem_available_gb()
    ok = free >= workers and mem >= workers * MEM_NEED_GB
    if verbose:
        print(f"cores {n}, load 1/5 min {l1:.1f}/{l5:.1f}, campaign jobs running {ours} -> free {free:.1f} "
              f"(need {workers}); memory available {mem:.0f} GB (need {workers * MEM_NEED_GB:.0f}) -> "
              f"{'OK' if ok else 'NOT ENOUGH: start nothing'}")
    return ok


def jobs(only=None):
    import importlib
    sys.path.insert(0, os.path.join(HERE, "scripts"))
    JOBS = importlib.import_module(os.environ.get("CAMPAIGN_JOBS", "campaign_jobs")).JOBS  # env: tests
    names = [j["name"] for j in JOBS]
    assert len(names) == len(set(names)), "duplicate job names"
    return [j for j in JOBS if not only or any(j["name"].startswith(p) for p in only.split(","))]


def state(name):
    for s in ("done", "failed", "running"):
        if os.path.exists(os.path.join(STATE, f"{name}.{s}")):
            return s
    return "pending"


def running_pids():
    """pids of campaign jobs still alive (from their .running markers)."""
    out = []
    if not os.path.isdir(STATE):
        return out
    for f in os.listdir(STATE):
        if f.endswith(".running"):
            try:
                pid = int(open(os.path.join(STATE, f)).read().split()[0])
                os.kill(pid, 0)
                out.append(pid)
            except (ValueError, OSError, IndexError):
                pass
    return out


def command(job):
    env = ("OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
           f"PYTHONPATH={HERE}/src:{HERE}/scripts")
    args = " ".join(job["args"])
    return (f"cd {HERE} && {env} prlimit --as={MEM_CAP_GB * 2 ** 30} nice -n 19 {PY} scripts/{job['script']} {args}")


def run(workers, only):
    os.makedirs(LOGS, exist_ok=True)
    if not check(workers):
        print("campaign NOT started (shared-machine rule)")
        return 1
    todo = [j for j in jobs(only) if state(j["name"]) in ("pending", "running")]
    for j in todo:  # stale .running markers from a crashed scheduler
        p = os.path.join(STATE, f"{j['name']}.running")
        if os.path.exists(p):
            os.remove(p)
    todo.sort(key=lambda j: -j.get("est_min", 0))  # longest first
    procs = {}
    print(f"{time.strftime('%H:%M')} campaign started: {len(todo)} jobs, {workers} workers", flush=True)
    while todo or procs:
        for name, p in list(procs.items()):
            if p.poll() is not None:
                os.remove(os.path.join(STATE, f"{name}.running"))
                ok = p.returncode == 0
                open(os.path.join(STATE, f"{name}.{'done' if ok else 'failed'}"), "w").write(time.strftime("%c\n"))
                print(f"{time.strftime('%H:%M')} {'done  ' if ok else 'FAILED'} {name} (exit {p.returncode})",
                      flush=True)
                del procs[name]
        ready = [j for j in todo if all(state(d) == "done" for d in j.get("after", []))]
        blocked = [j for j in todo if any(state(d) == "failed" for d in j.get("after", []))]
        for j in blocked:
            print(f"{time.strftime('%H:%M')} skipped {j['name']}: a dependency failed", flush=True)
            todo.remove(j)
        l1, _ = load()
        saturated = l1 - len(procs) > os.cpu_count() - workers + 2  # others' load came back: launch nothing new
        while ready and len(procs) < workers and not saturated:
            j = ready.pop(0)
            todo.remove(j)
            log = open(os.path.join(LOGS, f"{j['name']}.log"), "w")
            log.write(f"started {time.strftime('%c')}\n{command(j)}\n")
            log.flush()
            p = subprocess.Popen(["bash", "-c", command(j)], stdout=log, stderr=subprocess.STDOUT)
            open(os.path.join(STATE, f"{j['name']}.running"), "w").write(f"{p.pid} {time.time()}\n")
            procs[j["name"]] = p
            print(f"{time.strftime('%H:%M')} start  {j['name']} (~{j.get('est_min', '?')} min)", flush=True)
        time.sleep(POLL_S)
    print(f"{time.strftime('%H:%M')} campaign finished", flush=True)
    return 0


def status(only):
    js = jobs(only)
    counts = {}
    left = 0.0
    for j in js:
        s = state(j["name"])
        counts[s] = counts.get(s, 0) + 1
        if s in ("pending", "running"):
            left += j.get("est_min", 0)
        print(f"  {s:8s} {j['name']:40s} ~{j.get('est_min', '?')} min")
    print(" ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f" | remaining ~{left / 60:.1f} core-hours")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=("check", "launch", "run", "status", "list", "retry"))
    ap.add_argument("--workers", type=int, default=25)
    ap.add_argument("--only", default=None, help="comma-separated job-name prefixes")
    a = ap.parse_args()
    if a.action == "check":
        sys.exit(0 if check(a.workers) else 1)
    if a.action == "list":
        for j in jobs(a.only):
            print(f"{j['name']:40s} ~{j.get('est_min', '?'):>4} min  after {j.get('after', [])}\n    {command(j)}")
        print(f"{len(jobs(a.only))} jobs, ~{sum(j.get('est_min', 0) for j in jobs(a.only)) / 60:.1f} core-hours")
        return
    if a.action == "retry":
        for j in jobs(a.only):
            p = os.path.join(STATE, f"{j['name']}.failed")
            if os.path.exists(p):
                os.remove(p)
                print("re-queued", j["name"])
        return
    if a.action == "status":
        status(a.only)
        return
    if a.action == "run":
        sys.exit(run(a.workers, a.only))
    if a.action == "launch":
        if not check(a.workers):
            print("campaign NOT launched (shared-machine rule)")
            sys.exit(1)
        os.makedirs(LOGS, exist_ok=True)
        only = f" --only {a.only}" if a.only else ""
        env = " ".join(f"{k}={os.environ[k]}" for k in ("CAMPAIGN_JOBS", "CAMPAIGN_STATE") if k in os.environ)
        cmd = (f"cd {HERE} && {env} {PY} scripts/campaign.py run --workers {a.workers}{only} "
               f">> {STATE}/scheduler.log 2>&1")
        subprocess.run(["bash", "-c", "tmux has-session -t doppler 2>/dev/null || tmux new-session -d -s doppler -n idle"])
        subprocess.run(["tmux", "new-window", "-d", "-t", "doppler", "-n", "campaign", cmd], check=True)
        print(f"launched: tmux window doppler:campaign, log {STATE}/scheduler.log")


if __name__ == "__main__":
    main()
