"""
Experiment runner:  python -m uwofdm.run experiments/<name>.yaml [--dry-run] [--parallel N]

YAML keys
  name       experiment name (results/<name>/...)
  script     script to run, relative to the repo root
  args       argument list for a single run, or
  runs       list of argument lists (independent runs, executed in parallel)
  parallel   number of runs executed at once (default 1; overridden by --parallel)
  versioned  true (default): outputs go to results/<name>/<YYYYmmdd-HHMMSS>_<git-hash>/ with a manifest;
             false: outputs go to the canonical results/<name>/ (for results other experiments read as inputs)
Every run directory gets config.yaml, manifest.json (git commit, dirty flag, versions, host, timings, exit codes)
and one log per run.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import yaml

from .paths import ROOT, results_dir


def git_state():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip())
        return h or "nogit", dirty
    except OSError:
        return "nogit", False


def versions():
    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "h5py", "matplotlib", "uwa_channels"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "installed")
        except ImportError:
            out[mod] = None
    return out


def _one(script, args, log_path, env):
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.run([sys.executable, "-u", str(ROOT / script), *map(str, args)], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT)
    return {"args": list(map(str, args)), "log": os.path.basename(log_path), "exit": p.returncode,
            "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parallel", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    runs = cfg.get("runs") or [cfg.get("args", [])]
    par = a.parallel or cfg.get("parallel", 1)
    h, dirty = git_state()
    if cfg.get("versioned", True):
        out = results_dir(cfg["name"], f"{dt.datetime.now():%Y%m%d-%H%M%S}_{h}{'-dirty' if dirty else ''}")
    else:
        out = results_dir(cfg["name"])
    print(f"experiment {cfg['name']}: {len(runs)} run(s), parallel {par}, output {out}")
    if a.dry_run:
        for r in runs:
            print("  ", cfg["script"], *r)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.config, out / "config.yaml")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if cfg.get("versioned", True):
        env["UWOFDM_RUN_DIR"] = str(out)
    manifest = {"experiment": cfg["name"], "git": h, "dirty": dirty, "host": platform.node(), "versions": versions(),
                "started": dt.datetime.now().isoformat(timespec="seconds"), "runs": []}
    with cf.ThreadPoolExecutor(max_workers=par) as ex:
        futs = [ex.submit(_one, cfg["script"], r, out / f"run_{i:03d}.log", env) for i, r in enumerate(runs)]
        for f in cf.as_completed(futs):
            res = f.result()
            manifest["runs"].append(res)
            print(f"  exit {res['exit']} in {res['seconds']} s: {' '.join(res['args'])}", flush=True)
    manifest["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    json.dump(manifest, open(out / "manifest.json", "w", encoding="utf-8"), indent=2)
    failed = sum(r["exit"] != 0 for r in manifest["runs"])
    print(f"done: {len(runs) - failed} ok, {failed} failed -> {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
