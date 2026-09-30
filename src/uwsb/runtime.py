"""Run plumbing shared by Proposal B entry points (THEORY-B T-1).

- config_hash: same algorithm as the Proposal 0 runner (sha1 of the sorted JSON, first
  10 hex chars), so results dirs are keyed identically across the codebase.
- Paths are resolved from the repo root, never absolute in configs:
    data:    $UWSB_DATA_DIR    (default <repo>/data)
    results: $UWSB_RESULTS_DIR (default <repo>/results)
- run_metadata: Python/numpy/scipy versions, hostname and git revision, written into
  every results directory (kickoff "Running code").
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def data_dir() -> Path:
    return Path(os.environ.get("UWSB_DATA_DIR", REPO_ROOT / "data"))


def results_root() -> Path:
    return Path(os.environ.get("UWSB_RESULTS_DIR", REPO_ROOT / "results"))


def result_dir(cfg: dict) -> Path:
    return results_root() / f"{cfg.get('name', 'run')}-{config_hash(cfg)}"


def git_revision() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    return out.stdout.strip() if out.returncode == 0 else "not a git repository"


def run_metadata() -> dict:
    import numpy
    import scipy
    return {
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "git_revision": git_revision(),
    }
