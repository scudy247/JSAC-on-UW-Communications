"""
Data and results locations. Nothing else in the code hard-codes a path.

  UWOFDM_DATA     data root (default <repo>/data): channel_library/*.mat, uwa-channels-python/
  UWOFDM_RESULTS  canonical results root (default <repo>/results): inputs shared between experiments,
                  e.g. results/diagnostic/<file>/profiles.npz written by scripts/replay_diagnostic.py
  UWOFDM_RUN_DIR  set by `python -m uwofdm.run` for versioned runs: where the current run writes its outputs
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def data_dir(*parts):
    return Path(os.environ.get("UWOFDM_DATA", ROOT / "data")).joinpath(*parts)


def channel_file(name):
    """Channel Library file by short name, e.g. 'blue_1' -> <data>/channel_library/blue_1.mat."""
    return data_dir("channel_library", f"{name}.mat")


def results_dir(*parts):
    """Canonical results (read side)."""
    return Path(os.environ.get("UWOFDM_RESULTS", ROOT / "results")).joinpath(*parts)


def output_dir(name):
    """Where a script writes: the run directory of a versioned run, else the canonical results/<name>."""
    run = os.environ.get("UWOFDM_RUN_DIR")
    p = Path(run) if run else results_dir(name)
    p.mkdir(parents=True, exist_ok=True)
    return p
