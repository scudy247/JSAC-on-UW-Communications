"""End-to-end test of experiments/run_noise_check.py on synthetic local WAV files,
plus the shared run plumbing in uwsb.runtime."""

import json

import numpy as np
import pytest
import yaml
from scipy.io import wavfile

from experiments import run_noise_check
from experiments.run_synthetic import config_hash as p0_config_hash
from uwsb import runtime


def test_config_hash_matches_proposal0_runner():
    cfg = {"name": "x", "a": [1, 2], "b": {"c": 0.5}}
    assert runtime.config_hash(cfg) == p0_config_hash(cfg)


def test_run_metadata_has_required_fields():
    meta = runtime.run_metadata()
    for key in ("hostname", "python", "numpy", "scipy", "git_revision", "utc"):
        assert meta[key]


def _write_wav(path, x, fs_hz):
    wavfile.write(str(path), int(fs_hz), (x / np.max(np.abs(x)) * 0.5 * 32767).astype(np.int16))


def test_noise_check_end_to_end_separates_gaussian_from_impulsive(tmp_path, monkeypatch):
    monkeypatch.setenv("UWSB_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("UWSB_RESULTS_DIR", str(tmp_path / "results"))
    fs_hz = 64_000.0
    n = int(20 * fs_hz)                                    # 20 s per file
    rng = np.random.default_rng(7)
    gauss = rng.standard_normal(n)
    impulsive = rng.standard_normal(n)
    idx = rng.choice(n, size=400, replace=False)           # sparse large clicks
    impulsive[idx] += 60.0 * rng.choice([-1.0, 1.0], size=idx.size)
    _write_wav(tmp_path / "gauss.wav", gauss, fs_hz)
    _write_wav(tmp_path / "impulsive.wav", impulsive, fs_hz)

    cfg = {
        "name": "b_noise_check_test", "block_s": 5.0, "warmup_s": 0.5, "k_sigma": 5.0,
        "clip_level": 0.999,
        "bands_hz": {"blue": [10500.0, 15500.0], "red": [20000.0, 30000.0],
                     "too_high": [30000.0, 40000.0]},
        "files": [
            {"id": "gauss", "path": str(tmp_path / "gauss.wav"),
             "bands": ["blue", "red", "too_high"]},
            {"id": "impulsive", "path": str(tmp_path / "impulsive.wav"),
             "bands": ["blue", "red"]},
        ],
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    assert run_noise_check.main(["--config", str(cfg_path), "--n-workers", "2"]) == 0
    out = runtime.result_dir(cfg)
    summary = {s["id"]: s for s in json.loads((out / "summary.json").read_text())}

    g, i = summary["gauss"], summary["impulsive"]
    assert g["bands_skipped_above_nyquist"] == ["too_high"]   # 40 kHz > fs/2 = 32 kHz
    assert g["fs_hz"] == fs_hz and g["duration_s"] == pytest.approx(20.0)
    assert g["clip_frac"] == 0.0
    for band in ("blue", "red"):
        assert g["bands"][band]["n_blocks"] == 4
        assert g["bands"][band]["exceed_ratio_p10_p50_p90"][1] < 20.0
        assert i["bands"][band]["exceed_ratio_p10_p50_p90"][1] > 100.0
    for key in ("config.json", "metadata.json", "blocks.json", "run.log"):
        assert (out / key).exists()
    assert list(out.glob("*.png"))

    # idempotent: a second call finds summary.json and does nothing
    mtime = (out / "summary.json").stat().st_mtime
    assert run_noise_check.main(["--config", str(cfg_path)]) == 0
    assert (out / "summary.json").stat().st_mtime == mtime
