import json

import pytest

from experiments import run_layer1_b
from uwsb import runtime


def test_layer1_smoke_runs_all_agents(tmp_path, monkeypatch):
    monkeypatch.setenv("UWSB_RESULTS_DIR", str(tmp_path))
    cfg_path = "experiments/configs/b_layer1_smoke.yaml"
    assert run_layer1_b.main(["--config", cfg_path, "--smoke", "--n-workers", "2"]) == 0
    import yaml
    cfg = yaml.safe_load(open(cfg_path))
    summary = json.loads((runtime.result_dir(cfg) / "summary.json").read_text())
    assert set(summary["agents"]) == {a["name"] for a in cfg["agents"]}
    assert summary["true_means"] == cfg["arms"]["means"]


def test_layer1_smoke_branch_e_new_baselines(tmp_path, monkeypatch):
    monkeypatch.setenv("UWSB_RESULTS_DIR", str(tmp_path))
    cfg_path = "experiments/configs/b_layer1_smoke_E.yaml"
    assert run_layer1_b.main(["--config", cfg_path, "--smoke", "--n-workers", "2"]) == 0
    import yaml
    cfg = yaml.safe_load(open(cfg_path))
    summary = json.loads((runtime.result_dir(cfg) / "summary.json").read_text())
    assert set(summary["agents"]) == {a["name"] for a in cfg["agents"]}
    assert summary["best_fixed_arm_final_mean"] == 0.0           # stationary table
    mu = summary["true_means"]
    assert summary["ack_threshold"] == pytest.approx(0.5 * (mu[0] + mu[1]))
