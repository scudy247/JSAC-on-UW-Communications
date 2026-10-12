"""Paths and experiment-runner tests: environment overrides, config validity, versioned output layout."""
import glob
import os
import subprocess
import sys

import yaml

from uwofdm import paths

ROOT = paths.ROOT


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("UWOFDM_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("UWOFDM_RESULTS", str(tmp_path / "r"))
    monkeypatch.delenv("UWOFDM_RUN_DIR", raising=False)
    assert paths.channel_file("blue_1") == tmp_path / "d" / "channel_library" / "blue_1.mat"
    assert paths.results_dir("diagnostic") == tmp_path / "r" / "diagnostic"
    assert paths.output_dir("x") == tmp_path / "r" / "x" and (tmp_path / "r" / "x").is_dir()
    monkeypatch.setenv("UWOFDM_RUN_DIR", str(tmp_path / "run"))
    assert paths.output_dir("x") == tmp_path / "run"


def test_every_experiment_config_is_valid():
    cfgs = glob.glob(str(ROOT / "experiments" / "*.yaml"))
    assert cfgs
    for f in cfgs:
        c = yaml.safe_load(open(f, encoding="utf-8"))
        assert {"name", "script"} <= set(c), f
        assert (ROOT / c["script"]).exists(), f
        assert ("args" in c) != ("runs" in c), f"{f}: give either args or runs"


def test_runner_versioned_run(tmp_path):
    """A versioned run writes config, manifest and log into results/<name>/<stamp>_<git>/ and passes UWOFDM_RUN_DIR."""
    script = tmp_path / "probe.py"
    script.write_text("import os, sys; open(os.path.join(os.environ['UWOFDM_RUN_DIR'], 'probe.txt'), 'w').write(sys.argv[1])\n")
    cfg = tmp_path / "probe.yaml"
    cfg.write_text(yaml.safe_dump({"name": "probe", "script": str(script), "runs": [["a"], ["b"]], "parallel": 2}))
    env = dict(os.environ, UWOFDM_RESULTS=str(tmp_path / "results"),
               PYTHONPATH=os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]))
    p = subprocess.run([sys.executable, "-m", "uwofdm.run", str(cfg)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    (run_dir,) = (tmp_path / "results" / "probe").iterdir()
    for f in ("config.yaml", "manifest.json", "run_000.log", "run_001.log", "probe.txt"):
        assert (run_dir / f).exists(), f
