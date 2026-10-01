# JSAC on UW Communications — `uwsb`

Companion code for an IEEE JSAC submission (Special Issue: *Next-Generation Underwater Acoustic
Communication Systems*): multi-armed bandits for link adaptation over underwater acoustic (UWA)
links. Work in progress.

The `uwsb` package holds two lines of work that share infrastructure (config-hashed results,
seeding, delayed-feedback environments):

- **Proposal B — noise-informed robust bandits (current).** Link adaptation under impulsive UWA
  noise (α-stable, Middleton Class A, snapping shrimp). The receiver also observes a dense stream of
  noise-only samples; the code studies how that side information can set the robustness of the
  learner. Synthetic (Layer 1) experiments, noise models and estimators, and wrappers around the
  measured channels of the Underwater Acoustic Channel Library.
- **Proposal 0 — staleness-aware bandits.** Feedback delay comparable to the channel coherence time;
  forgetting set by measured physics (staleness number S = τ/T_c).

## Layout

```
src/uwsb/
  runtime.py                 config hash, data/results directories, run metadata
  noise/                     (B) impulsiveness diagnostics, SaS / complex isotropic SaS, Class A,
                             alpha-SGN(m) exact sampler, library noise model, recording loader
                             (FLAC, band-pass, robust SNR scaling), regime-switching generator
  channels/uwa_library.py    (B) Underwater Acoustic Channel Library wrapper: explicit seeding,
                             range-checked replay, cached replay (identical output, 9-20x faster)
  envs/table_env.py          (B) delayed-feedback environment over counterfactual outcome tables,
                             noise-stream delivery, best-fixed-arm reference
  estimation/                (B) tail-index estimators (Hill, quantile, log-moment, mean excess),
                             recursive noise state, CUSUM regime detector
                             (0) kernels.py
  bandits/robust.py          (B) naive and ACK baselines, truncated-mean / median-of-means /
                             Catoni UCB (Bubeck et al. 2013), AdaR-UCB (Genalti et al. 2024),
                             fixed clipping, median-filtered greedy, NIR-UCB (draft)
  bandits/predictive_ucb.py, bandits/baselines.py, channels/ou.py, envs/link_adaptation.py,
  metrics.py, theory.py      (0)
experiments/
  run_layer1_b.py            (B) Layer-1 regret runner (branches S and E); configs b_layer1_*.yaml
  make_b_layer1_sweep.py, run_b_layer1_sweep.sh   (B) alpha x gap sweep
  run_noise_check.py         (B) impulsiveness check on noise recordings
  analysis/                  (B) one-off analysis scripts (dated) and figure scripts
  run_synthetic.py           (0) Layer-1 runner; configs smoke, learning_curve, delay_tax, phase_*
tests/                       pytest; test_b_* for Proposal B
```

## Install

Python ≥ 3.10, in a virtual environment:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,b]"            # "b" = soundfile, uwa-channels 0.7.1, h5py (Proposal B)
pytest
```

Exact tested versions: `requirements-lock.txt`.

## Data (not in the repository)

- **Underwater Acoustic Channel Library** (Li et al., 2026): channel and noise `.mat` files from
  Zenodo record 21287414 (CC BY 4.0), read with the `uwa-channels` package. Place them in
  `$UWSB_DATA_DIR/uwa_library/` (e.g. `red_1.mat`, `blue_1.mat`, `red_noise.mat`).
- **Noise recordings** for the impulsiveness checks (e.g. NOAA/US Navy SanctSound, MBARI) go under
  `$UWSB_DATA_DIR/noise_census/`; see each script's docstring for the files it reads.

`UWSB_DATA_DIR` defaults to `<repo>/data`; `UWSB_RESULTS_DIR` defaults to `<repo>/results`.

## Running

```bash
# Proposal B, Layer 1
python -m experiments.run_layer1_b --config experiments/configs/b_layer1_smoke.yaml --smoke
python -m experiments.run_layer1_b --config experiments/configs/b_layer1_E_a16.yaml --dry-run   # time estimate
python -m experiments.run_layer1_b --config experiments/configs/b_layer1_E_a16.yaml --n-workers 4

# Proposal 0
make smoke
python -m experiments.run_synthetic --config experiments/configs/learning_curve.yaml
```

## Reproducibility

Every result comes from one YAML config; result directories are named `<name>-<config hash>` and
store the config, run metadata (versions, git commit) and summaries. All randomness goes through
explicitly seeded `numpy.random.Generator`s; seeds come from the config. Agents see only their own
delayed feedback (and, in Proposal B, the noise stream); true means and other arms' outcomes are
used for metrics only.

## License

MIT (see `LICENSE`). Data sets keep their own licences.
