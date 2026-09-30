# uwa-staleness-bandit

Staleness-aware multi-armed bandits for underwater acoustic (UWA) link
adaptation. Companion code for an IEEE JSAC paper (SI: *Next-Generation
Underwater Acoustic Communication Systems*).

**Idea.** In UWA links the feedback delay τ is comparable to or larger than the
channel coherence time T_c, so every reward is stale by a known amount. The PHY
can *measure* both the age (τ from ranging) and the decay rate (T_c from Doppler
spread), so the learner's forgetting is set by physics, not hand-tuned windows.
The central quantity is the **staleness number** S = τ/T_c.

Design docs (read these before the code):

- [`project.MD`](project.MD) — research brief, repo map, engineering conventions.
- [`THEORY.md`](THEORY.md) — worked-out model, locked decisions, the results we
  prove, and the falsifiable predictions the simulations must confirm.

## Layout

```
src/uwsb/
  channels/ou.py            OU/AR(1) reward process  mu_k(t)=m_k+b_k g(t)+e_k(t)
  estimation/kernels.py     R(dt) kernels (exponential + 2-exp mixture)
  bandits/predictive_ucb.py 2-state Kalman-per-arm PUCB v2 + Predictive TS
  bandits/baselines.py      SW-UCB, D-UCB, UCB1, TS, outdated-CSI AMC
  envs/link_adaptation.py   delayed-feedback simulation loop (queue -> release)
  metrics.py                staleness_number, regret vs the oracle ladder
experiments/run_synthetic.py   Layer-1 regret curves
tests/                          pytest: every estimator vs an analytic case
```

## Quickstart

```bash
pip install -e .[dev]
make smoke        # tiny end-to-end run, < 60 s  (project.MD §7.8)
pytest            # unit tests
python experiments/run_synthetic.py --config experiments/configs/learning_curve.yaml
```

## Reproducibility

Every figure regenerates from a single YAML config; result directories are keyed
by a config hash (project.MD §7.5). Seeds are drawn from the config and logged
with results. No oracle ever leaks into an agent (project.MD §7.2): agents see
only delayed rewards through the env's observation interface; oracles live in
`metrics.py`.
