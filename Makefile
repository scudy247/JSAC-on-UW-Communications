# Convenience targets. On Windows use `python -m` equivalents if `make` absent.

.PHONY: install smoke test lint figures clean

install:
	pip install -e ".[dev,b]"

# project.MD §7.8: tiny end-to-end run must pass before any grid search.
smoke:
	python -m experiments.run_synthetic --config experiments/configs/smoke.yaml --smoke
	python -m experiments.run_synthetic --config experiments/configs/phase_smoke.yaml --smoke

test:
	pytest

# Regenerate all Layer-1 figures from their configs.
figures:
	python -m experiments.run_synthetic --config experiments/configs/learning_curve.yaml
	python -m experiments.run_synthetic --config experiments/configs/delay_tax.yaml
	python -m experiments.run_synthetic --config experiments/configs/phase_transition.yaml

clean:
	rm -rf results/* figures/*.png
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
