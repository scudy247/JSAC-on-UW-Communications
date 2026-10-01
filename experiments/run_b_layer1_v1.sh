#!/usr/bin/env bash
# Run the NIR-UCB v1 checkpoint configs one after another (4 workers, nice 10); completed configs are
# skipped by the runner, so the script can be re-launched after an interruption.
#   nohup setsid experiments/run_b_layer1_v1.sh > reports/logs/layer1_v1_grid_<date>.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for cfg in experiments/configs/b_layer1_v1/b_l1v1_E_*.yaml experiments/configs/b_layer1_v1/b_l1v1_S_*.yaml; do
  echo "=== $(date -Is) $cfg"
  nice -n 10 .venv/bin/python -m experiments.run_layer1_b --config "$cfg" --n-workers 4 || echo "!!! FAILED $cfg"
done
echo "=== $(date -Is) ALL DONE"
