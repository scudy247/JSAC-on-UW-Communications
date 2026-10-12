#!/usr/bin/env bash
# One-time setup on the Ubuntu server, from anywhere:   bash code/setup_server.sh
# Needs Python >= 3.10 with venv (Ubuntu: sudo apt install python3-venv). Override the interpreter with PYTHON=python3.12.
set -euo pipefail
cd "$(dirname "$0")"  # code/: holds pyproject.toml; data/ and results/ default here (see src/uwofdm/paths.py)

PY=${PYTHON:-python3}
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 required, found " + sys.version'

if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# Exact versions used during development; fall back to the minimum versions in pyproject.toml if a pin is unavailable
python -m pip install -r requirements-lock.txt || echo "WARNING: pinned versions not installable here, using pyproject minimums"
python -m pip install -e ".[test]"

if [ -d data/uwa-channels-python ]; then
  python -m pip install -e data/uwa-channels-python
else
  echo "WARNING: data/uwa-channels-python not found (Channel Library replay toolbox); see README"
fi

python - <<'PY'
import numpy, scipy, h5py, matplotlib, yaml, uwofdm, uwa_channels
print("python", __import__("sys").version.split()[0], "| numpy", numpy.__version__, "| scipy", scipy.__version__)
PY
echo "Setup done. Activate with: source .venv/bin/activate   then run: python -m pytest"
