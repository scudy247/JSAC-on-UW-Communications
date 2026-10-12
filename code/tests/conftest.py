"""Make `uwofdm` importable from src/ when the package is not installed (e.g. a quick check without the venv)."""
import os
import sys

try:
    import uwofdm  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
