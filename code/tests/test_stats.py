"""Shared statistics (stats.py)."""
import numpy as np

from uwofdm import stats


def test_stats():
    assert stats.mcnemar([1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0, 1]) == (5, 1, 0.21875)
    assert stats.mcnemar([0, 0], [0, 0]) == (0, 0, 1.0)
    lo, hi = stats.wilson(0, 100)
    assert lo == 0.0 and 0.02 < hi < 0.05
    assert abs(stats.ci95(np.r_[np.zeros(50), np.ones(50)]) - 1.96 * 0.5025 / 10) < 1e-3
    lo, hi = stats.bootstrap_ci(np.arange(100.0))
    assert lo < 49.5 < hi
    assert stats.seed_of("buoy users", 4, 8, 30) == stats.seed_of("buoy users", 4, 8, 30) != stats.seed_of("x")
