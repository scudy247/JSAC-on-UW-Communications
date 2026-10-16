"""
Statistics shared by the experiment scripts and the figures (PLAN: Phase 2, campaign of 17 Oct 2026): confidence
intervals of means and proportions, paired tests, per-configuration seeds.
"""
import zlib
from math import comb

import numpy as np


def ci95(x):
    """95 % half-width of the mean of independent samples x (normal approximation; nan below two samples)."""
    x = np.asarray(x, float)
    return float(1.96 * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def wilson(k, n, z=1.96):
    """95 % Wilson interval [lo, hi] of a proportion k / n (sensible at 0 and n, unlike the normal interval)."""
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def mcnemar(err_a, err_b):
    """Paired comparison of two receivers on the same packets: (lost only by a, lost only by b, exact two-sided
    McNemar p-value, i.e. a binomial test of the discordant pairs with p = 1/2)."""
    b = sum(bool(x) and not bool(y) for x, y in zip(err_a, err_b))
    c = sum(bool(y) and not bool(x) for x, y in zip(err_a, err_b))
    n = b + c
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return b, c, float(p)


def bootstrap_ci(x, stat=np.mean, n_boot=2000, seed=0, level=0.95):
    """Percentile bootstrap interval [lo, hi] of stat(x) (e.g. a 10th percentile of per-packet SINR)."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed)
    b = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    a = (1 - level) / 2
    return [float(np.quantile(b, a)), float(np.quantile(b, 1 - a))]


def seed_of(*parts):
    """Stable 32-bit seed from a configuration (strings / numbers): the same configuration always draws the same
    packets, whatever the order or the split of the jobs (Python's hash() is salted per process, crc32 is not)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode())
