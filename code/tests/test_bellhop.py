"""Bellhop arrivals: geometry of the first arrival, merging, floor (skipped if the binary is missing)."""
import os

import numpy as np
import pytest

from uwofdm import bellhop

pytestmark = pytest.mark.skipif(not os.path.exists(bellhop.BELLHOP_EXE), reason="no Bellhop binary")


def test_direct_path_and_floor():
    arr = bellhop.merge(bellhop.arrivals(1000.0, 2.0, 99.5, 100.0))
    first = arr[0]
    assert first.n_top == 0 and first.n_bot == 0
    assert abs(first.delay - np.hypot(1000.0, 97.5) / bellhop.C) < 1e-4
    prof = bellhop.profile(arr, -20)
    assert prof[0]["delay_ms"] == 0.0 and max(r["power_db"] for r in prof) == 0.0
    assert bellhop.floor(prof, 0.1) > 0
    assert bellhop.floor([dict(r, n_top=0) for r in prof], 0.1) == 0.0


def test_rough_surface_removes_coherent_surface_paths():
    """|R| = exp(-2 (k sigma sin g)^2): ~1 at grazing incidence, ~0 at 10 deg for sigma 0.25 m at 16 kHz; with the
    table in Bellhop, a 2 m sea leaves no strong surface-bounced path on the buoy -> seabed link."""
    assert bellhop.rough_surface_coefficient(0.0, 16e3, 0.25) == 1.0
    assert bellhop.rough_surface_coefficient(10.0, 16e3, 0.25) < 1e-6
    prof = bellhop.profile(bellhop.merge(bellhop.arrivals(1000.0, 2.0, 99.5, 100.0, surface_rms=0.5)), -20)
    assert all(r["n_top"] == 0 for r in prof)
    assert bellhop.floor(prof, 0.1) == 0.0


def test_diffuse_floor_limits():
    """No scattered power: no floor; all the smooth surface's surface-path power scattered: the smooth-surface floor."""
    prof = bellhop.profile(bellhop.merge(bellhop.arrivals(1000.0, 2.0, 99.5, 100.0)), -20)
    d2 = bellhop.diffuse_doppler_factor(prof)
    assert bellhop.diffuse_floor(-200.0, d2, 0.1) < 1e-25
    p = np.array([10 ** (r["power_db"] / 10) for r in prof])
    share = p[[r["n_top"] > 0 for r in prof]].sum() / p.sum()
    rho_db = 10 * np.log10(share / (1 - share))
    assert np.isclose(bellhop.diffuse_floor(rho_db, d2, 0.1), bellhop.floor(prof, 0.1), rtol=1e-6)
