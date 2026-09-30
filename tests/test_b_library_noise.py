"""Seedable library noise model vs uwa_channels.noisegen (needs the library noise files)."""

import numpy as np
import pytest

from uwsb.noise import library_model as lm
from uwsb.runtime import data_dir

uwa = pytest.importorskip("uwa_channels")
import importlib  # noqa: E402
# the package re-exports the function `noisegen` under the submodule's name: import the module
ng_mod = importlib.import_module("uwa_channels.noisegen")

LIB = data_dir() / "uwa_library"
CASES = [("red_noise.mat", 96_000.0, [0, 1, 2]), ("blue_noise.mat", 48_000.0, [0, 5, 11])]


def _load(name):
    path = LIB / name
    if not path.exists():
        pytest.skip(f"{path} not present (set UWSB_DATA_DIR)")
    return uwa.load_noise(str(path))


@pytest.mark.parametrize("name,fs_hz,idx", CASES)
def test_mixing_identical_to_noisegen_for_same_drivers(name, fs_hz, idx, monkeypatch):
    noise = _load(name)
    n = 20_000
    alpha, _, beta = lm.model_params(noise)
    K, K_mix = lm.n_driver_samples(n, fs_hz, noise)
    z = np.random.default_rng(11).standard_normal((K + K_mix, beta.shape[0]))
    # feed the same drivers to the library's generator path
    monkeypatch.setattr(ng_mod.np.random, "randn", lambda *s: z)
    monkeypatch.setattr(ng_mod.levy_stable, "rvs", lambda *a, **k: z)
    ref = ng_mod.noisegen((n, len(idx)), fs_hz, idx, noise)
    ours = lm.mix_drivers(z, n, fs_hz, idx, noise)
    np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("name,fs_hz,idx", CASES)
def test_library_noise_reproducible_and_seed_sensitive(name, fs_hz, idx):
    noise = _load(name)
    a = lm.library_noise(5_000, fs_hz, idx, noise, np.random.default_rng(1))
    b = lm.library_noise(5_000, fs_hz, idx, noise, np.random.default_rng(1))
    c = lm.library_noise(5_000, fs_hz, idx, noise, np.random.default_rng(2))
    assert a.shape == (5_000, len(idx))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_red_model_is_impulsive_and_blue_is_gaussian():
    red, blue = _load("red_noise.mat"), _load("blue_noise.mat")
    assert lm.model_params(red)[0] == pytest.approx(1.7)
    assert lm.model_params(blue)[0] == 2.0
    kurt = lambda x: float(np.mean((x - x.mean()) ** 4) / np.var(x) ** 2 - 3)
    r = lm.library_noise(200_000, 96_000.0, [0], red, np.random.default_rng(3))[:, 0]
    b = lm.library_noise(200_000, 48_000.0, [0], blue, np.random.default_rng(3))[:, 0]
    assert kurt(b) == pytest.approx(0.0, abs=0.1)
    assert kurt(r) > 10.0
