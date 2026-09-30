import numpy as np
import pytest
import soundfile as sf

from uwsb.noise.impulsiveness import robust_sigma
from uwsb.noise.recordings import load_segments, load_window, noise_level, scale_to_snr
from uwsb.runtime import data_dir

FS = 48_000


def _write(path, x, fs=FS):
    sf.write(str(path), x, fs, subtype="PCM_24")
    return path


def test_skip_and_window_select_the_right_samples(tmp_path):
    t = np.arange(10 * FS) / FS
    x = 0.1 * np.sin(2 * np.pi * 13_000 * t)
    x[: 2 * FS] = 0.0                                   # "start-up" segment
    p = _write(tmp_path / "a.flac", x)
    w = load_window(p, (10_500, 15_500), FS, start_s=1.0, duration_s=3.0, skip_s=2.0)
    assert w.samples.size == 3 * FS
    assert np.mean(w.samples ** 2) == pytest.approx(0.005, rel=0.02)   # full tone, no zeros
    assert w.meta()["skip_s"] == 2.0 and w.meta()["start_s"] == 1.0


def test_band_pass_and_resampling(tmp_path):
    t = np.arange(6 * FS) / FS
    tone_in = 0.1 * np.sin(2 * np.pi * 13_000 * t)
    tone_out = 0.1 * np.sin(2 * np.pi * 3_000 * t)
    p = _write(tmp_path / "b.wav", tone_in + tone_out)
    w = load_window(p, (10_500, 15_500), 39_062.5, start_s=0.0, duration_s=4.0, skip_s=1.0)
    assert w.fs_out_hz == 39_062.5
    assert w.samples.size == pytest.approx(4.0 * 39_062.5, abs=2)
    assert np.mean(w.samples ** 2) == pytest.approx(0.005, rel=0.03)   # only the 13 kHz tone
    spec = np.abs(np.fft.rfft(w.samples))
    f = np.fft.rfftfreq(w.samples.size, 1 / 39_062.5)
    assert f[np.argmax(spec)] == pytest.approx(13_000, abs=5)


def test_impulse_position_and_amplitude_preserved(tmp_path):
    rng = np.random.default_rng(1)
    x = 0.01 * rng.standard_normal(8 * FS)
    k = 5 * FS + 1234
    x[k] = 0.8                                          # one strong click
    p = _write(tmp_path / "c.flac", x)
    w = load_window(p, (10_500, 15_500), FS, start_s=0.0, duration_s=6.0, skip_s=1.0)
    peak = int(np.argmax(np.abs(w.samples)))
    assert abs(peak - (k - FS)) <= 3                    # zero-phase: no delay
    # not normalised away: linear band-limiting to B scales the click peak by 2B/fs and the
    # white background sigma by sqrt(2B/fs), so peak/sigma = (0.8/0.01) * sqrt(2B/fs)
    expected = (0.8 / 0.01) * np.sqrt(2 * 5_000 / FS)
    assert np.max(np.abs(w.samples)) / robust_sigma(w.samples) == pytest.approx(expected, rel=0.1)


def test_scaling_methods_robust_vs_power():
    rng = np.random.default_rng(2)
    g = rng.standard_normal(1_000_000)
    for method in ("robust", "power"):
        y = scale_to_snr(g, signal_power=2.0, snr_db=10.0, method=method)
        assert 2.0 / noise_level(y, method) == pytest.approx(10.0, rel=1e-9)
    imp = g.copy()
    imp[rng.choice(g.size, 500, replace=False)] += 300.0          # sparse impulses
    assert noise_level(imp, "robust") / noise_level(g, "robust") == pytest.approx(1.0, abs=0.01)
    assert noise_level(imp, "power") / noise_level(g, "power") > 40.0
    with pytest.raises(ValueError):
        noise_level(g, "variance")


def test_validation(tmp_path):
    p = _write(tmp_path / "d.flac", np.zeros(3 * FS))
    with pytest.raises(ValueError):
        load_window(p, (20_000, 30_000), 96_000, 0.0, 1.0)         # above source Nyquist
    with pytest.raises(ValueError):
        load_window(p, (10_500, 15_500), FS, 0.0, 5.0, skip_s=0.0)   # past end of file
    with pytest.raises(TypeError):
        load_window([p, p], (10_500, 15_500), FS, 0.0, 1.0)


def test_real_fk01_window_loads():
    path = data_dir() / "noise_census" / "S1_fk01" / "SanctSound_FK01_01_671359016_20181218T214011Z.flac"
    if not path.exists():
        pytest.skip(f"{path} not present")
    w = load_window(path, (10_500, 15_500), 48_000, start_s=60.0, duration_s=30.0)
    assert w.samples.shape == (30 * 48_000,)
    assert np.all(np.isfinite(w.samples)) and w.clip_frac == 0.0
    assert w.robust_sigma_in_band > 0


def test_segments_joined_per_file_without_cross_boundary_filtering(tmp_path):
    t = np.arange(8 * FS) / FS
    tone = 0.1 * np.sin(2 * np.pi * 13_000 * t)
    p1 = _write(tmp_path / "s1.flac", tone)                      # 8 s of tone
    p2 = _write(tmp_path / "s2.flac", np.zeros(6 * FS))          # 6 s of silence
    tl = load_segments([p1, p2], (10_500, 15_500), FS, skip_s=2.0)
    assert tl.boundaries == (0, 6 * FS)                          # 8-2 s, then 6-2 s
    assert tl.samples.size == 10 * FS
    assert [m["source"] for m in tl.segments] == ["s1.flac", "s2.flac"]
    b = tl.boundaries[1]
    assert np.max(np.abs(tl.samples[b:b + 100])) == 0.0          # no ringing into segment 2
    assert np.mean(tl.samples[:b] ** 2) == pytest.approx(0.005, rel=0.02)


def test_default_scaling_method_is_robust():
    g = np.random.default_rng(3).standard_normal(100_000)
    assert noise_level(g) == noise_level(g, "robust")
    y = scale_to_snr(g, 1.0, 20.0)
    assert 1.0 / noise_level(y, "robust") == pytest.approx(100.0, rel=1e-9)
