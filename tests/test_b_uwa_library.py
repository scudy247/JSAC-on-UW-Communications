import numpy as np
import pytest
from scipy import signal

from uwsb.channels.uwa_library import LibraryChannel, wrap_start_s
from uwsb.runtime import data_dir

pytest.importorskip("uwa_channels")


def _chan(name):
    if not (data_dir() / "uwa_library" / f"{name}.mat").exists():
        pytest.skip(f"{name}.mat not present (set UWSB_DATA_DIR)")
    return LibraryChannel(name)


def _band_noise(fs_hz, band_hz, dur_s, seed):
    sos = signal.butter(8, band_hz, btype="bandpass", fs=fs_hz, output="sos")
    x = signal.sosfiltfilt(sos, np.random.default_rng(seed).standard_normal(int(dur_s * fs_hz)))
    return x / np.sqrt(np.mean(x ** 2))


def test_wrap_start_loops_through_the_record():
    assert wrap_start_s(3.0, 1.0, 10.0) == 4.0
    assert wrap_start_s(12.5, 1.0, 10.0) == pytest.approx(3.5)
    with pytest.raises(ValueError):
        wrap_start_s(1.0, 0.0, 0.0)


def test_metadata_red_1():
    ch = _chan("red_1")
    assert (ch.n_time, ch.n_elements, ch.n_taps) == (4588, 3, 768)
    assert ch.fc_hz == 25_000.0 and ch.fs_delay_hz == 19_200.0 and ch.fs_time_hz == 96.0
    assert ch.record_s == pytest.approx(47.79, abs=0.01)
    assert ch.band_hz == (20_000.0, 30_000.0)


def test_replay_deterministic_start_sensitive_and_range_checked():
    ch = _chan("red_1")
    fs = 96_000.0
    x = _band_noise(fs, ch.band_hz, 0.3, seed=1)
    a = ch.replay(x, fs, [0], start_s=10.0)
    b = ch.replay(x, fs, [0], start_s=10.0)
    c = ch.replay(x, fs, [0], start_s=20.0)
    assert np.array_equal(a, b) and not np.allclose(a, c)
    assert a.shape[1] == 1 and a.shape[0] >= x.size
    with pytest.raises(ValueError):
        ch.replay(x, fs, [0], start_s=-0.1)
    with pytest.raises(ValueError):
        ch.replay(x, fs, [0], start_s=ch.max_start_s(x.size, fs))      # would read past the end
    with pytest.raises(ValueError):
        ch.replay(x, fs, [3], start_s=1.0)                              # red_1 has 3 elements


def test_seeded_starts_and_unit_average_power():
    ch = _chan("blue_1")
    fs = 48_000.0
    x = _band_noise(fs, ch.band_hz, 0.25, seed=2)
    rng = np.random.default_rng(3)
    starts = [ch.draw_start_s(rng, x.size, fs) for _ in range(12)]
    rng_a, rng_b = np.random.default_rng(4), np.random.default_rng(4)
    assert ch.draw_start_s(rng_a, x.size, fs) == ch.draw_start_s(rng_b, x.size, fs)
    # library normalisation ([P] §VI-A): total power over the array = number of elements, i.e.
    # unit gain on average over elements (single elements differ by design, e.g. 0.46-1.39).
    # Compare ENERGIES: the output is longer than the input by the delay tail (13 % at 0.25 s),
    # so power averaged over the output would be diluted (measured 0.838 vs energy ratio 0.948).
    all_el = list(range(ch.n_elements))
    ratio = np.mean([np.sum(ch.replay(x, fs, all_el, s) ** 2) / len(all_el) / np.sum(x ** 2)
                     for s in starts])
    assert ratio == pytest.approx(1.0, abs=0.15)


def test_output_delay_within_the_impulse_response_span():
    ch = _chan("blue_1")
    fs = 48_000.0
    x = _band_noise(fs, ch.band_hz, 0.25, seed=5)
    y = ch.replay(x, fs, [0], start_s=5.0)[:, 0]
    xc = signal.correlate(y, x, mode="full", method="fft")
    lag_s = (np.argmax(np.abs(xc)) - (x.size - 1)) / fs
    assert 0.0 <= lag_s <= ch.n_taps / ch.fs_delay_hz              # within the 32.8 ms delay span
