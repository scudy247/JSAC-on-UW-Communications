import numpy as np
import pytest

from uwsb.estimation.noise_state import (
    ExceedanceCounter, LogMomentTracker, NoiseState, QuantileTracker,
)
from uwsb.estimation.tail_index import log_moment_fit
from uwsb.noise.asgn import asgn_m
from uwsb.noise.stable import sas_real


def _stream(tracker, x, block=1000):
    for i in range(0, x.size, block):
        tracker.update(x[i:i + block])
    return tracker.estimate()


@pytest.mark.parametrize("alpha", [1.2, 1.5, 1.8])
def test_both_trackers_converge_on_iid_sas(alpha):
    x = sas_real(alpha, 2.0, 200_000, np.random.default_rng(1))
    a_l, c_l = _stream(LogMomentTracker(), x)
    a_q, c_q = _stream(QuantileTracker(), x)
    assert a_l == pytest.approx(alpha, abs=0.05) and c_l == pytest.approx(2.0, rel=0.05)
    assert a_q == pytest.approx(alpha, abs=0.07) and c_q == pytest.approx(2.0, rel=0.05)


def test_log_moment_streaming_equals_batch_without_forgetting():
    x = sas_real(1.5, 1.0, 50_000, np.random.default_rng(2))
    a_s, c_s = _stream(LogMomentTracker(1.0), x, block=777)       # uneven blocks
    a_b, c_b = log_moment_fit(x)
    assert a_s == pytest.approx(a_b, abs=1e-9) and c_s == pytest.approx(c_b, rel=1e-9)


def test_block_update_equals_sample_by_sample_with_forgetting():
    x = sas_real(1.5, 1.0, 3_000, np.random.default_rng(3))
    a, b = LogMomentTracker(0.999), LogMomentTracker(0.999)
    a.update(x)
    for v in x:
        b.update([v])
    assert a.estimate() == pytest.approx(b.estimate(), rel=1e-9)
    assert a.n_eff == pytest.approx(b.n_eff, rel=1e-9)


def test_forgetting_tracks_a_change_in_alpha():
    rng = np.random.default_rng(4)
    x = np.concatenate([sas_real(1.8, 1.0, 100_000, rng), sas_real(1.3, 1.0, 100_000, rng)])
    lm = LogMomentTracker(forget=1 - 1 / 20_000)                   # memory ~ 20k samples
    for i in range(0, 100_000, 1000):
        lm.update(x[i:i + 1000])
    assert lm.estimate()[0] == pytest.approx(1.8, abs=0.08)
    for i in range(100_000, 200_000, 1000):
        lm.update(x[i:i + 1000])
    assert lm.estimate()[0] == pytest.approx(1.3, abs=0.08)
    qt = QuantileTracker(forget_blocks=1 - 1 / 20)                  # memory ~ 20 blocks
    _stream(qt, x[:100_000])
    assert _stream(qt, x[100_000:])[0] == pytest.approx(1.3, abs=0.1)


def test_consistent_on_asgn_memory_stream():
    x = asgn_m(1.715, 1.0, [1.0, 0.621, 0.237, 0.161, -0.054], 200_000, np.random.default_rng(5))
    assert _stream(LogMomentTracker(), x)[0] == pytest.approx(1.715, abs=0.08)
    assert _stream(QuantileTracker(), x)[0] == pytest.approx(1.715, abs=0.1)


def test_constant_memory_and_zero_handling():
    lm = LogMomentTracker()
    lm.update(np.zeros(10))
    assert lm.n_zero == 10 and np.isnan(lm.estimate()[0])
    x = sas_real(1.5, 1.0, 20_000, np.random.default_rng(6))
    lm.update(x)
    size_before = len(vars(lm))
    lm.update(x)
    assert len(vars(lm)) == size_before                              # no growing buffers


def test_noise_state_exceedances_use_past_scale_only():
    ns = NoiseState("log_moment")
    first = ns.update(np.full(1000, 1e6))                             # huge block, no prior scale
    assert first["n_exceed"] == 0                                      # nothing to compare with yet
    x = sas_real(1.5, 1.0, 200_000, np.random.default_rng(7))
    ns2 = NoiseState("log_moment", k_sigma=5.0)
    for i in range(0, x.size, 1000):
        out = ns2.update(x[i:i + 1000])
    # P(|X| > 5c) for SaS(1.5, c) is a few percent: exceedances are counted and plausible
    assert 0.005 < out["exceed_rate"] < 0.1
    with pytest.raises(ValueError):
        NoiseState("hill")
    ec = ExceedanceCounter(3.0)
    ec.update(np.array([0.0, 4.0, -4.0, 1.0]), 1.0)
    assert ec.n_exceed == 2 and ec.rate == 0.5
