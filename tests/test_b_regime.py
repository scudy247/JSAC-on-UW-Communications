import numpy as np
import pytest

from uwsb.estimation.regime import RegimeDetector, block_statistics
from uwsb.estimation.tail_index import stable_quantile_fit
from uwsb.noise.regimes import regime_switching_noise
from uwsb.noise.stable import sas_real

B = 1000


def _run(det, x):
    for i in range(0, x.size, B):
        det.update(x[i:i + B])
    return det


def test_regime_generator_follows_the_chain_and_the_regime_laws():
    regimes = [(1.8, 1.0), (1.2, 3.0)]
    P = [[0.95, 0.05], [0.10, 0.90]]
    x, z = regime_switching_noise(regimes, P, 2000, 200, np.random.default_rng(1))
    assert x.size == 2000 * 200 and set(np.unique(z)) == {0, 1}
    stay = np.mean(z[1:][z[:-1] == 0] == 0)
    assert stay == pytest.approx(0.95, abs=0.02)
    for r, (alpha, c) in enumerate(regimes):
        xr = x.reshape(2000, 200)[z == r].ravel()
        a_hat, c_hat = stable_quantile_fit(xr)
        assert a_hat == pytest.approx(alpha, abs=0.06) and c_hat == pytest.approx(c, rel=0.06)
    with pytest.raises(ValueError):
        regime_switching_noise(regimes, [[0.5, 0.4], [0.1, 0.9]], 10, 10, np.random.default_rng(0))


def test_few_false_alarms_on_a_stationary_stream():
    x = sas_real(1.6, 1.0, 2000 * B, np.random.default_rng(2))
    det = _run(RegimeDetector(), x)
    assert len(det.alarms) <= 2                                   # over 2000 blocks


@pytest.mark.parametrize("change", ["level", "tail"])
def test_detects_level_and_tail_changes_quickly(change):
    rng = np.random.default_rng(3)
    before = sas_real(1.8, 1.0, 300 * B, rng)
    after = sas_real(1.8, 3.0, 300 * B, rng) if change == "level" else sas_real(1.2, 1.0, 300 * B, rng)
    det = _run(RegimeDetector(), np.concatenate([before, after]))
    first = [a for a in det.alarms if a >= 300]
    assert first and first[0] - 300 <= 5                          # delay <= 5 blocks
    assert not [a for a in det.alarms if a < 300]                 # no alarm before the change
    assert len(first) <= 2                                        # no alarm storm afterwards


def test_tail_change_is_invisible_to_the_level_statistic_but_seen_by_impulsiveness():
    lev_a, imp_a = block_statistics(sas_real(1.8, 1.0, 20 * B, np.random.default_rng(4)))
    lev_b, imp_b = block_statistics(sas_real(1.2, 1.0, 20 * B, np.random.default_rng(5)))
    theo = lambda a: np.pi ** 2 / 6 * (1 / a ** 2 + 0.5)
    assert imp_a == pytest.approx(theo(1.8), abs=0.1) and imp_b == pytest.approx(theo(1.2), abs=0.15)
    assert abs(lev_b - lev_a) < 0.2 < abs(imp_b - imp_a)


def test_regime_id_and_validation():
    det = RegimeDetector(warmup=5)
    out = det.update(np.ones(20) * np.arange(1, 21))
    assert out["learning"] and out["regime_id"] == 0
    with pytest.raises(ValueError):
        RegimeDetector(warmup=2)
    with pytest.raises(ValueError):
        block_statistics(np.zeros(100))


def test_noise_state_reports_regime_changes():
    from uwsb.estimation.noise_state import NoiseState
    rng = np.random.default_rng(6)
    x = np.concatenate([sas_real(1.8, 1.0, 100 * B, rng), sas_real(1.8, 3.0, 100 * B, rng)])
    ns = NoiseState("log_moment", forget=1 - 1 / 20_000, detector=RegimeDetector())
    flags = [ns.update(x[i:i + B])["regime_change"] for i in range(0, x.size, B)]
    first = int(np.argmax(flags))
    assert any(flags) and 100 <= first <= 105


# --- recalibration for real noise (2026-10-01) ---------------------------------------------------
def _wander(n, sd_db, rho, rng, alpha=1.6):
    """SaS blocks whose level (dB) follows a stationary AR(1): correlated slow wander, no regime change."""
    lev = np.zeros(n)
    e = rng.normal(0.0, sd_db * np.sqrt(1 - rho ** 2), n)
    for i in range(1, n):
        lev[i] = rho * lev[i - 1] + e[i]
    return sas_real(alpha, 1.0, (n, B), rng) * 10 ** (lev[:, None] / 20)


def test_recalibrated_detector_ignores_correlated_wander_that_storms_the_original():
    # reproduces the real-noise failure (27-40 alarms/h on FK01/HI01): seeded regression facts,
    # original 30 alarms in 2000 blocks, recalibrated 0
    x = _wander(2000, 0.3, 0.9, np.random.default_rng(1))
    orig, recal = RegimeDetector(), RegimeDetector.for_real_noise()
    for blk in x:
        orig.update(blk)
        recal.update(blk)
    assert len(orig.alarms) > 10 and len(recal.alarms) == 0


@pytest.mark.parametrize("step_db, expect", [(0.5, False), (3.0, True)])
def test_recalibrated_detector_respects_the_minimum_shift(step_db, expect):
    rng = np.random.default_rng(5)
    x = np.concatenate([sas_real(1.8, 1.0, 400 * B, rng),
                        sas_real(1.8, 10 ** (step_db / 20), 400 * B, rng)])
    det = _run(RegimeDetector.for_real_noise(), x)
    if expect:
        assert det.alarms and 400 <= det.alarms[0] <= 405 and det.alarm_stat[0] == "level"
    else:
        assert det.alarms == []


def test_recalibrated_detector_still_sees_tail_changes_fast():
    rng = np.random.default_rng(6)
    x = np.concatenate([sas_real(1.8, 1.0, 400 * B, rng), sas_real(1.5, 1.0, 400 * B, rng)])
    det = _run(RegimeDetector.for_real_noise(), x)
    assert det.alarms and 400 <= det.alarms[0] <= 405 and det.alarm_stat[0] == "impuls"


def test_long_run_scale_matches_ar1_theory():
    rho, n = 0.8, 4000
    rng = np.random.default_rng(8)
    s = np.zeros((n, 2))
    e = rng.normal(0, 1, (n, 2))
    for i in range(1, n):
        s[i] = rho * s[i - 1] + e[i]
    plain, lr = RegimeDetector(warmup=n), RegimeDetector(warmup=n, long_run=True)
    for v in s:
        plain.update_stats(v)
        lr.update_stats(v)
    assert np.allclose(lr._sd / plain._sd, np.sqrt((1 + rho) / (1 - rho)), rtol=0.1)


def test_defaults_unchanged_and_update_stats_equivalent():
    x = sas_real(1.6, 1.0, 300 * B, np.random.default_rng(9))
    a, b = RegimeDetector(), RegimeDetector()
    for i in range(0, x.size, B):
        assert a.update(x[i:i + B]) == b.update_stats(block_statistics(x[i:i + B]))
    assert np.all(a._drift == a.k)                       # default: drift is k (original detector)
    assert RegimeDetector.for_real_noise(h=5.0).h == 5.0
    with pytest.raises(ValueError):
        RegimeDetector(min_shift_db=-1.0)


def test_nir_ucb_can_use_the_recalibrated_detector():
    from uwsb.bandits.robust import NIRUCB
    det = NIRUCB(2, detector_params=RegimeDetector.REAL_NOISE).ns.detector
    assert det.long_run and det.warmup == 120 and det.min_shift[0] > 0
    assert not NIRUCB(2).ns.detector.long_run                    # default unchanged
