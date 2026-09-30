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
