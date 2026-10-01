"""Layer-1 goodput model (THEORY-B G-1/G-2): arms = sub-band x MCS, reward = goodput = rate x success,
gamma-hat (pilot EVM in dB) as the soft signal that predicts success.

Per slot t and sub-band j a packet has n_pilot pilot and n_data data symbols with noise z (isotropic complex
SaS(alpha, c); c = 0.5 makes alpha = 2 unit-power Gaussian, so SNR is the Gaussian-equivalent SNR):
  gamma_db[t, j] = S_j - 10 log10(mean_pilots |z|^2)                         (same for every MCS on j)
  C[t, j]        = achievable rate of the packet's data symbols under `success_model` (s = 10^((S_j - gap_db)/10)):
      "mean_power"  log2(1 + s / mean_data |z|^2)                  impulse-BLIND decoder (Gaussian metric on the
                                                                   packet's average noise power)
      "blanking"    (n_kept / n) log2(1 + s / mean_kept |z|^2)      symbols with |z|^2 > 10^(blank_db/10) blanked
                                                                   (erased), the rest decoded as Gaussian
      "miesm"       mean_data log2(1 + s / |z_i|^2)                 impulse-AWARE decoder (per-symbol noise known;
                                                                   MIESM link abstraction)
  success[t, j, m] = 1{C[t, j] >= rate_m},   goodput = rate_m * success,    S_j = snr_db + subband_gains_db[j].
  Which model holds is a modelling choice (receiver dependent); step 4's real receiver settles it.
Broadband impulses (shared_impulses): the sub-Gaussian mixing variable of each symbol time is shared by all
sub-bands (a snap hits the whole band). Arm index a = j * M + m. Noise stream between packets: real parts of
the same law. truth_mean = expected goodput per arm (seeded Monte Carlo, `expected_goodput`).

SuccessTable: for the learners, on a grid of alpha and S, the success probability of every MCS and the law of
the gamma-hat noise X = -10 log10(mean_pilots |z|^2) (mean, sd, sub-exponential pairs as reward_law.py).
Defaults are Claude's technical proposal (THEORY-B Open: MCS table, success model, sub-bands), to confirm.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from ..estimation.reward_law import NU2_FACTORS, subexp_pairs
from ..noise.stable import positive_stable, sas_complex_isotropic
from .table_env import OutcomeTable


@dataclass(frozen=True)
class GoodputModel:
    alpha: float = 1.6
    c: float = 0.5
    snr_db: float = 7.0
    subband_gains_db: tuple = (0.0, -2.0, -4.0)
    rates: tuple = (0.5, 1.0, 1.5, 2.0, 3.0)
    gap_db: float = 2.0
    n_pilot: int = 64
    n_data: int = 256
    shared_impulses: bool = True
    success_model: str = "blanking"
    blank_db: float = 10.0

    @property
    def J(self) -> int:
        return len(self.subband_gains_db)

    @property
    def M(self) -> int:
        return len(self.rates)

    @property
    def arm_rates(self) -> np.ndarray:
        return np.tile(np.asarray(self.rates, float), self.J)

    def snr_j(self) -> np.ndarray:
        return self.snr_db + np.asarray(self.subband_gains_db, float)


def packet_noise(alpha, c, n_pkt, J, n_sym, rng, shared=True):
    """(n_pkt, J, n_sym) isotropic complex SaS; mixing variable shared across sub-bands if `shared`."""
    if not shared:
        return sas_complex_isotropic(alpha, c, (n_pkt, J, n_sym), rng)
    sigma = np.sqrt(2.0) * c
    g = rng.normal(0.0, sigma, (n_pkt, J, n_sym)) + 1j * rng.normal(0.0, sigma, (n_pkt, J, n_sym))
    if alpha == 2.0:
        return g
    return np.sqrt(positive_stable(alpha / 2.0, (n_pkt, 1, n_sym), rng)) * g


def achievable_rate(model: GoodputModel, s_lin, p2):
    """Packet rate C from data-symbol noise powers p2 (..., n_data) and linear SNR s_lin (broadcast on ...)."""
    s = np.asarray(s_lin, float)[..., None]
    if model.success_model == "mean_power":
        return np.log2(1.0 + s[..., 0] / np.mean(p2, axis=-1))
    if model.success_model == "miesm":
        return np.mean(np.log2(1.0 + s / p2), axis=-1)
    if model.success_model == "blanking":
        keep = p2 <= 10.0 ** (model.blank_db / 10.0)
        nk = keep.sum(axis=-1)
        pk = np.where(nk > 0, np.sum(np.where(keep, p2, 0.0), axis=-1) / np.maximum(nk, 1), np.inf)
        return nk / p2.shape[-1] * np.log2(1.0 + s[..., 0] / pk)
    raise ValueError(f"success_model must be mean_power, blanking or miesm, got {model.success_model!r}")


def _packet_stats(model: GoodputModel, z):
    """gamma_db (n, J) and achievable rate C (n, J) from packet noise z (n, J, n_pilot + n_data)."""
    S = model.snr_j()
    zp, zd = z[..., : model.n_pilot], z[..., model.n_pilot:]
    gamma = S - 10.0 * np.log10(np.mean(np.abs(zp) ** 2, axis=-1))
    s_lin = 10.0 ** ((S - model.gap_db) / 10.0)
    return gamma, achievable_rate(model, np.broadcast_to(s_lin, zd.shape[:-1]), np.abs(zd) ** 2)


def make_goodput_table(model: GoodputModel, T: int, nu: int, seed: int, truth=None, chunk: int = 1000):
    rng = np.random.default_rng(seed)
    J, M = model.J, model.M
    rates = np.asarray(model.rates, float)
    gamma = np.empty((T, J))
    mi = np.empty((T, J))
    for s in range(0, T, chunk):                       # bounded memory
        n = min(chunk, T - s)
        z = packet_noise(model.alpha, model.c, n, J, model.n_pilot + model.n_data, rng, model.shared_impulses)
        gamma[s:s + n], mi[s:s + n] = _packet_stats(model, z)
    stream = sas_complex_isotropic(model.alpha, model.c, (T, nu), rng).real
    succ = (mi[:, :, None] >= rates[None, None, :]).reshape(T, J * M).astype(float)
    g_arm = np.repeat(gamma, M, axis=1)
    truth = expected_goodput(model) if truth is None else np.asarray(truth, float)
    return OutcomeTable(fields={"gamma_db": g_arm, "ack": succ, "goodput": succ * model.arm_rates},
                        truth_mean=np.tile(truth, (T, 1)), noise=stream,
                        meta={"J": J, "M": M, "rates": list(model.rates)})


def expected_goodput(model: GoodputModel, n_mc: int = 100_000, seed: int = 12345) -> np.ndarray:
    """E[goodput] per arm (j * M + m) by a seeded Monte Carlo of n_mc packets."""
    rng = np.random.default_rng(seed)
    rates = np.asarray(model.rates, float)
    tot = np.zeros((model.J, model.M))
    done = 0
    while done < n_mc:
        n = min(10_000, n_mc - done)
        z = packet_noise(model.alpha, model.c, n, model.J, model.n_pilot + model.n_data, rng, model.shared_impulses)
        _, mi = _packet_stats(model, z)
        tot += np.sum(mi[:, :, None] >= rates[None, None, :], axis=0)
        done += n
    return (tot / n_mc * rates[None, :]).ravel()


class SuccessTable:
    """P_m(S; alpha) on grids, and the gamma-hat noise law X(alpha) = -10 log10(mean_pilots |z|^2).
    psucc: (n_alpha, n_S, M); xmean, xsigma: (n_alpha,); nu2, b: (n_alpha, n_factors)."""

    def __init__(self, model: GoodputModel, alphas, s_grid, n_mc: int, seed: int):
        self.alphas = np.asarray(sorted(alphas), float)
        self.s_grid = np.asarray(s_grid, float)
        self.rates = np.asarray(model.rates, float)
        rng = np.random.default_rng(seed)
        ps, xm, xs, nu2, b = [], [], [], [], []
        for a in self.alphas:
            z = packet_noise(a, model.c, n_mc, 1, model.n_pilot + model.n_data, rng, False)[:, 0, :]
            x = -10.0 * np.log10(np.mean(np.abs(z[:, : model.n_pilot]) ** 2, axis=1))
            p2 = np.abs(z[:, model.n_pilot:]) ** 2
            p = np.empty((self.s_grid.size, self.rates.size))
            for i, s in enumerate(self.s_grid):
                cap = achievable_rate(model, np.full(n_mc, 10.0 ** ((s - model.gap_db) / 10.0)), p2)
                p[i] = np.mean(cap[:, None] >= self.rates[None, :], axis=0)
            sg, v, bb = subexp_pairs(x)
            ps.append(p), xm.append(float(x.mean())), xs.append(sg), nu2.append(v), b.append(bb)
        self.psucc, self.xmean, self.xsigma = np.array(ps), np.array(xm), np.array(xs)
        self.nu2, self.b = np.array(nu2), np.array(b)

    def _w(self, alpha):
        a = float(np.clip(alpha, self.alphas[0], self.alphas[-1]))
        i = int(np.clip(np.searchsorted(self.alphas, a) - 1, 0, self.alphas.size - 2))
        return i, (a - self.alphas[i]) / (self.alphas[i + 1] - self.alphas[i])

    def law(self, alpha):
        """(mean, sigma, nu2[j], b[j]) of the gamma-hat noise X at alpha (linear interpolation)."""
        i, w = self._w(alpha)
        f = lambda t: (1 - w) * t[i] + w * t[i + 1]
        return float(f(self.xmean)), float(f(self.xsigma)), f(self.nu2), f(self.b)

    def psucc_at(self, alpha, S):
        """P_m(S; alpha) for an array S -> (len(S), M); S clipped to the grid."""
        i, w = self._w(alpha)
        P = (1 - w) * self.psucc[i] + w * self.psucc[i + 1]
        S = np.clip(np.atleast_1d(np.asarray(S, float)), self.s_grid[0], self.s_grid[-1])
        return np.stack([np.interp(S, self.s_grid, P[:, m]) for m in range(P.shape[1])], axis=-1)


def cached_success_table(model: GoodputModel, alphas, s_grid, n_mc, seed, cache_dir=None) -> SuccessTable:
    from ..runtime import results_root
    md = {k: v for k, v in asdict(model).items() if k not in ("alpha", "snr_db", "subband_gains_db", "shared_impulses")}
    key = json.dumps({"model": md, "alphas": [float(a) for a in sorted(alphas)], "s": [float(s) for s in s_grid],
                      "n_mc": int(n_mc), "seed": int(seed), "factors": NU2_FACTORS.tolist()}, sort_keys=True)
    path = (cache_dir or results_root() / "_cache") / f"success_table-{hashlib.sha1(key.encode()).hexdigest()[:10]}.npz"
    if path.exists():
        z = np.load(path)
        t = SuccessTable.__new__(SuccessTable)
        for k in ("alphas", "s_grid", "rates", "psucc", "xmean", "xsigma", "nu2", "b"):
            setattr(t, k, z[k])
        return t
    t = SuccessTable(model, alphas, s_grid, n_mc, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **{k: getattr(t, k) for k in ("alphas", "s_grid", "rates", "psucc", "xmean", "xsigma", "nu2", "b")})
    tmp.replace(path)
    return t
