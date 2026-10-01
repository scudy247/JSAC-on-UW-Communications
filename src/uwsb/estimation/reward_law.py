"""Noise -> reward-noise law for NIR-UCB v1 (THEORY-B G-0, G-5): the confidence widths are calibrated
from the noise stream instead of worst-case constants.

Branch E (gamma-hat in dB from a pilot EVM over n_sym symbols, experiments/run_layer1_b.py):
  gamma_db = P_dB - 10 log10(mean_j |z_j|^2),  z_j i.i.d. isotropic complex SaS(alpha, c).
Because z = c z1 with z1 ~ SaS(alpha, 1), the reward noise X = -10 log10(mean |z|^2) equals
-20 log10(c) + X1 with X1 depending on (alpha, n_sym) only: c shifts every arm equally and does not enter
the widths. EvmDbLaw tabulates, on a grid of alpha, by a seeded Monte Carlo of X1:
  sigma(alpha)          standard deviation (for the Gaussian-structured width);
  (nu2_j, b_j)(alpha)   sub-exponential parameter pairs: for nu2_j = f_j sigma^2 (f_j on NU2_FACTORS),
                        b_j = 1 / lambda_j with lambda_j the largest lambda such that the empirical
                        log-MGF of the centred noise satisfies psi(l) <= l^2 nu2_j / 2 for all 0 < l <= lambda,
                        on BOTH sides (the lower side is the impulsive one). Then (Wainwright 2019, Prop. 2.9)
                        P(|mean_n - mu| >= w) <= 2 exp(-L) for w = max(sqrt(2 nu2 L / n), 2 b L / n).
                        lambda is also capped where the largest sample would carry > 1 % of the empirical
                        MGF (beyond that the Monte Carlo does not resolve the tail).
  mean(alpha)           E[X1]; with scale c the reward-noise mean is E[X1] - 20 log10(c) (`bias`), used by
                        NIR-UCB v1 to debias observations across noise regimes.
Values at a given alpha are linearly interpolated (clipped to the grid).
Branch S (reward = mean + SaS(alpha, c)): sigma = sqrt(2) c (the Gaussian-equivalent scale, as the naive
baseline's oracle); no MGF exists for alpha < 2, so no sub-exponential pairs.
"""

from __future__ import annotations

import math

import numpy as np

from ..noise.stable import sas_complex_isotropic

NU2_FACTORS = np.array([1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0])
MGF_SHARE_CAP = 0.01


def evm_db_noise(alpha: float, n_sym: int, n_pkt: int, rng: np.random.Generator) -> np.ndarray:
    """X1 = -10 log10(mean |z|^2) over n_sym symbols of isotropic complex SaS(alpha, 1)."""
    z = sas_complex_isotropic(alpha, 1.0, (n_pkt, n_sym), rng)
    return -10.0 * np.log10(np.mean(np.abs(z) ** 2, axis=1))


def _lambda_max(y, nu2, lam_grid):
    """Largest grid lambda with psi(l) <= l^2 nu2 / 2 for every grid l <= lambda (y centred)."""
    best = 0.0
    for lam in lam_grid:
        a = lam * y
        m = a.max()
        e = np.exp(a - m)
        tot = e.sum()
        if 1.0 / tot > MGF_SHARE_CAP:                        # largest term's share of the empirical MGF
            break
        psi = m + math.log(tot / y.size)
        if psi > 0.5 * lam * lam * nu2:
            break
        best = lam
    return best


def subexp_pairs(x, factors=NU2_FACTORS, n_lam=400):
    """(sigma, nu2[j], b[j]) of the centred sample x; b = inf where no lambda > 0 qualifies."""
    x = np.asarray(x, dtype=float)
    d = x - x.mean()
    sigma = float(d.std())
    lam_grid = np.geomspace(1e-3, 100.0, n_lam) / sigma
    nu2 = factors * sigma ** 2
    b = np.empty(factors.size)
    for j, v in enumerate(nu2):
        lam = min(_lambda_max(d, v, lam_grid), _lambda_max(-d, v, lam_grid))
        b[j] = 1.0 / lam if lam > 0 else np.inf
    return sigma, nu2, b


class EvmDbLaw:
    """Tabulated branch-E reward-noise law; call law(alpha, c) -> (sigma, nu2[j], b[j])."""

    def __init__(self, n_sym: int, alphas, n_mc: int, seed: int):
        self.n_sym = int(n_sym)
        self.alphas = np.asarray(sorted(alphas), dtype=float)
        if self.alphas.size < 2 or self.alphas[0] <= 1.0 or self.alphas[-1] > 2.0:
            raise ValueError("need >= 2 grid values in (1, 2]")
        rng = np.random.default_rng(seed)
        rows, means = [], []
        for a in self.alphas:
            x = evm_db_noise(a, self.n_sym, int(n_mc), rng)
            rows.append(subexp_pairs(x))
            means.append(float(x.mean()))
        self.mean = np.array(means)
        self.sigma = np.array([r[0] for r in rows])
        self.nu2 = np.array([r[1] for r in rows])                # (n_alpha, n_factors)
        self.b = np.array([r[2] for r in rows])

    def __call__(self, alpha: float, c: float | None = None):
        a = float(np.clip(alpha, self.alphas[0], self.alphas[-1]))
        i = int(np.clip(np.searchsorted(self.alphas, a) - 1, 0, self.alphas.size - 2))
        w = (a - self.alphas[i]) / (self.alphas[i + 1] - self.alphas[i])
        lerp = lambda t: (1 - w) * t[i] + w * t[i + 1]
        return float(lerp(self.sigma)), lerp(self.nu2), lerp(self.b)

    def bias(self, alpha: float, c: float) -> float:
        """E[X] = E[X1](alpha) - 20 log10(c): the reward-noise mean in dB."""
        a = float(np.clip(alpha, self.alphas[0], self.alphas[-1]))
        return float(np.interp(a, self.alphas, self.mean)) - 20.0 * math.log10(float(c))


class SasLaw:
    """Branch S: reward noise = the stream's SaS law; Gaussian-equivalent scale only."""

    def __call__(self, alpha: float, c: float):
        return math.sqrt(2.0) * float(c), None, None

    def bias(self, alpha: float, c: float) -> float:
        return 0.0                                      # symmetric noise: the reward mean is the arm mean


def cached_evm_db_law(n_sym: int, alphas, n_mc: int, seed: int, cache_dir=None) -> EvmDbLaw:
    """EvmDbLaw, stored as .npz under cache_dir (default <results>/_cache) keyed by its parameters."""
    import hashlib
    import json
    from ..runtime import results_root
    key = json.dumps({"n_sym": int(n_sym), "alphas": [float(a) for a in sorted(alphas)], "n_mc": int(n_mc),
                      "seed": int(seed), "factors": NU2_FACTORS.tolist(), "cap": MGF_SHARE_CAP, "v": 2},
                     sort_keys=True)
    path = (cache_dir or results_root() / "_cache") / f"evm_db_law-{hashlib.sha1(key.encode()).hexdigest()[:10]}.npz"
    law = EvmDbLaw.__new__(EvmDbLaw)
    if path.exists():
        z = np.load(path)
        law.n_sym, law.alphas, law.sigma, law.nu2, law.b = int(z["n_sym"]), z["alphas"], z["sigma"], z["nu2"], z["b"]
        law.mean = z["mean"]
        return law
    law = EvmDbLaw(n_sym, alphas, n_mc, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, n_sym=law.n_sym, alphas=law.alphas, sigma=law.sigma, nu2=law.nu2, b=law.b, mean=law.mean)
    tmp.replace(path)
    return law
