"""
Coded data path of the end-to-end chain (PLAN: Phase 2): bits -> frame data symbols, and demodulated symbols ->
decoded bits / packet error.

Transmitter: info bits -> convolutional code (coding.py) -> interleaver -> Gray QPSK or 16-QAM on the data subcarriers
(used subcarriers minus pilots); pilots: a fixed QPSK sequence; in-band nulls empty (frame.Frame).
Receiver, per OFDM symbol: channel on all subcarriers from the pilots (delay-domain least squares,
experiment._estimator); noise + inter-carrier interference power measured on the in-band nulls (so the LLRs carry
the Doppler residual as noise: the per-subcarrier LLR weighting of Lee et al. 2026); exact QPSK / 16-QAM LLRs; deinterleave;
Viterbi. A packet is in error if any decoded info bit is wrong.
"""
import numpy as np

from . import coding
from .experiment import _estimator


def masks(F):
    """(data, pilot) subcarrier masks of the frame's data symbols."""
    pil = F.o.pilots() & F.used
    return F.used & ~pil, pil


def n_info_bits(F, mod="qpsk"):
    data, _ = masks(F)
    bps = coding.MODS[mod][0]
    return F.n_data * int(data.sum()) * bps // 2 - (coding.K - 1)  # rate 1/2


def pilot_symbols(F, seed=777):
    _, pil = masks(F)
    b = np.random.default_rng(seed).integers(0, 2, (F.n_data, 2 * int(pil.sum())))
    return np.stack([coding.qpsk(r) for r in b])


def build_data(info, F, mod="qpsk"):
    """Data symbols (n_data, n_sc) for the info bits (length n_info_bits(F, mod))."""
    data, pil = masks(F)
    coded = coding.encode(info)
    perm = coding.interleaver(len(coded))
    sym = coding.MODS[mod][1](coded[perm]).reshape(F.n_data, int(data.sum()))
    X = np.zeros((F.n_data, F.o.n_sc), complex)
    X[:, data] = sym
    X[:, pil] = pilot_symbols(F)
    return X


def decode(Y, F, mod="qpsk"):
    """Decoded info bits from demodulated data symbols Y (n_data, n_sc)."""
    o = F.o
    data, pil = masks(F)
    p_idx, d_idx, M = _estimator(o)  # pilots of the numerology (every 4th active) -> other active subcarriers
    P = pilot_symbols(F)
    pil_pos = np.where(pil)[0]
    llrs = []
    for k in range(F.n_data):
        y = Y[k]
        h_all = np.zeros(o.n_sc, complex)
        ratio = np.zeros(len(p_idx), complex)
        sel = np.isin(p_idx, pil_pos)
        ratio[sel] = y[p_idx[sel]] / P[k]
        h_all[d_idx] = M @ ratio
        h_all[p_idx] = ratio
        nv = max(float(np.mean(np.abs(y[F.nulls]) ** 2)), 1e-12)
        llrs.append(coding.MODS[mod][2](y[data], h_all[data], nv))
    llr = np.concatenate(llrs)
    perm = coding.interleaver(len(llr))
    deint = np.empty_like(llr)
    deint[perm] = llr
    return coding.viterbi(deint, n_info_bits(F, mod))


def sinr(Y, X, F):
    """SINR on the data subcarriers after the receiver's own pilot channel estimate (no genie): signal power of
    h x vs power of y - h x, accumulated over the packet."""
    o = F.o
    data, pil = masks(F)
    p_idx, d_idx, M = _estimator(o)
    pil_pos = np.where(pil)[0]
    sel = np.isin(p_idx, pil_pos)
    sig = err = 0.0
    for y, x in zip(Y, X):
        ratio = np.zeros(len(p_idx), complex)
        ratio[sel] = y[p_idx[sel]] / x[p_idx[sel]]
        h = np.zeros(o.n_sc, complex)
        h[d_idx] = M @ ratio
        sig += np.sum(np.abs(h[data] * x[data]) ** 2)
        err += np.sum(np.abs(y[data] - h[data] * x[data]) ** 2)
    return sig, err


def _branch_terms(Y, X_pilots_known, F):
    """Per-symbol channel on all subcarriers (pilots) and noise + interference power (nulls) of one branch."""
    o = F.o
    data, pil = masks(F)
    p_idx, d_idx, M = _estimator(o)
    P = pilot_symbols(F)
    sel = np.isin(p_idx, np.where(pil)[0])
    H, NV = [], []
    for k, y in enumerate(Y):
        ratio = np.zeros(len(p_idx), complex)
        ratio[sel] = y[p_idx[sel]] / P[k]
        h = np.zeros(o.n_sc, complex)
        h[d_idx] = M @ ratio
        h[p_idx] = ratio
        H.append(h)
        NV.append(max(float(np.mean(np.abs(y[F.nulls]) ** 2)), 1e-12))
    return np.array(H), np.array(NV)


def combine(Ys, F):
    """Maximum-ratio combining of branches (one per resampling scale): per symbol and subcarrier,
    z = sum_b conj(H_b) Y_b / nv_b / sum_b |H_b|^2 / nv_b, with effective noise 1 / sum_b |H_b|^2 / nv_b."""
    num = den = 0.0
    for Y in Ys:
        H, NV = _branch_terms(Y, None, F)
        num = num + np.conj(H) * Y / NV[:, None]
        den = den + np.abs(H) ** 2 / NV[:, None]
    return num / np.maximum(den, 1e-30), 1.0 / np.maximum(den, 1e-30)


def decode_combined(Ys, F, mod="qpsk"):
    """Decode after MRC of branches (equalised symbols with unit channel and effective noise per subcarrier)."""
    data, _ = masks(F)
    Z, NVe = combine(Ys, F)
    llr = np.concatenate([coding.MODS[mod][2](Z[k][data], np.ones(int(data.sum())), NVe[k][data])
                          for k in range(F.n_data)])
    perm = coding.interleaver(len(llr))
    deint = np.empty_like(llr)
    deint[perm] = llr
    return coding.viterbi(deint, n_info_bits(F, mod))


def sinr_combined(Ys, X, F):
    """SINR on data subcarriers after MRC of branches, each subcarrier weighted by its combined reliability
    (1 / effective noise), so that deep fades do not dominate (consistent with sinr())."""
    data, _ = masks(F)
    Z, NVe = combine(Ys, F)
    w = 1.0 / NVe[:, data]
    return float(np.sum(w * np.abs(X[:, data]) ** 2)), float(np.sum(w * np.abs(Z[:, data] - X[:, data]) ** 2))


def hard_decisions(Y, F, mod="qpsk"):
    """Frame data symbols from one branch: per-symbol pilot channel, equalise, slice the data subcarriers; pilots
    known, nulls zero (used to regenerate the signal for interference cancellation)."""
    data, pil = masks(F)
    H, _ = _branch_terms(Y, None, F)
    Z = Y / np.where(np.abs(H) > 1e-12, H, 1.0)
    X = np.zeros_like(Y)
    if mod == "qpsk":
        X[:, data] = (np.sign(Z[:, data].real) + 1j * np.sign(Z[:, data].imag)) / np.sqrt(2)
    else:
        lv = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        q = lambda u: lv[np.argmin(np.abs(u[..., None] - lv), axis=-1)]
        X[:, data] = q(Z[:, data].real) + 1j * q(Z[:, data].imag)
    X[:, pil] = pilot_symbols(F)
    return X
