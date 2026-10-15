"""
Coded data path of the end-to-end chain (PLAN: Phase 2): bits -> frame data symbols, and demodulated symbols ->
decoded bits / packet error.

Transmitter: info bits -> convolutional code (coding.py) -> interleaver -> Gray QPSK or 16-QAM on the data subcarriers
(used subcarriers minus pilots); pilots: a fixed QPSK sequence; in-band nulls empty (frame.Frame).
Receiver, per OFDM symbol: channel on all subcarriers from the pilots (channel_estimates); noise + inter-carrier
interference power measured on the in-band nulls (so the LLRs carry the Doppler residual as noise: the per-subcarrier
LLR weighting of Lee et al. 2026); exact QPSK / 16-QAM LLRs; deinterleave; Viterbi. A packet is in error if any
decoded info bit is wrong.

Channel estimators (CHANNEL_ESTIMATOR):
  'somp'  sparse delay-domain estimate shared by the packet (default from 12 Oct 2026): simultaneous orthogonal
          matching pursuit over the symbols of the packet (Berger et al., IEEE TSP 2010, sparse UWA channels) on the
          twice-oversampled delay grid of experiment._estimator; the delay support is common to all symbols, the tap
          gains are refitted per symbol (residual Doppler phase and drift change them). A tap is kept only while it
          explains more than TAP_GAIN times the noise energy it adds (noise from the in-band nulls), at most MAX_TAPS.
          The old estimator fitted ~390 taps from 248 pilots per symbol (regularised), so its error was about as large
          as the noise: ~4 dB lost (static AWGN, rate-1/2 QPSK: PER 0.15 at 9 dB vs error-free at 5 dB with a genie).
  'ls'    the old per-symbol regularised least squares on the whole delay span (experiment._estimator).
"""
import numpy as np

from . import coding
from .experiment import MAX_DELAY, MIN_DELAY, _estimator

CHANNEL_ESTIMATOR = "somp"
NOISE_SOURCE = "nulls"  # "nulls": noise + interference from the in-band nulls (default); "max": the larger of that and
# the pilot residual after the sparse channel fit (15 Oct 2026, scripts/b23_per_floor.py: cancellation residue and
# neighbouring users' leakage can sit on the data subcarriers without reaching the nulls)
MAX_TAPS = 40
TAP_GAIN = 4.0  # a tap must remove more than TAP_GAIN x (noise variance x symbols) of pilot residual energy
_DICT = {}


def _dictionary(o):
    """Pilot and active-subcarrier steering matrices on the delay grid of experiment._estimator (cached)."""
    key = (o.n_sc, o.pilot_spacing, o.guard_sc, o.bandwidth)
    if key not in _DICT:
        act = np.where(o.active)[0]
        pil = act[o.pilots()[act]]
        dtau = 0.5 / (len(pil) * o.pilot_spacing * o.df)
        taus = np.arange(int(np.floor(MIN_DELAY / dtau)), int(np.ceil(MAX_DELAY / dtau))) * dtau
        f = lambda idx: (idx - pil[0]) * o.df
        _DICT[key] = (pil, act, np.exp(-2j * np.pi * np.outer(f(pil), taus)), np.exp(-2j * np.pi * np.outer(f(act), taus)))
    return _DICT[key]


def somp(R, nv, A, max_taps=MAX_TAPS, tap_gain=TAP_GAIN):
    """Simultaneous OMP. R: (n_sym, n_pilots) pilot observations (y / x), nv: noise variance per observation,
    A: (n_pilots, n_grid) dictionary. Returns (support indices, coefficients (len(support), n_sym))."""
    n_sym = len(R)
    S, coef = [], np.zeros((0, n_sym), complex)
    res, e_prev = R.copy(), float(np.sum(np.abs(R) ** 2))
    An = A / np.linalg.norm(A, axis=0)
    while len(S) < max_taps:
        j = int(np.argmax(np.sum(np.abs(res @ np.conj(An)) ** 2, axis=0)))
        if j in S:
            break
        c = np.linalg.lstsq(A[:, S + [j]], R.T, rcond=None)[0]
        r = R - (A[:, S + [j]] @ c).T
        e = float(np.sum(np.abs(r) ** 2))
        if e_prev - e < tap_gain * nv * n_sym:
            break
        S, coef, res, e_prev = S + [j], c, r, e
    return S, coef


def channel_estimates(Y, F, P=None, method=None):
    """Channel on every subcarrier for each symbol of Y (n_sym, n_sc) from the pilots P (default: the frame's pilot
    symbols), and the noise + interference variance per symbol from the in-band nulls. Returns (H, NV)."""
    o = F.o
    method = method or CHANNEL_ESTIMATOR
    _, pil_mask = masks(F)
    P = pilot_symbols(F) if P is None else P
    NV = np.array([max(float(np.mean(np.abs(y[F.nulls]) ** 2)), 1e-12) for y in Y])
    H = np.zeros(Y.shape, complex)
    if method == "ls":
        p_idx, d_idx, M = _estimator(o)
        sel = np.isin(p_idx, np.where(pil_mask)[0])
        for k, y in enumerate(Y):
            ratio = np.zeros(len(p_idx), complex)
            ratio[sel] = y[p_idx[sel]] / P[k]
            H[k, d_idx] = M @ ratio
            H[k, p_idx] = ratio
        return H, NV
    pil, act, A_p, A_a = _dictionary(o)
    sel = np.isin(pil, np.where(pil_mask)[0])
    R = np.stack([y[pil[sel]] / P[k] for k, y in enumerate(Y)])
    S, coef = somp(R, float(np.mean(NV)), A_p[sel])
    if S:
        H[:, act] = (A_a[:, S] @ coef).T
    if NOISE_SOURCE == "max":  # per-symbol pilot residual (unit-power pilots), degrees of freedom of the fit removed
        res = R - (A_p[sel][:, S] @ coef).T if S else R
        nv_pil = np.sum(np.abs(res) ** 2, axis=1) / max(R.shape[1] - len(S), 1)
        NV = np.maximum(NV, nv_pil)
    return H, NV


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
    data, _ = masks(F)
    H, NV = channel_estimates(Y, F)
    llrs = [coding.MODS[mod][2](Y[k][data], H[k][data], NV[k]) for k in range(F.n_data)]
    llr = np.concatenate(llrs)
    perm = coding.interleaver(len(llr))
    deint = np.empty_like(llr)
    deint[perm] = llr
    return coding.viterbi(deint, n_info_bits(F, mod))


def sinr(Y, X, F):
    """SINR on the data subcarriers after the receiver's own pilot channel estimate (no genie): signal power of
    h x vs power of y - h x, accumulated over the packet."""
    data, pil = masks(F)
    H, _ = channel_estimates(Y, F, P=X[:, pil])
    sig = err = 0.0
    for y, x, h in zip(Y, X, H):
        sig += np.sum(np.abs(h[data] * x[data]) ** 2)
        err += np.sum(np.abs(y[data] - h[data] * x[data]) ** 2)
    return sig, err


def _branch_terms(Y, X_pilots_known, F):
    """Per-symbol channel on all subcarriers (pilots) and noise + interference power (nulls) of one branch."""
    return channel_estimates(Y, F)


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
