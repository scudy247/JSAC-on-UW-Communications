"""
Channel coding for coded PER (PLAN: Phase 2): rate-1/2, constraint-length-7 convolutional code (generators 133, 171
octal, the IEEE 802.11 / DVB code), zero-tail termination, soft-decision Viterbi decoding, a fixed pseudo-random
interleaver and Gray-mapped QPSK with exact LLRs.

LLR convention: positive LLR means bit 0 (QPSK maps bit 0 -> +1/sqrt(2) on each axis).
"""
import numpy as np

K = 7
G = (0o133, 0o171)
N_STATES = 1 << (K - 1)


def _parity(x):
    return bin(x).count("1") & 1


# trellis: from state s with input bit b -> next state, output bits (2,)
_NEXT = np.zeros((N_STATES, 2), int)
_OUT = np.zeros((N_STATES, 2, 2), int)
for _s in range(N_STATES):
    for _b in (0, 1):
        reg = (_b << (K - 1)) | _s
        _NEXT[_s, _b] = reg >> 1
        _OUT[_s, _b] = [_parity(reg & g) for g in G]
_SIGN = 1 - 2 * _OUT  # bit 0 -> +1


def encode(bits):
    """Zero-tail encoding: len(bits) + K - 1 input bits -> 2 (len(bits) + K - 1) coded bits."""
    s, out = 0, []
    for b in np.concatenate([np.asarray(bits, int), np.zeros(K - 1, int)]):
        out.extend(_OUT[s, b])
        s = _NEXT[s, b]
    return np.array(out, int)


def viterbi(llr, n_info):
    """Soft-decision Viterbi (max-sum on correlation metrics) for a zero-tail code; llr: 2 per input step."""
    n_steps = n_info + K - 1
    L = np.asarray(llr, float)[:2 * n_steps].reshape(n_steps, 2)
    metric = np.full(N_STATES, -np.inf)
    metric[0] = 0.0
    prev = np.zeros((n_steps, N_STATES), np.int32)
    pbit = np.zeros((n_steps, N_STATES), np.int8)
    src = np.repeat(np.arange(N_STATES), 2)
    inp = np.tile([0, 1], N_STATES)
    dst = _NEXT[src, inp]
    sgn = _SIGN[src, inp]  # (2 N_STATES, 2)
    for t in range(n_steps):
        cand = metric[src] + sgn @ L[t]
        new = np.full(N_STATES, -np.inf)
        order = np.argsort(cand)  # write the larger candidates last so they win
        new[dst[order]] = cand[order]
        best = np.zeros(N_STATES, np.int32)
        best[dst[order]] = src[order]
        bb = np.zeros(N_STATES, np.int8)
        bb[dst[order]] = inp[order]
        metric, prev[t], pbit[t] = new, best, bb
    s, bits = 0, np.zeros(n_steps, int)
    for t in range(n_steps - 1, -1, -1):
        bits[t] = pbit[t, s]
        s = prev[t, s]
    return bits[:n_info]


def interleaver(n, seed=12345):
    """Fixed pseudo-random permutation of n coded bits (known to both ends)."""
    return np.random.default_rng(seed).permutation(n)


def qpsk(bits):
    """Gray QPSK, unit energy: bit pairs (b0, b1) -> ((1 - 2 b0) + j (1 - 2 b1)) / sqrt(2)."""
    b = np.asarray(bits, int).reshape(-1, 2)
    return ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])) / np.sqrt(2)


def qpsk_llr(y, h, noise_var):
    """Exact LLRs of Gray QPSK for y = h x + n, n ~ CN(0, noise_var): per-subcarrier weighting by |h|^2 / noise."""
    z = np.conj(h) * y
    nv = np.broadcast_to(noise_var, z.shape)
    return np.stack([2 * np.sqrt(2) * z.real / nv, 2 * np.sqrt(2) * z.imag / nv], axis=-1).reshape(-1)


_PAM4 = {(0, 0): 3, (0, 1): 1, (1, 1): -1, (1, 0): -3}  # Gray 4-PAM per axis
_LEVELS = np.array([3, 1, -1, -3]) / np.sqrt(10)
_LEVEL_BITS = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])


def qam16(bits):
    """Gray 16-QAM, unit average energy: 4 bits (b0 b1 on I, b2 b3 on Q) per symbol."""
    b = np.asarray(bits, int).reshape(-1, 4)
    pam = lambda u, v: np.array([_PAM4[(int(x), int(y))] for x, y in zip(u, v)])
    return (pam(b[:, 0], b[:, 1]) + 1j * pam(b[:, 2], b[:, 3])) / np.sqrt(10)


def qam16_llr(y, h, noise_var):
    """Exact LLRs (log-sum-exp over the 4 levels per axis) of Gray 16-QAM for y = h x + n, n ~ CN(0, noise_var)."""
    g = np.abs(h) ** 2
    z = np.conj(h) * y / np.maximum(g, 1e-30)  # equalised
    s2 = np.broadcast_to(noise_var, z.shape) / (2 * np.maximum(g, 1e-30))  # per-axis noise variance after eq.
    out = []
    for u in (z.real, z.imag):
        d = -(u[:, None] - _LEVELS[None, :]) ** 2 / (2 * s2[:, None])  # log-likelihood per level
        for j in range(2):
            m0, m1 = _LEVEL_BITS[:, j] == 0, _LEVEL_BITS[:, j] == 1
            out.append(np.logaddexp.reduce(d[:, m0], axis=1) - np.logaddexp.reduce(d[:, m1], axis=1))
    return np.stack([out[0], out[1], out[2], out[3]], axis=-1).reshape(-1)


MODS = {"qpsk": (2, qpsk, qpsk_llr), "16qam": (4, qam16, qam16_llr)}


def bcjr(llr, n_info):
    """Max-log-MAP (BCJR) decoder for the zero-tail code: channel LLRs (2 per input step, positive = bit 0) ->
    (a posteriori LLRs of the coded bits, a posteriori LLRs of the info bits). Used for soft interference
    cancellation (soft symbols from the coded-bit LLRs) where Viterbi only gives hard decisions."""
    n_steps = n_info + K - 1
    L = np.asarray(llr, float)[:2 * n_steps].reshape(n_steps, 2)
    src = np.repeat(np.arange(N_STATES), 2)
    inp = np.tile([0, 1], N_STATES)
    dst = _NEXT[src, inp]
    sgn = _SIGN[src, inp].astype(float)  # (2 N_STATES, 2): +1 for coded bit 0
    gam = 0.5 * (L @ sgn.T)  # (n_steps, 2 N_STATES) branch metrics
    neg = -1e30
    alpha = np.full((n_steps + 1, N_STATES), neg)
    alpha[0, 0] = 0.0
    for t in range(n_steps):
        cand = alpha[t, src] + gam[t]
        a = np.full(N_STATES, neg)
        np.maximum.at(a, dst, cand)
        alpha[t + 1] = a - a.max()
    beta = np.full((n_steps + 1, N_STATES), neg)
    beta[n_steps, 0] = 0.0  # zero tail ends in state 0
    for t in range(n_steps - 1, -1, -1):
        cand = beta[t + 1, dst] + gam[t]
        b = np.full(N_STATES, neg)
        np.maximum.at(b, src, cand)
        beta[t] = b - b.max()
    tot = alpha[:-1, src] + gam + beta[1:, dst]  # (n_steps, 2 N_STATES)
    out = np.empty((n_steps, 2))
    for j in range(2):
        b0 = sgn[:, j] > 0
        out[:, j] = tot[:, b0].max(1) - tot[:, ~b0].max(1)
    info = tot[:, inp == 0].max(1) - tot[:, inp == 1].max(1)
    return out.reshape(-1), info[:n_info]


def soft_qpsk(llr):
    """Expected Gray-QPSK symbols from coded-bit LLRs (pairs per symbol): (tanh(L0/2) + j tanh(L1/2)) / sqrt 2."""
    t = np.tanh(0.5 * np.clip(np.asarray(llr, float), -40, 40)).reshape(-1, 2)
    return (t[:, 0] + 1j * t[:, 1]) / np.sqrt(2)


def soft_qam16(llr):
    """Expected Gray 16-QAM symbols from coded-bit LLRs (4 per symbol: b0 b1 on I, b2 b3 on Q)."""
    p0 = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(llr, float), -40, 40))).reshape(-1, 4)  # P(bit = 0)
    def axis(pa, pb):
        e = 0.0
        for (b0, b1), lev in _PAM4.items():
            e = e + (pa if b0 == 0 else 1 - pa) * (pb if b1 == 0 else 1 - pb) * lev
        return e
    return (axis(p0[:, 0], p0[:, 1]) + 1j * axis(p0[:, 2], p0[:, 3])) / np.sqrt(10)


SOFT = {"qpsk": soft_qpsk, "16qam": soft_qam16}
