"""
Partial-FFT demodulation (PLAN: Phase 5, fair signal-only baselines): the signal-only answer to a Doppler that varies
within an OFDM symbol (S. Yerramalli, M. Stojanovic, U. Mitra, "Partial FFT demodulation: a detection method for
highly Doppler distorted OFDM systems", IEEE TSP 2012).

The FFT window of each symbol is cut into n_seg equal segments; segment p alone gives Y_p = FFT(window * 1_p) (their
sum is the ordinary FFT). A residual phase drift within the window turns, to first order, into a different phase per
segment, which the combination Z = sum_p w_p Y_p undoes. Weights here are estimated per symbol from the pilots, jointly
with the channel on a sparse delay support (the support found by receiver.somp on the packet), by linear least
squares with w_1 = 1:
    Y_1[k] = - sum_{p >= 2} w_p Y_p[k] + X_k sum_s A[k, s] h_s        for every pilot subcarrier k,
unknowns w_2..w_P and the taps h (the original papers adapt the weights by RLS on pilots / decisions; the joint LS is
the block version of the same criterion). Z then goes to the ordinary channel estimate, LLRs and decoder.
"""
import numpy as np


def segment_ffts(seg, o, n_seg):
    """(n_seg, n_sc) partial FFTs of one FFT window (same normalisation as OFDM.demodulate)."""
    L = len(seg) // n_seg
    out = np.empty((n_seg, o.n_sc), complex)
    for p in range(n_seg):
        part = np.zeros(len(seg), complex)
        b = len(seg) if p == n_seg - 1 else (p + 1) * L
        part[p * L:b] = seg[p * L:b]
        out[p] = np.fft.fft(part)[o.bins()] * np.sqrt(o.n_sc) / o.n_fft
    return out


def combine_weights(Yp, x_pil, pil, A_pil):
    """Per-symbol weights (n_seg,) with w_1 = 1 and taps h from the pilots: Yp (n_seg, n_sc), x_pil pilot symbols at
    subcarriers pil, A_pil (len(pil), n_taps) steering matrix of the support."""
    n_seg = len(Yp)
    M = np.concatenate([-Yp[1:, pil].T, x_pil[:, None] * A_pil], axis=1)
    sol = np.linalg.lstsq(M, Yp[0, pil], rcond=None)[0]
    return np.concatenate([[1.0], sol[:n_seg - 1]]), sol[n_seg - 1:]
