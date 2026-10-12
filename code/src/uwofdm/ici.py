"""
ICI-aware equalisation (PLAN: Phase 5, fair signal-only baselines): the receiver models the inter-carrier
interference that a Doppler varying within the FFT window leaves after resampling, and equalises it jointly instead
of treating it as noise (decision-directed ICI-aware OFDM receivers of the sparse-UWA-channel literature, e.g.
Berger, Zhou, Willett, Huang; basis-expansion channel models).

Model of one FFT window: the channel is a sparse multipath channel (delay support S from receiver.somp on the packet)
whose taps are constant within each of n_seg equal segments (piecewise-constant basis expansion; partial-FFT
demodulation uses the same segments but only re-weights them). The FFT of the window is then
    Y = sum_p C_p diag(H_p) X + noise,      H_p = A c_p (frequency response of segment p's taps),
with C_p the known leakage of segment p between subcarriers: C_p[m, k] = w_p[(m - k) mod n_fft], w_p the
normalised FFT of the segment's indicator.
Receiver per symbol, `iters` times: decisions on the data subcarriers (pilots known, nulls zero), taps of every
segment by least squares on all active subcarriers, then linear MMSE equalisation of the full active band:
    X_hat = (H^H H + nv I)^-1 H^H Y,  H = sum_p C_p diag(H_p)  (active subcarriers only),
nv from the model residual on the in-band nulls. The first decisions come from partial-FFT demodulation.
Output: equalised symbols (unit channel), for genie SINR or for receiver.decode (its pilot estimate sees ~1).
"""
import numpy as np


def leakage(o, n_seg, act):
    """List of (len(act), len(act)) leakage matrices C_p between active subcarriers."""
    n = o.n_fft
    L = n // n_seg
    diff = (act[:, None] - act[None, :]) % n
    out = []
    for p in range(n_seg):
        ind = np.zeros(n)
        ind[p * L:(n if p == n_seg - 1 else (p + 1) * L)] = 1.0
        w = np.fft.fft(ind) / n
        out.append(w[diff])
    return out


def qpsk_slice(z):
    return (np.sign(z.real) + 1j * np.sign(z.imag)) / np.sqrt(2)


def equalise_symbol(Y, X0, C, A, data, pilots_mask, null_mask, slicer=qpsk_slice, iters=2):
    """One symbol. Y: received active subcarriers (len(act)); X0: initial symbol estimate on active subcarriers
    (pilots and nulls already exact); C: leakage list; A: (len(act), n_taps) steering matrix of the support;
    data / pilots_mask / null_mask: boolean masks over active subcarriers. Returns equalised symbols."""
    Xd = X0.copy()
    Xd[data] = slicer(X0[data])
    n_seg, n_taps = len(C), A.shape[1]
    Xe = X0
    for _ in range(iters):
        G = np.concatenate([C[p] @ (Xd[:, None] * A) for p in range(n_seg)], axis=1)  # columns (p, tap)
        c = np.linalg.lstsq(G, Y, rcond=None)[0].reshape(n_seg, n_taps)
        H = sum(C[p] * (A @ c[p])[None, :] for p in range(n_seg))
        nv = max(float(np.mean(np.abs(Y[null_mask] - (H @ Xd)[null_mask]) ** 2)), 1e-12)
        Xe = np.linalg.solve(H.conj().T @ H + nv * np.eye(len(Y)), H.conj().T @ Y)
        Xd = Xe.copy()
        Xd[data] = slicer(Xe[data])
        Xd[pilots_mask] = X0[pilots_mask]
        Xd[null_mask] = 0
    return Xe
