"""
E1: Doppler side information (PLAN: Phase 4). The transmitter measures its own share of the link Doppler while it
emits (IMU / INS: placement.NodeSensors) and carries it in the same packet as B bits; the receiver rebuilds it and
uses it as the transmitter's share of its Doppler estimate.

Representation: the share over the packet [0, T] is projected on K orthonormal Legendre polynomials
phi_k(t) = sqrt(2k+1) P_k(2t/T - 1) (mean square 1 over the packet); coefficient k of a component exp(j 2 pi f t)
has energy (2k+1) j_k(pi f T)^2, so its variance follows from the motion spectrum (coefficient_std) and the residual
of keeping K coefficients is theory.kernel_poly(f, T, K-1). Each coefficient is quantised uniformly with b bits over
+-(|mean| + LOAD sigma_k): both ends know the ranges (they depend on the sea state, not on the packet).

Scoring (time domain, per packet): residual r(t) after the receiver's whole estimate -> interference = mean over FFT
windows of the variance of 2 pi fc int r dt (the quantity of theory.kernel_ici, computed on the packet itself).
"""
import numpy as np
from numpy.polynomial import legendre
from scipy.special import spherical_jn

LOAD = 4.0  # quantiser range in standard deviations of each coefficient


def basis(t, T, K):
    """Orthonormal Legendre basis on [0, T] (mean square 1), shape (len(t), K)."""
    x = 2 * np.asarray(t, float) / T - 1
    return np.stack([np.sqrt(2 * k + 1) * legendre.legval(x, [0] * k + [1]) for k in range(K)], axis=1)


def project(x, t, T, K):
    """Least-squares coefficients of x(t) on the basis (equals the mean of x phi_k on a uniform grid)."""
    B = basis(t, T, K)
    return np.linalg.lstsq(B, np.asarray(x, float), rcond=None)[0]


def reconstruct(c, t, T):
    return basis(t, T, len(c)) @ np.asarray(c, float)


def coefficient_std(f, S, T, K):
    """Std of each coefficient for a stationary share with one-sided spectrum S(f) (same units squared per Hz)."""
    df = f[1] - f[0]
    x = np.pi * np.asarray(f, float) * T
    return np.array([np.sqrt(np.sum(S * (2 * k + 1) * spherical_jn(k, x) ** 2) * df) for k in range(K)])


def quantise(c, bits, ranges):
    """Uniform mid-rise quantiser with 2^bits levels over [-range, range] per coefficient (clipped)."""
    c, ranges = np.asarray(c, float), np.asarray(ranges, float)
    if bits is None:
        return c.copy()
    step = 2 * ranges / 2 ** bits
    idx = np.clip(np.floor((c + ranges) / step), 0, 2 ** bits - 1)
    return -ranges + (idx + 0.5) * step


def interference(r, t, windows, fc):
    """Mean over FFT windows of var(2 pi fc int r dt) — interference power relative to the signal."""
    th = np.concatenate([[0.0], np.cumsum(0.5 * (r[1:] + r[:-1]) * np.diff(t))]) * 2 * np.pi * fc
    return float(np.mean([np.var(th[i:j]) for i, j in windows]))


def allocate(std, total_bits, load=LOAD):
    """Greedy bit allocation (reverse water-filling): each bit goes to the coefficient whose current error is largest.
    Error of coefficient k: std_k^2 if it gets no bits (not sent, rebuilt as 0), else (2 load std_k)^2 / 12 / 4^b_k."""
    std = np.asarray(std, float)
    bits = np.zeros(len(std), int)
    err = lambda k: std[k] ** 2 if bits[k] == 0 else (2 * load * std[k]) ** 2 / 12 / 4.0 ** bits[k]
    for _ in range(int(total_bits)):
        k = max(range(len(std)), key=err)
        bits[k] += 1
    return bits


def quantise_alloc(c, bits, ranges):
    """Quantise coefficient k with bits[k] bits over +-ranges[k]; coefficients with 0 bits are not sent (0)."""
    c, out = np.asarray(c, float), np.zeros(len(bits))
    for k, b in enumerate(bits):
        if b > 0:
            out[k] = quantise(c[k:k + 1], int(b), np.asarray(ranges[k:k + 1]))[0]
    return out


def stage_weights(t, t_samples, W, windows, fc, K):
    """Interference caused by a unit error in each coefficient after the receiver's stage: the error phi_k(t) minus
    what the stage's fit (weights W from its estimates at t_samples) removes of it. W None: no signal stage."""
    T = t[-1]
    B = basis(t, T, K)
    Bs = basis(t_samples, T, K) if W is not None else None
    w = np.empty(K)
    for k in range(K):
        r = B[:, k] if W is None else B[:, k] - W @ Bs[:, k]
        w[k] = interference(r, t, windows, fc)
    return w


def allocate_weighted(std, weights, total_bits, load=LOAD):
    """Greedy allocation by marginal decrease of weighted error per bit. A coefficient is started with 2 bits (1 bit
    over +-load sigma has error 1.33 sigma^2, worse than not sending it); coefficients whose error the receiver's
    stage removes (weight ~ 0) get no bits."""
    std, weights = np.asarray(std, float), np.asarray(weights, float)
    bits = np.zeros(len(std), int)
    err = lambda k, b: std[k] ** 2 if b == 0 else (2 * load * std[k]) ** 2 / 12 / 4.0 ** b
    left = int(total_bits)
    while left > 0:
        best, gain = None, 0.0
        for k in range(len(std)):
            step = 1 if bits[k] > 0 else 2
            if step > left:
                continue
            g = weights[k] * (err(k, bits[k]) - err(k, bits[k] + step)) / step
            if g > gain:
                best, gain = (k, step), g
        if best is None:
            break
        bits[best[0]] += best[1]
        left -= best[1]
    return bits


# --------------------------------------------------------------------------- T2: closed form for the bits (PLAN: Phase 4)
def coefficient_distortion(std, bits, load=LOAD):
    """Mean-square error of each coefficient: std^2 if it gets no bits (rebuilt as 0), else the uniform quantiser's
    (2 load std)^2 / 12 / 4^b (granular noise; overload neglected at load = 4)."""
    std, bits = np.asarray(std, float), np.asarray(bits, int)
    return np.where(bits > 0, (2 * load * std) ** 2 / 12 / 4.0 ** bits, std ** 2)


def predicted_interference(std, weights, bits, floor, load=LOAD):
    """T2 closed form: interference after the receiver's stage when the transmitter's share is sent with the bit
    allocation `bits`: floor (what remains when the transmitter applies its estimate exactly) + sum_k w_k D_k."""
    return float(floor + np.sum(np.asarray(weights) * coefficient_distortion(std, bits, load)))


def rd_bits(std, weights, target):
    """Rate-distortion (reverse water-filling) minimum bits for sum_k w_k E[e_k^2] <= target with Gaussian
    coefficients and ideal vector quantisation: lam_k = w_k std_k^2, D = sum min(theta, lam_k),
    B = sum max(0, log2(lam_k / theta) / 2). Returns (bits, theta); bits = 0 if the target needs nothing."""
    lam = np.sort(np.asarray(weights, float) * np.asarray(std, float) ** 2)[::-1]
    lam = lam[lam > 0]  # coefficients the receiver's stage removes cost nothing
    if np.sum(lam) <= target:
        return 0.0, float(lam[0]) if len(lam) else 0.0
    lo, hi = 0.0, float(lam[0])
    for _ in range(200):  # bisection on theta: D(theta) increasing
        th = 0.5 * (lo + hi)
        if np.sum(np.minimum(th, lam)) > target:
            hi = th
        else:
            lo = th
    th = lo
    return float(np.sum(np.maximum(0.0, 0.5 * np.log2(lam / th)))), float(th)
