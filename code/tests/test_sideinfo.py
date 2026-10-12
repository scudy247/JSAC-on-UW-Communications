"""E1 side information: basis, projection, coefficient statistics, quantiser, time-domain interference."""
import numpy as np

from uwofdm import sideinfo, theory


def test_basis_orthonormal_and_projection():
    T, t = 2.0, np.linspace(0, 2.0, 4001)
    B = sideinfo.basis(t, T, 5)
    G = B.T @ B / len(t)
    assert np.allclose(G, np.eye(5), atol=2e-3)
    c = np.array([0.3, -1.0, 0.2, 0.0, 0.05])
    assert np.allclose(sideinfo.project(sideinfo.reconstruct(c, t, T), t, T, 5), c, atol=1e-9)


def test_coefficient_std_and_residual_match_kernel():
    """For a tone, the energy outside K coefficients is theory.kernel_poly(f, T, K-1)."""
    T, f0 = 1.5, 0.3
    t = np.linspace(0, T, 3001)
    x = np.sqrt(2) * np.cos(2 * np.pi * f0 * t + 0.7)
    res = []
    for ph in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        x = np.sqrt(2) * np.cos(2 * np.pi * f0 * t + ph)
        c = sideinfo.project(x, t, T, 3)
        res.append(np.mean((x - sideinfo.reconstruct(c, t, T)) ** 2))
    assert abs(np.mean(res) - theory.kernel_poly(f0, T, 2)) < 2e-3
    f = np.linspace(0, 1, 1001)
    S = np.where(np.abs(f - f0) < 5e-4, 1.0 / 1e-3, 0.0)  # unit power at f0
    std = sideinfo.coefficient_std(f, S, T, 3)
    assert np.isclose(np.sum(std ** 2) + theory.kernel_poly(f0, T, 2), 1.0, atol=0.02)


def test_quantiser():
    c = np.array([0.0, 0.9, -2.0])
    q = sideinfo.quantise(c, 4, np.array([1.0, 1.0, 1.0]))
    assert np.all(np.abs(q[:2] - c[:2]) <= 2 / 16 / 2 + 1e-12)
    assert q[2] == -1 + 1 / 16  # clipped to the lowest level
    assert np.array_equal(sideinfo.quantise(c, None, np.ones(3)), c)


def test_interference_offset_limit():
    fc, T = 16e3, 0.128
    t = np.linspace(0, T, 513)
    e = 3e-6
    I = sideinfo.interference(np.full(len(t), e), t, [(0, len(t))], fc)
    assert np.isclose(I, (np.pi * fc * e * T) ** 2 / 3, rtol=1e-2)


def test_allocation():
    std = np.array([1.0, 0.3, 0.1, 0.01])
    bits = sideinfo.allocate(std, 10)
    assert bits.sum() == 10 and bits[0] >= bits[1] >= bits[2] >= bits[3]
    assert bits[3] == 0  # a coefficient 100x smaller is not worth a bit at this budget
    q = sideinfo.quantise_alloc(np.array([0.5, -0.2, 0.05, 0.003]), bits, 4 * std)
    assert q[3] == 0.0 and abs(q[0] - 0.5) <= 8 / 2 ** bits[0] / 2 + 1e-12


def test_stage_aware_allocation():
    """A line through two estimates removes most of an error in coefficients 0-1, so they get no bits under that
    stage; without a stage they get the most."""
    from uwofdm import placement as P
    rxm = P.ReceiverModel(packet_s=1.0)
    t, t_s, W, win = P.stage_estimator("line2", rxm)
    w_line = sideinfo.stage_weights(t, t_s, W, win, 16e3, 6)
    w_none = sideinfo.stage_weights(t, t_s, None, win, 16e3, 6)
    assert w_line[0] < 1e-3 * w_none[0] and w_line[1] < 0.05 * w_none[1]
    std = np.array([4e-4, 2e-4, 5e-5, 1e-5, 3e-6, 1e-6])
    b_line = sideinfo.allocate_weighted(std, w_line, 16)
    b_none = sideinfo.allocate_weighted(std, w_none, 16)
    assert b_line[0] == 0 and b_none[0] > 0 and b_line.sum() <= 16
    assert all(b != 1 for b in b_line) and all(b != 1 for b in b_none)


def test_t2_closed_form_and_rate_distortion():
    """T2: predicted interference is the floor plus weighted coefficient errors; reverse water-filling needs no more
    bits than the uniform quantiser with greedy allocation for the same target, and zero bits if the target is loose."""
    std = np.array([1e-4, 5e-5, 2e-5, 5e-6])
    w = np.array([1.0, 2.0, 3.0, 4.0]) * 1e6
    none = sideinfo.predicted_interference(std, w, np.zeros(4, int), 0.0)
    assert abs(none - np.sum(w * std ** 2)) < 1e-12
    target = 0.01 * none
    b_rd, _ = sideinfo.rd_bits(std, w, target)
    b_un = next(b for b in range(1, 200)
                if sideinfo.predicted_interference(std, w, sideinfo.allocate_weighted(std, w, b), 0.0) <= target)
    assert 0 < b_rd <= b_un
    assert sideinfo.rd_bits(std, w, 2 * none)[0] == 0.0
