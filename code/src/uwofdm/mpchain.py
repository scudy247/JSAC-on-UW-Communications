"""
Multipath receivers with per-path Doppler (PLAN: Phase 5, redesign of 14 Oct 2026 for the realistic channel of
multipath.py: fractional delays, path clusters that fade within the packet, receiver angles from a nominal
geometry). One machinery, two sources for the path description:

  receive_sic_signal   signal-only: path delays from the preamble correlation (sub-sample), each path's mean Doppler
                       from its own preamble / postamble, shared within-packet shape from the signal-only Kalman;
                       per-path search (Doppler offset and sub-sample delay)
  receive_sic_sensor   sensor-aided: per-path Doppler predicted from the side information, the receiver's own IMU and
                       the path angles of a nominal-geometry Bellhop run, matched to the measured arrivals by delay and
                       Doppler; common residual from the chirps; narrow per-path search; guard against the signal-only
                       receiver by null-subcarrier SINR (signal_branches: that receiver's output if already computed)

Shared (both sides): soft interference cancellation (max-log-MAP decoder -> expected symbols, coding.bcjr), path
gains varying within the packet (least squares per data symbol and on the known chirps, interpolated), fractional
delays (waveform.warp_delayed), per-branch partial-FFT demodulation (optional), maximum-ratio combining.
"""
import numpy as np

from . import chain, coding, estimators, receiver
from .waveform import unwarp, warp_delayed


def pilot_frame(F):
    """(n_data, n_sc) with the frame's pilot symbols in place (known to every receiver)."""
    X = np.zeros((F.n_data, F.o.n_sc), complex)
    _, pil = receiver.masks(F)
    X[:, pil] = receiver.pilot_symbols(F)
    return X


def _start(y, F, search):
    n_pre, _ = estimators.detect_chirp(y, F, *search)
    return max(0, int(round(n_pre))), n_pre


def demod_branch(z, F, n_seg=0, backoff=0):
    if n_seg:
        return chain._demod_pfft(z, F, pilot_frame(F), n_seg, False, backoff)[0]
    return chain._demod(z, F, cfo=False, backoff=backoff)[0]


def soft_frame(branches, F, mod="qpsk"):
    """MRC of the branches -> coded-bit LLRs -> max-log-MAP -> expected data symbols (pilots exact, nulls 0) and the
    decoded info bits."""
    data, pil = receiver.masks(F)
    Z, NVe = receiver.combine(branches, F)
    llr = np.concatenate([coding.MODS[mod][2](Z[k][data], np.ones(int(data.sum())), NVe[k][data])
                          for k in range(F.n_data)])
    perm = coding.interleaver(len(llr))
    de = np.empty_like(llr)
    de[perm] = llr
    coded, info = coding.bcjr(de, receiver.n_info_bits(F, mod))
    post = coded[:len(de)][perm]  # back to channel order
    X = np.zeros((F.n_data, F.o.n_sc), complex)
    X[:, data] = coding.SOFT[mod](post).reshape(F.n_data, int(data.sum()))
    X[:, pil] = receiver.pilot_symbols(F)
    return X, (info < 0).astype(int)


def gain_windows(F, n):
    """Sample windows for the time-varying gains: the two chirps and every data symbol (frame coordinates)."""
    o = F.o
    w = [(0, F.n_chirp)]
    w += [(F.data_start + k * o.n_symbol, F.data_start + (k + 1) * o.n_symbol) for k in range(F.n_data)]
    w += [(F.post_start, min(F.post_start + F.n_chirp, n))]
    return w


def fit_gains(seg, regen, windows):
    """Least-squares path gains per window, linearly interpolated between window centres (held at the ends).
    Returns (n_paths, len(seg)) complex gains."""
    n = len(seg)
    G, c = [], []
    for a, b in windows:
        A = np.stack([r[a:b] for r in regen], 1)
        if np.linalg.norm(A) < 1e-9:
            continue
        G.append(np.linalg.lstsq(A, seg[a:b], rcond=None)[0])
        c.append(0.5 * (a + b))
    G, c = np.array(G), np.array(c)
    t = np.arange(n)
    return np.stack([np.interp(t, c, G[:, p].real) + 1j * np.interp(t, c, G[:, p].imag) for p in range(len(regen))])


TAPS = (-4.0, -2.0, 0.0, 2.0, 4.0)  # per-path tap offsets [samples]: a path is a short filter, not one ray (a
# boundary-interacting path with ~0.05 ms of delay spread has a frequency response of its own across the 8 kHz band)


def frame_signal(X, F, n):
    s = F.build(np.zeros((F.n_header, F.o.n_sc), complex), X)
    return np.pad(s, (0, max(0, n - len(s))))[:n]


def path_taps(s, F, tr, d, n, taps=TAPS):
    """Unit-gain contributions of one path's taps (delays d + offsets) along its Doppler trajectory."""
    o = F.o
    return [warp_delayed(s, o.fs, tr[:n], o.fc, d + k / o.fs) for k in taps]


def regenerate(X, F, trs, delays, n, taps=TAPS):
    """Per path, the list of its taps' contributions for data symbols X (chirps known, header left out)."""
    s = frame_signal(X, F, n)
    return [path_taps(s, F, tr, d, n, taps) for tr, d in zip(trs, delays)]


def fit_paths(seg, regen, windows):
    """Time-varying least-squares gains of every tap of every path (fit_gains on the flattened taps); returns the
    fitted contribution of each path, (n_paths, len(seg))."""
    flat = [r for taps in regen for r in taps]
    g = fit_gains(seg, flat, windows)
    out, i = [], 0
    for taps in regen:
        out.append(sum(g[i + j] * taps[j] for j in range(len(taps))))
        i += len(taps)
    return np.array(out)


def sic(y, F, trs, delays, search=(0, None), iters=3, mod="qpsk", n_seg=0, backoff=0):
    """Soft per-path interference cancellation with time-varying gains. trs[p]: Doppler trajectory of path p per
    sample of the frame (index 0 = the locked preamble, path 0), delays[p]: its delay relative to path 0 [s,
    fractional]. Returns (branches, info) with info["bits"] the decoded info bits."""
    o = F.o
    i0, _ = _start(y, F, search)
    n = min(len(trs[0]), len(y) - i0)
    seg = y[i0:i0 + n]
    win = gain_windows(F, n)
    branches = [demod_branch(unwarp(seg, o.fs, tr[:n], o.fc), F, n_seg, backoff) for tr in trs]
    bits = None
    for _ in range(iters):
        X, bits = soft_frame(branches, F, mod)
        contrib = fit_paths(seg, regenerate(X, F, trs, delays, n), win)
        branches = []
        for p, tr in enumerate(trs):
            clean = seg - sum(contrib[q] for q in range(len(trs)) if q != p)
            branches.append(demod_branch(unwarp(clean, o.fs, tr[:n], o.fc), F, n_seg, backoff))
    X, bits = soft_frame(branches, F, mod)
    return branches, dict(bits=bits, trajectories=trs, delays=delays)


def refine(y, F, trs, delays, search=(0, None), eps_grid=np.linspace(-6e-5, 6e-5, 9),
           delay_grid=np.array([-0.5, -0.25, 0.0, 0.25, 0.5]), passes=2, mod="qpsk", n_seg=0):
    """Per-path coordinate search of a constant Doppler offset and a sub-sample delay offset [samples], minimising
    the residual of the regenerated paths (soft decisions from a one-round cancellation, time-varying gains)."""
    o = F.o
    trs, delays = [np.array(t, float) for t in trs], list(delays)
    i0, _ = _start(y, F, search)
    n = min(len(trs[0]), len(y) - i0)
    seg = y[i0:i0 + n]
    win = gain_windows(F, n)

    def resid(regen):
        return float(np.sum(np.abs(seg - fit_paths(seg, regen, win).sum(0)) ** 2))

    for _ in range(passes):
        br, _ = sic(y, F, trs, delays, search, iters=1, mod=mod, n_seg=n_seg)
        X, _ = soft_frame(br, F, mod)
        s = frame_signal(X, F, n)
        regen = [path_taps(s, F, tr, d, n) for tr, d in zip(trs, delays)]
        for p in range(len(trs)):
            best = (resid(regen), 0.0, 0.0, regen[p])
            for de in eps_grid:
                for dd in (delay_grid if p > 0 else [0.0]):  # path 0 is the timing reference
                    if de == 0.0 and dd == 0.0:
                        continue
                    cand = path_taps(s, F, trs[p] + de, delays[p] + dd / o.fs, n)
                    r = resid(regen[:p] + [cand] + regen[p + 1:])
                    if r < best[0]:
                        best = (r, de, dd, cand)
            trs[p] = trs[p] + best[1]
            delays[p] = delays[p] + best[2] / o.fs
            regen[p] = best[3]
    return trs, delays


def arrivals(y, F, search=(0, None), max_delay=0.025, thresh=0.3, eps_max=3e-3):
    """Signal-measured arrivals: delays (sub-sample, relative to the first) from the preamble up-sweep correlation
    peaks above thresh of the strongest, at least 1 ms apart; each arrival's mean Doppler scale from its own
    preamble-to-postamble gap. Returns (delays [s], scales)."""
    o = F.o
    n0, _ = estimators.detect_chirp(y, F, *search)
    up = F.chirp_up()
    lo = max(int(n0) - 32, 0)
    hi = int(n0 + max_delay * o.fs) + 32
    corr = np.abs(np.correlate(y[lo:hi + len(up)], up, mode="valid"))
    peaks = [i for i in range(1, len(corr) - 1) if corr[i] >= corr[i - 1] and corr[i] >= corr[i + 1]
             and corr[i] >= thresh * corr.max()]
    sel = []
    for i in sorted(peaks, key=lambda k: -corr[k]):
        if all(abs(i - j) > 0.001 * o.fs for j in sel):
            sel.append(i)
    pos = []
    for i in sorted(sel):
        l, c, r = corr[i - 1], corr[i], corr[i + 1]
        den = l - 2 * c + r
        pos.append(lo + i + (float(np.clip(0.5 * (l - r) / den, -0.5, 0.5)) if den != 0 else 0.0))
    pos = [p for p in pos if p >= n0 - 8] or [float(n0)]
    scales, eps0 = [], None
    for p in pos:
        n_pre, _ = estimators.detect_chirp(y, F, int(p) - 6, int(p) + 6)
        if eps0 is None:
            n_post, _ = estimators.detect_chirp(y, F, *chain._post_window(n_pre, F, eps_max))
            eps0 = estimators.scale_pre_post(n_pre, n_post, F.gap)
        guess = n_pre + F.gap / (1 + eps0)
        w = int(4e-4 * F.gap) + 8
        n_u, _ = estimators.detect(y, F.chirp_up(), int(guess) - w, int(guess) + w, first_path=None)
        n_d, _ = estimators.detect(y, F.chirp_down(), int(n_u + F.n_half) - 16, int(n_u + F.n_half) + 17,
                                   first_path=None)
        scales.append(estimators.scale_pre_post(n_pre, 0.5 * (n_u + n_d - F.n_half), F.gap))
    return [(p - pos[0]) / o.fs for p in pos], scales


def signal_description(y, F, search=(0, None), eps_max=3e-3):
    """Signal-only path description: measured arrivals (delays, mean scales) + the signal-only Kalman's
    within-packet shape, common to all paths."""
    o = F.o
    delays, scales = arrivals(y, F, search, eps_max=eps_max)
    _, info = chain.receive_signal_fused(y, F, search, eps_max=eps_max)
    n = int(F.length * (1 + 2 * eps_max)) + 4096
    shape = np.interp(np.arange(n) / o.fs, info["t_grid"], info["r_hat"])
    shape = shape - np.mean(shape[:F.length])
    return [sc + shape for sc in scales], delays, scales


def receive_sic_signal(y, F, search=(0, None), iters=3, mod="qpsk", n_seg=0, passes=2):
    trs, delays, _ = signal_description(y, F, search)
    if len(trs) == 1:
        return [demod_branch(unwarp(y[_start(y, F, search)[0]:][:len(trs[0])], F.o.fs, trs[0][:len(y)], F.o.fc),
                             F, n_seg)], dict(paths=1)
    trs, delays = refine(y, F, trs, delays, search, passes=passes, mod=mod, n_seg=n_seg)
    return sic(y, F, trs, delays, search, iters, mod, n_seg)


def associate(pred, pred_delays, meas_delays, meas_scales, F, tol_delay=1.5e-3, tol_scale=1e-4):
    """Match predicted paths (Doppler trajectories, nominal relative delays) to measured arrivals by delay and mean
    Doppler (cost (dd / 0.5 ms)^2 + (de / 3e-5)^2, greedy, within tolerances). Returns a list over measured arrivals
    of the matched predicted index or None."""
    n = F.length
    pm = [float(np.mean(p[:n])) for p in pred]
    cost = [((md - pd) / 5e-4) ** 2 + ((ms - pmi) / 3e-5) ** 2
            for md, ms in zip(meas_delays, meas_scales) for pd, pmi in zip(pred_delays, pm)]
    cost = np.array(cost).reshape(len(meas_delays), len(pred))
    out, used = [None] * len(meas_delays), set()
    for flat in np.argsort(cost, axis=None):
        i, j = divmod(int(flat), len(pred))
        if out[i] is None and j not in used and abs(meas_delays[i] - pred_delays[j]) < tol_delay \
                and abs(meas_scales[i] - pm[j]) < tol_scale:
            out[i] = j
            used.add(j)
    return out


def receive_sic_sensor(y, F, pred, pred_delays, search=(0, None), iters=3, mod="qpsk", n_seg=0, passes=1,
                       eps_grid=np.linspace(-2e-5, 2e-5, 9), guard=True, signal_branches=None):
    """Sensor-aided per-path cancellation. pred: per-path Doppler trajectories predicted from side information, own
    IMU and nominal path angles (per sample of the frame, index 0 = preamble); pred_delays: the nominal relative
    delays (used only to match paths). Delays come from the signal; each measured arrival takes the matched
    prediction (or, unmatched, its signal-only description); the common residual is measured on the chirps along
    path 0; a narrow per-path search absorbs angle errors. Guard: the signal-only cancellation is run too and the
    output with the higher null-subcarrier SINR is kept."""
    o = F.o
    sig_trs, delays, scales = signal_description(y, F, search)
    match = associate(pred, pred_delays, delays, scales, F)
    trs = [np.array(pred[j], float) if j is not None else sig_trs[i] for i, j in enumerate(match)]
    n = min(len(t) for t in trs)
    trs = [t[:n] for t in trs]
    i0, _ = _start(y, F, search)
    z0 = unwarp(y[i0:i0 + n], o.fs, trs[0][:len(y) - i0], o.fc)
    m_pre, _ = estimators.detect_chirp(z0, F, 0, 8)
    m_post, _ = estimators.detect_chirp(z0, F, *chain._post_window(m_pre, F, 3e-4))
    common = estimators.scale_pre_post(m_pre, m_post, F.gap)
    trs = [t + common if match[i] is not None else t for i, t in enumerate(trs)]
    if len(trs) > 1:
        trs, delays = refine(y, F, trs, delays, search, eps_grid=eps_grid, passes=passes, mod=mod, n_seg=n_seg)
        br, info = sic(y, F, trs, delays, search, iters, mod, n_seg)
    else:
        br = [demod_branch(unwarp(y[i0:i0 + n], o.fs, trs[0][:len(y) - i0], o.fc), F, n_seg)]
        info = dict(paths=1)
    info.update(matched=match, sensor_branches=br)
    if not guard:
        return br, info
    br_s = signal_branches if signal_branches is not None else receive_sic_signal(y, F, search, iters, mod, n_seg)[0]
    q = lambda b: null_sinr_branches(b, F)
    info.update(choice="sensor" if q(br) >= q(br_s) else "signal")
    return (br if info["choice"] == "sensor" else br_s), info


def null_sinr_branches(branches, F):
    """SINR the receiver can measure after combining: power on used subcarriers vs the in-band nulls."""
    Z, _ = receiver.combine(branches, F)
    return float(np.mean(np.abs(Z[:, F.used]) ** 2) / max(np.mean(np.abs(Z[:, F.nulls]) ** 2), 1e-12))
