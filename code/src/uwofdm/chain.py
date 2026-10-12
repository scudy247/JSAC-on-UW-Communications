"""
End-to-end receivers (PLAN: Phase 5). Both take the received baseband y (unknown start, unknown Doppler) and return
the demodulated data symbols with what they estimated.

receive_signal_only  the signal-only front end of Li et al. 2008 / Lee et al. 2026: find the preamble, search the
                     postamble over the full Doppler range, one scale per packet from their gap, resample, then remove
                     a residual frequency offset per OFDM symbol from the null subcarriers.
receive_aided        sensor-aided: find the preamble (the chirp tolerates Doppler), resample along the sensor prior
                     (side information + own IMU, as a Doppler trajectory over the frame), then estimate only what the
                     prior missed: a residual scale from preamble/postamble in a narrow window, and the per-symbol
                     residual offset as above. Steps can be switched off to measure what each source contributes.
The prior is given per emitted sample of the frame (index 0 = start of the preamble).
"""
import numpy as np

from . import estimators, fusion, header as hdr
from .waveform import unwarp


CFO_SPAN = 4.0  # residual-CFO search in subcarrier spacings. Was 2.0 until 12 Oct 2026: rough-sea residuals after the
# chirps' mean scale reach 2-3.7 subcarriers on buoy-buoy 4 s packets and the +-2 search failed there (signal Kalman
# + pFFT 10-27 dB -> 28-29 dB with +-4; no change at 0-10 dB SNR). Results with +-2 understate signal-only receivers.


def _demod(z, F, cfo=True, backoff=0):
    """Data symbols of a resampled frame z (index 0 = preamble), with optional per-symbol residual-CFO removal."""
    o = F.o
    Y, offsets = np.empty((F.n_data, o.n_sc), complex), []
    for k in range(F.n_data):
        a = F.data_start + k * o.n_symbol + o.n_cp - backoff
        seg = z[a:a + o.n_fft]
        if len(seg) < o.n_fft:
            seg = np.pad(seg, (0, o.n_fft - len(seg)))
        d = estimators.residual_cfo(seg, o, span=CFO_SPAN, n_grid=81, null_bins=F.null_bins()) if cfo else 0.0
        seg = seg * np.exp(-2j * np.pi * d * np.arange(o.n_fft) / o.fs)
        Y[k] = np.fft.fft(seg)[o.bins()] * np.sqrt(o.n_sc) / o.n_fft
        offsets.append(d)
    return Y, offsets


def _post_window(n_pre, F, eps_max):
    """Search window for the postamble when the scale is within +-eps_max."""
    lo = n_pre + F.gap / (1 + eps_max)
    hi = n_pre + F.gap / (1 - eps_max)
    return int(np.floor(lo)) - 2, int(np.ceil(hi)) + 2


def receive_signal_only(y, F, search=(0, None), eps_max=3e-3, cfo=True, backoff=0):
    o, c = F.o, F.chirp()
    n_pre, _ = estimators.detect_chirp(y, F, *search)
    n_post, _ = estimators.detect_chirp(y, F, *_post_window(n_pre, F, eps_max))
    eps = estimators.scale_pre_post(n_pre, n_post, F.gap)
    i0 = int(round(n_pre))
    seg = y[i0:i0 + int(F.length * (1 + 2 * eps_max)) + 4096]
    z = unwarp(seg, o.fs, np.full(len(seg), eps), o.fc)
    Y, offs = _demod(z, F, cfo, backoff)
    return Y, dict(n_pre=n_pre, eps=eps, cfo=offs, post_window=_post_window(n_pre, F, eps_max))


def receive_aided(y, F, prior, search=(0, None), eps_res_max=3e-4, prepost=True, cfo=True, backoff=0):
    """prior: Doppler trajectory for emitted samples 0 .. len(prior)-1 of the frame (side information + own IMU)."""
    o, c = F.o, F.chirp()
    n_pre, _ = estimators.detect_chirp(y, F, *search)
    i0 = int(round(n_pre))
    seg = y[i0:i0 + len(prior)]
    z = unwarp(seg, o.fs, prior[:len(seg)], o.fc)
    eps_r = 0.0
    if prepost:
        m_pre, _ = estimators.detect_chirp(z, F, 0, 8)
        m_post, _ = estimators.detect_chirp(z, F, *_post_window(m_pre, F, eps_res_max))
        eps_r = estimators.scale_pre_post(m_pre, m_post, F.gap)
        z = unwarp(z, o.fs, np.full(len(z), eps_r), o.fc)
    Y, offs = _demod(z, F, cfo, backoff)
    return Y, dict(n_pre=n_pre, eps_residual=eps_r, cfo=offs, post_window=_post_window(0, F, eps_res_max))


def read_header(y, F, rx_prior, alloc, search=(0, None), eps_max=3e-3, backoff=0):
    """Side-information header only: own IMU -> mean scale from preamble / postamble -> residual CFO of the header
    symbol -> soft DBPSK bits -> CRC. Returns (ok, payload bits, info)."""
    o = F.o
    n_pre, _ = estimators.detect_chirp(y, F, *search)
    i0 = int(round(n_pre))
    seg = y[i0:i0 + len(rx_prior)]
    z = unwarp(seg, o.fs, rx_prior[:len(seg)], o.fc)
    m_pre, _ = estimators.detect_chirp(z, F, 0, 8)
    m_post, _ = estimators.detect_chirp(z, F, *_post_window(m_pre, F, eps_max))
    eps_mean = estimators.scale_pre_post(m_pre, m_post, F.gap)
    z = unwarp(z, o.fs, np.full(len(z), eps_mean), o.fc)
    a = F.header_start + o.n_cp - backoff
    hseg = z[a:a + o.n_fft]
    d = estimators.residual_cfo(hseg, o, span=CFO_SPAN, n_grid=81, null_bins=F.null_bins())
    hseg = hseg * np.exp(-2j * np.pi * d * np.arange(o.n_fft) / o.fs)
    Yh = np.fft.fft(hseg)[o.bins()] * np.sqrt(o.n_sc) / o.n_fft
    bits, soft = hdr.demodulate(Yh, int(np.sum(alloc)) + 8, o, F.used)
    payload, ok = hdr.check(bits)
    return ok, payload, dict(header_ok=ok, header_margin=float(np.min(np.abs(soft))), eps_mean=eps_mean, header_cfo=d)


def receive_with_header(y, F, rx_prior, alloc, ranges, basis_fn, search=(0, None), eps_max=3e-3, backoff=0,
                        receiver="aided"):
    """Full sensor-aided chain with a real header. rx_prior: the receiver's own IMU share per emitted sample of the
    frame; alloc / ranges: the side-information quantiser (known to both ends); basis_fn(c, m): rebuild the
    transmitter's share at emitted samples m from coefficients c. Returns data symbols, info (header ok, bits, ...).
    Order: own IMU -> mean scale from preamble / postamble -> residual CFO of the header symbol -> header bits -> CRC;
    if it passes, resample along side information + own IMU and run receive_aided, else fall back to
    receive_signal_only. receiver="safe": receive_fused_safe after the header, and the signal-only Kalman
    (receive_signal_fused) as the fallback."""
    ok, payload, info = read_header(y, F, rx_prior, alloc, search, eps_max, backoff)
    if not ok:
        fall = receive_signal_fused if receiver == "safe" else receive_signal_only
        Y, inf2 = fall(y, F, search, eps_max=eps_max, backoff=backoff)
        info.update(inf2, fallback=True)
        return Y, info
    coef = hdr.unpack(payload, alloc, ranges)
    prior = basis_fn(coef, np.arange(len(rx_prior))) + rx_prior
    if receiver == "safe":
        Y, inf2 = receive_fused_safe(y, F, prior, search, eps_max=eps_max, backoff=backoff)
    else:
        Y, inf2 = receive_aided(y, F, prior, search, backoff=backoff)
    info.update(inf2, fallback=False, coefficients=coef)
    return Y, info


FUSION = dict(sigma_r0=3e-4, sigma_dr0=3e-4, q=(2e-4) ** 2, cfo_std_0db=0.5, timing_std_samples=0.3)


def symbol_snr(seg, o, F):
    """Per-subcarrier SNR of one FFT window: power on used subcarriers vs power on in-band nulls."""
    Yk = np.fft.fft(seg)[o.bins()] * np.sqrt(o.n_sc) / o.n_fft
    pn = max(float(np.mean(np.abs(Yk[F.nulls]) ** 2)), 1e-12)
    ps = float(np.mean(np.abs(Yk[F.used]) ** 2))
    return max(ps / pn - 1.0, 1e-3)


def receive_signal_fused(y, F, search=(0, None), eps_max=3e-3, params=None, backoff=0):
    """Signal-only Kalman baseline (fair counterpart of receive_fused): the same smoother, but the 'prior' is only
    the mean scale from preamble / postamble searched over +-eps_max, with wide uncertainty and fast dynamics, since
    nothing is known about the motion inside the packet."""
    o, c = F.o, F.chirp()
    n_pre, _ = estimators.detect_chirp(y, F, *search)
    n_post, _ = estimators.detect_chirp(y, F, *_post_window(n_pre, F, eps_max))
    eps = estimators.scale_pre_post(n_pre, n_post, F.gap)
    i0 = int(round(n_pre))
    prior = np.full(int(F.length * (1 + 2 * eps_max)) + 4096, eps)
    p = dict(SIGNAL_FUSION, **(params or {}))
    Y, info = receive_fused(y[i0 - 64:], F, prior, (0, 200), eps_res_max=3e-4, params=p, backoff=backoff)
    info.update(eps_mean=eps)
    return Y, info


# tuned 12 Oct 2026 (scripts/b1_tune_signal_kalman.py: best mean over 4 cases x 2 SNRs; flat for q_std 5e-4 ... 1e-3,
# worse below; the hand-set (2e-3, (4e-3)^2) was 0.13 dB lower on average, up to 0.5 dB on 4 s packets)
SIGNAL_FUSION = dict(sigma_r0=2e-4, sigma_dr0=5e-4, q=(1e-3) ** 2)


def receive_fused(y, F, prior, search=(0, None), eps_res_max=3e-4, params=None, backoff=0):
    """Sensor-aided receiver with Kalman fusion: resample along the prior, measure the postamble timing and each
    symbol's residual frequency offset (with SNR-dependent variance), smooth the residual Doppler over the packet
    (fusion.smooth; or the wave-state model fusion.wave_model / smooth_lti if params["wave"] = (f_p, zeta,
    sigma_w)), resample once along it, demodulate without per-symbol corrections."""
    p = dict(FUSION, **(params or {}))
    o, c = F.o, F.chirp()
    n_pre, _ = estimators.detect_chirp(y, F, *search)
    i0 = int(round(n_pre))
    seg = y[i0:i0 + len(prior)]
    z0 = unwarp(seg, o.fs, prior[:len(seg)], o.fc)
    m_pre, _ = estimators.detect_chirp(z0, F, 0, 8)
    m_post, _ = estimators.detect_chirp(z0, F, *_post_window(m_pre, F, eps_res_max))
    events = []
    tau_post = (F.gap - (m_post - m_pre)) / o.fs  # y(t) = s(t + tau): a feature arrives tau earlier
    events.append((F.post_start / o.fs, "tau", tau_post, (p["timing_std_samples"] / o.fs) ** 2))
    starts = [F.header_start] + [F.data_start + k * o.n_symbol for k in range(F.n_data)]
    snrs = []
    for a0 in starts:
        a = a0 + o.n_cp - backoff
        w = z0[a:a + o.n_fft]
        d = estimators.residual_cfo(w, o, span=CFO_SPAN, n_grid=81, null_bins=F.null_bins())
        snr = symbol_snr(w * np.exp(-2j * np.pi * d * np.arange(o.n_fft) / o.fs), o, F)
        var = (p["cfo_std_0db"] ** 2 / snr + 0.01 ** 2) / o.fc ** 2
        events.append(((a + o.n_fft / 2) / o.fs, "r", d / o.fc, var))
        snrs.append(snr)
    t_grid = np.arange(0, len(z0), 160) / o.fs  # every 5 ms
    if p.get("wave") is not None:  # wave-state model (fusion.wave_model): params["wave"] = (f_p, zeta, sigma_w)
        A, Qc, P0 = fusion.wave_model(*p["wave"], sigma_b0=p["sigma_r0"])
        r_hat, r_std = fusion.smooth_lti(events, t_grid, A, Qc, P0)
    else:
        r_hat, r_std = fusion.smooth(events, t_grid, p["sigma_r0"], p["sigma_dr0"], p["q"])
    r_full = np.interp(np.arange(len(z0)) / o.fs, t_grid, r_hat)
    z = unwarp(z0, o.fs, r_full, o.fc)
    Y, _ = _demod(z, F, cfo=False, backoff=backoff)
    return Y, dict(n_pre=n_pre, tau_post=tau_post, snr_est_db=10 * np.log10(np.mean(snrs)), r_hat=r_hat,
                   r_std=r_std, t_grid=t_grid, z=z)


def null_sinr(Y, F):
    """SINR the receiver can measure without a genie: power on used subcarriers vs noise + interference on the in-band
    nulls, over the packet's data symbols."""
    pn = max(float(np.mean(np.abs(Y[:, F.nulls]) ** 2)), 1e-12)
    return max(float(np.mean(np.abs(Y[:, F.used]) ** 2)) / pn - 1.0, 1e-3)


PRIOR_POOR = 2e-4  # prior error [scale] at which fusion takes the signal-only Kalman settings


def receive_fused_safe(y, F, prior, search=(0, None), prior_std=None, eps_max=3e-3, backoff=0, wave=None):
    """Sensor-aided Kalman fusion that is never much worse than the signal alone (12 Oct 2026: with poor sensors the
    fixed-setting receive_fused fell 3-19 dB below the signal-only Kalman).
    1. Prior quality, per packet: the mean scale of the packet from preamble / postamble over the full +-eps_max
       range, against the mean of the prior over the same span; s = max(prior_std, 2 |difference|), prior_std being
       the sensor's specified error if known.
    2. Fusion adapted to s: residual window max(3e-4, 4 s); prior uncertainty sigma_r0 >= 2 s; rate and process noise
       moved from the sensor settings (FUSION) to the signal-only ones (SIGNAL_FUSION) as s grows to PRIOR_POOR
       (geometric interpolation).
    3. Guard: the signal-only Kalman is run too and the output with the higher null-subcarrier SINR is kept."""
    o = F.o
    n_pre, _ = estimators.detect_chirp(y, F, *search)
    n_post, _ = estimators.detect_chirp(y, F, *_post_window(n_pre, F, eps_max))
    eps_sig = estimators.scale_pre_post(n_pre, n_post, F.gap)
    diff = eps_sig - float(np.mean(prior[:F.gap]))
    s = max(prior_std or 0.0, 2 * abs(diff))
    w = min(s / PRIOR_POOR, 1.0)
    blend = lambda a, b: float(np.exp((1 - w) * np.log(a) + w * np.log(b)))
    p = dict(sigma_r0=max(FUSION["sigma_r0"], 2 * s), sigma_dr0=blend(FUSION["sigma_dr0"], SIGNAL_FUSION["sigma_dr0"]),
             q=blend(FUSION["q"], SIGNAL_FUSION["q"]))
    if wave is not None:  # wave=(sensor residual params, signal-only params), each (f_p, zeta, sigma_w)
        p["wave"] = wave[0]
    window = min(max(3e-4, 4 * s), eps_max)
    Ya, ia = receive_fused(y, F, prior, search, eps_res_max=window, params=p, backoff=backoff)
    Yb, ib = receive_signal_fused(y, F, search, eps_max=eps_max, backoff=backoff,
                                  params=None if wave is None else dict(wave=wave[1]))
    sa, sb = null_sinr(Ya, F), null_sinr(Yb, F)
    info = dict(prior_offset=diff, prior_quality=s, window=window, params=p, null_sinr_fused=sa, null_sinr_signal=sb,
                choice="fused" if sa >= sb else "signal")
    return (Ya if sa >= sb else Yb), dict(ia if sa >= sb else ib, **info)


def receive_cho_kim(y, F, search=(0, None), eps_max=3e-3, rho_min=0.4, backoff=0):
    """Cho and Kim (IET Commun. 2026) baseline: constant velocity + constant acceleration over the frame. Velocity
    (mean scale) from preamble / postamble; acceleration from the per-symbol frequency offsets given by the
    cyclic-prefix correlation (symbols with normalised correlation below rho_min discarded), fitted as a line over
    the symbol centres (slope = acceleration; the intercept also corrects the mean, which gives the baseline its best
    version); one resampling along the resulting linear Doppler trajectory, no per-symbol correction."""
    o, c = F.o, F.chirp()
    n_pre, _ = estimators.detect_chirp(y, F, *search)
    n_post, _ = estimators.detect_chirp(y, F, *_post_window(n_pre, F, eps_max))
    eps = estimators.scale_pre_post(n_pre, n_post, F.gap)
    i0 = int(round(n_pre))
    seg = y[i0:i0 + int(F.length * (1 + 2 * eps_max)) + 4096]
    z0 = unwarp(seg, o.fs, np.full(len(seg), eps), o.fc)
    starts = [F.header_start] + [F.data_start + k * o.n_symbol for k in range(F.n_data)]
    t_c, r = [], []
    for a in starts:
        d, rho = estimators.cp_offset(z0, a, o)
        if rho >= rho_min:
            t_c.append((a + o.n_symbol / 2) / o.fs)
            r.append(d / o.fc)
    t_mid = F.gap / 2 / o.fs
    slope, icpt = np.polyfit(np.array(t_c) - t_mid, r, 1) if len(t_c) >= 2 else (0.0, 0.0)
    t = np.arange(len(z0)) / o.fs
    z = unwarp(z0, o.fs, icpt + slope * (t - t_mid), o.fc)  # line about the frame centre (intercept: best version)
    Y, _ = _demod(z, F, cfo=False, backoff=backoff)
    return Y, dict(n_pre=n_pre, eps=eps + icpt, accel_scale_per_s=slope, accel_m_s2=slope * 1500.0, n_used=len(t_c))


def receive_branches(y, F, trajectories, search=(0, None), backoff=0):
    """Multi-branch front end (Tu et al. 2013, multiple resampling): one resampling per Doppler trajectory (each a
    full trajectory per emitted sample of the frame, index 0 = preamble), each branch demodulated without per-symbol
    corrections. Returns the list of branch outputs (combine with receiver.combine / decode_combined)."""
    o = F.o
    n_pre, _ = estimators.detect_chirp(y, F, *search)
    i0 = int(round(n_pre))
    out = []
    for tr in trajectories:
        seg = y[i0:i0 + len(tr)]
        z = unwarp(seg, o.fs, tr[:len(seg)], o.fc)
        out.append(_demod(z, F, cfo=False, backoff=backoff)[0])
    return out


def grid_trajectories(y, F, offsets=(-1.5e-4, 0.0, 1.5e-4), search=(0, None), eps_max=3e-3):
    """Signal-only branch set (Tu-like): the signal-only Kalman bulk trajectory plus fixed scale offsets."""
    o = F.o
    _, info = receive_signal_fused(y, F, search, eps_max=eps_max)
    n = int(F.length * (1 + 2 * eps_max)) + 4096
    t = np.arange(n) / o.fs
    bulk = info["eps_mean"] + np.interp(t, info["t_grid"], info["r_hat"])
    return [bulk + d for d in offsets]


def receive_sic(y, F, trajectories, rel_delays, search=(0, None), iters=3, mod="qpsk", backoff=0, decisions=None,
                refine_common=False, eps_res_max=3e-4):
    """Per-path interference cancellation. trajectories[p]: Doppler trajectory of path p per emitted sample of the
    frame (index 0 = preamble; path 0 = the path the receiver locks on), rel_delays[p]: its delay relative to path 0 [s]
    (from the link geometry, e.g. Bellhop). Each iteration: decisions on the data (decoder-aided by default: decode,
    re-encode), regenerate every path from them with its own Doppler and delay, fit the path gains by least squares
    on the data part, and for every path p subtract all the others and resample along p's Doppler (one cleaned
    branch per path); the branches are combined (MRC) for the next decisions and the output. With trajectories from
    sensors (own motion, side information, path angles) only the gains are estimated from the signal.
    decisions: None = decoder-aided on the combined branches (hard slicing does not converge at 2-6 dB SINR; decisions
    from path 0 alone locked in at 9 bit errors on a two-path test, from the combined branches they keep improving),
    "hard", or a callable branches -> X. Returns (branches, info); decode with receiver.decode_combined."""
    from .receiver import build_data, combine, decode_combined, hard_decisions
    from .waveform import warp_delayed
    o = F.o
    n_pre, _ = estimators.detect_chirp(y, F, *search)
    i0 = int(round(n_pre))
    n = min(len(trajectories[0]), len(y) - i0)
    seg = y[i0:i0 + n]
    common = 0.0
    if refine_common:  # sensor drift is common to all paths: measure it once from preamble / postamble along path 0
        z0 = unwarp(seg, o.fs, trajectories[0][:n], o.fc)
        m_pre, _ = estimators.detect_chirp(z0, F, 0, 8)
        m_post, _ = estimators.detect_chirp(z0, F, *_post_window(m_pre, F, eps_res_max))
        common = estimators.scale_pre_post(m_pre, m_post, F.gap)
        trajectories = [tr + common for tr in trajectories]
    branches = [_demod(unwarp(seg, o.fs, tr[:n], o.fc), F, cfo=False, backoff=backoff)[0] for tr in trajectories]
    a, b = F.header_start, F.post_start - F.n_guard
    gains, cleaned = None, []
    for _ in range(iters):
        if decisions == "hard":
            Z, _ = combine(branches, F)
            Xh = hard_decisions(Z, F, mod)
        elif decisions is None:
            Xh = build_data(decode_combined(branches, F, mod), F, mod)
        else:
            Xh = decisions(branches)
        s_hat = F.build(np.zeros((F.n_header, o.n_sc), complex), Xh)
        s_hat = np.pad(s_hat, (0, max(0, n - len(s_hat))))[:n]
        regen = [warp_delayed(s_hat, o.fs, tr[:n], o.fc, d) for tr, d in zip(trajectories, rel_delays)]
        A = np.stack([r[a:b] for r in regen], axis=1)
        gains = np.linalg.lstsq(A, seg[a:b], rcond=None)[0]
        branches, cleaned = [], []
        for p, tr in enumerate(trajectories):
            clean_p = seg - sum(g * r for q, (g, r) in enumerate(zip(gains, regen)) if q != p)
            zp = unwarp(clean_p, o.fs, tr[:n], o.fc)
            cleaned.append(zp)
            branches.append(_demod(zp, F, cfo=False, backoff=backoff)[0])
    return branches, dict(n_pre=n_pre, gains=gains, common_correction=common, cleaned=cleaned,
                          trajectories=trajectories)


def signal_path_estimates(y, F, search=(0, None), max_delay=0.025, thresh=0.3, eps_max=3e-3):
    """Signal-only multipath description for cancellation (baseline for the sensor-aided version):
    delays: local peaks of the preamble's up-sweep correlation within max_delay of the first arrival, above thresh of
    the strongest; per-path mean Doppler: each path's own preamble-to-postamble gap (multi-scale estimate);
    within-packet variation: the signal-only Kalman trajectory, shared by all paths.
    Returns (trajectories per emitted sample of the frame, relative delays [s], per-path mean scales)."""
    o = F.o
    n0, _ = estimators.detect_chirp(y, F, *search)
    up = F.chirp_up()
    lo = max(int(n0) - 32, 0)
    hi = int(n0 + max_delay * o.fs) + 32
    seg = y[lo:hi + len(up)]
    corr = np.abs(np.correlate(seg, up, mode="valid"))
    peaks = [i for i in range(1, len(corr) - 1) if corr[i] >= corr[i - 1] and corr[i] >= corr[i + 1]
             and corr[i] >= thresh * corr.max()]
    sel = []
    for i in sorted(peaks, key=lambda k: -corr[k]):  # strongest first, at least 1 ms apart
        if all(abs(i - j) > 0.001 * o.fs for j in sel):
            sel.append(i)
    sel = sorted(sel)
    pos = [lo + i for i in sel]
    pos = [p for p in pos if p >= n0 - 8] or [int(n0)]
    eps0 = None
    scales = []
    for p in pos:
        n_pre, _ = estimators.detect_chirp(y, F, p - 6, p + 6)
        if eps0 is None:
            n_post, _ = estimators.detect_chirp(y, F, *_post_window(n_pre, F, eps_max))
            eps0 = estimators.scale_pre_post(n_pre, n_post, F.gap)
        guess = n_pre + F.gap / (1 + eps0)
        win = int(4e-4 * F.gap) + 8  # +-4e-4 around the first path's scale
        n_post, _ = estimators.detect(y, F.chirp_up(), int(guess) - win, int(guess) + win, first_path=None)
        n_post_d, _ = estimators.detect(y, F.chirp_down(), int(n_post + F.n_half) - 16, int(n_post + F.n_half) + 17,
                                        first_path=None)
        scales.append(estimators.scale_pre_post(n_pre, 0.5 * (n_post + n_post_d - F.n_half), F.gap))
    rel = [(p - pos[0]) / o.fs for p in pos]
    _, info = receive_signal_fused(y, F, search, eps_max=eps_max)
    n = int(F.length * (1 + 2 * eps_max)) + 4096
    t = np.arange(n) / o.fs
    shape = np.interp(t, info["t_grid"], info["r_hat"])
    shape = shape - np.mean(shape[:F.length])
    trajectories = [sc + shape for sc in scales]
    return trajectories, rel, scales


def _refine_path(z, F, tr, p):
    """Per-path residual Doppler from a cleaned, resampled branch: per-symbol offsets on the nulls (with SNR-dependent
    variance), smoothed by the Kalman smoother; returns the corrected trajectory."""
    o = F.o
    events = []
    for a0 in [F.header_start] + [F.data_start + k * o.n_symbol for k in range(F.n_data)]:
        a = a0 + o.n_cp
        w = z[a:a + o.n_fft]
        if len(w) < o.n_fft:
            continue
        d = estimators.residual_cfo(w, o, span=CFO_SPAN, n_grid=81, null_bins=F.null_bins())
        snr = symbol_snr(w * np.exp(-2j * np.pi * d * np.arange(o.n_fft) / o.fs), o, F)
        events.append(((a + o.n_fft / 2) / o.fs, "r", d / o.fc, (p["cfo_std_0db"] ** 2 / snr + 0.01 ** 2) / o.fc ** 2))
    t_grid = np.arange(0, len(tr), 160) / o.fs
    r_hat, _ = fusion.smooth(events, t_grid, p["sigma_r0"], p["sigma_dr0"], p["q"])
    return tr + np.interp(np.arange(len(tr)) / o.fs, t_grid, r_hat)


def _fit_residual(seg, regen, a, b):
    A = np.stack([r[a:b] for r in regen], axis=1)
    g = np.linalg.lstsq(A, seg[a:b], rcond=None)[0]
    return float(np.sum(np.abs(seg[a:b] - A @ g) ** 2))


def refine_paths_by_search(y, F, trs, rel, search=(0, None), grid=np.linspace(-6e-5, 6e-5, 9), passes=2,
                           mod="qpsk", backoff=0):
    """Correct each path's Doppler trajectory by the constant offset in `grid` that best explains the received signal
    (least-squares fit of the paths regenerated from decoder-aided decisions), every path in turn, `passes` times with
    fresh decisions. Used by the signal-only cancellation (wide grid around chirp estimates) and, around a sensor
    prediction with a narrow grid, to absorb path-angle errors."""
    from .receiver import build_data, decode_combined
    from .waveform import warp_delayed
    o = F.o
    trs = [np.array(t, float) for t in trs]
    n_pre, _ = estimators.detect_chirp(y, F, *search)
    i0 = int(round(n_pre))
    n = min(len(trs[0]), len(y) - i0)
    seg = y[i0:i0 + n]
    a, b = F.header_start, F.post_start - F.n_guard
    for _pass in range(passes):
        br, _ = receive_sic(y, F, trs, rel, search, iters=1, mod=mod, backoff=backoff)
        Xh = build_data(decode_combined(br, F, mod), F, mod)
        s_hat = F.build(np.zeros((F.n_header, o.n_sc), complex), Xh)
        s_hat = np.pad(s_hat, (0, max(0, n - len(s_hat))))[:n]
        regen = [warp_delayed(s_hat, o.fs, tr[:n], o.fc, d) for tr, d in zip(trs, rel)]
        for p in range(len(trs)):
            best = None
            for dlt in grid:
                cand = warp_delayed(s_hat, o.fs, trs[p][:n] + dlt, o.fc, rel[p])
                res = _fit_residual(seg, regen[:p] + [cand] + regen[p + 1:], a, b)
                if best is None or res < best[0]:
                    best = (res, dlt, cand)
            trs[p] = trs[p] + best[1]
            regen[p] = best[2]
    return trs


def receive_signal_sic(y, F, search=(0, None), iters=3, mod="qpsk", backoff=0, outer=0,
                       grid=np.linspace(-6e-5, 6e-5, 9), passes=2, **kw):
    """Signal-only per-path cancellation (fair counterpart of the sensor-aided one). Path delays and mean Doppler
    from the chirps (signal_path_estimates; under overlapping chirps the means are off by ~3e-5 while cancellation
    needs ~5e-6), then refine_paths_by_search (wide grid, every path, two passes), then the full cancellation.
    outer > 0 adds rounds of per-path refinement from the cleaned branches (it degraded results when the first decode
    failed, so it is off by default)."""
    trs, rel, scales = signal_path_estimates(y, F, search, **kw)
    if len(trs) == 1:
        return receive_branches(y, F, trs, search, backoff), dict(rel=rel, scales=scales)
    trs = refine_paths_by_search(y, F, trs, rel, search, grid, passes, mod, backoff)
    br, info = receive_sic(y, F, trs, rel, search, iters=iters, mod=mod, backoff=backoff)
    pp = dict(FUSION)
    for _ in range(outer):
        trs = [_refine_path(z, F, tr, pp) for z, tr in zip(info["cleaned"], info["trajectories"])]
        br, info = receive_sic(y, F, trs, rel, search, iters=iters, mod=mod, backoff=backoff)
    info.update(rel=rel, scales=scales)
    return br, info


def receive_sensor_sic(y, F, trs, rel, search=(0, None), iters=3, mod="qpsk", backoff=0, refine_common=True,
                       grid=np.linspace(-2e-5, 2e-5, 9), passes=1):
    """Sensor-aided per-path cancellation with a narrow signal search around the sensor prediction (absorbs path-angle
    and geometry errors): common residual from preamble / postamble, refine_paths_by_search on `grid`, cancellation.
    grid None: no search (receive_sic only)."""
    if refine_common:
        _, inf0 = receive_sic(y, F, trs, rel, search, iters=0, refine_common=True)
        trs = inf0["trajectories"]
    if grid is not None and len(trs) > 1:
        trs = refine_paths_by_search(y, F, trs, rel, search, grid, passes, mod, backoff)
    return receive_sic(y, F, trs, rel, search, iters=iters, mod=mod, backoff=backoff)


def _demod_pfft(z, F, pilots, n_seg=4, cfo=False, backoff=0):
    """Data symbols of a resampled frame z by partial-FFT demodulation (pfft.py): optional per-symbol residual CFO
    first, sparse delay support from the packet's ordinary FFT outputs (receiver.somp), then per-symbol segment
    weights and taps by least squares on the pilots. pilots: (n_data, n_sc) with the pilot symbols in place."""
    from . import pfft, receiver
    o = F.o
    _, pil_mask = receiver.masks(F)
    pil_all, _, A_p, _ = receiver._dictionary(o)
    sel = np.isin(pil_all, np.where(pil_mask)[0])
    pil, A = pil_all[sel], A_p[sel]
    segs, Y = [], np.empty((F.n_data, o.n_sc), complex)
    for k in range(F.n_data):
        a = F.data_start + k * o.n_symbol + o.n_cp - backoff
        seg = z[a:a + o.n_fft]
        if len(seg) < o.n_fft:
            seg = np.pad(seg, (0, o.n_fft - len(seg)))
        if cfo:
            d = estimators.residual_cfo(seg, o, span=CFO_SPAN, n_grid=81, null_bins=F.null_bins())
            seg = seg * np.exp(-2j * np.pi * d * np.arange(o.n_fft) / o.fs)
        segs.append(seg)
        Y[k] = np.fft.fft(seg)[o.bins()] * np.sqrt(o.n_sc) / o.n_fft
    nv = float(np.mean(np.abs(Y[:, F.nulls]) ** 2))
    S, _ = receiver.somp(Y[:, pil] / pilots[:, pil], max(nv, 1e-12), A)
    if not S:
        return Y, dict(weights=None)
    Z, W = np.empty_like(Y), []
    for k, seg in enumerate(segs):
        Yp = pfft.segment_ffts(seg, o, n_seg)
        w, _ = pfft.combine_weights(Yp, pilots[k, pil], pil, A[:, S])
        Z[k] = w @ Yp
        W.append(w)
    return Z, dict(weights=np.array(W), support=S)


def receive_pfft(y, F, pilots, search=(0, None), front="kalman", n_seg=4, eps_max=3e-3, backoff=0):
    """Signal-only receiver with partial-FFT demodulation after a signal-only front end: front 'kalman' = the
    resampling of the signal-only Kalman (receive_signal_fused); 'lilee' = Li et al. 2008 one scale per packet +
    per-symbol residual CFO, as in Yerramalli et al. 2012 (resampling then partial FFT)."""
    o = F.o
    if front == "kalman":
        _, info = receive_signal_fused(y, F, search, eps_max=eps_max, backoff=backoff)
        z, cfo = info["z"], False
    else:
        n_pre, _ = estimators.detect_chirp(y, F, *search)
        n_post, _ = estimators.detect_chirp(y, F, *_post_window(n_pre, F, eps_max))
        eps = estimators.scale_pre_post(n_pre, n_post, F.gap)
        i0 = int(round(n_pre))
        seg = y[i0:i0 + int(F.length * (1 + 2 * eps_max)) + 4096]
        z, cfo = unwarp(seg, o.fs, np.full(len(seg), eps), o.fc), True
    return _demod_pfft(z, F, pilots, n_seg, cfo, backoff)


_LEAK = {}


def _demod_ici(z, F, pilots, n_seg=8, iters=2, backoff=0):
    """ICI-aware equalisation (ici.py) of a resampled frame z: first decisions from partial-FFT demodulation, then
    per symbol `iters` rounds of per-segment sparse taps (least squares on pilots + decisions) and joint MMSE over the
    active band. pilots: (n_data, n_sc) with the pilot symbols in place. Returns equalised symbols (unit channel)."""
    from . import ici, receiver
    o = F.o
    Z, inf = _demod_pfft(z, F, pilots, n_seg, False, backoff)
    S = inf.get("support")
    if not S:
        return Z, dict(inf, ici=False)
    _, act, _, A_a = receiver._dictionary(o)
    key = (o.n_sc, o.n_fft, n_seg)
    if key not in _LEAK:
        _LEAK[key] = ici.leakage(o, n_seg, o.bins()[act])
    C = _LEAK[key]
    A = A_a[:, S]
    _, pil_mask = receiver.masks(F)
    pm, nm = pil_mask[act], F.nulls[act]
    data = F.used[act] & ~pm
    Hz, _ = receiver.channel_estimates(Z, F, P=pilots[:, pil_mask])
    out = np.zeros((F.n_data, o.n_sc), complex)
    for k in range(F.n_data):
        a = F.data_start + k * o.n_symbol + o.n_cp - backoff
        seg = z[a:a + o.n_fft]
        if len(seg) < o.n_fft:
            seg = np.pad(seg, (0, o.n_fft - len(seg)))
        Y = (np.fft.fft(seg)[o.bins()] * np.sqrt(o.n_sc) / o.n_fft)[act]
        X0 = np.where(np.abs(Hz[k, act]) > 1e-12, Z[k, act] / np.where(Hz[k, act] == 0, 1, Hz[k, act]), 0)
        X0[pm] = pilots[k, act][pm]
        X0[nm] = 0
        out[k, act] = ici.equalise_symbol(Y, X0, C, A, data, pm, nm, iters=iters)
    return out, dict(inf, ici=True)


def receive_pilot_track(y, F, pilots, search=(0, None), eps_max=3e-3, params=None, backoff=0, iters=3):
    """Signal-only pilot tracking (PLAN: Phase 5, estimators.pilot_track), closed loop. Li et al. front end (one mean
    scale from the chirps, resampling), then `iters` rounds of: for each pair of consecutive data symbols the
    carrier-phase rotation of the known pilots, c_k = sum_pilots P_k conj(P_(k-1)) exp(-j b m) (P = Y / pilot
    symbol; b the maximum-likelihood linear phase across the band, i.e. the timing change between the symbols, m the
    pilot index about the centre), so that angle(c_k) = 2 pi fc r_k T_sym with r_k the mean residual Doppler between
    the symbol centres; the 2 pi ambiguity (|r| ~ 2e-4) is resolved with the coarse null-subcarrier estimate; variance
    from the pilot SNR (noise + ICI from the nulls); Kalman smoother (signal-only settings) with the postamble timing;
    resampling along the smoothed residual. Each round measures on the signal corrected by the previous ones: in a
    rough sea the first round sees pilots blurred by ICI (|c_k| / N down to 0.15), the next ones see clean pilots.
    Returns (Y, info) with info["z"] (resampled frame)."""
    from . import receiver
    from .experiment import _ramp
    o = F.o
    p = dict(FUSION, **SIGNAL_FUSION, **(params or {}))
    n_pre, _ = estimators.detect_chirp(y, F, *search)
    n_post, _ = estimators.detect_chirp(y, F, *_post_window(n_pre, F, eps_max))
    eps = estimators.scale_pre_post(n_pre, n_post, F.gap)
    i0 = int(round(n_pre))
    seg = y[i0:i0 + int(F.length * (1 + 2 * eps_max)) + 4096]
    z = unwarp(seg, o.fs, np.full(len(seg), eps), o.fc)
    _, pil_mask = receiver.masks(F)
    pil = np.where(pil_mask)[0]
    m = np.arange(len(pil)) - (len(pil) - 1) / 2
    t_sym = o.n_symbol / o.fs
    turn = 1.0 / (o.fc * t_sym)  # one 2 pi ambiguity in r
    t_grid = np.arange(0, len(z), 160) / o.fs
    for it in range(iters):
        P, coarse, nvs = [], [], []
        for k in range(F.n_data):
            a = F.data_start + k * o.n_symbol + o.n_cp - backoff
            w = z[a:a + o.n_fft]
            if len(w) < o.n_fft:
                w = np.pad(w, (0, o.n_fft - len(w)))
            Yk = np.fft.fft(w)[o.bins()] * np.sqrt(o.n_sc) / o.n_fft
            P.append(Yk[pil] / pilots[k, pil])
            nvs.append(max(float(np.mean(np.abs(Yk[F.nulls]) ** 2)), 1e-12))
            coarse.append(estimators.residual_cfo(w, o, span=CFO_SPAN, n_grid=81, null_bins=F.null_bins()) / o.fc)
        m_pre, _ = estimators.detect_chirp(z, F, 0, 8)
        m_post, _ = estimators.detect_chirp(z, F, *_post_window(m_pre, F, 3e-4))
        events = [(F.post_start / o.fs, "tau", (F.gap - (m_post - m_pre)) / o.fs,
                   (p["timing_std_samples"] / o.fs) ** 2)]
        for k in range(1, F.n_data):
            q = P[k] * np.conj(P[k - 1])
            c = np.sum(q * np.exp(-1j * _ramp(q) * m))
            r = np.angle(c) / (2 * np.pi * o.fc * t_sym)
            r = r + turn * np.round((0.5 * (coarse[k] + coarse[k - 1]) - r) / turn)
            pw = float(np.mean(np.abs(P[k]) ** 2))
            var = 2 * 0.5 * (nvs[k] + nvs[k - 1]) / max(len(pil) * pw, 1e-12) / (2 * np.pi * o.fc * t_sym) ** 2 \
                + (1e-7) ** 2
            events.append(((F.data_start + (k - 0.5) * o.n_symbol + o.n_cp + o.n_fft / 2 - backoff) / o.fs, "r", r,
                           var))
        r_hat, r_std = fusion.smooth(events, t_grid, p["sigma_r0"], p["sigma_dr0"], p["q"])
        z = unwarp(z, o.fs, np.interp(np.arange(len(z)) / o.fs, t_grid, r_hat), o.fc)
    Y, _ = _demod(z, F, cfo=False, backoff=backoff)
    return Y, dict(eps=eps, r_hat=r_hat, r_std=r_std, t_grid=t_grid, z=z)
