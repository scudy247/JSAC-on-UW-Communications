"""
Replay diagnostic, inherited from an earlier OFDM subcarrier-selection project (uwofdm).
Still useful here for its per-frame common-Doppler estimate (pre/postamble timing) on Channel Library replays.

Question: on Underwater Acoustic Channel Library replays, what per-subcarrier
structure survives (residual-Doppler ICI vs. channel gain vs. colored noise),
does residual ICI grow with subcarrier frequency, and how stable is the
long-term per-subcarrier profile over time / feedback delay?

Method
------
OFDM frames (LFM preamble + K ZP-OFDM symbols + LFM postamble). Even subcarriers
carry known QPSK, odd subcarriers are null: energy on a null subcarrier is
ICI (+ noise). Two receiver conditions:
  A  "realistic": replay with the library's common Doppler drift (phi_hat);
     receiver estimates the common Doppler from pre/postamble timing,
     resamples, then estimates residual CFO by null-subcarrier energy
     minimisation.
  B  "oracle": replay with the common drift removed (phi_hat dropped), i.e.
     perfect common-Doppler compensation; remaining ICI comes only from the
     time variation of the drift-free taps (what replay keeps of
     path-specific Doppler).
A separate noise-only pass (library noise model) gives the per-subcarrier
noise power profile.

Outputs go to results/diagnostic/<channel>/ : metrics.json, profiles.npz, figures.
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.signal as sg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uwa_channels import replay, noisegen, load_channel, load_noise

from uwofdm.paths import channel_file, output_dir

FS = 48_000.0  # passband sampling rate [Hz]
FREEZE = False  # sanity mode: condition B uses a time-invariant (frozen) channel
RNG = np.random.default_rng(1)


# --------------------------------------------------------------------------- #
# Waveform
# --------------------------------------------------------------------------- #
def make_frame(fc, B, N, K, t_zp, t_lfm=0.05, t_gap=0.04):
    df = B / N
    T = 1.0 / df
    Ns = int(round(T * FS))
    Nzp = int(round(t_zp * FS))
    n = np.arange(Ns)
    k_off = np.arange(N) - N // 2  # baseband subcarrier index offsets
    active = (np.arange(N) % 2) == 0
    X = np.zeros((K, N), complex)
    qpsk = (RNG.choice([-1, 1], (K, active.sum())) + 1j * RNG.choice([-1, 1], (K, active.sum()))) / np.sqrt(2)
    X[:, active] = qpsk
    E = np.exp(2j * np.pi * np.outer(k_off, n) / Ns)  # (N, Ns)
    syms = X @ E / np.sqrt(active.sum())  # (K, Ns), unit average power
    blocks = [np.concatenate([s, np.zeros(Nzp)]) for s in syms]

    nl = int(round(t_lfm * FS))
    tl = np.arange(nl) / FS
    lfm = np.exp(1j * np.pi * (B / t_lfm) * tl ** 2 - 1j * np.pi * B * tl)  # -B/2 -> +B/2
    gap = np.zeros(int(round(t_gap * FS)))
    bb = np.concatenate([lfm, gap] + blocks + [gap, lfm])
    idx = {
        "pre": 0,
        "sym0": nl + len(gap),
        "post": len(bb) - nl,
        "Ns": Ns,
        "Nzp": Nzp,
        "Nb": Ns + Nzp,
        "nl": nl,
    }
    pad = np.zeros(int(0.05 * FS))
    bb = np.concatenate([pad, bb, pad])
    for key in ("pre", "sym0", "post"):
        idx[key] += len(pad)
    passband = np.real(bb * np.exp(2j * np.pi * fc * np.arange(len(bb)) / FS)) * np.sqrt(2)
    return passband, bb, X, active, lfm, idx, df


def to_baseband(r, fc, B):
    v = r * np.exp(-2j * np.pi * fc * np.arange(len(r)) / FS) * np.sqrt(2)
    b, a = sg.butter(6, 0.6 * B / (FS / 2))
    return sg.filtfilt(b, a, v)


# --------------------------------------------------------------------------- #
# Receiver
# --------------------------------------------------------------------------- #
def detect(v, lfm, approx, search=0.15):
    lo = max(0, approx - int(search * FS))
    hi = min(len(v), approx + int(search * FS) + len(lfm))
    c = np.abs(np.correlate(v[lo:hi], lfm, "valid"))
    i = int(np.argmax(c))
    # parabolic refinement
    if 0 < i < len(c) - 1:
        y0, y1, y2 = c[i - 1], c[i], c[i + 1]
        frac = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-30)
    else:
        frac = 0.0
    return lo + i + frac


def first_arrival_backoff(v, lfm, peak, span, thr=0.01, margin=0.003):
    """Samples between the strongest arrival and the first arrival above thr*peak (from preamble MF), plus a margin."""
    pk = int(round(peak))
    lo = max(0, pk - span)
    c = np.abs(np.correlate(v[lo : pk + len(lfm) + 1], lfm, "valid"))
    if len(c) == 0:
        return int(margin * FS)
    first = int(np.argmax(c >= thr * c.max()))
    return (len(c) - 1 - first) + int(margin * FS)


def fold(seg, Ns):
    """ZP-OFDM overlap-add for any ZP length: fold the segment modulo Ns."""
    L = int(np.ceil(len(seg) / Ns)) * Ns
    seg = np.pad(seg, (0, L - len(seg)))
    return seg.reshape(-1, Ns).sum(axis=0)


def demod_symbols(v, idx, N, K, offset0, scale=1.0):
    """FFT each ZP-OFDM symbol (overlap-add of ZP tail). Positions scaled by `scale`."""
    Ns, Nzp, Nb = idx["Ns"], idx["Nzp"], idx["Nb"]
    Y = np.zeros((K, N), complex)
    bins = (np.arange(N) - N // 2) % Ns
    for k in range(K):
        s = int(round(offset0 + k * Nb * scale))
        seg = v[s : s + Ns + Nzp]
        if len(seg) < Ns + Nzp:
            seg = np.pad(seg, (0, Ns + Nzp - len(seg)))
        Yf = np.fft.fft(fold(seg, Ns)) / Ns
        Y[k] = Yf[bins]
    return Y


def cfo_search(v, idx, N, K, offset0, active, grid):
    t = np.arange(len(v)) / FS
    best, best_e = 0.0, np.inf
    for eps in grid:
        Y = demod_symbols(v * np.exp(-2j * np.pi * eps * t), idx, N, min(K, 4), offset0)
        e = np.mean(np.abs(Y[:, ~active]) ** 2) / np.mean(np.abs(Y[:, active]) ** 2)
        if e < best_e:
            best, best_e = eps, e
    return best


def receive(r, fc, B, idx, lfm, N, K, active, realistic):
    v = to_baseband(r, fc, B)
    t_pre = detect(v, lfm, idx["pre"])
    back = first_arrival_backoff(v, lfm, t_pre, idx["span"])
    if realistic:
        t_post = detect(v, lfm, idx["post"])
        ratio = (t_post - t_pre) / (idx["post"] - idx["pre"])  # received/transmitted duration
        a_hat = 1.0 / ratio - 1.0
        # resample so that the frame duration matches the nominal one
        tq = t_pre + np.arange(int((len(v) - t_pre) / ratio) - 1) * ratio
        vr = np.interp(tq, np.arange(len(v)), v.real) + 1j * np.interp(tq, np.arange(len(v)), v.imag)
        off = idx["sym0"] - idx["pre"] - back
        coarse = cfo_search(vr, idx, N, K, off, active, np.arange(-30.0, 30.01, 0.5))
        eps = cfo_search(vr, idx, N, K, off, active, coarse + np.arange(-0.5, 0.51, 0.025))
        vr = vr * np.exp(-2j * np.pi * eps * np.arange(len(vr)) / FS)
        Y = demod_symbols(vr, idx, N, K, off)
        return Y, a_hat, eps
    off = t_pre + (idx["sym0"] - idx["pre"]) - back
    return demod_symbols(v, idx, N, K, off), 0.0, 0.0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(channel_path, noise_path, out_dir, N, K, elements, frame_step, B_frac, max_frames):
    ch = load_channel(channel_path)
    p = ch["params"]
    fs_delay, fs_time, fc = (float(np.asarray(p[k]).ravel()[0]) for k in ("fs_delay", "fs_time", "fc"))
    T_ch = np.asarray(ch["h_hat"]["real"]).shape[0] / fs_time
    L = np.asarray(ch["h_hat"]["real"]).shape[2]
    t_zp = 1.5 * L / fs_delay
    B = B_frac * fs_delay
    passband, bb, X, active, lfm, idx, df = make_frame(fc, B, N, K, t_zp)
    idx["span"] = int(L / fs_delay * FS)
    T_frame = len(passband) / FS
    sym_period = idx["Nb"] / FS
    print(f"{os.path.basename(channel_path)}: fc={fc:.0f} Hz, B={B:.0f} Hz (B/fc={B/fc:.2f}), "
          f"N={N}, df={df:.2f} Hz, T_sym={1/df*1e3:.1f} ms, ZP={t_zp*1e3:.1f} ms, frame={T_frame:.2f} s, channel={T_ch:.1f} s")

    # oracle channel: drop the common drift (phi_hat) -> pure time-varying convolution
    ch_oracle = {k: ch[k] for k in ch.keys() if k not in ("phi_hat", "theta_hat", "#refs#")}
    if FREEZE:
        hr = np.asarray(ch["h_hat"]["real"]); hi_ = np.asarray(ch["h_hat"]["imag"])
        i0 = hr.shape[0] // 2
        ch_oracle = dict(ch_oracle)
        ch_oracle["h_hat"] = {"real": np.repeat(hr[i0:i0 + 1], hr.shape[0], 0), "imag": np.repeat(hi_[i0:i0 + 1], hr.shape[0], 0)}
        ch_oracle["params"] = {k: np.asarray(ch["params"][k]) for k in ("fs_delay", "fs_time", "fc")}
        ch_oracle["version"] = np.asarray(ch["version"])
        ch_A = dict(ch_oracle)
        ch_A["phi_hat"] = ch["phi_hat"]  # frozen taps + real common Doppler drift
    else:
        ch_A = ch

    starts_s = np.arange(0.5, T_ch - T_frame - 1.0, frame_step)[:max_frames]
    f_k = fc - B / 2 + np.arange(N) * df
    res = {c: {"P": [], "t": [], "a": [], "eps": []} for c in ("A", "B")}
    t0 = time.time()
    for s_sec in starts_s:
        start = int(s_sec * fs_delay)
        for cond, chan, realistic in (("A", ch_A, True), ("B", ch_oracle, False)):
            out = replay(passband, FS, elements, chan, start=start)
            for m in range(len(elements)):
                Y, a_hat, eps = receive(out[:, m], fc, B, idx, lfm, N, K, active, realistic)
                res[cond]["P"].append(np.abs(Y) ** 2)  # (K, N)
                res[cond]["t"].append(s_sec + idx["sym0"] / FS + np.arange(K) * sym_period)
                res[cond]["a"].append(a_hat)
                res[cond]["eps"].append(eps)
        print(f"  frame @ {s_sec:5.1f} s done ({time.time()-t0:.0f} s elapsed)")

    # noise-only pass
    nz = load_noise(noise_path)
    w = noisegen((int(T_frame * FS) * 4, len(elements)), FS, elements, nz)
    Pn = []
    for m in range(len(elements)):
        v = to_baseband(w[:, m], fc, B)
        nb = idx["Nb"]
        for k in range((len(v) - nb) // nb):
            seg = v[k * nb : k * nb + nb]
            Yf = np.fft.fft(fold(seg, idx["Ns"])) / idx["Ns"]
            Pn.append(np.abs(Yf[(np.arange(N) - N // 2) % idx["Ns"]]) ** 2)
    Pn = np.array(Pn)
    noise_prof = np.median(Pn, axis=0)  # median: robust for alpha-stable noise

    os.makedirs(out_dir, exist_ok=True)
    metrics = {"channel": os.path.basename(channel_path), "fc": fc, "B": B, "B_over_fc": B / fc, "N": N, "df": df,
               "K": K, "elements": list(map(int, elements)), "frames": int(len(starts_s)), "sym_period_s": sym_period}
    npz = {"f_k": f_k, "noise_profile": noise_prof, "active": active}
    odd_all = np.where(~active)[0][1:-1]
    act_all = np.where(active)[0]
    # valid band: largest contiguous run of active subcarriers within 10 dB of the median long-term gain
    Pb = np.concatenate(res["B"]["P"], axis=0)
    g_db = 10 * np.log10(np.mean(Pb[:, act_all], axis=0))
    ok = g_db > np.median(g_db) - 10.0
    runs, cur = [], []
    for i, v in enumerate(ok):
        if v:
            cur.append(i)
        elif cur:
            runs.append(cur); cur = []
    if cur:
        runs.append(cur)
    run_ = max(runs, key=len)
    f_lo, f_hi = f_k[act_all[run_[0]]] + 2 * df * 4, f_k[act_all[run_[-1]]] - 2 * df * 4  # trim 4 active bins each side
    act_idx = act_all[(f_k[act_all] >= f_lo) & (f_k[act_all] <= f_hi)]
    odd = odd_all[(f_k[odd_all] >= f_lo) & (f_k[odd_all] <= f_hi)]
    fmid = 0.5 * (f_lo + f_hi)
    metrics["valid_band_hz"] = [float(f_lo), float(f_hi)]
    metrics["valid_B_over_fmid"] = float((f_hi - f_lo) / fmid)
    for cond in ("A", "B"):
        P = np.concatenate(res[cond]["P"], axis=0)  # (Nsym, N)
        tt = np.concatenate(res[cond]["t"])
        nb_act = 0.5 * (P[:, odd - 1] + P[:, odd + 1])
        ici = P[:, odd] / nb_act
        ici_mean = np.mean(ici, axis=0)
        f_odd = f_k[odd]
        gain = np.mean(P[:, act_idx], axis=0)
        ici_db = 10 * np.log10(ici_mean)
        slope_db_per_khz = np.polyfit((f_odd - fmid) / 1e3, ici_db, 1)[0]
        q = (f_hi - f_lo) / 4
        lo, hi = f_odd < f_lo + q, f_odd > f_hi - q
        hi_lo_db = float(np.mean(ici_db[hi]) - np.mean(ici_db[lo]))
        # bootstrap over frames for the top-minus-bottom ICI difference (uncertainty)
        nfr = len(res[cond]["P"])
        boot = []
        rng = np.random.default_rng(0)
        Pf = [x for x in res[cond]["P"]]
        for _ in range(200):
            pick = rng.integers(0, nfr, nfr)
            Pq = np.concatenate([Pf[i] for i in pick], axis=0)
            ic = 10 * np.log10(np.mean(Pq[:, odd] / (0.5 * (Pq[:, odd - 1] + Pq[:, odd + 1])), axis=0))
            boot.append(np.mean(ic[hi]) - np.mean(ic[lo]))
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
        lg = 10 * np.log10(P[:, act_idx] + 1e-30)
        dev = lg - lg.mean(axis=0, keepdims=True)
        lags = np.array([0.1, 0.25, 0.5, 1, 2, 5, 10, 20])
        rho = []
        dt = tt[None, :] - tt[:, None]
        for lag in lags:
            tol = max(0.5 * sym_period, 0.2 * lag)
            ii, jj = np.where(np.abs(dt - lag) < tol)
            if len(ii) < 10:
                rho.append(float("nan"))
                continue
            rho.append(float(np.corrcoef(dev[ii].ravel(), dev[jj].ravel())[0, 1]))
        # long-term profile stability: 10 s windows, compared on the valid band after removing each window's mean level
        win_edges = np.arange(tt.min(), tt.max() + 1e-9, 10.0)
        profs = []
        for w0 in win_edges[:-1]:
            sel = (tt >= w0) & (tt < w0 + 10.0)
            if sel.sum() >= K:
                pr = 10 * np.log10(np.mean(P[sel][:, act_idx], axis=0))
                profs.append(pr - pr.mean())
        profs = np.array(profs)
        # smooth profiles over ~8 subcarriers (the slow structure), then compare windows
        ker = np.ones(8) / 8
        sm = np.array([np.convolve(x, ker, "same") for x in profs]) if len(profs) else profs
        if len(profs) >= 2:
            cc = np.corrcoef(sm[:, 8:-8])
            stab = float(np.mean(cc[np.triu_indices(len(profs), 1)]))
            across_win_std = float(np.mean(np.std(sm[:, 8:-8], axis=0)))
            across_freq_std = float(np.mean(np.std(sm[:, 8:-8], axis=1)))
        else:
            stab = across_win_std = across_freq_std = float("nan")
        ns = noise_prof[act_idx] / np.mean(noise_prof[act_idx]) * np.mean(gain) / 100.0
        ici_at_act = np.interp(f_k[act_idx], f_odd, ici_mean) * gain
        sinr_db = 10 * np.log10(gain / (ns + ici_at_act))
        a_arr = np.abs(np.array(res[cond]["a"]))
        metrics[cond] = {
            "ici_ratio_db_mean": float(np.mean(ici_db)),
            "ici_slope_db_per_khz": float(slope_db_per_khz),
            "ici_top_minus_bottom_quarter_db": hi_lo_db,
            "ici_top_minus_bottom_95ci_db": ci,
            "gain_slope_db_per_khz": float(np.polyfit((f_k[act_idx] - fmid) / 1e3, 10 * np.log10(gain), 1)[0]),
            "rho_lags_s": lags.tolist(),
            "rho_loggain_dev": rho,
            "longterm_profile_corr_10s_windows_smoothed": stab,
            "longterm_profile_std_across_windows_db": across_win_std,
            "longterm_profile_std_across_freq_db": across_freq_std,
            "n_windows": int(len(profs)),
            "a_hat_abs_median": float(np.median(a_arr)),
            "a_hat_abs_max": float(np.max(a_arr)),
            "eps_hat_abs_max_hz": float(np.max(np.abs(res[cond]["eps"]))),
            "sinr_profile_db_at_20dB": {"min": float(sinr_db.min()), "max": float(sinr_db.max()),
                                        "argmax_freq_hz": float(f_k[act_idx][np.argmax(sinr_db)]),
                                        "top_quarter_minus_best_db": float(np.mean(sinr_db[f_k[act_idx] > f_hi - q]) - sinr_db.max())},
        }
        npz[f"{cond}_ici_db"] = ici_db
        npz[f"{cond}_gain_db"] = 10 * np.log10(gain)
        npz[f"{cond}_sinr_db"] = sinr_db
        npz[f"{cond}_profiles_db"] = profs
        # raw per-symbol data for follow-up checks (block bootstrap, held-out, fixed-band tests)
        npz[f"{cond}_P"] = P.astype(np.float32)
        npz[f"{cond}_t"] = tt
        npz[f"{cond}_frame"] = np.concatenate([np.full(len(x), i) for i, x in enumerate(res[cond]["t"])])
    noise_db = 10 * np.log10(noise_prof / np.mean(noise_prof[act_idx]))
    metrics["noise_slope_db_per_khz"] = float(np.polyfit((f_k[act_idx] - fmid) / 1e3, noise_db[act_idx], 1)[0])
    np.savez(os.path.join(out_dir, "profiles.npz"), **npz)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # figures
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    f_odd = f_k[odd] / 1e3
    f_act = f_k[act_idx] / 1e3
    for cond, lab in (("A", "A: realistic receiver"), ("B", "B: oracle common-Doppler removal")):
        ax[0, 0].plot(f_odd, npz[f"{cond}_ici_db"], label=lab)
        ax[0, 1].plot(f_act, npz[f"{cond}_gain_db"], label=lab)
        ax[1, 0].plot(f_act, npz[f"{cond}_sinr_db"], label=lab)
        ax[1, 1].plot(metrics[cond]["rho_lags_s"], metrics[cond]["rho_loggain_dev"], "o-", label=lab)
    ax[0, 0].set(title="Residual ICI ratio on null subcarriers", xlabel="f [kHz]", ylabel="dB")
    ax[0, 1].set(title="Long-term gain on active subcarriers", xlabel="f [kHz]", ylabel="dB")
    ax[0, 1].plot(f_act, noise_db[act_idx] + np.mean(npz["A_gain_db"]) - 10, "k--", label="noise PSD (shape, offset)")
    ax[1, 0].set(title="Long-term SINR profile (20 dB ref. SNR)", xlabel="f [kHz]", ylabel="dB")
    ax[1, 1].set(title="Temporal corr. of per-subcarrier log-gain deviation", xlabel="lag τ [s]", ylabel="ρ(τ)", xscale="log")
    for a_ in ax.ravel():
        a_.grid(True, alpha=0.3)
        a_.legend(fontsize=7)
    fig.suptitle(f"Replay diagnostic: {metrics['channel']} (B/fc={B/fc:.2f}, N={N})")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "diagnostic.png"), dpi=130)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("channel", help="path to a .mat file, or a short name such as blue_1 (looked up in $UWOFDM_DATA)")
    ap.add_argument("noise", help="path or short name, e.g. blue_noise")
    ap.add_argument("--out", default=None)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--elements", type=int, nargs="+", default=[0])
    ap.add_argument("--frame_step", type=float, default=2.0)
    ap.add_argument("--B_frac", type=float, default=0.8)
    ap.add_argument("--max_frames", type=int, default=1000)
    ap.add_argument("--freeze", action="store_true", help="sanity test: B uses a frozen channel")
    ap.add_argument("--fs", type=float, default=48000.0, help="passband sampling rate (must exceed 2*(fc+B/2))")
    a = ap.parse_args()
    FREEZE = a.freeze
    FS = a.fs
    resolve = lambda x: x if os.path.exists(x) else str(channel_file(x))
    a.channel, a.noise = resolve(a.channel), resolve(a.noise)
    out = a.out or str(output_dir("diagnostic") / os.path.splitext(os.path.basename(a.channel))[0])
    run(a.channel, a.noise, out, a.N, a.K, np.array(a.elements), a.frame_step, a.B_frac, a.max_frames)
