"""Evidence for the group (D-B1/D-B2/D-B8): does impulsive noise make the per-packet SNR
estimate gamma-hat heavy-tailed, in linear or in dB units?

Model: per packet, N known pilot symbols s_i (|s_i| = 1) and r_i = s_i + n_i. Pilot EVM:
  Pn_hat = mean |r_i - s_i|^2   (noise-power estimate, linear)
  g_lin  = 1 / Pn_hat           (gamma-hat, linear, since |s|^2 = 1)
  g_db   = 10 log10(g_lin)
Noise is scaled so that the median per-symbol |n|^2 equals the Gaussian median at the
target SNR (robust scaling; SaS has no variance), i.e. comparable "typical" SNR for all models.
Receiver has no impulse blanking/clipping (worst case for tails; a real receiver may add it).

Tail diagnostic: mean excess e(u) = E[X - u | X > u] at thresholds u = empirical quantiles
0.9, 0.99, 0.999 of X. Known cases (checked below before use, project.MD §7.1):
exponential -> e(u) constant; Pareto(a) -> e(u) = u/(a-1), grows linearly; Gaussian -> e(u)
shrinks like 1/u; a variable with no mean (Pareto a <= 1) -> e(u) grows and is unstable.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.stats import levy_stable

N_SYM = 256              # pilot symbols per packet
SNR_DB = 10.0            # target "typical" SNR
N_PKT = 20_000           # packets per synthetic model
QS = (0.9, 0.99, 0.999)
SKIP_S = 5.0            # skip at the start of each recording file
from uwsb.runtime import data_dir  # noqa: E402

D = str(data_dir() / "noise_census")
rng = np.random.default_rng(20260930)


def mean_excess(x, qs=QS):
    x = np.asarray(x, float)
    out = []
    for q in qs:
        u = np.quantile(x, q)
        out.append((u, float(np.mean(x[x > u] - u))))
    return out


def known_case_checks():
    r = np.random.default_rng(0)
    e = mean_excess(r.exponential(1.0, 2_000_000))
    assert all(abs(v - 1.0) < 0.1 for _, v in e), e                      # constant = 1
    a = 1.7
    p = (1.0 - r.random(2_000_000)) ** (-1.0 / a)                           # Pareto(1.7), xm = 1
    for u, v in mean_excess(p)[:2]:
        assert abs(v / (u / (a - 1.0)) - 1.0) < 0.2, (u, v)                # e(u) = u/(a-1)
    g = mean_excess(r.standard_normal(2_000_000))
    assert g[0][1] > g[1][1] > g[2][1], g                                   # shrinking
    print("known-case checks passed: exponential flat, Pareto(1.7) linear, Gaussian shrinking")


def robust_scale(n):
    """Scale complex noise so median |n|^2 matches CN(0, s2) at SNR_DB (median = s2 ln 2)."""
    s2 = 10 ** (-SNR_DB / 10)
    return n * np.sqrt(s2 * np.log(2) / np.median(np.abs(n) ** 2))


def gaussian(n):
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


def sas_subgaussian(n, alpha):
    """Isotropic complex SaS (sub-Gaussian): sqrt(A) * CN(0,1), A ~ S_{alpha/2}(beta=1)."""
    a = levy_stable.rvs(alpha / 2, 1.0, loc=0.0,
                        scale=np.cos(np.pi * alpha / 4) ** (2 / alpha), size=n, random_state=rng)
    return np.sqrt(a) * gaussian(n)


def class_a(n, A=0.1, Gamma=0.01):
    """Middleton Class A: per sample m ~ Poisson(A), variance ∝ (m/A + Gamma)/(1 + Gamma)."""
    m = rng.poisson(A, n)
    var = (m / A + Gamma) / (1 + Gamma)
    return np.sqrt(var) * gaussian(n)


def recording_baseband(path, band, minutes):
    """Band-pass, mix to baseband, low-pass, decimate to symbol rate = bandwidth (fs/10)."""
    lo, hi = band
    fc = (lo + hi) / 2
    out = []
    with sf.SoundFile(path) as f:
        fs = f.samplerate
        dec = 10
        sos_bp = signal.butter(8, band, btype="bandpass", fs=fs, output="sos")
        sos_lp = signal.butter(8, (hi - lo) / 2, fs=fs, output="sos")
        zi_bp = np.zeros((sos_bp.shape[0], 2))
        zi_lr = np.zeros((sos_lp.shape[0], 2))
        zi_li = np.zeros((sos_lp.shape[0], 2))
        t0 = 0
        for blk in f.blocks(blocksize=int(60 * fs), dtype="float64", frames=int(minutes * 60 * fs)):
            y, zi_bp = signal.sosfilt(sos_bp, blk, zi=zi_bp)
            t = (t0 + np.arange(blk.size)) / fs
            t0 += blk.size
            bb_r, zi_lr = signal.sosfilt(sos_lp, y * np.cos(2 * np.pi * fc * t), zi=zi_lr)
            bb_i, zi_li = signal.sosfilt(sos_lp, -y * np.sin(2 * np.pi * fc * t), zi=zi_li)
            out.append((bb_r + 1j * bb_i)[::dec])
    # drop filter transient and the recorder start-up segment: HI01 files are ~27 dB
    # quieter for the first ~2.8 s (checked 2026-09-30); skip 5 s of every recording
    z = np.concatenate(out)[int(SKIP_S * fs / dec):]
    return z, fs / dec


def evaluate(name, noise):
    n_pkt = noise.size // N_SYM
    nz = robust_scale(noise[: n_pkt * N_SYM]).reshape(n_pkt, N_SYM)
    pn = np.mean(np.abs(nz) ** 2, axis=1)               # |r - s|^2 with known pilots
    g_lin = 1.0 / pn
    g_db = 10 * np.log10(g_lin)
    d_db = np.median(g_db) - g_db                       # impulse side of gamma-hat in dB (drops)
    g_lin_drop = np.median(g_lin) - g_lin               # impulse side in linear (bounded by median)
    kurt = lambda x: float(np.mean((x - x.mean()) ** 4) / np.var(x) ** 2 - 3)
    fmt = lambda e: " ".join(f"u={u:.3g}:e={v:.3g}" for u, v in e)
    print(f"== {name}: {n_pkt} packets of {N_SYM} symbols")
    print(f"   gamma_dB: median {np.median(g_db):.2f} dB, std {g_db.std():.2f} dB, excess kurtosis {kurt(g_db):.1f}")
    print(f"   q0.1% / q1% below median: {np.quantile(d_db, 0.999):.1f} / {np.quantile(d_db, 0.99):.1f} dB")
    print(f"   mean excess, dB drop  (impulse side):  {fmt(mean_excess(d_db))}")
    print(f"   mean excess, Pn_hat/median (EVM):      {fmt(mean_excess(pn / np.median(pn)))}")
    print(f"   gamma_lin impulse side is bounded: max drop / median = {g_lin_drop.max() / np.median(g_lin):.3f}"
          f"; excess kurtosis gamma_lin {kurt(g_lin):.1f}")
    return dict(name=name, n_pkt=n_pkt, g_db=g_db, pn=pn)


if __name__ == "__main__":
    known_case_checks()
    n = N_PKT * N_SYM
    evaluate("Gaussian", gaussian(n))
    evaluate("SaS alpha=1.7 (library Red model, i.i.d.)", sas_subgaussian(n, 1.7))
    evaluate("Middleton Class A (A=0.1, Gamma=0.01)", class_a(n))
    z, rs = recording_baseband(f"{D}/S1_fk01/SanctSound_FK01_01_671359016_20181218T214011Z.flac",
                               (10500.0, 15500.0), minutes=60)
    evaluate(f"SanctSound FK01, blue band, 60 min, {rs:.0f} sym/s", z)
    zs = []
    for stamp in ("000002Z", "003002Z", "010002Z", "013002Z"):
        z, rs = recording_baseband(
            f"{D}/S2_hi01/SanctSound_HI01_01_671129638_20181115T{stamp}.flac", (20000.0, 30000.0), minutes=15)
        zs.append(z)
    evaluate(f"SanctSound HI01, red band, 4 x 15 min, {rs:.0f} sym/s", np.concatenate(zs))
