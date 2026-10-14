"""Multi-user uplink (multiuser.py): disjoint subbands, subband chirps, and per-user branches that recover two users
with opposite Doppler at high SNR."""
import numpy as np

from uwofdm import multiuser as mu, waveform


def test_user_frames():
    o = waveform.OFDM()
    frames = mu.contiguous(o, 4, 8)
    used = np.array([F.used for F in frames])
    assert used.sum(0).max() == 1  # disjoint
    for F in frames:
        assert not np.any(used[:, list(F.quiet)])  # guards are empty
        c = F.chirp_up()
        S = np.abs(np.fft.fft(c, 8 * len(c))) ** 2
        f = np.fft.fftfreq(len(S), 1 / o.fs)
        lo, hi = F.band
        f_lo, f_hi = (lo - o.n_sc // 2 - 1) * o.df, (hi - o.n_sc // 2) * o.df
        assert S[(f >= f_lo) & (f <= f_hi)].sum() > 0.9 * S.sum()


def test_two_users_opposite_doppler():
    o = waveform.OFDM()
    rng = np.random.default_rng(3)
    frames = mu.contiguous(o, 2, 16)
    lead, n = 1000, frames[0].length + 9000
    pk, xs, eps, truths = [], [], [], []
    for k, (F, e) in enumerate(zip(frames, (8e-4, -8e-4))):
        info, X, s = mu.user_packet(F, rng)
        x = np.zeros(n, complex)
        x[lead:lead + len(s)] = s
        pk.append((info, X))
        xs.append(x)
        eps.append(np.full(n, e))
        truths.append(np.full(F.length + 4000, e))
    y = mu.add_noise(mu.uplink(xs, eps, o.fs, o.fc), o, 35.0, rng)
    Ys, _ = mu.receive_users(y, frames, truths, truths, [(0, lead + 1500)] * 2,
                             which=("branch_signal", "branch_genie"))
    for r in ("branch_signal", "branch_genie"):
        for k, F in enumerate(frames):
            s_db, err = mu.score(Ys[r][k], pk[k][1], pk[k][0], F)
            assert not err and s_db > 15
