"""
Inherited from the earlier subcarrier-selection project: receiver-conditioned impairment model theta, band/MCS
action family and expected goodput. Only the MCS table and the EESM + logistic PER mapping (goodput_mcs) carry
over to the Doppler project; the plan sections, followup_checks.py, gate notes and certificate module cited below
belong to that project and are not in this repository.

theta (d = 7, all in dB), carriers on x in [-1, 1] across the band:
  a0..a3  long-term channel power relative to the mean noise level: gdB(x) = sum_p a_p P_p(x)  (a0 = mean SNR)
  b1      noise tilt: ndB(x) = b1 x  (noise level absorbed in a0)
  c0, c1  residual-leakage level: ldB(x) = c0 + c1 x  (total leakage fraction at carrier x)
Leakage kernel (declared family, to be checked against the sparse-probe measurement): fraction w_d of the
total leakage lambda(x) comes from neighbour distance d, w_d ~ 1/d^2 for 1 <= |d| <= D, sum over +-d = 1.
Action a = (contiguous band on a block grid, MCS); power convention "total" (fixed total power, p = M/|S|)
or "percarrier" (p = 1). Interference is occupancy dependent: only active carriers of the band leak.
Utility U(a, theta) = expected goodput (info bits per OFDM symbol) from the long-term SINR via EESM + logistic PER.
Fast fading: without `fading`, the long-term profile is plugged into the EESM directly (no fading). With
`fading=(PG, PL)`, U is the average over R measured fading patterns (dB deviations of the per-symbol gain and leakage
profiles from their window means, taken from calibration data at the receiver): a fading-marginalised expected
goodput. On replay this cuts the decision-level mismatch against per-symbol goodput from 55-160 % to 10-17 % of the
best goodput (gate notes, 5 Oct). The patterns are fixed (common random numbers), so U stays deterministic and
monotone in every profile value, and all interval and gradient bounds remain rigorous for this U.
"""
import numpy as np
from numpy.polynomial import legendre

# (info bits per carrier per symbol, AWGN 10%-PER threshold dB, EESM beta linear), as in followup_checks.py
MCS = np.array([(0.5, 1.0, 1.5), (1.0, 3.0, 1.6), (1.5, 6.0, 2.0), (2.0, 9.0, 5.0), (3.0, 12.5, 7.0)])
NAMES = ["a0", "a1", "a2", "a3", "b1", "c0", "c1"]
# quantisation ranges (dB) of each parameter: the message encodes theta_j uniformly on [lo, hi]
RANGES = np.array([(-5, 35), (-8, 8), (-8, 8), (-8, 8), (-8, 8), (-45, -5), (-10, 10)], float)
D_THETA = len(NAMES)


class Model:
    def __init__(self, M=160, n_blocks=16, min_blocks=3, D=8, power="total", fading=None):
        self.M, self.D, self.power = M, D, power
        # fading patterns (R, M) in dB for gain and leakage; None: no fading (R = 1, zero deviation)
        if fading is None:
            self._FG, self._FL = np.ones((1, M)), np.ones((1, M))
        else:
            PG, PL = (np.atleast_2d(np.asarray(f, float)) for f in fading)
            assert PG.shape == PL.shape and PG.shape[1] == M
            self._FG, self._FL = 10 ** (PG / 10), 10 ** (PL / 10)
        self.n_fading = len(self._FG)
        self.x = np.linspace(-1.0, 1.0, M)
        self.Pg = np.vstack([legendre.legval(self.x, np.eye(4)[p]) for p in range(4)])  # (4, M)
        self.Pn = self.x[None, :]
        self.Pc = np.vstack([np.ones(M), self.x])
        d = np.arange(1, D + 1)
        wd = 1.0 / d ** 2
        wd /= 2 * wd.sum()
        self.w = np.concatenate([wd[::-1], [0.0], wd])  # length 2D+1, zero at the centre
        self.w_odd = 2 * wd[(d % 2) == 1].sum()  # weight seen by a null carrier under the alternating pattern
        edges = np.linspace(0, M, n_blocks + 1).round().astype(int)
        self.bands = np.array([(edges[i], edges[j]) for i in range(n_blocks) for j in range(i + min_blocks, n_blocks + 1)])
        self.n_mcs = len(MCS)
        self.nA = len(self.bands) * self.n_mcs
        # vectorised evaluation: band masks, per-band power and the leakage matrix W[k, m] = w_{m-k}
        k = np.arange(M)
        self._mask = (k[None, :] >= self.bands[:, :1]) & (k[None, :] < self.bands[:, 1:])
        self._nk = self._mask.sum(axis=1)
        self._p = self.M / self._nk if self.power == "total" else np.ones(len(self.bands))
        dk = k[None, :] - k[:, None]
        self._W = np.where(np.abs(dk) <= D, self.w[np.clip(dk + D, 0, 2 * D)], 0.0)
        # gradient enclosure: range of each gain basis P_j over the carriers that leak into carrier k within band b
        nbr = (np.abs(dk) <= D) & (dk != 0)
        allowed = nbr[None, :, :] & self._mask[:, None, :]  # (bands, k, m)
        self._nbr_min = np.empty((len(self.bands), M, 4))
        self._nbr_max = np.empty((len(self.bands), M, 4))
        for j in range(4):
            Pj = self.Pg[j][None, None, :]
            self._nbr_min[:, :, j] = np.where(allowed, Pj, np.inf).min(axis=2)
            self._nbr_max[:, :, j] = np.where(allowed, Pj, -np.inf).max(axis=2)
        outside = ~allowed.any(axis=2)  # carriers outside the band: no leakage, value unused
        self._nbr_min[outside], self._nbr_max[outside] = 0.0, 0.0

    # ---- action indexing: a = band_index * len(MCS) + mcs_index
    def action(self, a):
        return tuple(self.bands[a // len(MCS)]), int(a % len(MCS))

    # ---- profiles (dB) from theta
    def profiles_db(self, th):
        th = np.asarray(th, float)
        return th[:4] @ self.Pg, th[4] * self.x, th[5] + th[6] * self.x

    def profile_bounds_db(self, lo, hi, mis=(0.0, 0.0, 0.0)):
        """Exact per-carrier interval of each profile over the box [lo, hi], widened by the declared mismatch mis (dB)."""
        c, h = 0.5 * (lo + hi), 0.5 * (hi - lo)
        g, n, l = self.profiles_db(c)
        hg = h[:4] @ np.abs(self.Pg) + mis[0]
        hn = h[4] * np.abs(self.x) + mis[1]
        hl = h[5] + h[6] * np.abs(self.x) + mis[2]
        return (g - hg, g + hg), (n - hn, n + hn), (l - hl, l + hl)

    # ---- SINR and goodput
    def _interf(self, gb):
        full = np.convolve(gb, self.w)
        return full[self.D:self.D + len(gb)]

    def sinr_band(self, g_num, g_int, n, lam, band):
        """Per-carrier SINR on the band; signal power from g_num, leakage computed from g_int (lets bounds decouple)."""
        i, j = band
        p = self.M / (j - i) if self.power == "total" else 1.0
        return p * g_num[i:j] / (n[i:j] + p * lam[i:j] * self._interf(g_int[i:j]))

    @staticmethod
    def goodput_mcs(gam):
        """Goodput of every MCS for one band's SINR vector (EESM + logistic PER); monotone increasing in every gam_k."""
        r, thr, beta = MCS[:, 0], MCS[:, 1], MCS[:, 2]
        e = np.mean(np.exp(-gam[None, :] / beta[:, None]), axis=1)
        geff_db = 10 * np.log10(-beta * np.log(np.maximum(e, 1e-300)))
        per = 1.0 / (1.0 + np.exp(np.clip(2.0 * (geff_db - thr), -60, 60)))
        return len(gam) * r * (1 - per)

    def _utilities_nofading(self, g_num, g_int, n, lam):
        """All actions at once for one realisation (vectorised over bands; same values as sinr_band + goodput_mcs)."""
        p = self._p[:, None]
        interf = (self._mask * g_int[None, :]) @ self._W  # only the band's own carriers leak
        gam = p * g_num[None, :] / (n[None, :] + p * lam[None, :] * interf)  # (bands, M); used only on the mask
        r, thr, beta = MCS[:, 0], MCS[:, 1], MCS[:, 2]
        e = np.einsum("bk,mbk->bm", self._mask, np.exp(-gam[None, :, :] / beta[:, None, None])) / self._nk[:, None]
        geff_db = 10 * np.log10(-beta[None, :] * np.log(np.maximum(e, 1e-300)))
        per = 1.0 / (1.0 + np.exp(np.clip(2.0 * (geff_db - thr[None, :]), -60, 60)))
        return (self._nk[:, None] * r[None, :] * (1 - per)).ravel()

    def utilities_lin(self, g_num, g_int, n, lam):
        """Expected goodput of every action: average over the fading patterns (one pattern without fading)."""
        if self.n_fading == 1 and np.all(self._FG == 1) and np.all(self._FL == 1):
            return self._utilities_nofading(g_num, g_int, n, lam)
        return np.mean([self._utilities_nofading(g_num * fg, g_int * fg, n, lam * fl)
                        for fg, fl in zip(self._FG, self._FL)], axis=0)

    def utilities_lin_loop(self, g_num, g_int, n, lam):
        """Reference implementation (one band and one fading pattern at a time); kept for the equivalence test."""
        out = np.zeros(self.nA)
        for fg, fl in zip(self._FG, self._FL):
            for b, band in enumerate(self.bands):
                out[b * len(MCS):(b + 1) * len(MCS)] += self.goodput_mcs(
                    self.sinr_band(g_num * fg, g_int * fg, n, lam * fl, band))
        return out / self.n_fading

    def utilities_db(self, gdb, ndb, ldb):
        g, n, l = 10 ** (gdb / 10), 10 ** (ndb / 10), 10 ** (ldb / 10)
        return self.utilities_lin(g, g, n, l)

    def U(self, th):
        """Point utility of all actions under theta."""
        return self.utilities_db(*self.profiles_db(th))

    def U_bounds(self, lo, hi, mis=(0.0, 0.0, 0.0)):
        """Rigorous lower/upper utility of every action over the box (SINR is monotone in each profile value:
        increasing in the carrier's own gain, decreasing in noise, leakage level and the other carriers' gains)."""
        (gl, gh), (nl, nh), (ll, lh) = self.profile_bounds_db(lo, hi, mis)
        L = lambda x: 10 ** (x / 10)
        return (self.utilities_lin(L(gl), L(gh), L(nh), L(lh)),
                self.utilities_lin(L(gh), L(gl), L(nl), L(ll)))

    def U_grad_bounds(self, lo, hi):
        """Rigorous enclosure of U and dU/dtheta over the box (see _grad_bounds_one); with fading, the average of the
        per-pattern enclosures (each pattern shifts the gain and leakage profiles by a fixed dB vector)."""
        if self.n_fading == 1 and np.all(self._FG == 1) and np.all(self._FL == 1):
            return self._grad_bounds_one(lo, hi, 0.0, 0.0)
        out = [self._grad_bounds_one(lo, hi, 10 * np.log10(fg), 10 * np.log10(fl)) for fg, fl in zip(self._FG, self._FL)]
        return tuple(np.mean([o[i] for o in out], axis=0) for i in range(4))

    def _grad_bounds_one(self, lo, hi, sg, sl):
        """Rigorous enclosure of U and of dU/dtheta over the box [lo, hi] (no mismatch slack).

        Returns Ul, Uh (nA,) and dlo, dhi (nA, D_THETA) with Ul <= U(a, theta) <= Uh and dlo <= dU(a, theta)/dtheta
        <= dhi for every theta in the box. Built from log-derivatives, which are exact constants for the gain, noise
        and leakage profiles (linear in theta in dB); the interference log-derivative is a convex combination of the
        gain log-derivatives of the leaking carriers, so it lies in the range of the basis over those carriers. Every
        other factor is enclosed from the monotone value bounds (as in U_bounds). Used by the mean-value form of the
        certificate (certificate.bb_certificate), whose error is second order in the box width."""
        c = np.log(10.0) / 10.0
        nb, M = len(self.bands), self.M
        (gl, gh), (nl, nh), (ll, lh) = self.profile_bounds_db(lo, hi)
        gl, gh, ll, lh = gl + sg, gh + sg, ll + sl, lh + sl  # fixed fading pattern (dB); derivatives unchanged
        L = lambda x: 10 ** (x / 10)
        Gl, Gh, Nl, Nh, Ll, Lh = L(gl), L(gh), L(nl), L(nh), L(ll), L(lh)
        p = self._p[:, None]
        Il = (self._mask * Gl[None, :]) @ self._W
        Ih = (self._mask * Gh[None, :]) @ self._W
        den_l, den_h = Nl[None, :] + p * Ll[None, :] * Il, Nh[None, :] + p * Lh[None, :] * Ih
        gam_l, gam_h = p * Gl[None, :] / den_h, p * Gh[None, :] / den_l
        al_l = Nl[None, :] / (Nl[None, :] + p * Lh[None, :] * Ih)  # alpha = noise share of the denominator
        al_h = Nh[None, :] / (Nh[None, :] + p * Ll[None, :] * Il)
        # exact log-derivatives (M, 7) and interval ones (nb, M, 7)
        dG = np.zeros((M, 7)); dG[:, :4] = c * self.Pg.T
        dN = np.zeros((M, 7)); dN[:, 4] = c * self.x
        dL = np.zeros((M, 7)); dL[:, 5] = c; dL[:, 6] = c * self.x
        dI_l = np.zeros((nb, M, 7)); dI_h = np.zeros((nb, M, 7))
        dI_l[:, :, :4], dI_h[:, :, :4] = c * self._nbr_min, c * self._nbr_max
        B_l, B_h = dL[None] + dI_l, dL[None] + dI_h
        A = dN[None]
        cand = [a[:, :, None] * A + (1 - a[:, :, None]) * b for a in (al_l, al_h) for b in (B_l, B_h)]
        dden_l, dden_h = np.minimum.reduce(cand), np.maximum.reduce(cand)
        dlg_l, dlg_h = dG[None] - dden_h, dG[None] - dden_l  # d ln gamma
        r, thr, beta = MCS[:, 0], MCS[:, 1], MCS[:, 2]
        bt = beta[:, None, None]
        lu_l, lu_h = -gam_h[None] / bt, -gam_l[None] / bt  # (5, nb, M), lu = -gamma / beta
        # d lu = -(gamma / beta) d ln gamma: positive factor times a signed interval, negated
        fl, fh = (gam_l[None] / bt)[..., None], (gam_h[None] / bt)[..., None]
        pl = np.where(dlg_l[None] >= 0, fl * dlg_l[None], fh * dlg_l[None])
        ph = np.where(dlg_h[None] >= 0, fh * dlg_h[None], fl * dlg_h[None])
        dlu_l, dlu_h = -ph, -pl  # (5, nb, M, 7)
        msk = self._mask[None]
        neg = np.where(msk, 0.0, -np.inf)
        lse = lambda z: np.logaddexp.reduce(z + neg, axis=-1)  # log-sum-exp over the band's carriers
        LSE_l, LSE_h = lse(lu_l), lse(lu_h)  # (5, nb)
        s_l = np.where(msk, np.exp(lu_l - LSE_h[..., None]), 0.0)[..., None]  # softmax weights, enclosure
        s_h = np.where(msk, np.minimum(1.0, np.exp(lu_h - LSE_l[..., None])), 0.0)[..., None]
        ql = np.where(dlu_l >= 0, s_l * dlu_l, s_h * dlu_l)
        qh = np.where(dlu_h >= 0, s_h * dlu_h, s_l * dlu_h)
        m4 = msk[..., None]
        dle_l = np.maximum(ql.sum(axis=2), np.where(m4, dlu_l, np.inf).min(axis=2))  # d ln e: weights sum to 1
        dle_h = np.minimum(qh.sum(axis=2), np.where(m4, dlu_h, -np.inf).max(axis=2))  # (5, nb, 7)
        lnN = np.log(self._nk)[None, :]
        t_l = np.maximum(-(LSE_h - lnN), 1e-300)  # t = -ln e > 0
        t_h = np.maximum(-(LSE_l - lnN), 1e-300)
        k10 = 10.0 / np.log(10.0)
        # d geff_dB = -k10 * d ln e / t
        il, ih = (1.0 / t_h)[..., None], (1.0 / t_l)[..., None]
        ul = np.where(dle_l >= 0, ih * dle_l, il * dle_l)
        uh = np.where(dle_h >= 0, il * dle_h, ih * dle_h)
        dge_l, dge_h = -k10 * uh, -k10 * ul
        ge_l = k10 * (np.log(beta)[:, None] + np.log(t_l))
        ge_h = k10 * (np.log(beta)[:, None] + np.log(t_h))
        z_l, z_h = 2 * (ge_l - thr[:, None]), 2 * (ge_h - thr[:, None])
        sp = lambda z: 1.0 / (2.0 + np.exp(np.clip(z, -60, 60)) + np.exp(np.clip(-z, -60, 60)))  # sigma (1 - sigma)
        sp_h = np.where((z_l <= 0) & (z_h >= 0), 0.25, sp(np.minimum(np.abs(z_l), np.abs(z_h))))
        sp_l = sp(np.maximum(np.abs(z_l), np.abs(z_h)))
        f_l = (self._nk[None, :] * r[:, None] * 2 * sp_l)[..., None]
        f_h = (self._nk[None, :] * r[:, None] * 2 * sp_h)[..., None]
        dU_l = np.where(dge_l >= 0, f_l * dge_l, f_h * dge_l)
        dU_h = np.where(dge_h >= 0, f_h * dge_h, f_l * dge_h)
        Ul = self._utilities_nofading(Gl, Gh, Nh, Lh)
        Uh = self._utilities_nofading(Gh, Gl, Nl, Ll)
        # (5, nb, 7) -> action order a = band * n_mcs + mcs
        return Ul, Uh, dU_l.transpose(1, 0, 2).reshape(-1, 7), dU_h.transpose(1, 0, 2).reshape(-1, 7)
