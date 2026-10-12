# G1 check: a design rule for Doppler compensation inside the packet (9 Oct 2026)

**Gap under test (G1):** compensation of a time-varying Doppler inside a packet exists in many forms; a design rule
linking the estimate accuracy, the trajectory model (constant, linear, quadratic, piecewise) and the packet length to
the residual interference is not established.

## Verdict
**G1 is only partly open. It cannot be a headline claim.**
- **Already analysed:**
  - The residual from a *constant* Doppler-scale error, including guidelines for the scale resolution (Mason et al.
    2008).
  - The exact interference pattern of Doppler scaling in multicarrier signals, with an upper bound (Stojanovic,
    Djurovic, Djukanovic 2012; Berger et al. 2010 for the per-path pattern).
  - SINR in a geometry-based channel (Ha, Nguyen, Pätzold 2015).
  - In radio: exact interference from frequency offset, sampling offset and channel variation (García, Oberli 2009);
    the effect of an uncompensated Doppler rate on synchronisation (Salberg, Swami 2005); closed-form interference
    and SINR from residual Doppler in LEO links (Tanash et al. 2026).
- **Thresholds, not rules:**
  - Doppler rate below about 10⁻⁵ makes the residual interference negligible (Stojanovic 2006).
  - Above a scale spread of about 10⁻³, one resampling factor is not enough (Yerramalli, Mitra 2011).
  - Acceleration up to about ±1 m/s² over a packet is tolerated by Abdelkareem et al. (2015/2016).
- **Not found:** a rule that takes the errors of each estimated coefficient, the order of the trajectory model, the
  packet length *and the motion statistics* (real spectra) and returns the residual interference, so as to pick the
  model order for a given motion and packet. Compensators are compared by BER in chosen scenarios.

## Consequences for the plan
1. **Reword G1** as: "the residual from a constant-scale error and the interference from Doppler scaling or a linear
   frequency drift are analysed; no rule combines estimate errors, trajectory model order and packet length with real
   motion statistics."
2. **Keep C1 and R1 as a supporting result**, not the main claim (consistent with the README's risks). The main claim
   stays on placement and splitting (G2, G3, R2).
3. **One analysis for R1 and R2.** Both reduce to one question: how well v(t) is represented from limited
   information, given its spectrum S_v(f).
   - R2 asks it *across time*: how well an estimate H seconds old represents v(t).
   - R1 asks it *within the packet*: how well a polynomial of order N fitted over the packet duration T represents
     v(t).

   Both errors are integrals of S_v(f) against a kernel (1 − cos 2πfH for staleness; the residual of the order-N fit
   over T for the model). This makes the two results one framework, evaluated on the measured buoy spectra
   (*my proposal, to be derived*).
4. **Build on the existing derivations** (Mason 2008; García–Oberli 2009 for the decomposition with a time-varying
   offset; Tanash 2026 for the closed form) rather than re-deriving the interference of a constant or linear error.

## Full texts read (9 Oct 2026)
- **Mason et al. 2008** (OCEANS 2008 version, author copy at users.ece.cmu.edu/~crberger/Oceans2008.pdf), Section V.
  - Residual: a *constant* scale mismatch a − b after resampling, in ZP-OFDM (fc 27 kHz, B 12 kHz, K 1024).
  - Bound: an SNR upper bound per subcarrier.
  - Result: to keep self-interference 20 dB below the signal, Δv < 0.06 m/s without fine frequency compensation,
    Δv < 0.2 m/s with it; correlator spacing of 0.4 m/s suggested.
  - Not covered: time variation inside the packet, trajectory models, motion statistics.
- **Tanash et al. 2026** (arXiv 2603.20447, HTML), Eqs. 4–8.
  - Residual: a constant frequency offset per symbol, with useful power sinc²(χT) and interference 1 − sinc²(χT),
    for the central subcarrier and many subcarriers; LoS only.
  - Not covered: Doppler rate, estimation error.
  - For small offsets this is (πχT)²/3, the form used in `code/src/uwofdm/theory.py`.
- Neither changes the verdict above. Stojanovic et al. 2012 and Ha et al. 2015 have no open copy found; still to read
  through the library's access.
- Closed form found while writing `theory.py` (to verify against the literature): the residual of the best order-N
  polynomial fit over T to a component at frequency f is K_N = 1 − Σ_{k≤N} (2k+1) j_k(πfT)² (spherical Bessel
  functions); checked numerically in `code/tests/test_theory_compensate.py`.

## What was searched
- Local library: abstracts of all 34,745 records (residual or time-varying Doppler, acceleration, Doppler rate with
  ICI / SINR / analysis) and full text of 1,406 PDFs (residual Doppler, acceleration or Doppler rate near ICI or SINR).
- Web: 3 searches (closed-form ICI with residual scale and rate in UWA OFDM; ICI from time-varying frequency offset in
  high-speed-train and LEO OFDM; model order vs packet duration in UWA compensation). OpenAlex for the references.
- Not done: Google Scholar (blocked by a CAPTCHA earlier on this network); full text of Mason 2008, Stojanovic et al.
  2012, Ha et al. 2015, Tanash et al. 2026. These should be read before writing R1.

## References found
| Work | What it gives |
|---|---|
| S. Mason, C. Berger, S. Zhou, P. Willett, "Detection, synchronization, and Doppler scale estimation with multicarrier waveforms in UWA communication", IEEE JSAC 26(9), 2008 | Impact of Doppler-scale estimation accuracy on data performance; guidelines for the scale resolution (constant scale) |
| Stojanovic, Djurovic, Djukanovic, "The effects of Doppler scaling in underwater acoustic OFDM communication", 2012 | Exact interference power and upper bound for Doppler scaling (venue to confirm) |
| C. R. Berger et al., IEEE Trans. Signal Process. 2010 (sparse channel estimation) | Exact ICI pattern per path with its own Doppler scale |
| D. V. Ha, N. V. Duc, M. Pätzold, "SINR analysis of OFDM systems using a geometry-based UWA channel model", PIMRC 2015, [doi:10.1109/pimrc.2015.7343385](https://doi.org/10.1109/pimrc.2015.7343385) | SINR vs maximum Doppler, bandwidth, power |
| M. Stojanovic, "Low complexity OFDM detector for underwater acoustic channels", OCEANS 2006 | Residual Doppler treated as noise; Doppler rate below 10⁻⁵ makes ICI negligible |
| S. Yerramalli, U. Mitra, "Optimal resampling of OFDM signals for multiscale–multilag UWA channels", IEEE JOE 2011, [doi:10.1109/joe.2010.2093752](https://doi.org/10.1109/joe.2010.2093752) | One resampling factor inadequate above a scale spread of about 10⁻³ |
| A. E. Abdelkareem, B. S. Sharif, C. C. Tsimenidis, Ad Hoc Netw. 2016; J. Electr. Comput. Eng. 2012 (linear multiscale Doppler) | Linear Doppler variation within the symbol; tolerance about ±1 m/s² |
| M. García, C. Oberli, "Intercarrier interference in OFDM: a general model for transmissions in mobile environments with imperfect synchronization", EURASIP JWCN 2009, [doi:10.1155/2009/786040](https://doi.org/10.1155/2009/786040) | Exact ICI from frequency offset, sampling offset and channel variation, decomposed |
| A.-B. Salberg, A. Swami, "Doppler and frequency-offset synchronization in wideband OFDM", IEEE TWC 2005, [doi:10.1109/twc.2005.858337](https://doi.org/10.1109/twc.2005.858337) | Uncompensated Doppler rate degrades effective SNR (radio) |
| I. M. Tanash, N. González-Prelcic, R. Wichman, "Performance analysis of LEO-terrestrial systems in presence of Doppler effect", IEEE WCL 2026, [doi:10.1109/lwc.2026.3711220](https://doi.org/10.1109/lwc.2026.3711220) | Closed-form ICI and SINR from residual Doppler (radio) |
| H. Jin et al., OES COA 2024; Zhu et al., IEEE Access 2022; Sensors 2019 (time-varying filter) | Further time-varying compensators, compared by BER |

Sources (web): [IEEE 7343385](https://ieeexplore.ieee.org/document/7343385); [ResearchGate 261267543](https://www.researchgate.net/publication/261267543_The_effects_of_doppler_scaling_in_underwater_acoustic_OFDM_communication);
[JWCN 2009](https://jwcn-eurasipjournals.springeropen.com/articles/10.1155/2009/786040);
[arXiv 2603.20447](https://arxiv.org/pdf/2603.20447); [Salberg–Swami](https://www.researchgate.net/publication/3433310_Doppler_and_frequency-offset_synchronization_in_wideband_OFDM);
[Abdelkareem 2016](https://www.sciencedirect.com/science/article/abs/pii/S157087051500116X);
[Abdelkareem 2012](https://onlinelibrary.wiley.com/doi/10.1155/2012/139416); [Sensors 2019](https://doi.org/10.3390/s19010105).
