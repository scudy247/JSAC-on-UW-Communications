# Code plan

What has to be built for the JSAC paper on sensor-aided Doppler **estimation**, phase by phase: the question each
phase answers, which code does what, how it is tested and when it is done. Derived from [`../README.md`](../README.md)
and the deck (`project_idea/doppler_estimation.pdf`). Findings are logged in [`RESULTS.md`](RESULTS.md).
Rewritten on 12 Oct 2026 for the estimation focus; the compensation-era plan is in
`project_idea/archive/PLAN_compensation_2026-10-09.md`.

Status marks: ✅ done · 🟡 partly done · 🔄 running · ⬜ to do · ♻️ done in the compensation phase, reused.

## Principles
- **Estimation is the object; compensation is a tool.** Estimators are now written and compared; compensation is the
  standard resampler (`waveform.unwarp`) driven by the estimated trajectory, or the transmitter applying its estimate.
- **Every estimate has a source and a cost:** sensor (drift, latency, lever arm, direction), side information (bits),
  signal (pilots, preamble length, SNR, search range).
- **Real motion as input** (CDIP, OOI); models only for AUVs and seabed nodes.
- **Theory next to simulation:** every experiment prints the closed-form prediction (`placement.evaluate`,
  `theory.kernel_ici`) next to the simulated value.
- **Light on the shared machine:** `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19`, in the
  background. **Reproducible:** each figure from one command; band-limited interpolation; padded packets.

## Package layout
```
code/
  src/uwofdm/
    paths.py, run.py     ♻️ locations; versioned experiment runner
    buoy.py              ♻️ CDIP / OOI records -> velocity, acceleration, gyro rates
    motion.py            ♻️ tracks, lever arm, link Doppler and its transmitter / receiver / slow shares
    estimates.py         ♻️ noise, age, Gauss-Markov drift (sensor error models)
    waveform.py          ♻️ OFDM, wideband Doppler warp / unwarp, band-limited interpolator
    channel.py           ♻️ multipath with bulk + path-specific Doppler
    bellhop.py           ♻️ Bellhop arrivals, rough-surface loss, scattered-surface floor
    compensate.py        ♻️ trajectory fits + resampling (the compensation tool)
    theory.py            ♻️ kernels: polynomial, sampled fit, staleness, extrapolation, interference (kernel_ici)
    placement.py         ♻️ evaluator: who estimates / removes what -> residual -> SIR (E3)
    experiment.py        ♻️ link harness (packet loop, strategies incl. generic placements, genie and pilot metrics)
    sideinfo.py          ⬜ E1: representation, quantisation, header model of the transmitter's motion
    estimators.py        ⬜ E2: signal-based Doppler estimators (preamble/postamble, pilots, residual CFO) and baselines
    fusion.py            ⬜ E2: Kalman / RTS smoother fusing side information, own IMU and signal measurements
    receiver.py, per.py  ⬜ channel estimation, equalisation, LLR weighting, coded PER
  scripts/               one script per analysis or figure
  tests/                 pytest (33 passing on 11 Oct 2026)
```

---

## Phase 0 — Real motion and staleness ✅
**Question.** How large and how predictable is wave-driven Doppler? **Answer:** not predictable across a round trip
(normalised error 0.76–1.0 at 2–3 s), well known below 1 s — so the transmitter's motion must travel *with* the packet
(E1), never as feedback.

| Code | What it does | Status |
|---|---|---|
| `scripts/fetch_cdip.py`, `fetch_ooi.py` | CDIP Waverider and OOI motion-package records, swell / mixed / wind days | ✅ |
| `buoy.py` | Displacement / accelerometer + gyro -> earth-frame velocity, acceleration, rates | ✅ |
| `scripts/motion_predictability.py`, `ooi_motion.py` | Prediction error vs horizon; Hs check | ✅ |

---

## Phase 1 — Models 🟡 (core done)
| Code | What it has to do | Status |
|---|---|---|
| `motion.*` | Tracks (buoy, static, AUV leg, manoeuvring AUV), lever arm, link Doppler split | ✅ |
| `estimates.*`, `placement.NodeSensors` | Sensor error model: Gauss–Markov drift, latency (+ accelerometer bridge), lever arm, direction error | ✅ |
| `waveform.*`, `channel.*` | OFDM, Doppler warp / unwarp, multipath with per-path Doppler | ✅ |
| `bellhop.py` | Arrivals, coherent rough-surface loss, scattered-surface floor (swept) | 🟡 ✅ Bellhop paths in the chain (`multipath.py`: per-path Doppler from angles, diffuse scattering; first-arrival detection); experiment `scripts/e2_multipath.py` ⏳; open: literature value for scattered power; per-path compensation 🟡: Tu-like multi-branch MRC ✅, sensor-aided decoder-aided per-path cancellation ✅ (decodes real buoy–buoy two-path packets where all others fail); realistic per-path prediction ✅ (`sensors.py`: 3-axis IMU, two-component side information 64 bits, Bellhop angles; still decodes); signal-only cancellation baseline ✅ (`receive_signal_sic`: chirp delays + per-path search; decodes 1/3 real buoy–buoy packets vs 3/3 with sensors); angle errors ✅ (sensors alone ≤ 0.5°, with a narrow search ≈ 1°; delays must come from the signal: a 0.15 ms geometric delay error broke the search); next: statistics from `scripts/e2_multipath.py` |
| synthetic spectra | JONSWAP motion for sea-state sweeps (exists in `scripts/c2_map.py`; move to `motion`) | 🟡 |
| moving geometry | u(t), τ(t) for long AUV runs | ⬜ |

---

## Phase 2 — Metric: estimation error → performance ▶️ (T1)
**Question.** How much does a given Doppler-estimation error cost? Needed by every later phase.

| Code | What it has to do | Status |
|---|---|---|
| `theory.kernel_ici` | Interference = phase variance within each FFT window, for any linear estimator; offset and ramp limits | ✅ |
| `theory.kernel_samples`, `noise_samples` | Error of the fit the receiver really makes (estimates at given instants) | ✅ |
| `experiment.genie_powers`, `pilot_powers` | Genie (ML phase ramp) and pilot-based (delay-domain LS) SIR | ✅ |
| `receiver.py` | Pilot channel estimate, channel-estimate smoothing, per-subcarrier LLR weighting (as in Lee et al. 2026), equalisation | ✅ pilot delay-domain LS channel, noise + ICI from in-band nulls, exact QPSK LLRs, deinterleave, Viterbi (12 Oct); smoothing beyond the LS fit ⬜ |
| `per.py` | Convolutional / LDPC-like code, soft decoding; PER vs residual Doppler and SNR; tables for DESERT | 🟡 `coding.py` ✅ (rate 1/2 K=7, Viterbi, interleaver, Gray QPSK); PER on real motion `scripts/e2_per.py` 🔄; 16-QAM, longer packets, tables ⬜ |
| complexity counters | Operations per packet: resampling, search, fusion | ⬜ |

**Done when** coded PER reproduces the Lee et al. regime (frame-level estimate enough at ≈ 9 dB, QPSK, short frames)
and shows where it is not (high SNR / 16-QAM, long frames, rough sea, both ends moving).

---

## Phase 3 — E3: who estimates what ♻️ (done in the compensation phase; reframe)
**Question.** Which part of the Doppler should come from which source, given sensor quality, latency, packet length
and the receiver's estimate budget?

| Code | What it does | Status |
|---|---|---|
| `placement.py` | Enumerates sources per share (node IMU on/off × receiver stage), predicts residual and SIR | ✅ validated: 81/85 winners, ±0.7 dB median |
| `experiment.run_link` | Simulates any placement name; scenarios buoy→seabed, buoy–buoy, AUV→buoy | ✅ (open: theory column for `rx_sym`/`split` still the old formula — switch to the evaluator) |
| `scripts/c2_*.py`, `c2_sweeps.py`, `c2_robustness.py`, `c2_new_placements.py`, `c2_map.py`, `c2_rule.py`, `surface_scatter_sweep.py` | Scenarios, sweeps, robustness, maps, validation, scattered floor | ✅ |
| reframing | Rename strategies as estimation sources in figures and text ("sensor share + signal share"); equal-budget comparison as the headline | ⬜ |

---

## Phase 4 — E1: Doppler side information 🔄 (T2)
**Question.** What should the transmitter send about its own motion, in how many bits, to reach a target accuracy;
how does the receiver read it before it knows the Doppler; how does sending compare with applying (pre-compensation)?

| Code | What it has to do | Status |
|---|---|---|
| `sideinfo.represent` | Transmitter's measured velocity along the path over the packet → samples, Legendre-polynomial or DCT coefficients (order / count as parameter) | 🟡 orthonormal Legendre (`basis`, `project`, `reconstruct`, `coefficient_std`) ✅ 12 Oct; samples / DCT ⬜ |
| `sideinfo.quantise`, `bits` | Uniform / Lloyd–Max quantisation per coefficient, bit allocation from the motion spectrum; total bits per packet | ✅ uniform ±4σ; size-based (`allocate`) and stage-aware (`stage_weights`, `allocate_weighted`) allocation |
| `sideinfo.header` | Header model: robust low-rate field (e.g. BPSK, rate-1/3 code) after coarse sync; decoding-error probability vs SNR and residual Doppler; fallback when the header fails | 🟡 `header.py` ✅ (quantiser indices + CRC-8, DBPSK across subcarriers with repetition, read after own IMU + preamble/postamble scale + header CFO, fallback to signal-only); real-motion sweep ✅ (read in 100% of packets ≥ 5 dB, 95–100% at 0 dB with 16–32 bits; full chain = ideal side information) |
| `theory` (T2) | Residual variance vs bits: polynomial / sampled representation error (kernels) + quantisation noise, from the motion spectrum; → SIR through `kernel_ici` | ⬜ |
| `scripts/e1_sideinfo.py` | Real motion, three scenarios: SIR vs bits per packet; vs packet length; header failure rate; comparison with transmit-side pre-compensation (`tx_own`) and with the receiver alone | 🟡 bits sweep, allocation, stage-aware allocation ✅ (≈ 10–30 bits per second of packet within 1 dB of pre-compensation); next: waveform check, header ⬜ |
| tests | Representation round trip; quantisation error vs theory; header success vs SNR | 🟡 basis, kernel match, quantiser, interference offset limit ✅ (`tests/test_sideinfo.py`); header ⬜ |

**Done when** bits-vs-SIR curves exist with theory within a few dB, and the header design survives the Doppler range
of the scenarios.

---

## Phase 5 — E2: end-to-end chain and fusion ▶️ (T3, main result)
**Question.** How much better, cheaper or more robust is a receiver that fuses side information, its own IMU and the
signal than one that uses the signal alone?

| Code | What it has to do | Status |
|---|---|---|
| `estimators.preamble_scale` | Frame-level Doppler-scale estimate from a preamble / postamble (correlation over a Doppler grid, or two-chirp delay), on the simulated waveform; search range and grid as parameters | ✅ `frame.py` (chirp preamble / postamble, header, data with in-band nulls) + `estimators.detect`, `scale_pre_post` |
| `estimators.residual_cfo` | Per-symbol residual CFO from pilots / null subcarriers (Li et al. 2008 style) | ✅ null-subcarrier energy over ±2 subcarriers; in-band nulls (every 16th) — 0.5 Hz rms at 0 dB per subcarrier |
| `estimators.pilot_track` | Per-symbol Doppler-scale tracking from pilots (signal-only Kalman tracker baseline) | ⬜ |
| baselines | Li et al. 2008 two-step; Lee et al. 2026 tiers (frame scale + per-symbol CFO); Cho–Kim constant acceleration | 🟡 `chain.receive_signal_only` (Li / Lee front end) ✅; `chain.receive_aided`, `receive_with_header` ✅; Cho–Kim ✅ (`chain.receive_cho_kim`, best version); signal-only Kalman ✅; chirp coupling bias fixed (up/down sweeps); real-motion SNR sweep ✅ (fusion vs signal-only Kalman: 0 dB CDIP, up to +14.6 dB buoy–buoy 4 s at 30 dB) |
| `fusion.WaveState` | State model: Doppler share(s), rate, slow drift, IMU bias; process noise from the motion spectrum | ⬜ |
| `fusion.smooth` | Kalman filter + RTS smoother over the packet: inputs side information (with its quantisation error), own IMU samples (with latency), signal measurements (with SNR-dependent error); output trajectory + covariance | ⬜ |
| acquisition | Prior-centred Doppler search window for the preamble; probability of wrong acquisition vs window and SNR | ⬜ |
| `theory` (T3) | Bayesian Cramér–Rao bound on the trajectory with / without the sensor prior vs SNR and pilot count | ⬜ |
| `scripts/e2_fusion.py` | SIR / PER vs SNR, pilots per packet, preamble length, search range; the three scenarios; Lee-regime check | ⬜ |
| tests | Kalman vs analytic steady state; fused error ≤ each source; acquisition with a correct prior | ⬜ |

**Done when** fusion is compared with the signal-only baselines on accuracy, overhead and robustness, with the bound,
and the regime where it matters is mapped (SNR × packet length × sea state × modulation).

---

## Phase 6 — Multi-user extension ⬜ (optional for the first submission)
Per-user side information sets each gateway branch from priors (Tu et al. 2013 baseline). Code as in the archived
plan (`multiuser.py`: uplink signal, single / multiple resampling, prior-driven branches, guard bands).

---

## Phase 7 — DESERT extension ⬜ (in parallel; reframe)
Estimation-aware PHY: per packet, the Doppler-estimation error of the chosen sources (side information, IMUs, pilots)
→ interference → SINR → PER; side-information bits and pilots as overhead. Architecture and steps in
[`DESERT_PLAN.md`](DESERT_PLAN.md) (written for compensation on 9 Oct; the model library and wave mobility carry
over, the PHY's "placement" becomes "estimation sources"). Decisions pending (addons vs core, per-packet vs tables,
MAC, multi-user hook, WOSS per-path step).

---

## Phase 8 — Experiments, figures, writing ▶️
| Code | What it has to do | Status |
|---|---|---|
| `experiments/*.yaml` | One config per paper figure through `run.py` | ⬜ |
| `scripts/paper_stats.py` | High-statistics PER (150–300 packets per point) and multipath (60) runs for the figures, files `*_hi.json` | 🔄 (13 Oct, ≈ 6 h, two queues) |
| figure scripts | From result JSON files (dataviz rules: fixed colours per source, one axis) | 🟡 `scripts/fig_results.py` ✅ (SINR rough link, PER 4 s, multipath; `_hi` suffix for the high-statistics files); deck section "Results" ✅ |
| housekeeping | Move `model.py`, `replay_diagnostic.py` to `legacy/`; README and deck in sync | ⬜ |
| paper | JSAC LaTeX skeleton | ⬜ |

---

## Timeline
About 24 weeks from 12 Oct 2026 to 31 March 2027. Critical path: Phase 2 coded PER (needed to answer Lee et al.)
→ Phase 4 (E1) → Phase 5 (E2, main result) → Phase 8. Phase 3 is reframing only. Phases 6–7 alongside; if time runs
short, multi-user and the DESERT network experiment move to the revision.

## DESERT and WOSS (checked 9 Oct 2026)
- DESERT's OFDM PHY gives PER from SNR (`UwOFDMPhy::getOFDMPER`); nothing models Doppler; mobility models have no
  waves. WOSS 2.0.1 has no Doppler; its Bellhop arrival angles are read and dropped (`TimeArr`).
- The build runs (OFDM-ALOHA sample) with paths corrected to /home/dario.
