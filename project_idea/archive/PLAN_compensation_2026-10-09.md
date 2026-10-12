# Code plan

What has to be built for the JSAC paper on Doppler **compensation**, phase by phase: the question each phase answers,
what it needs, which code does what, how it is tested and when it is done. Derived from
[`../README.md`](../README.md) and the deck (`project_idea/doppler_compensation.pdf`). Findings are logged separately
in [`RESULTS.md`](RESULTS.md). Rewritten in detail on 9 Oct 2026.

Status marks: ✅ done · 🟡 partly done · 🔄 running · ⬜ to do.

## Principles
- **Compensation is the object; estimation is an input.** No estimator is written. Every compensator receives estimates
  made as *true value + error*, with a stated error level and age.
- **Real motion as input.** CDIP and OOI buoy records drive the simulations; models only where no record exists (AUV,
  seabed nodes).
- **Two fidelity levels.** Waveform level (sampled OFDM through a time-varying channel, Python) for residuals,
  interference, BER/PER and complexity; network level (DESERT extension) for many nodes and MAC, fed with tables from
  the waveform level.
- **Theory next to simulation.** Every experiment prints the closed-form prediction next to the simulated value.
- **Light on the shared machine.** Single thread, low priority (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1 nice -n 19`); experiments run in the background while other work goes on.
- **Reproducible.** Each figure from one command; motion records interpolated band-limited (never straight lines);
  packets padded so that stretching never cuts them.

## Package layout
```
code/
  src/uwofdm/
    paths.py, run.py     ✅ data/results locations; versioned experiment runner (inherited)
    buoy.py              ✅ CDIP / OOI records -> earth-frame velocity, acceleration, gyro rates
    motion.py            ✅ tracks (buoy, static, AUV leg, manoeuvring AUV), lever arm, link Doppler and its split
    estimates.py         ✅ estimation input model (noise, age, IMU drift)
    waveform.py          ✅ OFDM, wideband Doppler warp and its inverse, band-limited interpolator
    channel.py           🟡 multipath with bulk + path-specific Doppler (Bellhop paths to do)
    compensate.py        🟡 receiver trajectory models (complexity counters to do)
    theory.py            🟡 closed forms for R1, R2 (R3 and refinements to do)
    experiment.py        ⬜ shared harness for C1/C2/C3 (packet loop, strategies, metric, theory rows)
    multiuser.py         ⬜ C3 transmitters and receivers for K users
    receiver.py          ⬜ channel estimation, equalisation, decoding (for BER/PER)
    per.py               ⬜ residual -> SINR -> PER tables (also exported for DESERT)
  scripts/               one script per analysis or figure
  tests/                 unit tests per module (pytest, 21 passing on 9 Oct 2026)
  legacy/                ⬜ move model.py and replay_diagnostic.py here
```

---

## Phase 0 — Real motion and staleness ✅
**Question.** How large and how predictable is the Doppler of wave-driven nodes, from real data?

| Code | What it does | Status |
|---|---|---|
| `scripts/fetch_cdip.py` | Downloads CDIP Waverider wave summaries, picks swell / mixed / wind-sea days, downloads one day of 3-axis displacement (1.28 Hz) per type | ✅ |
| `scripts/fetch_ooi.py` | Downloads OOI wave-sensor logs (one day in four), picks the same day types, downloads the motion-package bursts (10 Hz, 20 min per hour) | ✅ |
| `buoy.py` | CDIP: displacement -> velocity, acceleration by spectral differentiation. OOI: decodes 0xCB records, attitude filter (gyro offset removed), earth-frame acceleration, spectral integration | ✅ |
| `scripts/motion_predictability.py`, `scripts/ooi_motion.py` | Prediction error vs horizon (hold, per-packet extrapolation, best linear predictor), with noise; Hs check against the wave sensor | ✅ |

**Done when** the staleness of a good estimate is measured on both buoy types and three sea states. ✅ (RESULTS.md)

---

## Phase 1 — Models 🟡 (core done)
**Question.** None by itself: provides motion, estimates, signals and channels that all later phases use.

| Code | What it has to do | Status |
|---|---|---|
| `motion.Track` | Velocity of one transducer on a time grid, split into fast (wave-driven), slow (cruise, drift) and lever-arm parts; `velocity_at(times, part)` band-limited with edge padding | ✅ |
| `motion.buoy_track` | Recorded buoy motion as the fast part; for OOI, lever-arm velocity ω × r from the gyro | ✅ |
| `motion.static_track`, `auv_leg`, `auv_maneuvering` | Seabed node; constant AUV leg; AUV with cruise, turns, speed changes, small heave | ✅ |
| `motion.link_doppler` | ε(t) = (v_A(t) − v_B(t + τ))·u / c on the emission grid, split into transmitter, receiver and slow shares | ✅ |
| `motion.synthetic_track` | Fast motion from a wave spectrum (JONSWAP, NDBC spectra) to sweep sea states beyond the 9 recorded days | ⬜ |
| moving geometry | u(t) and τ(t) from positions, for AUVs that travel kilometres during a run (now fixed geometry) | ⬜ |
| `estimates` | `noisy`, `imu`, `aged`, `link`, `gauss_markov` (IMU velocity drift) | ✅ |
| `waveform.warp` / `unwarp` | Wideband Doppler y(t) = s(w(t)) e^{j2πf_c(w(t)−t)} and its inverse along any trajectory (receiver compensation and transmit-side pre-compensation) | ✅ |
| `waveform.resample_at` | Band-limited interpolation: polyphase windowed-sinc table, unit gain at zero frequency | ✅ |
| `waveform.OFDM` | CP-OFDM numerology (16 kHz, 8 kHz, 1024 subcarriers), modulation / demodulation, pilots, QPSK / 16-QAM, K-user subbands with guard bands | ✅ |
| `channel.apply`, `exponential_paths`, `awgn` | Multipath with per-path Doppler offsets; random sparse channels; noise | ✅ |
| `bellhop.py` (was `channel.bellhop_paths`) | Paths (delay, amplitude, angles) from Bellhop arrivals along the node positions (`bellhop.exe` in the DESERT build); per-path Doppler from the angles | 🟡 arrivals, merging, profile, floor ✅; coherent rough-surface loss ✅ (9 Oct); diffuse surface scattering swept ✅ (placement matters below rho ≈ −20…−25 dB); open: literature values for rho, sound-speed profiles, wiring into `experiment.Multipath` (bounce counts), projection of node motion on each path's angles |

**Tests (✅):** sign conventions, split sums, reception-time reading, estimate statistics, Gauss–Markov statistics and
spectrum, lever arm, AUV turns, OFDM round trip, analytic tone Doppler, inverse warps (< −50 dB), pre-compensation with
padding at ±1e-3 (< −45 dB), CDIP track statistics vs Phase 0.
**Done when** the synthetic spectra and Bellhop paths exist and are tested (needed for sea-state sweeps and for
path-specific Doppler in Phases 3–4).

---

## Phase 2 — C1: compensation inside the packet 🟡 (supporting result R1)
**Question.** With good estimates inside the packet, which trajectory model and update rate leave how much
interference, as a function of packet length and motion? (G1 is only partly open: supporting result.)

| Code | What it has to do | Status |
|---|---|---|
| `compensate.fit_trajectory` | Fit a model to M estimates at given instants: order 0, 1 (constant acceleration, Cho–Kim-like), 2, piecewise constant | ✅ |
| `compensate.compensate` | Resample the received signal along the fitted trajectory | ✅ |
| complexity counters | Operations per packet for each model (resampling, fitting, number of estimates needed) | ⬜ |
| `theory.kernel_poly` | Residual of the best order-N fit over T: K_N = 1 − Σ (2k+1) j_k(πfT)² | ✅ |
| `theory.ici_sir_db` | Interference between subcarriers from a residual Doppler (small-offset form) | ✅ |
| large-offset form | Use 1 − sinc²(f_c e T_s) instead of the small-offset form when interference is strong | ⬜ |
| piecewise kernel | Exact kernel for per-symbol constant compensation (the order-0 approximation is 5–7 dB pessimistic) | ⬜ |
| `scripts/c1_residual.py` | Real buoy motion -> packets of 1, 2, 4 s -> models 0, 1, 2, piecewise, estimate error 0 / 3e-6 -> power-averaged SIR (genie removes per-symbol phase and timing) and theory | ✅ (rerun after padding fix: results to log) |
| `receiver.py` + `per.py` | Channel estimation from pilots, equalisation, a convolutional code with soft Viterbi; BER/PER vs residual, to replace the genie metric | ⬜ (after Phase 3, as decided) |

**Done when** the design rule (model order vs packet length vs motion) is stated with theory within a few dB of
simulation, and confirmed with coded PER.

---

## Phase 3 — C2: where to compensate ▶️ (main message, R2)
**Question.** Who should remove which part of the Doppler (receiver, transmitter, split), given the accuracy and age of
the estimates each node has? Decided 9 Oct 2026: main message; own IMU counts as a good estimate; SIR first.

| Code | What it has to do | Status |
|---|---|---|
| IMU model | Own velocity projected on the path + Gauss–Markov drift, latency extrapolated with the measured acceleration, lever arm from the gyro (or not) | ✅ (in scripts) |
| `NodeIMU` | Per node: true share of the link Doppler, IMU estimate at any time, residual spectrum for the theory; option for nodes whose INS also knows the slow motion (AUV) | ✅ (in `c2_buoy_buoy.py`) |
| strategies | rx_sym, rx_pkt2, rx_stale_H, tx_imu, rx_imu, both, split (line), split0 (constant) | ✅ |
| `theory` (R2) | Staleness kernel, averaged over a packet; extrapolation kernel; Gauss–Markov spectrum; strategy residuals as filtered spectra | ✅ |
| cross terms | Correlated residual parts (latency error and lever arm, two-point fit) in the split prediction (now 3–9 dB optimistic) | ⬜ |
| placement rule | Closed-form condition for when transmit-side, receive-side or split is best, from the spectra and the estimate errors. `placement.py`: enumerates node stages x receiver stages, node stage as a transfer function (cross terms), receiver stage via `theory.kernel_samples`, path-Doppler floor; `scripts/c2_rule.py` validates against every simulated point | 🔄 evaluator ✅ (direction check: 15/15 winners, split within 0.3 dB median); validation on the 85 rerun points ✅ (81/85 winners, all strategies within ±0.7 dB median, with `theory.kernel_ici`); generic placements in the harness ✅; new placements simulated ✅ (17/18 winners; IMU + per-symbol receiver best, +4…+15 dB over receiver per symbol at equal budget); dimensionless maps ✅ (first statement of the rule in RESULTS.md); open: switch `run_link` theory column to the evaluator |
| `experiment.py` | Move the shared harness (packet loop, padding, strategies, genie metric, theory rows, JSON output) out of the scripts | ✅ (genie ramp fixed 9 Oct: ML estimate; all genie results being rerun) |
| `scripts/c2_placement.py` | Scenario 1, buoy -> seabed node | ✅ (rerun done: results to log) |
| `scripts/c2_buoy_buoy.py` | Scenario 2, buoy -> buoy (independent motions, same sea state) | ✅ (rerun done: results to log) |
| `scripts/c2_auv.py` | Scenario 3, AUV -> buoy (modelled AUV, real buoy) | ✅ (rerun done: results to log) |
| sweeps | IMU error (0.005–0.1 m/s), latency (0–200 ms), lever arm (0–5 m), range (τ), packet length (0.5–8 s), more sea states (synthetic spectra) | ⬜ |
| figures + deck slide | SIR vs strategy and scenario; staleness and placement curves with theory | 🟡 deck: 5 slides "C2 results (preliminary)" (9 Oct); paper figures ⬜ |

**Done when** all three scenarios are rerun cleanly, the sweeps show where each strategy wins, the placement rule is
derived and matches simulation within a few dB, and the results are in the deck.

---

## Phase 4 — C3: several Doppler at once ⬜ (R3)
**Question.** In a multi-user link, how much interference between users remains for each way of splitting the
compensation, and what does each cost (resampling branches, estimates)?

| Code | What it has to do | Status |
|---|---|---|
| `multiuser.uplink_signal` | K users, each on its own subband (contiguous or interleaved) with guard bands, each warped by its own link Doppler (own motion + gateway motion projected on its direction), summed at the gateway with noise | ⬜ |
| `multiuser.rx_single` | One resampling for all users (baseline) | ⬜ |
| `multiuser.rx_multi` | Tu et al.: one resampling branch per user, each followed by an FFT that keeps that user's subcarriers; branch Doppler estimated per user | ⬜ |
| `multiuser.split_uplink` | Each user pre-compensates its own motion (IMU); the gateway removes its own motion per user from its IMU and the users' directions (branches set, not estimated); users with similar directions share a branch | ⬜ |
| `multiuser.downlink` | The gateway pre-compensates each user's subband separately (separate warps summed) before sending | ⬜ |
| guard bands | Size each guard band from the residual Doppler after compensation (not from the raw mobility) | ⬜ |
| metrics | Per-user SINR split into own interference and interference from other users; branches and estimates needed per packet | ⬜ |
| `theory` (R3) | Interference between users vs residual differences and guard bands; users supported vs branches | ⬜ |
| `scripts/c3_multiuser.py` | Scenarios: buoy gateway with seabed nodes, with AUVs; K = 2–8; compared with Tu et al. | ⬜ |

**Done when** split compensation is compared with the multiple-resampling receiver on interference and complexity,
with theory, for uplink and downlink.

---

## Phase 5 — DESERT Doppler extension ⬜ (in parallel; about 2–3 calendar weeks)
**Question.** Does the link-level gain hold in a network (MAC, traffic, many nodes)? Also an open contribution.

| Code (C++ / Tcl, in DESERT_Addons) | What it has to do | Status |
|---|---|---|
| rebuild | Rebuild DESERT on /home/dario (the build's `environment` points to /home/fabio); check the samples | ⬜ |
| swell mobility module | Node position and velocity from a replayed record or a wave spectrum (new module next to `uwdriftposition`) | ⬜ |
| Doppler per packet | Link Doppler from the two nodes' velocities at emission and reception times; optional per-path Doppler from WOSS/Bellhop arrival angles | ⬜ |
| OFDM PHY extension | Residual Doppler after the node's compensation strategy -> SINR loss and inter-user interference in `uwofdmphy`; textbook formula first, then tables from `per.py` | ⬜ |
| compensation option | Per node: receiver-only, transmit-side, split; IMU error and latency as parameters | ⬜ |
| Tcl samples + checks | Uplink network scenario; single-link results checked against Python | ⬜ |

**Needs from Python:** residual and PER tables exported by `per.py` (CSV), per strategy, packet length and sea state.

**Detailed plan (9 Oct 2026):** [`DESERT_PLAN.md`](DESERT_PLAN.md) — architecture (model library, wave mobility, Doppler OFDM PHY, registry), steps with done criteria, validation chain, decisions needed.

---

## Phase 6 — Experiments, figures, writing ⬜
| Code | What it has to do | Status |
|---|---|---|
| `experiments/*.yaml` | One config per paper figure, run through `run.py` (versioned outputs with manifests) | ⬜ |
| figure scripts | Publication figures from the result JSON files | ⬜ |
| coded BER/PER | Rerun the headline cases with `receiver.py` + `per.py` | ⬜ |
| housekeeping | Move `model.py`, `replay_diagnostic.py` to `legacy/`; README and deck kept in sync | ⬜ |
| paper | JSAC LaTeX skeleton; sections fed by the figures | ⬜ |

---

## Timeline
About 24 weeks to 31 March 2027 (from 9 Oct 2026). Critical path: Phase 3 (C2, main message) -> Phase 4 (C3) ->
Phase 6. Phase 2's remaining items and Phase 5 run alongside. If time runs short, the downlink and the DESERT network
experiment move to the revision.

## DESERT and WOSS (checked 9 Oct 2026)
- DESERT's OFDM PHY gives PER from SNR and packet length (`UwOFDMPhy::getOFDMPER(snr, nbits)`); nothing models
  Doppler. Its mobility models (linear, Gauss–Markov, drift) have no swell.
- WOSS 2.0.1 (in the build, latest release) does not model Doppler either: Bellhop at current positions, time-evolving
  environment, waypoint mobility, one power gain per packet (`WossMPropagation::getGain`). Its Bellhop arrivals (delay,
  amplitude, angles) are the input for per-path Doppler.
- The build runs: the OFDM-ALOHA sample completes with the paths corrected to /home/dario.
