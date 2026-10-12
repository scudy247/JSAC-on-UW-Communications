# Sensor-aided Doppler estimation for underwater acoustic links

**Target:** IEEE JSAC special issue *Next-Generation Underwater Acoustic Communication Systems*, deadline
31 March 2027. **Roadmap:** [`project_idea/doppler_estimation.pdf`](project_idea/doppler_estimation.pdf)
(group-discussion deck, 12 Oct 2026); this README summarises it. Code plan: [`code/PLAN.md`](code/PLAN.md); results
log: [`code/RESULTS.md`](code/RESULTS.md).

**Ground rule (12 Oct 2026, reversing the rule of 9 Oct).** The contribution is the Doppler **estimation**: who
estimates which part of the Doppler, from which information, how accurately, at what cost (pilots, bits, search),
and how the pieces are combined. **Compensation** (resampling along an estimated trajectory, or the transmitter
applying its own estimate) is taken as a working tool: once the estimate is right, existing compensators do their job.

## Main idea
The Doppler of a packet is the sum of parts that are best known in different places:

| Part of the Doppler | Who knows it best, and when |
|---|---|
| transmitter's own motion during emission | the transmitter's IMU, live, while it sends |
| receiver's own motion during reception | the receiver's IMU, live, while it receives |
| slow part (drift, currents, cruise, path direction errors) and whatever the sensors miss | the received signal (preamble, postamble, pilots) |
| path-specific part (surface scattering) | nobody: a floor |

Today every UWA receiver estimates the whole Doppler from the received signal alone. The proposal is to estimate
each part where it is known and **fuse** them at the receiver:
- **E1 – Doppler side information.** The transmitter measures its own motion while it emits and carries it *in the
  same packet* (a few samples or coefficients of its velocity along the path). It is never stale, because it
  describes the packet it travels with. How many bits are needed for a target accuracy, and where to place them so
  that the receiver can read them before it knows the Doppler, are part of the design. (Alternative use of the same
  estimate: the transmitter applies it itself, i.e. transmit-side pre-compensation; the two are compared.)
- **E2 – Fusion at the receiver.** The receiver combines E1, its own IMU and the signal-based estimates (preamble,
  postamble, pilots) into one Doppler trajectory over the packet (Kalman / MAP smoother on a wave-motion model).
  Sensors give a prior that narrows the Doppler search and replaces most per-symbol pilots; the signal corrects the
  sensors' drift and the parts they cannot see.
- **E3 – Who estimates what.** Given sensor quality (IMU drift, latency, lever arm, direction error), packet length and
  the receiver's estimate budget (two estimates per packet vs one per symbol), which part should come from which
  source. Our placement results (Phase 3) answer this and are reframed here.

**Candidate claim:** the first sensor-aided Doppler estimation for UWA links: transmitter motion as in-packet side
information, fused at the receiver with its own IMU and signal-based estimates; sized in bits and pilots; with closed
forms from estimation error to inter-carrier interference; validated on real buoy motion.

**Scope and regime.** OFDM; mobile or surface-coupled nodes (buoys, AUVs) talking to fixed or moving nodes; one link
first, multi-user as an extension. The claim targets the regime where a frame-level signal estimate is not enough or
too costly: high SNR / high-order modulation, frames longer than about 1 s, rough seas, both ends moving, low SNR at
acquisition, and links that cannot afford per-symbol pilots. In a noise-limited link with short frames and a moderate
sea, a preamble-based frame-level estimate is enough (Lee et al. 2026, below): the paper must say so.

## Evidence so far (from the compensation phase, reread as estimation)
Details in [`code/RESULTS.md`](code/RESULTS.md).
- **Feedback across a round trip does not work; in-packet side information does.** Phase 0, real CDIP and OOI buoy
  motion: wave-driven velocity cannot be predicted 2–3 s ahead (normalised error 0.76–1.0); estimates one round trip
  old give −12 to −5 dB SIR at every range. A transmitter's own motion during the packet, measured live, has no such
  age.
- **Sensor information is worth 4–15 dB at the same signal-estimate budget.** IMU stages plus a receiver that uses one
  estimate per symbol beat the receiver alone with the same estimates by +4 dB (CDIP swell), +12 dB (OOI Irminger),
  +15 dB (buoy–buoy), on real motion, 30 packets per point; with two estimates per packet, IMUs plus a receiver line
  ("split") give 37–41 / 32–38 / 33–40 dB (buoy→seabed / buoy–buoy / AUV→buoy) against 27–36 / 24–33 / 26–35 dB for
  the receiver per symbol.
- **What the sensors must achieve:** IMU drift up to about ten times worse than 0.01 m/s is tolerated when the signal
  corrects it; latency below about 100 ms (0.01 wave periods), bridged by the accelerometer; lever-arm correction from
  the gyro; path direction within about 5°.
- **Estimation error → performance in closed form.** Interference = phase variance of the residual within each FFT
  window (a residual ramp costs 5× less than an offset of the same variance); the evaluator predicts the simulated
  winner at 81/85 points and every strategy within ±0.7 dB (median). This is the metric for E1–E3.
- **A floor no estimate removes.** Diffusely scattered surface power with its own Doppler: estimation matters only
  while it stays about 20–25 dB below the coherent channel (Bellhop with measured surface roughness; literature value
  still to find).

## End-to-end results (12 Oct 2026, after fair baselines — supersedes the earlier gains)
Details and every correction in [`code/RESULTS.md`](code/RESULTS.md).
- **Single path: no SINR gain against a fair signal-only receiver.** With a tuned signal-only Kalman, partial-FFT
  demodulation and a wide enough null-subcarrier search (+-4 subcarriers), sensor fusion gains 0-0.3 dB even in the
  tail of rough, long, buoy-buoy packets (30 packets per case). The earlier +6 / +14.6 dB came from weaker baselines
  (+-2 search, no partial FFT) and few packets.
- **What sensors still buy:** the same SINR without partial FFT or pilot tracking; 8-30x smaller Doppler search;
  3-10x fewer wrong postamble locks near -9 dB (~1 dB of acquisition SNR); per the T3 bound, the sensor prior can
  replace pilot-phase tracking (+5-7 dB on the bound, invisible at the noise level).
- **Realistic sensors:** GNSS-aided industrial MEMS behave like the original model inside the fusion; consumer MEMS
  and unaided sensors need the prior-quality guard (`chain.receive_fused_safe`), which never falls below signal-only.
- **Side information (E1):** closed form for the bits (T2) validated within 1 dB (90th pct); ~15-30 bits per
  second of packet in rough seas; per-class quantiser ranges cost < 0.5 dB if per platform type; header read
  >= 90 % from 0 dB (single path), 4-6 dB worse in multipath; a wrong header is harmless with the guard.
- **Multipath:** sensor-predicted per-path cancellation was the strongest result, but it was measured with the old
  channel and receivers; the channel is now realistic (fractional delays, path clusters, sound-speed profile,
  receiver angles from a nominal geometry) and the evaluation is **on hold** with the other reruns.
- **Open decision (PLAN.md):** framing — (a) cost / complexity and pilot overhead, (b) multipath per-path
  cancellation, (c) transmit-side / E3.

## Gaps with the literature
Checks: local library (19 queries), web (11 queries) and OpenAlex (4), 9–12 Oct 2026
([note](literature/estimation_pivot_local_check.md)); earlier checks in [`literature/`](literature/).

| | Gap | Closest work |
|---|---|---|
| G1 | No UWA communication receiver uses motion-sensor data (IMU / INS / DVL) as an input or prior to Doppler estimation | All sensor + Doppler work runs the other way, communication Doppler into navigation filters (Aubry 2019, Garin 2021, Choi et al. RA-L 2022); only Oiwa et al. 2020 (accelerometer, IEICE Tech. Rep., one end). Radio analogues, mostly patents: GNSS-aided Doppler pre-compensation (3GPP NTN; US 10,826,595), IMU-driven predistortion on UAVs (Gan, US 10,015,762 and family) |
| G2 | No transmitter tells the receiver its motion; how many bits of side information a packet needs is not analysed | DE-Sync 2018 / APE-Sync 2019 return measured Doppler in replies, only to set clocks; packet headers carry addresses, SNR or MCS |
| G3 | Signal-only estimation: frame-level estimates suffice in noise-limited links, per-symbol tracking costs pilots; the accuracy / overhead / search trade-off with a sensor prior is not analysed | Lee et al. 2026 (surface buoy, 10.5 h: per-symbol tracking gains ×1.15 in PER at ≈ 9 dB SNR); Li et al. 2008; Mason et al. 2008; Sharif 2000; Cho–Kim 2026; Kalman Doppler trackers (Sun 2022, Hu 2024); pilot overhead (Berger 2011) |
| G4 | (Extension) several users' Doppler estimated at the gateway one branch at a time; per-user sensor priors not used | Tu, Duman, Stojanovic, Proakis 2011, 2013 (multiple resampling); Daoud et al. 2015 |

**Counter-evidence to address:** Lee et al. 2026 (Doppler is not the limit in their noise-limited regime) and AsyMov
(arXiv 2412.10387, radio: channel-based Doppler beats the IMU when enough paths exist; the IMU helps when they do
not). Our claim: sensors help where signal estimation is weak or costly, not everywhere.

## Proposed methodology
**Model.** Relative Doppler scale ε(t) = (v_A(t) − v_B(t + τ))·u/c, split into transmitter, receiver and slow shares
(`motion.link_doppler`). **Sensor model:** each node's IMU / INS measures its own velocity with drift (Gauss–Markov),
latency bridged by the accelerometer, lever arm from the gyro, direction error. **Signal model:** preamble, postamble
and pilots give noisy Doppler measurements at known instants with SNR-dependent error.

1. **E1 – Side information.** Encode the transmitter's measured motion over the packet (samples, polynomial or
   spectral coefficients of its velocity along the path), quantise to B bits, carry it in a robust header read after
   a coarse synchronisation. Output: residual Doppler and SIR vs bits; header robustness; comparison with transmit-side
   pre-compensation (no bits, but one intended receiver and real-time resampling at the transmitter).
2. **E2 – Fusion estimator.** Kalman / MAP smoother over the packet on a wave-motion state (velocity, acceleration,
   slow drift, IMU bias), fed by E1, the receiver's IMU and the signal measurements. Output: Doppler-trajectory error
   and SIR / PER vs SNR, pilots per packet, preamble length and Doppler search range; bound with and without the
   sensor prior. Baselines: preamble/postamble frame-level estimate with residual CFO tracking (Li et al. 2008,
   as in Lee et al. 2026); pilot-based per-symbol tracking; Cho–Kim constant acceleration; signal-only Kalman tracker.
3. **E3 – Who estimates what.** The placement evaluator (Phase 3) reread as an estimation design: which share from
   which source as a function of IMU quality, latency, packet length and estimate budget; rule and maps in
   wave-period units.
4. **Extension – multi-user.** Per-user side information lets the gateway set each user's branch from priors instead
   of estimating it.

**Evaluation.** Real buoy motion (CDIP, OOI) and modelled AUVs; waveform-level OFDM simulator with Bellhop channels
and the scattered-surface floor; SIR first, then coded PER (needed to answer Lee et al. 2026 on their own metric);
overhead in pilots and bits; complexity (Doppler search). Network level: DESERT extension, reframed as an
estimation-aware PHY ([`code/DESERT_PLAN.md`](code/DESERT_PLAN.md)).

## Theoretical objectives
| | Result |
|---|---|
| T1 | Estimation error → interference: phase variance within each FFT window for any linear estimator; closed form validated (done, `theory.kernel_ici`) |
| T2 | Side information: residual Doppler variance vs bits per packet for the transmitter's motion, from the motion spectrum (quantised representation over the packet; rate–distortion form) |
| T3 | Fusion: error of the fused trajectory with and without the sensor prior vs SNR and pilots (Bayesian Cramér–Rao bound); pilots and preamble length saved for a target SIR; reduction of the Doppler search range |
| T4 | Staleness: why feedback across a round trip fails (staleness kernel on the motion spectrum, done) — the argument for in-packet side information |

## Risks
- **Lee et al. 2026:** in noise-limited links the frame-level estimate is enough; our regime must be stated, and coded
  PER shown where it matters.
- **AsyMov-type objection:** signal-based Doppler can beat sensors; the fusion must show gains where signal estimates
  are weak or expensive, and no loss elsewhere.
- **Chicken and egg:** side information must be decoded before the Doppler is known; needs a robust header or
  side information carried by the preamble, and a coarse first estimate.
- **Scattered surface floor:** if real channels carry more than about −20 dB of scattered surface power, accuracy gains
  vanish; literature value needed.
- **Prior art in radio** (GNSS/IMU-aided Doppler, mostly patents): the claim is "first for UWA, with real wave motion,
  side-information sizing and fusion", not "first sensor-aided Doppler".
- **Data:** motion is real; channels and multi-node geometry are simulated.

## Open questions (for the group)
- Which nodes have an IMU / INS, and can a modem read it at symbol rate with known latency?
- Header budget: how many bits per packet for side information are acceptable?
- Is transmit-side pre-compensation (applying the estimate) in scope next to side information (sending it)?
- Metric for the paper: SIR and coded PER (decided: SIR first, PER next — now more urgent because of Lee et al. 2026).

## Layout
```
README.md            this plan (estimation focus, 12 Oct 2026)
followup_bandit.md   earlier follow-up idea (learning link adaptation), not part of the current plan
project_idea/        group-discussion deck doppler_estimation.tex/.pdf
  archive/           earlier decks, READMEs and plans: compensation (9 Oct), forecast-based design (8-9 Oct), bandit
literature/
  estimation_pivot_local_check.md   gap check for E1/E2 (local, web, OpenAlex) and notes on Lee et al. 2026
  prior_art_summary.md, prior_art_notes/   prior-art search of 8 Oct 2026 (forecast design; two-way fusion, Block 3)
  baseline_cho_kim_2026.md, g1_..., g5_... earlier checks (compensation phase)
code/                Python package uwofdm (motion, IMU model, OFDM, Doppler warp, channel, Bellhop, theory,
                     placement evaluator, experiment harness); PLAN.md, RESULTS.md, DESERT_PLAN.md
```

## Status
- **12 Oct 2026 (evening):** fair signal-only baselines built (tuned Kalman, partial FFT, ICI-aware, pilot tracking,
  numerology); realistic IMU model; sparse channel estimate (removed a 4 dB loss); T2 and T3; Phase 4 complete; the
  single-path sensor gain vanished against the fair baselines. Reruns (PER, sea states, numerology, multipath) on
  hold until the framing decision. Experiments run in tmux (`code/run_job.sh`). Study guide: [`STUDY_GUIDE.md`](STUDY_GUIDE.md).
- **Done before (9–11 Oct, compensation phase, all reusable):** Phase 0 on real buoy motion; models (motion, IMU,
  OFDM, wideband Doppler, multipath, Bellhop with surface loss and scattered power); closed forms from estimation
  error to interference, validated; placement evaluator and maps; three link scenarios, sweeps and robustness checks.
- **History:** 8 Oct forecast-driven link adaptation with a bandit; 9 Oct forecast-based pre-compensation; 9 Oct
  compensation focus after Phase 0; 12 Oct estimation focus. Earlier versions in `project_idea/archive/`.

## Data and literature
- **Motion:** CDIP wave buoys (15 MB) and OOI surface moorings (144 MB) in `code/data/buoy/`.
- **Channels:** Bellhop (binary in the DESERT build); Underwater Acoustic Channel Library (Zenodo, only if needed).
- **Literature:** `~/Underwater/papers/` (`python3 tools/search.py ...`); project notes in [`literature/`](literature/);
  Lee et al. 2026 at `~/Underwater/ssrn-7134057.pdf`.
