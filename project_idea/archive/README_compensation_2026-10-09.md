# Doppler compensation for underwater acoustic links

**Target:** IEEE JSAC special issue *Next-Generation Underwater Acoustic Communication Systems*, deadline
31 March 2027. **Roadmap:** [`project_idea/doppler_compensation.pdf`](project_idea/doppler_compensation.pdf)
(group-discussion deck, 9 Oct 2026); this README summarises it. Code plan: [`code/PLAN.md`](code/PLAN.md).

**Ground rule.** The contribution is the *compensation*: where, when and how a known Doppler is removed, how much of
it remains, and at what cost. Doppler *estimation* is taken as good: it enters the work only as an input model (error
level and age of the estimate), and no new estimator is developed.

## Main idea
A good Doppler estimate is not enough: it has to be applied at the right place and in time. Three facts shape how:
- **A good estimate goes stale in about a second.** Measured on real buoy motion (Phase 0, below): wave-driven
  velocity cannot be predicted across a 2–3 s acoustic round trip, but is well predicted below 1 s.
- **Fast motion is measurable locally.** Each node can measure its own motion live (IMU, at the instant it
  transmits or receives); only the slow part (geometry, drift, path direction) needs to cross the link.
- **Several Doppler at once** (several users, several paths) cannot all be removed by one resampling at the receiver.

The proposal is **split compensation**: each end removes the Doppler of its own motion, projected on the path, at
the instant it transmits or receives; the slow, shared part (relative geometry and drift) is exchanged over the link
and predicted; the receiver removes only what remains, inside the packet. In multi-user links this assigns each
Doppler component to the node that knows it: each user removes its own motion before sending; the gateway removes its
own motion, per user, from its IMU and the known directions, instead of estimating every user's Doppler.

**Candidate claim:** a design and analysis of Doppler compensation for underwater acoustic links that decides, for
each component of the Doppler, *who* removes it (transmitter, receiver or both) and *how* (resampling along a
trajectory, frequency correction, per-user branches), from the accuracy and the age of the estimates available at
that node. Supported by real motion data and closed-form residual bounds, and shown on single links and multi-user
uplinks and downlinks.

**Scope:** OFDM links (single carrier as an extension); mobile or surface-coupled nodes; one link, or K nodes and a
gateway (e.g. a buoy with seabed nodes or AUVs); only the bulk (common) Doppler of each link is compensated by
resampling, path-specific Doppler is left to the equaliser and counted as residual.

## Phase 0 evidence: how fast a good estimate goes stale
Real motion of moored surface buoys, from [`code/RESULTS.md`](code/RESULTS.md) (Phase 0) and `code/results/`:
- 0.9 m CDIP Waverider, three Pacific sites, and 3.3 m / 4 t OOI buoys, Oregon and Irminger Sea; swell, mixed and
  wind-sea days.
- Best linear prediction of the velocity from noisy per-packet measurements every 2.3–2.4 s, 2.3–2.4 s ahead:
  normalised error 0.76–1.0 (1 = no better than the mean); 70–95% of the velocity variance is in waves shorter than
  10 s, even on swell days. With measurements every 0.8 s, the error is 0.27–0.47 at 0.8 s ahead.
- Consequence: for surface nodes, an estimate of the *far end's* fast motion cannot be carried across a round trip;
  compensation of fast motion has to happen where it is measured, at the instant it is measured.

## Gaps with the literature
| | Gap | Closest work |
|---|---|---|
| G1 | *Partly open* ([check](literature/g1_intrapacket_compensation_check.md)): the residual from a constant-scale error and the interference from Doppler scaling or a linear frequency drift are analysed, and thresholds exist; no rule combines estimate errors, trajectory model order and packet length with real motion statistics. Supporting result, not the headline | Mason et al. 2008 (scale-estimation accuracy vs performance); Stojanovic, Djurovic, Djukanovic 2012 (exact ICI of Doppler scaling); García–Oberli 2009, Tanash et al. 2026 (radio, closed forms); Stojanovic 2006, Yerramalli–Mitra 2011 (thresholds); compensators: Sharif 2000, Li 2008, Abdelkareem 2016, Yoshizawa 2018, Cho–Kim 2026 ([comparison](literature/baseline_cho_kim_2026.md)) |
| G2 | No UWA transmitter compensates Doppler; where to compensate (transmitter, receiver, split) as a function of the age of the estimate is not analysed | Checked in the library, on the web, Scholar and 159 citing works ([note](literature/g5_web_scholar_check.md)); 3GPP NTN (transmit-side pre-compensation from ephemeris); Cuji Dután 2025 (stale feedback at the transmitter, for beamforming). In UWA papers "Doppler pre-compensation" means a receiver step: always say "transmit-side" |
| G3 | Several Doppler at once are handled only at the receiver, with one resampling branch per user; no split of the compensation between users and gateway; guard bands follow mobility | Tu, Duman, Stojanovic, Proakis 2011 (OFDMA) and JOE 2013 (multiple resampling); Daoud et al. 2015; Pottier et al. 2024; Zhang et al. 2026; Hasan et al. 2025 (guard bands from Doppler) |
| G4 | Motion sensors drive navigation from modem Doppler, not compensation; sensor-driven compensation appears only in one technical report, at one end | Oiwa et al. 2020 (accelerometer, IEICE Tech. Rep.); Choi et al. 2022, Aubry 2019 (modem Doppler into navigation) |

## Proposed methodology
**Model.** Relative Doppler scale of a link, ε(t) = (v_A(t) − v_B(t))·u(t)/c, with v_A, v_B the velocities of the two
transducers and u the direction of the path. It splits into a fast part (each node's wave-driven motion) and a slow
part (drift, vehicle cruise, geometry). **Estimation input model:** each node knows its own velocity from an IMU
with error σ_imu at negligible delay; the direction u and the slow part from the link with error σ_slow and age Δ;
the receiver's estimate of the residual inside the packet with error σ_rx. All errors are parameters, not designs.

1. **C1 – Compensation inside the packet (receiver).** Resampling along an estimated trajectory ε̂(t) (constant,
   linear, quadratic, piecewise) plus frequency correction, in baseband. Output: residual inter-carrier interference
   and SINR loss as functions of the estimate error, the trajectory model and the packet length; complexity.
   Baselines: one scale per packet; Cho–Kim (constant acceleration); per-symbol correction.
2. **C2 – Where to compensate: transmitter, receiver or split.** The transmitter removes its own motion, measured
   live, while it emits (real-time resampling of the outgoing waveform); the receiver removes its own motion while it
   receives; the slow part is exchanged and predicted; the receiver removes the remainder (C1). Compared with
   receiver-only compensation, as a function of estimate age, sea state and packet length, using the measured buoy
   motion as input.
3. **C3 – Several Doppler at once.**
   - *Uplink (K nodes to a gateway):* each node removes its own motion before sending (C2), so users differ only by
     the gateway's motion projected on their directions and by slow terms. A moving gateway still needs one
     resampling branch per user for its own motion, as in Tu et al., but each branch is set from its IMU and the
     known direction rather than estimated; users in similar directions can share a branch, the difference being
     left as residual. Compared with the multiple-resampling receiver of Tu et al. on residual interference between
     users and on complexity (number of branches, estimation needed per user).
   - *Downlink:* the gateway knows its own motion live, so it can pre-compensate each user's subband before
     sending, with no prediction.
   - *Guard bands* sized by the residual left after compensation, not by the raw mobility.

## Theoretical objectives
| | Result |
|---|---|
| R1 | Residual after intra-packet compensation: interference and SINR loss for OFDM as functions of estimate error, trajectory model order and packet length; design rule for the model order from the motion spectrum (building on Mason 2008, García–Oberli 2009, Tanash 2026) |
| R2 | Staleness and placement: compensation error as a function of the age of the estimate, from the motion spectrum (E[(v(t+H) − v(t))²] = 2∫S_v(f)(1 − cos 2πfH) df, and its prediction counterpart); condition under which transmit-side, receive-side or split compensation is best; validated on the CDIP and OOI records |
| R3 | Several Doppler: interference between users as a function of the residuals and guard bands; users supported vs complexity (branches) for split compensation and for multiple resampling |

R2 carries the paper's main message and is backed by real data; R1 and R3 make it a design. R1 and R2 can share one
framework: both are errors in representing v(t) from limited information (an old sample; a polynomial over the
packet), i.e. integrals of the motion spectrum S_v(f) against a kernel (proposal, to be derived). None is proved yet.

## Risks
- **Compensation is a classic topic.** Resampling, acceleration-aware and multi-branch receivers are well studied
  (G1); the novelty must lie in placement and splitting (G2, G3), backed by the staleness analysis (R2), not in a new
  resampler.
- **Sensors.** IMU errors, the lever arm between IMU and transducer, and the path direction u all enter the
  transmit-side correction; their accuracy must be modelled and stated. "Why not just an IMU?" becomes a design
  element, not an objection.
- **Path-specific Doppler** cannot be removed by one resampling per link; it stays as residual and must be counted.
- **Data.** Motion is real (buoys); the acoustic channel and multi-node geometry are simulated (Bellhop/Waymark-style),
  unless a multi-node sea-trial dataset is found. AUV and seabed scenarios need motion models or logs.
- **Not yet checked:** split or sensor-driven compensation outside the local library; full text of Mason 2008,
  Stojanovic et al. 2012, Ha et al. 2015 and Tanash et al. 2026 before writing R1.

## Open questions (for the group)
- Target scenarios: buoy gateway with seabed nodes, with AUVs, or both? Which nodes have an IMU?
- Can the modem resample the outgoing waveform in real time from the IMU, or only per packet?
- Which residual metric for the paper: SINR loss, BER/PER with a fixed code, or both?
- Is the downlink (gateway pre-compensating per user) in scope for the first paper?

## Layout
```
README.md            this plan (compensation focus, 9 Oct 2026)
followup_bandit.md   earlier follow-up idea (learning link adaptation), not part of the current plan
project_idea/        group-discussion deck doppler_compensation.tex/.pdf
  archive/           earlier decks and READMEs: forecast-based design (8-9 Oct 2026) and the bandit version
literature/
  prior_art_summary.md       prior-art search of 8 Oct 2026 (written for the forecast-based design)
  prior_art_notes/           detailed notes of that search
  baseline_cho_kim_2026.md   comparison with the acceleration-aware receiver of Cho-Kim (IET Commun. 2026)
  g5_web_scholar_check.md    web, Scholar and citation check: transmit-side and multi-user compensation (now G2, G3)
  g1_intrapacket_compensation_check.md   check of G1: residual analyses and thresholds for intra-packet compensation
code/                Python package uwofdm
  PLAN.md                        code plan: per phase, what each module and script has to do, status
  RESULTS.md                     results log of all experiments (current numbers first)
  scripts/fetch_cdip.py, motion_predictability.py   CDIP buoy data and staleness analysis
  scripts/fetch_ooi.py, ooi_motion.py               OOI large-buoy data and staleness analysis
  scripts/replay_diagnostic.py   OFDM receiver on Channel Library replay (inherited)
  src/uwofdm/                    paths, experiment runner; model.py is inherited and to be replaced
  data/, results/                downloaded data and outputs (not part of the sources)
```

## Status
- **Done (9 Oct 2026):** Phase 0 on real buoy motion (CDIP, OOI): staleness of estimates measured; it rules out the
  earlier forecast-based design for surface nodes and motivates split compensation.
- **Done (9 Oct 2026):** Phase 1 models (motion, estimates, waveform with the wideband Doppler warp, channel) and a
  first C1 study on real buoy motion (`code/results/c1_residual/`). On wave-following buoys, how often the
  compensation is updated matters more than the trajectory model: one quadratic per packet gives 27–33 dB of
  signal-to-interference at 1 s packets but 1–12 dB at 4 s; a per-symbol update holds 28–39 dB at any length. A
  closed form for the residual (spherical-Bessel kernel on the measured velocity spectrum) predicts the polynomial
  cases within 0–5 dB.
- **C2 results, three scenarios (9 Oct 2026, rerun after the packet-padding fix; `code/RESULTS.md`):** split
  compensation (each end removes its own motion with its IMU or INS, including the lever arm from the gyro; the
  receiver removes the rest from only two estimates per packet) gives 36–40 dB of signal-to-interference buoy →
  seabed, 31–38 dB buoy → buoy and 32–40 dB AUV → buoy, at 1 and 4 s packets, with a 0.01 m/s IMU; a receiver that
  updates every symbol gives 22–36 dB; stale estimates fail (−10 to 0 dB); compensating at one end only fails when
  both ends move; without lever-arm correction the IMU-based strategies fail. Closed forms agree with simulation
  within about −1 to 6 dB for split.
- **Decided (9 Oct 2026):** main message C2 (placement / split); a node's own IMU counts as a good estimate; Phase 3
  starts with a buoy talking to a seabed node, then buoy–buoy and buoy–AUV; SIR first, coded BER/PER later.
- **To build:** C2, C3; the evaluation with measured motion as input; the DESERT Doppler extension (see
  `code/PLAN.md`).
- **History:** 8 Oct 2026 forecast-driven link adaptation with a bandit; 9 Oct without the bandit, with transmit-side
  pre-compensation from a forecast; 9 Oct (later) compensation focus after Phase 0. Earlier versions are in
  `project_idea/archive/`.

## Data and literature
- **Motion:** CDIP wave buoys (THREDDS, 15 MB) and OOI surface moorings (raw-data server, 144 MB) in
  `code/data/buoy/`; scripts in `code/scripts/`.
- **Channels:** Bellhop (binary in the DESERT build) for geometry-driven channels; Underwater Acoustic Channel Library
  (Zenodo 10.5281/zenodo.21287414, 24 GB, only a few records if needed) for real one-way channels.
- **Literature:** `~/Underwater/papers/` (searchable database; `python3 tools/search.py ...`); project notes in
  [`literature/`](literature/).
