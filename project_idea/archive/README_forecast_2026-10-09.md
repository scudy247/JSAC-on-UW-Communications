> **Archived 9 Oct 2026.** Forecast-based design, superseded by the compensation-focused [`../../README.md`](../../README.md) after the Phase 0 measurements (`../../code/PLAN.md`). Links below are as they were; the deck it describes is [`doppler_link_adaptation_nomab.pdf`](doppler_link_adaptation_nomab.pdf) in this folder.

# Doppler-aided link adaptation for underwater acoustic links

**Target:** IEEE JSAC special issue *Next-Generation Underwater Acoustic Communication Systems*, deadline
31 March 2027. **Roadmap:** [`project_idea/doppler_link_adaptation_nomab.pdf`](project_idea/doppler_link_adaptation_nomab.pdf)
(group-discussion deck, 8 Oct 2026, revised after the internal review and the prior-art search on 9 Oct 2026); this
README summarises it. The contextual bandit is left to a follow-up paper: [`followup_bandit.md`](followup_bandit.md).

## Main idea
Every packet, data or ACK, carries a free measurement of the Doppler scale ε = v_rel / c, the same at both ends of
a common path. Both ends fuse these measurements, and the nodes' own motion sensors when they have them, into one
**kinematic forecast** of the drift d = dε/dt and its uncertainty. The forecast drives four blocks of the chain:
- **frame:** extra pilots placed where the forecast is uncertain, on top of a coherence-limited grid; packet length
  and modulation;
- **receiver:** compensation pre-warped along the forecast, tracking only its error;
- **transmitter:** the bulk Doppler pre-compensated before sending, so that several nodes with different Doppler can
  share the uplink to a gateway at the same time;
- **decision:** a threshold rule on the forecast-predicted PER, with a margin calibrated from delayed ACKs; in the
  uplink, the forecast uncertainty also sets each user's guard band and who may transmit concurrently.

The framework is written for K ≥ 1 nodes sending to one gateway; the point-to-point link is the case K = 1.

**Candidate claim:** kinematic prediction has driven frame design and reception in radio ISAC (Yuan et al. 2021),
and Kalman error covariance has driven pilot placement (Simeone–Spagnolini 2004), but underwater links must bridge
second-scale round trips with no external sensing. We build one forecast of the swell-driven Doppler drift from the
link's own two-way measurements, fused at both ends, and use it where no per-packet estimate can help: at the
transmitter, which has no current Doppler measurement. Pre-compensating each node's bulk Doppler lets several nodes
share the uplink, with guard bands and admission set by the forecast uncertainty; the same forecast pre-warps the
receiver, places extra training symbols and chooses packet length and modulation, with a PER safety layer
calibrated from delayed ACKs. Transmit pre-compensation from a motion model is known in 3GPP NTN (from satellite
ephemeris); the contribution is the underwater instance, where the motion model must be learned from the link, and
its analysis (R1–R4).

**Scope:** delay-tolerant traffic; mobile or surface-coupled nodes; seas with a swell component; paths resolvable
in delay or Doppler; one link, or an uplink from K nodes (seabed sensors, AUVs) to a gateway (e.g. a buoy). On
still, fixed links the system falls back to the baseline ("no harm" once overheads are counted is still to be shown).

## Gaps with the literature
Reworded after the prior-art search ([`literature/prior_art_summary.md`](literature/prior_art_summary.md)): no work
combines even two of the blocks, but each block has partial precedent. G5 was checked on 9 Oct 2026 in the local
library, on the web, on Google Scholar (until a CAPTCHA stopped it) and through the OpenAlex citation lists:
[`literature/g5_web_scholar_check.md`](literature/g5_web_scholar_check.md).

| | Gap | Closest work |
|---|---|---|
| G1 | Pilot density is adapted to mobility *statistics* and UWA frames adapt numerology to measured spreads or segment size to predicted SNR; none places training within a packet from the error variance of a kinematic Doppler forecast | Zakharov et al. 2019 (offline, for a motion range); Simeone–Spagnolini 2004 (Kalman-covariance pilots, radio); NR DMRS, 802.11bd; Mangione 2021; Kim–Park–Im 2026; Van der Werf et al. 2024 (OTFS) |
| G2 | Kinematic Doppler models are confined to one packet by design, and inter-packet information is used only as a heuristic gating window; no receiver carries a state-space Doppler forecast with uncertainty across packets | Abdelkareem 2012/2016 (previous packet's acceleration as search range); Cho–Kim 2026 and Hu et al. 2024 (acceleration within one frame); Ceballos Carrascosa–Stojanovic 2010; Thales US 8,023,596; Oiwa et al. 2020 (accelerometer-aided compensation, technical report); 3GPP NTN pre-compensation (external kinematics) |
| G3 | Doppler measured at both ends is exchanged and averaged for time sync, and modem Doppler feeds navigation filters one way; neither tracks the two-way series as a kinematic process nor uses it for the link | DE-Sync 2018, APE-Sync 2019, DA-Sync; Aubry 2019, Choi et al. 2022; Li et al. arXiv 2606.16717 (wave-driven state model, RF) |
| G4 | *Follow-up paper*, see [`followup_bandit.md`](followup_bandit.md) | |
| G5 | No UWA transmitter pre-compensates its own Doppler; per-user Doppler in multi-user links is separated at the receiver, with one resampling branch per user; transmitter-side allocation uses predicted channel gains, not Doppler; guard bands follow mobility, not a forecast | Tu, Duman, Stojanovic, Proakis 2011 (OFDMA, one resampling branch per user) and JOE 2013 (multiple resampling); Daoud et al. 2015; Pottier et al. 2024; Li et al. 2020; Zhang et al. 2026; downlink OFDMA with outdated or predicted CSI (2018, 2019, 2025); Hasan et al. 2025 (guard bands from Doppler); Cuji Dután 2025 (stale feedback at the transmitter, solved with slowly varying angles); 3GPP NTN (UE pre-compensation from ephemeris). Wording: UWA papers use "Doppler pre-compensation" for a *receiver-side* coarse correction, so always say "transmit-side" |

Closest precedent for the decision rule: Park et al., Remote Sens. 2026 (prediction instead of feedback, Doppler as a
table index, uncertainty margin, PER ≤ 0.05 target; adapts Raptor overhead only).

## Proposed methodology
**Network:** K ≥ 1 nodes send to one gateway; with K = 1 this is the point-to-point link between nodes A and B used
in Blocks 1–3. Action per packet: a = (M, N_sym, 𝒫): modulation (QPSK or 16-QAM), OFDM symbols per packet,
positions of the full training symbols; in the uplink, each node k also gets a subband S_k and a guard band g_k, or
is deferred. Symbol duration, cyclic prefix and code are fixed in the first paper; waiting time is set by the MAC.

1. **Block 1 – forecast-aided receiver.** Node B resamples along the forecast ε̂(t) over the whole packet; the
   tracker sees only the forecast error, so PER depends on the forecast uncertainty σ_e rather than on the drift.
2. **Block 2 – pilot placement.** Training symbols also estimate the channel taps, so 𝒫 always contains a uniform
   grid 𝒫₀ at the largest spacing K_max that the tap coherence time (Doppler spread) allows. The forecast places
   only the *extra* pilots, to minimise the forecast-error variance over the packet under a pilot budget. For a
   swell, extra pilots are expected to cluster around the zero crossings of d, where the jerk peaks. K_max depends
   on the receiver: comb pilots in every OFDM symbol or decision-directed tracking relax it. With K_max ≥ N_sym the
   grid is empty and the problem reduces to the unconstrained placement, so the constraint never makes the
   formulation worse. Greedy placement has no guarantee for free: see Risks.
3. **Block 3 – two-way Kalman filter.** A Kalman filter on a drift-plus-damped-oscillators model (EKF while the sea
   parameters are learned) fuses ε̂ measured by node A on ACKs and by node B on data (returned in the ACK header,
   ≈ 20 bits). The largest sampling gap drops from 2τ + T_a to ≈ τ + T_a: for an 8 s swell and 1 s packets, the
   no-aliasing range grows from 2.25 to 4.5 km.
   - *Ordering:* B's sample arrives with A's own sample on the same ACK and is older; in stop-and-wait both are
     processed in time order. Out-of-sequence updates are needed only with pipelined packets.
   - *Measurement model:* ε̂_j(t_k) = ε(t_k) + b_j(t_k) + w_{j,k}, w_{j,k} ~ N(0, R_j), for direction j ∈ {A, B}.
     ε is the common bulk state; b_j is a slowly varying bias of the path that direction j locks onto (a state);
     R_j is a per-direction noise, adapted from the innovations. Each direction may lock onto a different path
     (beam patterns, depths, surface change over τ), and the locked path can switch over time.
   - *Path switches:* innovation gating flags samples far from the prediction; on a flagged jump, b_j is reset or
     its covariance inflated while the bulk state is kept; path-resolved estimates (delay of the locked arrival)
     are used where the receiver provides them. First check: how often ε̂ jumps on Channel Library records.
   - *Motion sensors, when available:* a node with an IMU or GNSS feeds its own measured motion to the filter as an
     extra measurement of its own component; the projection onto the propagation path is a state learned from the
     link Doppler. Link Doppler still supplies what the sensor cannot: the far end's motion and the projection on
     the path actually used. Nodes without sensors run on link Doppler alone, so the comparison "link only vs
     sensors only vs both" is part of the evaluation.
   - *Uplink sampling:* the gateway's periodic beacon gives every node a Doppler sample at the same time; the
     gateway measures each uplink packet and returns each node's value in the next beacon (≈ 20 bits per node).
4. **Block 4 – transmitter pre-compensation.** Node k resamples its own transmitted signal along its forecast of
   ε_k over the receive interval, so the gateway receives only the residual e_k. Here a forecast is necessary, not
   just helpful: the newest measurement at the transmitter is at least τ old (at 2 km, 2–3 s from beacon to
   arrival, about a third of an 8 s swell period). With K nodes in separate subbands, a receiver can separate the users
   only with one resampling branch per user (Tu et al. 2011, 2013), and inter-user interference remains;
   pre-compensation at the source removes the bulk part before it arises, so the gateway uses one FFT and tracks
   each user's residual. The multiple-resampling receiver is the baseline, for interference and for complexity. Rough estimate: two users 0.45 m/s apart
   at 20 kHz are offset by ≈ 6 Hz (half a subcarrier); a forecast error of 0.05 m/s leaves ≈ 0.7 Hz, interference
   near −18 dB. Only the bulk Doppler can be pre-compensated; path-specific Doppler remains. With K = 1,
   pre-compensation is an alternative or a complement to Block 1.
5. **Decision rule and safety layer.** Predicted PER per action: the probability, under the forecast distribution
   of e over the receive interval, that the residual leaves the calibrated tolerance. Safe set: actions with
   predicted PER ≤ α − m_t, where α is the PER target and m_t a margin updated from delayed ACKs (outer-loop / ACI
   style). Choice: the highest expected goodput B_a (1 − p̂_fail) / T_a in the safe set; if it is empty, fall back,
   wait or drop. The PER model is not learned online in this paper.
   - *Uplink:* each node's forecast uncertainty σ_e,k sets its guard band g_k: the frequency offset its residual
     can reach at the target confidence over the receive interval. Nodes whose required guard band would cost more
     than it is worth are deferred to a later slot. The gateway chooses subbands, guard bands and admitted nodes to
     maximise expected sum goodput under each node's PER target.

Failure model: the receiver turns the forecast error e = d − d̂ into a residual impairment φ_a(e);
SNR_eff = SNR / (1 + SNR·φ_a(e)). In the uplink, the impairment also includes the interference from the other
users' residuals across the guard bands. Only the common (bulk) Doppler component is modelled at first.

## Theoretical objectives
| | Result |
|---|---|
| R1 | Predictability: how long Doppler stays predictable (≈ 13 s for an 8 s swell vs ≈ 1 s for fading, per the idea document); graceful degradation in mixed seas |
| R2 | Sampling: forecast error vs range, one- vs two-way; observability with delayed, irregular samples and per-direction bias; gain from motion sensors when present |
| R3 | Pilot placement: optimal extra positions on top of the grid 𝒫₀ for one swell component, in closed form; in general, a greedy bound via weak submodularity or curvature, or Jawaid–Smith-type conditions checked for the model |
| R4 | Concurrency: inter-user interference as a function of the forecast residuals and guard bands; how many nodes can share the uplink at a given range, sea state and PER target, compared with the multiple-resampling receiver of Tu et al. (contiguous and interleaved allocation) |

R1 and R2 carry the JSAC paper, with R4 as the main application; R3 if the single-swell case closes. Supporting:
long-run PER guarantee and stability of the safety layer (known structure: OLLA / ACI). Extensions: quiet windows,
range of Doppler vs SNR feedback, path separation. None of the results is proved yet.

## Risks (from the prior-art search)
- **R3 greedy bound.** Kalman sensor-scheduling costs are not submodular in general (Jawaid–Smith, Automatica 2015),
  so the Krause et al. 2008 bound does not apply. Routes: prove the single-swell case directly, bound the
  submodularity ratio (Hashemi et al., TAC 2020), or check Jawaid–Smith conditions for the drift-plus-oscillator model.
- **Block 2 gain may be small.** Periodic pilots are MSE-optimal for a stationary Gauss–Markov channel
  (Dong–Tong–Sadler 2002). Extra pilots help only where σ_e²(u) is non-stationary within the packet; state this as a
  precondition rather than discover it in simulation.
- **Block 1 gain may be small.** A 10.5 h surface-buoy trial found that per-symbol Doppler refinement lowers PER by
  at most 1.15× (Lee, Park, Kim, Im, SSRN 2026), and Cho–Kim 2026 restore near-ideal BER for ≈ 1 s frames at no
  overhead. Block 1 must be shown in a regime defined in advance (long packets, strong swell, few pilots, low SNR),
  against acceleration-aware baselines (Zakharov 2019, Cho–Kim 2026, Hu 2024); see
  [`literature/baseline_cho_kim_2026.md`](literature/baseline_cho_kim_2026.md).
- **Block 4 removes only the bulk Doppler.** Path-specific Doppler stays (Zhang et al. 2026 make this point for
  concurrent access); the residual interference must be modelled, not assumed away.
- **No real data for the uplink.** The Channel Library is one-way and single-link; the two-way and multi-node parts
  need simulation (Waymark/Bellhop trajectories, as in Cho–Kim 2026) unless a multi-node sea-trial dataset is found.
- **"Why not an IMU?"** Reviewers will ask; the answer is the sensor comparison in Block 3, which must show what
  link Doppler adds over the sensors.
- **"Obvious" framing.** DE-Sync's Doppler exchange, navigation filters fed by modem Doppler and NTN pre-compensation
  can make closing the loop look obvious; the defence is the analysis (R1–R4), not the plumbing.
- **Not yet checked:** Google Scholar cited-by of DE-Sync, APE-Sync, DA-Sync and Yuan et al. 2021 (priority 1 in the
  prior-art summary); Scholar cited-by for G5 (blocked by a CAPTCHA; OpenAlex lists used instead); full text of Tu et
  al. 2011 and of the Cuji Dután 2025 thesis.

## Open questions (for the group)
- First test: does a forecast prior still help acceleration-aware receivers, in PER or in pilots saved (see Risks)?
- Bandit moved to a follow-up: is a calibrated threshold rule enough for the JSAC paper?
- Which receiver structure sets K_max: comb pilots in every symbol, or full training symbols only?
- Can the modem or simulator carry ≈ 20 header bits for the forecast and the far end's Doppler, and K such fields
  in the gateway beacon?
- Target uplink scenario: a buoy gateway with seabed nodes, AUVs, or both? Which nodes have an IMU or GNSS?
- May the policy abstain, drop or fall back, and how is service counted?

## Layout
```
README.md            this plan (the no-bandit version, formerly README_nomab.md)
followup_bandit.md   the contextual bandit (G4, result F1), moved to a follow-up paper
project_idea/        group-discussion deck doppler_link_adaptation_nomab.tex/.pdf
  archive/           superseded deck with the bandit, and the README that described it
literature/
  prior_art_summary.md   prior-art search (8 Oct 2026): verdict per block, gap rewording, deck errors, risks
  prior_art_notes/       detailed notes: one per block, whole system, local-library sweep, gap check
  baseline_cho_kim_2026.md   comparison with the acceleration-aware receiver baseline (IET Commun. 2026)
  g5_web_scholar_check.md    web, Scholar and citation check of G5 (transmit-side pre-compensation, multi-user)
code/                starting points, inherited from an earlier project (package name uwofdm)
  PLAN.md                        code plan: phases, modules, tests, timeline, DESERT decision
  scripts/replay_diagnostic.py   OFDM receiver on Channel Library replay, with Doppler estimation
  src/uwofdm/model.py            old band/MCS impairment model; only the EESM + logistic PER mapping carries over
  src/uwofdm/paths.py, run.py    data/results locations, YAML experiment runner
  experiments/                   replay-diagnostic configs (Blue, Red)
  pyproject.toml, requirements-lock.txt, setup_server.sh, tests/
```
The literature notes refer to `README_nomab.md`, which is now this file. They predate Block 4 and the uplink.

## Status
- **Exists:** replay of public channels; OFDM reception with one common-Doppler estimate per frame (pre/postamble
  timing; only its median and max are saved, not the per-frame series); EESM/PER mapping (`Model.goodput_mcs`).
- **To build:** receiver calibration (forecast error → PER for blockwise, tracking, acceleration-aware and
  forecast-aided OFDM); Blocks 1–4; the decision rule and safety layer; the two-way closed-loop simulation; a
  multi-node uplink simulation.
- **Audited (8 Oct 2026):** stale references fixed in `pyproject.toml`, `setup_server.sh` (it also changed to the
  wrong directory) and docstrings. `model.py` is still the old subcarrier-selection model (θ, bands, certificate
  bounds); to be replaced by a slim PER module when the receiver calibration starts.
- **9 Oct 2026:** gaps, claim, related work and risks updated from the prior-art search, in this README and the deck.
  Later the same day: transmitter pre-compensation and the multi-user uplink (Block 4, G5, R4), guard bands and
  admission from σ_e, optional motion sensors in Block 3; the point-to-point link is now the case K = 1.

## Data and literature
- **Underwater Acoustic Channel Library** (keeps the bulk Doppler drift; one-way, ≈ 50–60 s records): Zenodo
  10.5281/zenodo.21287414, into `$UWOFDM_DATA/channel_library/`; replay toolbox github.com/uwa-channels (MIT).
  Results go to `$UWOFDM_RESULTS`.
- Other planned inputs: Qarabaqi–Stojanovic simulator, NOAA NDBC wave spectra, Waymark simulator (Liu–Zakharov–Chen,
  TVT 2012; Bellhop-based, arbitrary trajectories) for swell-driven and multi-node scenarios.
- **Literature:** `~/Underwater/papers/` (searchable database; `python3 tools/search.py ...`); prior art for this
  project in [`literature/`](literature/).

## Build the roadmap
```bash
cd project_idea && pdflatex doppler_link_adaptation_nomab.tex && pdflatex doppler_link_adaptation_nomab.tex
```
