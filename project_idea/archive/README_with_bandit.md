> **Archived 9 Oct 2026.** Superseded by [`../../README.md`](../../README.md) (no-bandit plan) and [`../../followup_bandit.md`](../../followup_bandit.md). Paths below are as they were on 8 Oct 2026; the deck it describes is [`doppler_link_adaptation.pdf`](doppler_link_adaptation.pdf) in this folder.

# Doppler-aided link adaptation for underwater acoustic links

**Target:** IEEE JSAC special issue *Next-Generation Underwater Acoustic Communication Systems*, deadline
31 March 2027. **Roadmap:** [`project_idea/doppler_link_adaptation.pdf`](project_idea/doppler_link_adaptation.pdf)
(group-discussion deck, 8 Oct 2026); this README summarises it.

## Main idea
Every packet, data or ACK, carries a free measurement of the Doppler scale ε = v_rel / c, the same at both ends of
a common path. Both ends fuse these measurements into one **kinematic forecast** of the drift d = dε/dt and its
uncertainty. The forecast drives three blocks of the chain:
- **frame:** pilots placed where the forecast is uncertain; packet length and modulation;
- **receiver:** compensation pre-warped along the forecast, tracking only its error;
- **learner:** a contextual bandit that calibrates PER online under delayed ACKs.

**Candidate claim:** one kinematic forecast, shared by both ends, drives frame design and receiver compensation,
calibrated online; PER control is a safety layer, not the headline.

**Scope:** delay-tolerant traffic; mobile or surface-coupled nodes; seas with a swell component; paths resolvable
in delay or Doppler. On still, fixed links the system falls back to the baseline ("no harm" once overheads are
counted is still to be shown).

## Gaps with the literature
| | Gap | Closest work |
|---|---|---|
| G1 | Pilot layout and packet length are fixed offline for an assumed motion range | Zakharov et al. 2019; Hong et al. 2025; OTFS pilot design (Van der Werf et al. 2024) |
| G2 | Receivers re-estimate Doppler on every packet, without a kinematic prior | Doppler predicted across OFDM blocks inside the receiver (Ceballos Carrascosa–Stojanovic 2010) |
| G3 | Doppler measured at both ends is not fused for the link | Two-way Doppler for time sync and ranging only (UWGS 2024, geo-Doppler 2026) |
| G4 | Learning link adaptation ignores kinematic context and arm-dependent feedback delay | Contextual bandits for band/relay selection (Schutz et al. 2023); Wu–Chitre–Anjangi 2025; AoI bandit (Busacca et al. 2025) |

## Proposed methodology
Action per packet: a = (M, N_sym, 𝒫): modulation (QPSK or 16-QAM), OFDM symbols per packet, positions of the full
training symbols. Symbol duration, cyclic prefix and code are fixed in the first paper; waiting time is set by the MAC.

1. **Block 1 – forecast-aided receiver.** Node B resamples along the forecast ε̂(t) over the whole packet; the
   tracker sees only the forecast error, so PER depends on the forecast uncertainty σ_e rather than on the drift.
2. **Block 2 – pilot placement.** Training symbols also estimate the channel taps, so 𝒫 always contains a uniform
   grid 𝒫₀ at the largest spacing K_max that the tap coherence time (Doppler spread) allows. The forecast places
   only the *extra* pilots, to minimise the forecast-error variance over the packet under a pilot budget (greedy,
   with a bound if the objective is submodular). For a swell, extra pilots are expected to cluster around the zero
   crossings of d, where the jerk peaks. K_max depends on the receiver: comb pilots in every OFDM symbol or
   decision-directed tracking relax it. With K_max ≥ N_sym the grid is empty and the problem reduces to the
   unconstrained placement, so the constraint never makes the formulation worse.
3. **Block 3 – two-way Kalman filter.** A Kalman filter on a drift-plus-damped-oscillators model (EKF while the sea
   parameters are learned) fuses ε̂ measured by node A on ACKs and by node B on data (returned in the ACK header,
   ≈ 20 bits). The largest sampling gap drops from 2τ + T_a to ≈ τ + T_a: for an 8 s swell and 1 s packets, the
   no-aliasing range grows from 2.25 to 4.5 km.
4. **Block 4 – physics-informed contextual bandit.** Context: forecast (d̂, σ_e), estimated SNR, range. Arms: (M,
   N_sym), with 𝒫 from Block 2. Reward: delivered bits per unit airtime. PER model generalised linear in physical
   features, initialised from receiver calibration and learned online. Feedback delay depends on the arm.
5. **Safety layer.** The bandit plays only arms whose upper-bound PER, plus a margin m_t updated from delayed ACKs
   (outer-loop / ACI style), meets the PER target α.

Failure model: the receiver turns the forecast error e = d − d̂ into a residual impairment φ_a(e);
SNR_eff = SNR / (1 + SNR·φ_a(e)). Only the common (bulk) Doppler component is modelled at first.

## Theoretical objectives
| | Result |
|---|---|
| R1 | Predictability: how long Doppler stays predictable (≈ 13 s for an 8 s swell vs ≈ 1 s for fading, per the idea document); graceful degradation in mixed seas |
| R2 | Sampling: forecast error vs range, one- vs two-way; observability with delayed, irregular samples |
| R3 | Pilot placement: optimal extra positions on top of the grid 𝒫₀ for one swell component; greedy with a bound in general |
| R4 | Learning: regret of the physics-informed bandit with arm-dependent feedback delay |

Supporting: long-run PER guarantee and stability of the safety layer (known structure: OLLA / ACI). Extensions:
quiet windows, range of Doppler vs SNR feedback, path separation. None of the results is proved yet.

## Open questions (for the group)
- First test: does a forecast prior still help acceleration-aware receivers, in PER or in pilots saved?
- One paper, or the bandit (Block 4) as a follow-up?
- Can the modem or simulator carry ≈ 20 header bits for the forecast and the far end's Doppler?
- May the policy abstain, drop or fall back, and how is service counted?

## Layout
```
project_idea/      group-discussion decks (LaTeX Beamer source and PDF; _nomab: variant in README_nomab.md)
code/              starting points, inherited from an earlier project (package name uwofdm)
  scripts/replay_diagnostic.py   OFDM receiver on Channel Library replay, with Doppler estimation
  src/uwofdm/model.py            old band/MCS impairment model; only the EESM + logistic PER mapping carries over
  src/uwofdm/paths.py, run.py    data/results locations, YAML experiment runner
  experiments/                   replay-diagnostic configs (Blue, Red)
  pyproject.toml, requirements-lock.txt, setup_server.sh, tests/
```

## Status
- **Exists:** replay of public channels; OFDM reception with one common-Doppler estimate per frame (pre/postamble
  timing; only its median and max are saved, not the per-frame series); EESM/PER mapping (`Model.goodput_mcs`).
- **To build:** receiver calibration (forecast error → PER for blockwise, tracking, acceleration-aware and
  forecast-aided OFDM); Blocks 1–4; the safety layer; the two-way closed-loop simulation.
- **Audited (8 Oct 2026):** stale references fixed in `pyproject.toml`, `setup_server.sh` (it also changed to the
  wrong directory) and docstrings. `model.py` is still the old subcarrier-selection model (θ, bands, certificate
  bounds); to be replaced by a slim PER module when the receiver calibration starts.

## Data and literature
- **Underwater Acoustic Channel Library** (keeps the bulk Doppler drift; one-way, ≈ 50–60 s records): Zenodo
  10.5281/zenodo.21287414, into `$UWOFDM_DATA/channel_library/`; replay toolbox github.com/uwa-channels (MIT).
  Results go to `$UWOFDM_RESULTS`.
- Other planned inputs: Qarabaqi–Stojanovic simulator, NOAA NDBC wave spectra.
- **Literature:** `~/Underwater/papers/` (searchable database; `python3 tools/search.py ...`).

## Build the roadmap
```bash
cd project_idea && pdflatex doppler_link_adaptation.tex && pdflatex doppler_link_adaptation.tex
```
