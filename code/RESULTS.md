# Results log

Findings of the experiments in `code/scripts/`, newest sections last; outputs in `code/results/<experiment>/`. The plan
of what to build is in [`PLAN.md`](PLAN.md). Moved here from PLAN.md on 9 Oct 2026 (text unchanged).

## Current numbers (rerun 9 Oct 2026, after the genie fix)
Same setting as the table below (three days, 1 and 4 s packets, receiver estimate error 3e-6, IMU 0.01 m/s, lever arm
corrected); simulated SIR with the corrected genie (ML phase ramp). Results above ~30 dB moved by +0.3–0.4 dB, those
below 25 dB by +3–5 dB; no ranking changed.

| Strategy | Buoy → seabed | Buoy → buoy | AUV → buoy |
|---|---|---|---|
| Receiver, stale estimate (1 s) | −5 to 6 dB | −5 to 2 dB | −4 to 4 dB |
| Receiver, quadratic per packet, 1 s / 4 s | up to 36 / down to 5 dB | up to 34 / down to 2 dB | up to 35 / down to 4 dB |
| Receiver, per symbol | 27–36 dB | 24–33 dB | 26–35 dB |
| One end only (IMU / INS) | 30–34 dB (only one end moves) | −4 to 7 dB | −2 to 7 dB |
| Both ends (IMUs / INS), no receiver estimates | — | 28–29 dB | 28–30 dB |
| **Split** (own motion at each end + receiver line from two estimates) | **37–41 dB** | **32–38 dB** | **33–40 dB** |
| Split with a 0.05 m/s IMU | 29–40 dB | 26–38 dB | 27–36 dB |

- Sea-state sweep (15 days, 1 and 4 s): split best in all 30 cases, median 9.2 dB above receiver per symbol (1–16 dB).
- Packet sweep (buoy–buoy): split 38 → 32 dB from 0.5 to 8 s; receiver per symbol 21–22 dB; quadratic 39 → −3 dB.
- IMU sweep: IMU-only strategies degrade less than reported before (0.1 m/s: 11 instead of 5 dB); split unchanged.
- Closed form: split, IMU and quadratic within ±2–3 dB; receiver per symbol 5–7 dB pessimistic — explained below
  (interference kernel), not a measurement error.

Placement evaluator on all 85 corrected points (`scripts/c2_rule.py`, offset formula for the SIR): same winner at
82 (misses cost ≤ 0.6 dB); prediction − simulation median / |.| p90: split 0.0 / 2.0 dB, split0 −0.1 / 1.6, tx_own
−0.5 / 2.9, both −0.7 / 3.0, quadratic −0.4 / 2.3, per symbol −5.6 / 7.3 dB.

### Interference kernel (`theory.kernel_ici`)
SIR = 3 / ((pi fc e T)^2 var) treats every residual as a constant offset within the FFT window. After per-symbol
compensation the residual inside a window is a zero-mean ramp (estimate at the symbol centre): its phase is quadratic
and gives 5x (7 dB) less interference than an offset of equal variance — the 5–7 dB gap. General form: interference =
mean over FFT windows of the variance of Theta(t) = 2 pi fc int r(t) dt, r the residual after any linear estimator
(weights from its estimates to the compensation trajectory). Checked: offset limit within 0.8%, ramp limit = 1/5 of
the offset formula. The evaluator now ranks by this kernel (`sir_db`), keeping the offset formula as `sir_offset_db`.

## Current numbers (rerun 9 Oct 2026, after the packet-padding fix)  — superseded by the table above
Transmit-side pre-compensation stretches a packet and, when the Doppler stretches the signal, the end of the last
symbol was being cut; every packet is now padded with 128 ms of zeros. All four experiments were rerun; the sections
further down keep the earlier numbers for the record and are superseded by this table.

Power-averaged SIR between subcarriers after compensation, range over the three days (CDIP 029 swell, CDIP 029 wind
sea, OOI Irminger swell), 1 and 4 s packets, receiver estimate error 3e-6, IMU drift 0.01 m/s unless noted; "theory −
sim" is the closed-form prediction minus the simulation.

| Strategy | Buoy → seabed | Buoy → buoy | AUV → buoy | theory − sim |
|---|---|---|---|---|
| Receiver, stale estimate (1 s) | −10 to 0 dB | −10 to −3 dB | −9 to −2 dB | −3 to 7 dB |
| Receiver, quadratic per packet, 1 s / 4 s | 27–36 / −1–9 dB | 24–33 / −2–5 dB | 25–35 / −1–7 dB | 0–7 dB |
| Receiver, per symbol | 26–36 dB | 22–33 dB | 25–34 dB | −7 to −4 dB (kernel approximation) |
| One end only (IMU / INS) | 29–34 dB (only one end moves) | −9 to 3 dB | −30 to 1 dB | 2–6 dB (AUV cruise mean not in theory) |
| Both ends (IMUs / INS), no receiver estimates | — | 27–28 dB | 27–29 dB | −1 to 2 dB |
| **Split** (own motion at each end + receiver line from two estimates) | **36–40 dB** | **31–38 dB** | **32–40 dB** | −1 to 6 dB |
| Split with a 0.05 m/s IMU | 27–39 dB | 24–38 dB | 25–36 dB | 0–7 dB |

- Buoy → seabed on the OOI buoy: without lever-arm correction the IMU-based strategies give −1 to 10 dB; with it,
  split gives 36–38 dB.
- C1 (`c1_residual`, buoy → seabed, receiver only): piecewise per symbol 28–39 dB at any packet length; quadratic per
  packet 29–38 dB at 1 s, 10–22 dB at 2 s, 0–9 dB at 4 s; constant per packet at most 12 dB. Theory within −1 to 5 dB
  for the quadratic, 3–7 dB optimistic for orders 0–1 (strong interference), 5–8 dB pessimistic for per symbol.

## Phase 0: real motion and staleness (CDIP, OOI)
- Best linear prediction of velocity from noisy per-packet measurements every 2.3–2.4 s, 2.3–2.4 s ahead, error
  normalised by the standard deviation: 0.77–1.08 for the Waverider, 0.76–0.93 for the OOI buoys. 70–95% of the
  velocity variance is in waves shorter than 10 s, even on swell days. Holding the last value is worse than the mean
  (1.2–1.9); per-packet extrapolation v + aH is worse still.
- With measurements every 0.8 s, prediction 0.8 s ahead is good (0.27–0.47) and mostly lost by 1.6 s.
- Checks and caveats: Hs from the reconstructed OOI motion matches each buoy's wave sensor within 1–5%. Noiseless dense
  CDIP samples look almost perfectly predictable, an artifact of the buoy's onboard low-pass filter that disappears
  with measurement noise. A 0.1 Hz low-pass "large platform" stand-in was far too optimistic, as the OOI data showed.
- **Conclusion:** a good estimate of a surface node's fast motion is useful for about a second; it cannot be carried
  across a 2–3 s round trip. This is the basis of split compensation and of R2.

## Phase 2 / C1: compensation inside the packet
**First C1 results (9 Oct 2026, `results/c1_residual/`).** Buoy transmitter (CDIP 029 swell and wind sea, OOI
Irminger swell) to a static receiver 1 km away; 20 packets per case; one Doppler estimate per symbol (0.15 s);
interference between subcarriers after removing per-symbol common phase and timing (power-averaged SIR):
- One polynomial per packet degrades fast with packet length: order 2 gives 29–37 dB at 1 s, 10–22 dB at 2 s and
  0–9 dB at 4 s; order 0 is at most 12 dB, below 3 dB on the rougher days.
- Piecewise per-symbol compensation stays at 28–38 dB whatever the packet length; estimate errors of 3e-6
  (0.005 m/s) cost at most 3 dB.
- Closed form vs simulation: within 0–6 dB for the polynomial models (optimistic when interference is strong, where
  the small-offset formula weakens); 5–7 dB pessimistic for the piecewise model (kernel approximated by order 0 over
  one symbol, to be refined).
- Motion records are interpolated band-limited (`at_times`): straight lines between the 1.28 Hz CDIP samples put kinks
  in the motion that inflate residuals (rerun 9 Oct 2026; numbers above are from the corrected run).
- Design message so far: on wave-following platforms the update rate of the compensation matters more than the order
  of the trajectory model; a single trajectory per packet only works for packets of about 1 s.
- Bug found and fixed on the way: averaging per-symbol SIR ratios (instead of powers) overstated the SIR by 13–20 dB.

## Phase 3 / C2: where to compensate
### Scenario 1: buoy to seabed node
**First C2 results (9 Oct 2026, `scripts/c2_placement.py`, `results/c2_placement/`).** Buoy (CDIP 029 swell and wind
sea, OOI Irminger swell) to a seabed node 1 km away, 100 m deep; 1 and 4 s packets; 20 packets per case. Receiver
estimates error 3e-6; IMU: slow drift (Gauss–Markov, 20 s) of std 0.01 or 0.05 m/s, 50 ms latency extrapolated with
the measured acceleration; on the OOI buoy the transducer is 2 m below the motion sensor (real gyro rates).
Power-averaged SIR between subcarriers:
- Receiver only: per-symbol 26–36 dB (needs an estimate every symbol); one quadratic per packet 26–35 dB at 1 s,
  −1 to 9 dB at 4 s; a stale estimate (1 or 2.5 s old) held over the packet −9 to 0 dB.
- Transmitter only (IMU): 28–34 dB with a 0.01 m/s IMU, 13–15 dB with 0.05 m/s.
- **Split** (IMU at the transmitter + receiver line or constant from only two estimates per packet): 35–40 dB with a
  0.01 m/s IMU at 1 and 4 s, 26–38 dB with 0.05 m/s. The transmitter removes the fast wave motion, the receiver the
  slow IMU drift: their errors are complementary.
- Lever arm: without the gyro-based correction, the IMU-based strategies fail on the OOI buoy (−1 to 10 dB).
- Closed form vs simulation: within 0–4 dB in most cases, up to 6 dB in the hardest.
- Bugs and artefacts fixed on the way: linear interpolation of the 1.28 Hz CDIP motion (kinks); acceleration by
  finite differences at 1.28 Hz; IMU reading evaluated on the coarse grid. All numbers above are after the fixes.

### Scenario 2: buoy to buoy
**Scenario 2, buoy to buoy (9 Oct 2026, `scripts/c2_buoy_buoy.py`, `results/c2_buoy_buoy/`).** Two surface buoys
1 km apart, transducers 10 m deep (horizontal path); independent motions in the same sea state (CDIP: same day 3 h
apart; OOI: two bursts of the same day, 2 m lever arms with gyro correction); same estimates as scenario 1;
IMU 0.01 m/s. Power-averaged SIR:
- One end only (transmitter IMU or receiver IMU): −9 to 3 dB: the other buoy's motion is left untouched.
- Both ends remove their own motion with their IMUs, no receiver estimation at all: 26–28 dB.
- **Split** (both IMUs + receiver line or constant from two estimates per packet): 30–36 dB at 1 and 4 s, above the
  receiver updating every symbol (21–32 dB) by 4–10 dB, with far fewer receiver estimates.
- Receiver only, one quadratic per packet: 23–33 dB at 1 s, −2 to 4 dB at 4 s; stale estimates: −10 to −3 dB.
- Closed form vs simulation: 1–3 dB for both IMUs; 3–9 dB optimistic for split (cross terms and the two-point fit
  are not modelled); 4–5 dB pessimistic for the per-symbol receiver (approximate kernel, as in C1).
- Fixed on the way: `Track.velocity_at` interpolated linearly (kinks in the receiver's motion read at reception time);
  now band-limited with edge padding. The interpolator's taps are normalised to unit gain at zero frequency (the
  previous gain was 1e-4 off; negligible for the earlier results, which were not rerun).

### Scenario 3: AUV to buoy (first run, 9 Oct 2026; superseded by the rerun)
AUV (modelled: 1.5 m/s cruise, turns, speed changes; 50 m deep) to the buoy (CDIP 029 swell and wind sea, OOI Irminger
swell), 1 and 4 s packets. Before the padding fix: buoy IMU only −30 to −19 dB (AUV cruise uncompensated); AUV
navigation only −7 to 1 dB; both 25–27 dB; split 27–33 dB; receiver per symbol 23–31 dB. Split stalled near 29 dB on
the CDIP days: traced to the packet cut by pre-compensation at bulk Doppler ~1e-3 (exact pre-compensation gave only
27 dB at 1e-3), which led to the padding fix.

## Phase 3 / C2: sweeps (9 Oct 2026, `scripts/c2_sweeps.py`, `results/c2_sweeps/`)
One factor at a time around the default (IMU drift 0.01 m/s, latency 50 ms extrapolated with the measured
acceleration, 2 m lever arm corrected, 1 s packets, 1 km), 10 packets per point; single path, genie metric.
Power-averaged SIR, simulation (closed form).
- **IMU error 0.005 → 0.1 m/s:** split stays at 37–40 → 27–32 dB in all three set-ups (the receiver's line removes the
  slow IMU drift); transmitter-only and both-IMU strategies fall from 31–39 dB to 2–5 dB. Split tolerates an IMU about
  ten times worse.
- **Latency (OOI buoy → seabed), extrapolated:** split 42 / 37 / 29 / 15 dB at 0 / 50 / 100 / 200 ms (theory 39 / 37
  / 31 / 24). Without extrapolation: 22 dB at 50 ms, 15 dB at 100 ms. Requirement: latency below about 100 ms and
  extrapolation with the measured acceleration.
- **Lever arm (OOI buoy → seabed):** corrected, split 41–38 dB from 0 to 2 m, 33 dB at 5 m; uncorrected, 22 dB already
  at 0.5 m and 1.5 dB at 5 m. The gyro-based correction is mandatory (theory optimistic when uncorrected: cross terms).
- **Packet length (OOI buoy → buoy):** split 38 → 32 dB from 0.5 to 8 s; receiver per symbol about 19 dB at any length
  (hard OOI day); receiver quadratic per packet 38 dB at 0.5 s, 20 dB at 1 s, below 4 dB from 2 s. Below about 0.5 s a
  receiver-only quadratic matches split; the advantage of split grows with packet length.
- **Range 0.5–4 km (OOI buoy → buoy):** split 35–37 dB independent of range; a receiver estimate from the previous
  round trip (1.7–6.3 s old) −18 to −10 dB at every range.
- **Sea states (buoy → seabed, all 15 downloaded days, 9 CDIP + 6 OOI, 1 and 4 s):** split 34–41 dB on every day and
  packet length, the best strategy in all 30 cases; transmitter IMU only 28–39 dB; receiver per symbol 19–35 dB;
  receiver quadratic 18–38 dB at 1 s, −7 to 11 dB at 4 s.
- Closed form vs simulation: within about 3 dB for split and the IMU strategies in nominal conditions; optimistic by
  5–10 dB at large latency or with an uncorrected lever arm (correlated residual parts not modelled); pessimistic by
  4–8 dB for the per-symbol receiver (approximate kernel, as before).

## Placement rule: does the closed form pick the right strategy? (9 Oct 2026)

Check on the existing sweep outputs (`results/c2_sweeps/*.json`, 70 operating points, genie metric): at each point,
the strategy with the highest closed-form SIR (`sir_theory_db`) is compared with the one with the highest simulated SIR.
- Same winner at 64 of 70 points; at the other 6 the closed form picks the constant (split0) instead of the line
  (split), losing 0.2–0.6 dB in simulation.
- So the closed form already ranks the strategies correctly, even where its absolute SIR is off by 5–10 dB (large
  latency, uncorrected lever arm). This is the basis of the placement rule (PLAN Phase 3).

Preliminary (robustness run still going): with surface-bounced paths carrying their own Doppler (specular, full buoy
heave, grazing 8–18°), every strategy falls to about 11 dB (CDIP) and 4 dB (OOI); without per-path Doppler the
multipath results match the single path. To be confirmed when the run ends.

**Superseded (same day):** the ranking check above used simulations measured with a biased genie (next section); it is
redone with `scripts/c2_rule.py` once the reruns are in.

## Genie measurement bug (9 Oct 2026) — all genie-metric results being rerun

`genie_powers` estimated the per-symbol linear phase across subcarriers from adjacent products. Inter-carrier
interference biases that estimate (~3e-3 rad per subcarrier at a 0.2-subcarrier offset, i.e. 3 rad across the band),
so measured SIRs came out too low once interference is strong: 0 dB error at 35 dB, 3 dB at 17 dB, 12 dB at 9 dB
(constant offset; worse for a varying one). Replaced by the maximum-likelihood ramp (peak of a zero-padded FFT,
parabolic refinement); the genie now matches (pi fc e T)^2 / 3 within 0.1–0.8 dB from 35 to 6 dB (new test
`test_genie_matches_ici_closed_form`). Only one implementation is left (`c1_residual.py` imports the harness one).
- Results above ~30 dB (split, most of C1) move by < 1 dB; low ones (receiver per symbol at 19–22 dB, IMU-only, stale,
  large latency, uncorrected lever arm) were too pessimistic. Earlier remarks "the per-symbol kernel is 4–8 dB
  pessimistic" and "theory optimistic by 5–10 dB at large latency / uncorrected lever" were at least partly this bug.
- Old outputs kept in the session scratchpad (`results_before_genie_fix/`); reruns of C1, scenarios 1–3, sweeps,
  direction in progress. Pilot-metric results (multipath check) are not affected.

## Robustness checks (PLAN Phase 3)

### Multipath (pilot-based receiver, ceiling ~40 dB; `results/c2_robustness/multipath.json`)
| scenario, packet | single path: best (split) | multipath, common Doppler: split | multipath + surface-path Doppler: all strategies |
|---|---|---|---|
| seabed CDIP, 1 s / 4 s | 34.3 / 31.4 | 34.2 / 33.3 | 11.5–11.7 / 8.5–10.6 |
| seabed OOI, 1 s / 4 s | 32.5 / 30.8 | 30.5 / 29.9 | 4.1–4.2 / 0.5–4.6 |
| buoy–buoy OOI, 1 s / 4 s | 32.7 / 30.1 | 30.3 / 29.7 | 4.1–4.2 (tx_own −1.6) / −0.1–4.6 |
- Multipath with a common Doppler changes nothing: split remains best by 2–10 dB.
- Surface-bounced paths with their own Doppler (specular, 2 sin(grazing) w / c with the buoy's full heave, 8–18°,
  40% of the power) set a floor that no end-point compensation removes, and all strategies collapse onto it.
- Closed form for the floor, `placement.path_floor`: sum_k p_k (2 sin g_k / c)^2 var(w) -> 14.4 dB (CDIP, w std 0.32
  m/s) and 3.6 dB (OOI, 1.12 m/s), against 11–12 and 4–5 dB simulated. The rule therefore has a third term: placement
  matters only where the end-point residual is above this floor. The model is a worst case (coherent specular
  bounce, rough Irminger day); a rough surface spreads and weakens these paths.

### Path-direction error (genie, after the fix; `results/c2_robustness/direction.json`)
| azimuth error | 0° | 2° | 5° | 10° | 20° |
|---|---|---|---|---|---|
| seabed CDIP: tx_own / split | 33.1 / 39.4 | 29.7 / 39.8 | 24.3 / 35.8 | 21.5 / 33.5 | 14.3 / 31.3 |
| seabed OOI: tx_own / split | 30.4 / 36.7 | 20.8 / 33.7 | 12.9 / 29.6 | 6.6 / 24.5 | 0.2 / 18.2 |
| buoy–buoy OOI: both / split | 31.6 / 36.7 | 15.7 / 32.4 | 7.4 / 26.4 | 3.4 / 20.9 | 0.5 / 14.7 |
- IMU-only strategies need the path direction within ~2° on the large OOI buoy (1 m/s horizontal motion across the
  path); split tolerates ~5° for 30 dB, because the receiver's line absorbs the slow part of the projection error.

## Placement evaluator (`src/uwofdm/placement.py`, `scripts/c2_rule.py`)

Enumerates every placement (node IMU stage on/off at each end x receiver stage none / per symbol / const or line from
the first and last symbol / quadratic from all symbols / fed-back value) and predicts its residual variance and SIR.
Node stage applied as a transfer function to the measured motion (cross terms of latency, lever arm and direction
included); receiver stage through the kernel of the fit it actually makes (`theory.kernel_samples`: the line through
two estimates leaves 2.7x (4.3 dB) more curvature error than the least-squares line, which the old closed form missed).
- Direction check (15 points): same winner 15/15; prediction − simulation median +0.3 dB, |.| p90 1.7 dB for split
  (old closed form: 3–6 dB optimistic at 5–20°); both +0.1 / 3.0 dB; tx_own −1.5 / 3.0 dB.
- Placements not simulated yet come out on top: own IMU stage(s) + receiver quadratic from every symbol's estimate,
  e.g. 30.8 vs 14.7 dB (split) on buoy–buoy at 20° error; and on a seabed link the static receiver should not run an
  IMU stage (it only adds drift). To be added to the harness and simulated after the reruns.

### Evaluator validation with the interference kernel (all 85 corrected points)
Same winner at 81 (misses cost ≤ 0.6 dB). Prediction − simulation, median / |.| p90: receiver per symbol +0.3 / 1.9 dB
(offset formula: −5.6 / 7.3), split +0.2 / 2.0, split0 0.0 / 1.5, tx_own −0.4 / 3.1, both −0.7 / 3.1, quadratic −0.5 /
2.2, stale −2.6 / 5.8 (4 points). Offset-formula version kept in the scratchpad (`validation_offset_formula.json`).

### Generic placements in the harness
`run_link` accepts any evaluator name (`own[tx|rx|tx+rx|none]+rx[none|sym|const2|line2|quad]`), predicted by the
evaluator. Check: own[tx+rx]+rx[line2] reproduces split (37.5 vs 37.4 dB over 40 packets; identical without estimate
noise; test added). Note: with 10 packets per point, strategies that differ only by estimate noise can differ by
±2–3 dB; the new-placement runs use 30 (`scripts/c2_new_placements.py`, queued after the maps).

## Bellhop paths (Phase 1, first step; `src/uwofdm/bellhop.py`)
Bellhop (DESERT build binary) in arrivals mode, isovelocity 1500 m/s, 100 m deep, smooth (vacuum) surface,
acoustic half-space bottom (1600 m/s, 1.8 g/cm3, 0.8 dB/wavelength); near-duplicate beams merged; paths within 20 dB.

| Geometry | Paths | Delay spread | Surface-bounced power | Floor CDIP / OOI | Hand-made profile |
|---|---|---|---|---|---|
| Buoy (2 m) → seabed (99.5 m), 1 km | 11 | 76 ms | 64% | 8.4 / −2.5 dB | 14.4 / 3.6 dB (40%, 13.5 ms) |
| Buoy (10 m) – buoy (10 m), 1 km | 10 | 56 ms | 68% | 9.1 / −1.8 dB | — |

- With a mirror-like surface, Bellhop puts more power on surface-bounced paths than the hand-made profile, and with
  more bounces, so the path-Doppler floor is ~6 dB worse. This is an upper bound: at 16 kHz with Hs of 1–5 m the
  surface is very rough (Rayleigh parameter >> 1), coherent reflection is small and the scattered energy arrives as
  a Doppler spread; a surface-loss model (or Bellhop's top reflection coefficient) is the next step.
- **Delay spread exceeds the 22 ms cyclic prefix** in both geometries (arrivals at 25–76 ms): in this waveguide our
  OFDM design would also suffer inter-symbol interference. Either a longer prefix, a summer (downward-refracting)
  profile, or the surface loss above will change this — to settle before the multipath figures.
- Not yet wired into the harness (`experiment.Multipath` has one surface flag per path and no bounce count); waits for
  the running background job, which imports the harness.

### Surface loss (coherent rough-surface reflection, `surface_rms`)
Surface as a tabulated reflection coefficient (Bellhop top option 'F'): |R| = exp(−2 (k sigma sin g)^2), phase 180°,
sigma = rms elevation from each buoy's own heave (CDIP swell day: 0.54 m, Hs ≈ 2.1 m; OOI Irminger: 1.59 m, Hs ≈ 6.4 m).
At 16 kHz, |R| = 0.84 at 1° grazing for sigma 0.25 m, 0.50 at 2°, 0.01 at 5°.

| Geometry, day | Smooth surface: paths / spread / surface share / floor | Rough surface (coherent): paths / spread / surface share / floor |
|---|---|---|
| seabed, CDIP | 11 / 76 ms / 64% / 8.4 dB | 2 / 0.1 ms / 0% / none |
| seabed, OOI | 11 / 76 ms / 64% / −2.5 dB | 2 / 0.1 ms / 0% / none |
| buoy–buoy, CDIP | 10 / 56 ms / 68% / 9.1 dB | 3 / 10.7 ms / 7% / 41 dB |
| buoy–buoy, OOI | 10 / 56 ms / 68% / −1.8 dB | 2 / 10.7 ms / 0% / none |

- With the measured sea states, coherent surface reflection at 16 kHz is gone: the coherent channel is the direct
  and bottom paths, inside the cyclic prefix, with no coherent path-Doppler floor.
- The truth lies between the two columns: the energy removed from the specular paths is scattered, and the part that
  reaches the receiver arrives spread in delay and Doppler. The two brackets (no floor vs a −2 to 9 dB floor) are too
  far apart to settle the floor; it depends on the bistatic surface scattering strength, which this model does not
  have. Proposal: treat the diffusely scattered surface power (relative to the direct path) as a swept parameter
  with its Doppler spread from the surface velocity, and anchor its range in measured values from the literature.

### Scattered surface power as a swept parameter (`scripts/surface_scatter_sweep.py`, `results/surface_scatter/`)
rho = diffusely scattered surface power reaching the receiver / coherent channel, −40 … 0 dB; its Doppler from the
smooth-surface Bellhop surface paths of the same geometry (`bellhop.diffuse_doppler_factor`) and the buoy's surface
velocity; added to each placement's endpoint interference from the evaluator (1 s packets, IMU 0.01 m/s, 50 ms).

| Scenario | Endpoint SIR (rho → −inf): per symbol / split / IMU + quadratic | rho where split's margin < 1 dB | SIR at rho = −20 / −10 / 0 dB (all placements) |
|---|---|---|---|
| buoy → seabed, CDIP swell | 35.2 / 40.1 / 42.6 | −25 dB | 26 / 17 / 9 |
| buoy → seabed, OOI Irminger | 25.9 / 37.6 / 40.5 | −24 dB | 15–16 / 6 / −1.5 |
| buoy – buoy, OOI Irminger | 22.8 / 36.3 / 39.3 | −20 dB | 16–17 / 7 / −0.5 |

- **Placement matters only while the scattered surface power stays ~20–25 dB below the coherent channel.** Above that,
  every placement converges to the same scattered-Doppler floor. This is now the key external number for C2: the paper
  must show where real channels sit (literature on surface-scattered power at 10–30 kHz in shallow water), or restrict
  the claim to geometries / conditions where the surface contribution is small (downward-refracting profiles, steep
  geometries, directional receivers, low sea states).
- The model is simple: scattered energy around the specular directions of the smooth-surface paths, independent
  bounce points, no delay spread beyond the coherent paths. It brackets correctly (rho → −inf: rough-surface result;
  rho = smooth surface-path share: smooth-surface floor, test `test_diffuse_floor_limits`).

## New placements simulated (`scripts/c2_new_placements.py`, `results/c2_new_placements/points.json`)
Three scenarios × packets 1 / 4 s × direction error 0 / 5 / 20°, 30 packets per point, IMU 0.01 m/s, 50 ms.
- Evaluator picks the simulated winner at 17/18 points. For the generic placements (evaluator theory) prediction −
  simulation is within ±0.6 dB median, ≤ 1.7 dB p90. (The `rx_sym` and `split` theory columns of this run still come
  from the old closed form inside `run_link`: −6.6 and +2.4 dB median; to be switched to the evaluator.)
- **The new placements win everywhere**: node IMU stage(s) + receiver stage on per-symbol estimates.
  - 1 s packets: IMU + receiver quadratic best: 43.9 / 41.0 / 39.6 dB (seabed CDIP / seabed OOI / buoy–buoy, 0°),
    against split 39.8 / 37.1 / 36.0 and receiver per symbol 35.3 / 26.2 / 23.0.
  - 4 s packets or direction error: IMU + receiver per symbol best and robust: 38.6–39.2 (CDIP), 32.9–38.8 (OOI),
    30.4–38.0 dB (buoy–buoy), while split falls to 16 / 4 / 3 dB at 4 s and 20°.
- **At equal estimate budget the node stage is worth +4 to +15 dB**: IMU + per symbol vs receiver per symbol, 1 s,
  0°: +4.0 (CDIP), +12.2 (OOI), +14.8 dB (buoy–buoy).
- Seabed links: tx IMU only + receiver line beats split by 0.5–1 dB (the static receiver's IMU stage only adds drift).

## Dimensionless maps (`scripts/c2_map.py`, `results/c2_map/`; one moving end, JONSWAP f_p 0.1 Hz, 0.3 m/s)
- A, 2 estimates per packet (latency 0.005 wave periods): tx IMU alone if its drift < ~2% of the motion; IMU +
  receiver constant up to ~5–10%; split (IMU + line) above, most of the map at long packets; 35 → 15 dB as the drift
  grows towards the motion itself.
- A, per-symbol estimates: IMU + per-symbol receiver everywhere beyond one symbol (≈35 dB, insensitive to IMU drift).
- B, 2 estimates (IMU drift 3%): IMU stages win below L f_p ≈ 0.006–0.009 (60–90 ms for 10 s waves); above, the
  receiver line (short packets) or IMU alone (long packets), all ≤ 25 dB.
- B, per-symbol estimates: IMU + quadratic below L f_p ≈ 0.003–0.004, IMU + per symbol up to ≈ 0.009, receiver per
  symbol alone above (the IMU no longer helps).
- **Placement rule (first statement):** (1) a moving node removes its own motion if its bridged latency is below
  ≈ 0.01 wave periods; a static node does not; (2) the receiver's stage follows its estimate budget: with 2 per packet,
  nothing / constant / line as the IMU drift grows; with one per symbol, quadratic for very small latency, per symbol
  otherwise; (3) at equal budget, the node stage adds 4–15 dB on real buoy links.

## E1: Doppler side information, bits sweep (12 Oct 2026; `scripts/e1_sideinfo.py`, `results/e1_sideinfo/`)
Transmitter's IMU estimate of its share (0.01 m/s drift, 50 ms latency bridged, lever arm corrected) sent as K
orthonormal Legendre coefficients × b bits (uniform, ±4σ ranges from the spectrum; K 1–8, b 2–10); receiver subtracts
it (and its own IMU share on buoy–buoy) and fits the remainder from the signal (none / 2 estimates per packet / one
per symbol, error 3e-6). Time-domain scoring per packet (`sideinfo.interference`), 200 real-motion packets per point.

| Link, packet | Signal estimates | Receiver alone | Transmitter applies (no bits) | Bits to come within 1 dB of applying |
|---|---|---|---|---|
| seabed CDIP, 1 s | none / 2 / per symbol | 6.9 / 19.9 / 35.2 | 32.1 / 39.9 / 38.8 | 32 / 30 / 9 |
| seabed OOI, 1 s | none / 2 / per symbol | −4.4 / 9.3 / 26.1 | 31.9 / 37.7 / 38.8 | 40 / 64 / 24 |
| buoy–buoy OOI, 1 s | none / 2 / per symbol | −4.9 / 9.2 / 25.6 | 28.0 / 36.2 / 38.0 | 32 / 40 / 20 |
| 4 s packets (all) | none / 2 / per symbol | −5 to 35 | 28–39 | not reached with K ≤ 8 (≤ 80 bits) except CDIP per symbol (32) |

- **1 s packets: 20–64 bits recover the gain of pre-compensation.** Gains over the receiver alone: +25 to +36 dB with
  no signal estimate, +20 to +28 dB with two per packet, +3 to +13 dB with one per symbol. 40 bits are ≈ 0.3% of a 1 s
  QPSK packet (≈ 14 kbit raw), against ≈ 25% for pilots in every symbol.
- **4 s packets need more coefficients, not more bits per coefficient:** the coefficient stds still carry 2e-5 at
  K = 8 on OOI (a 4 s packet spans about half a wave period of high-frequency content). Expected ≈ 2–4 coefficients
  per second of packet; to rerun with K up to ~24 and with bit allocation.
- Equal bits per coefficient is wasteful: the stds fall by ×2–5 per order on 1 s packets; reverse water-filling should
  cut the bit counts above.
- Caveat: time-domain scoring of the Doppler residual, not a waveform simulation; the reference levels match the
  waveform runs (e.g. 37.0 vs 37.8 dB). To confirm a few points with the waveform harness.

### E1 with bit allocation and up to 24 coefficients (`scripts/e1_allocation.py`, `sir_vs_bits_alloc.png`)
Greedy allocation by coefficient size (`sideinfo.allocate`), budgets 4–192 bits, 200 packets per point.

| Link | 1 s: bits (coefficients) within 1 dB of pre-compensation, stage none / line2 / per symbol | 4 s: same |
|---|---|---|
| seabed CDIP | 24 (4) / 32 (5) / 12 (3) | 64 (11) / 96 (12) / 48 (10) |
| seabed OOI | 32 (5) / 96 (18) / 24 (4) | 128 (24) / not reached (35.0 vs 36.4 dB at 192) / 96 (17) |
| buoy–buoy OOI | 32 (5) / 96 (18) / 24 (4) | 128 (24) / 192 (24) / 96 (17) |

- With sensors only or with one signal estimate per symbol: **≈ 25–30 bits per second of packet** (12–32 bits for 1 s,
  48–128 for 4 s) reach pre-compensation within 1 dB. Allocation cut the 1 s counts (e.g. 40 → 32, 20–24 → 24, 9 → 12
  bits: similar; gains mostly in the 4 s cases, which now reach the target).
- **Allocation by size is wrong when the receiver also fits a line:** with 2 signal estimates the curves are not
  monotonic (OOI 1 s: 35.3 dB at 32 bits, 31–35 at 48–64). Two reasons: (1) bits on coefficients 0–1 are wasted,
  because the receiver's line re-estimates them; (2) a first bit with ±4σ range has error 1.33σ², worse than not
  sending. Next: stage-aware allocation — weight each coefficient by the interference its error causes *after* the
  receiver's stage (computable exactly from the basis and the stage weights) and allocate by marginal decrease.

### E1 with stage-aware allocation (`scripts/e1_stage_alloc.py`, `sir_vs_bits_stage_alloc.png`)
Each coefficient weighted by the interference its error causes after the receiver's stage (`sideinfo.stage_weights`),
bits by marginal decrease per bit, coefficients started with 2 bits (`sideinfo.allocate_weighted`).

| Link | 1 s: bits within 1 dB of pre-compensation, none / line2 / per symbol (size-based) | 4 s: same |
|---|---|---|
| seabed CDIP | 24 / **12** / **8** (24 / 32 / 12) | 64 / **64** / **32** (64 / 96 / 48) |
| seabed OOI | 32 / **48** / **16** (32 / 96 / 24) | **96** / not reached (35.2 vs 36.4 at 192) / **64** (128 / – / 96) |
| buoy–buoy OOI | 32 / **64** / **16** (32 / 96 / 24) | **96** / 192 / **48** (128 / 192 / 96) |

- **What to send depends on what the receiver estimates itself:** under a receiver line the level and slope
  coefficients get no bits (sent at 32 bits: k = 2…8); with per-symbol signal estimates the transmitter sends
  k = 1–4 and some high orders (11–14), whose intra-symbol curvature the per-symbol estimates cannot follow; with no
  signal estimate, k = 0–4.
- Bits needed drop by up to ×2.7 (line2) and ×1.5–2 (per symbol); curves are now nearly monotonic (one small dip:
  OOI 1 s line2, 36.5 → 35.6 dB at 16 → 24 bits). Sensors-only cases are unchanged (the size order was already right).
- Rule of thumb now: **≈ 10–20 bits per second of packet with per-symbol signal estimates, ≈ 25–30 with none or two.**

### E1 waveform check (`scripts/e1_waveform_check.py`, `results/e1_sideinfo/waveform_check.json`)
Side information as a strategy of the OFDM waveform harness (`si<B>+own[..]+rx[..]`, experiment.py): 8–64 bits and
both references, three scenarios, 1 and 4 s, three receiver stages, 30 packets per point. Waveform − time-domain
scoring over 108 points: median +0.2 dB, |.| p90 1.6 dB, max 3.7 dB. The E1 bit counts stand.

## End-to-end chain (12 Oct 2026; `frame.py`, `estimators.py`, `chain.py`, `header.py`, `coding.py`,
`receiver.py`, `fusion.py`)
Frame: chirp | header | data (pilots, in-band nulls every 16th subcarrier) | chirp. Receivers: Li/Lee signal-only
front end (preamble/postamble scale + per-symbol CFO over ±2 subcarriers); sensor-aided (resample along side
information + own IMU, residual scale in ±3e-4, per-symbol CFO); real header (CRC, fallback); coded data path (rate
1/2 K=7, QPSK / 16-QAM, pilot channel estimate, LLRs weighted by noise + ICI measured on nulls); Kalman fusion
(state τ, r, r'; postamble timing + per-symbol offsets with SNR-dependent variance; RTS smoother; one resampling) and
the same smoother as a signal-only baseline.
- Fixes found on the way: per-symbol CFO search widened from ±0.5 to ±2 subcarriers (the narrow one saturated in rough
  seas and would have exaggerated our gain); in-band nulls (out-of-band bins made the CFO estimate random at low SNR:
  9.3 → 0.5 Hz rms at 0 dB); side information for the chain must send level and slope (the per-symbol-optimised
  allocation left the mean out and the narrow residual search saturated).
- Synthetic: fusion beats constant + per-symbol corrections by +7 to +12 dB at 35 dB SNR when the prior's residual
  varies within the packet (never worse); **the signal-only Kalman baseline gains almost as much from smoothing**
  (26.0 → 34.3 dB on a 6e-4 wave at 35 dB). Sensor gains must be stated against this baseline.
- Real motion, buoy–buoy OOI, 3 packets, SNR 30 dB: Li/Lee 16.5, signal-only Kalman 25.8, sensor-aided 29.8 dB
  (+4 dB over the best signal-only receiver; +13 over Li/Lee). Full sweeps running.

### Range-Doppler coupling of the chirps (fixed 12 Oct 2026) and the Cho–Kim baseline
- A single linear-FM preamble / postamble biases the packet's mean scale when the Doppler differs between them (its
  arrival time shifts by fc eps / k): 8.75e-4 vs 8.39e-4 true for 0.4 m/s² over 1.3 s. This handicapped the
  signal-only receivers (Cho–Kim, signal-only Kalman) more than the sensor-aided ones. Fix: each chirp is an
  up-sweep then a down-sweep, detected separately and averaged (`estimators.detect_chirp`): bias 6.6e-6 (−50 dB),
  constant Doppler exact. (A merged V-chirp lost Doppler tolerance: peak 0.38 at 1e-3.) All chain experiments rerun.
- Cho–Kim 2026 (`chain.receive_cho_kim`): mean scale from preamble / postamble + constant acceleration from the
  cyclic-prefix offsets (threshold 0.4), given its best version (intercept fitted too). 0.4 m/s² recovered within
  10%.
- Real motion, SNR 30 dB, 3 packets (Li/Lee, signal-only Kalman, Cho–Kim, sensor-aided, sensor fusion):
  CDIP 1 s 29.2 / 29.9 / 25.3 / 29.9 / 29.9; CDIP 4 s 27.9 / 29.9 / 16.3 / 29.7 / 29.9;
  OOI buoy–buoy 1 s 16.5 / 29.1 / 3.9 / 29.9 / 29.9; **OOI buoy–buoy 4 s 9.4 / 12.7 / −1.6 / 28.1 / 29.6**.
  With the bias fixed, the best signal-only receiver matches the sensors in moderate seas or short packets; in a
  rough sea with long packets the sensors gain +17 dB over it. Cho–Kim degrades with frame length (as predicted in
  `literature/baseline_cho_kim_2026.md`) and fails in the rough sea (residual beyond its ±½-subcarrier range,
  waves are not a constant acceleration). Full sweeps running.

### Bellhop multipath in the chain (`multipath.py`, `scripts/e2_multipath.py`)
Coherent Bellhop arrivals with measured rough-surface loss, **each path with its own Doppler from its departure /
arrival angles**, plus optional diffusely scattered surface power with per-patch surface Doppler; metrics: SINR from
the receiver's own pilot channel estimate (`receiver.sinr`) and coded PER. Detection now locks on the first
significant arrival (earliest peak ≥ 0.5 of the maximum).
- Buoy → seabed (CDIP): direct + bottom bounce 0.07 ms apart at nearly the same angle (5.6°): 23–32 dB per symbol at
  30 dB SNR, decoded; with ρ = −20 dB of scattered surface power: 14–18 dB per symbol, 1.8% coded-bit errors that the
  code does not fully correct (energy beyond the cyclic prefix gives confident errors).
- **Buoy – buoy (OOI, 10 m depth): the bottom bounce (±10.2°) sees both buoys' heave × sin 10°, the direct path does
  not: differential Doppler up to ~1.5e-4 (≈ 2.4 Hz) on half the received power.** SINR 2–8 dB even with the exact
  bulk Doppler (27 dB on the same packets with the direct path only; 27 dB with two paths when the nodes do not move).
  One resampling cannot fix two paths with different heave projections: this needs per-path (multi-branch)
  compensation — and the per-path Doppler is predictable from the IMU heave and the path angles (sensor-aided
  multi-branch: a natural extension, link to C3 / Tu et al.).

### Per-path multi-branch compensation and cancellation (`chain.receive_branches`, `grid_trajectories`, `receive_sic`; `receiver.combine`, `decode_combined`)
- Multi-branch MRC (Tu et al. 2013 style: one resampling per Doppler scale, per-branch pilot channel and null noise,
  maximum-ratio combining; LLRs from the combined symbols): signal-only grid (Kalman bulk ±1.5e-4) and sensor-set
  (one branch per Bellhop path at its predicted Doppler). Both gain only 1–2 dB on the buoy–buoy two-path channel:
  each branch keeps the other path's inter-carrier interference.
- **Sensor-aided per-path cancellation** (decoder-aided: decode the MRC of the per-path branches, re-encode,
  regenerate every path with its sensor-predicted Doppler and geometric delay, least-squares path gains, subtract the
  other paths per branch, MRC): synthetic two-path test (second path +1.5e-4) 5.5 → 25.2 dB, 0 bit errors after one
  iteration (decisions from path 0 alone locked in at 9 bit errors; hard slicing did not converge). Real buoy–buoy
  OOI packets at 30 dB SNR (single / Tu grid MRC / sensor MRC / sensor SIC): 5.9 / 7.5 / 7.7 / **24.8 ok**;
  1.7 / 3.0 / 3.1 / **12.6 ok**; 4.2 / 5.8 / 5.8 / **16.1 ok** — the only receiver that decodes.
- Caveats: per-path Doppler from ideal sensors so far (IMU error and the transmitter's vertical velocity in the side
  information still to add); no signal-only cancellation baseline yet (it would have to estimate the second path's
  Doppler trajectory). Full statistics: `scripts/e2_multipath.py` (queued).

### Per-path prediction from realistic sensors (`sensors.py`; `receive_sic(refine_common=True)`)
3-axis IMU model per node (per-axis Gauss–Markov drift 0.01 m/s, 50 ms latency bridged by the accelerometer, lever arm
from the gyro); two-component side information (transmitter's velocity along the link's horizontal direction and
vertical velocity, 32 + 32 bits, Legendre coefficients); per-path Doppler predicted from the decoded side information,
the receiver's IMU and the Bellhop angles; one common residual from preamble / postamble added to all paths.
- Prediction error per path 5–7e-6 (std over the packet). Exact sensors reproduce the path Doppler to 2e-9 (test).
- Real buoy–buoy OOI packets, 30 dB SNR (ideal-sensor SIC / IMU SIC / IMU SIC + common / Tu grid MRC):
  24.8 ok / 20.9 ok / 21.2 ok / 7.5 fail; 16.6 ok / 16.2 ok / 16.2 ok / 3.0 fail; 16.1 ok / 12.4 ok / 18.0 ok /
  5.8 fail. Realistic sensors keep the result: cancellation decodes, the state of the art does not.

### Signal-only cancellation baseline (`chain.signal_path_estimates`, `receive_signal_sic`)
Same decoder-aided cancellation, with the per-path description estimated from the signal: path delays from the
preamble correlation peaks (within 0.1 ms of the geometry), per-path mean Doppler from each path's own chirp gap
(errors 2.5–3.5e-5: the two paths' 25 ms chirps overlap when they arrive 10.7 ms apart), shared within-packet shape
from the signal-only Kalman; then a search of every path's Doppler (grid ±6e-5) on the least-squares fit of the
regenerated paths, two passes with fresh decisions (≈ 6 s per packet).
- Real buoy–buoy OOI packets, 30 dB SNR: chirp estimates only 8.1 ok / 5.6 / 7.9; + per-path refinement from the
  cleaned branches degraded (2.3–7.6; biased when the first decode fails, now off by default); + search of path 2
  9.1 ok / 5.2 / 9.0; **+ search of all paths, 2 passes: 10.8 ok / 6.1 fail / 10.0 fail**, against 21.2 / 16.2 / 18.0
  dB, all decoded, with realistic sensors (IMU SIC).
- Reading: cancelling a path needs its Doppler to ≈ 5e-6; the signal alone does not reach it here (overlapping
  chirps, poor first decisions), the sensors do (5–7e-6 prediction error). This is the sensor contribution in
  multipath, stated against a signal-only receiver with the same cancellation machinery.

### Path-angle errors (`chain.receive_sensor_sic`, `refine_paths_by_search`)
Sensor-aided cancellation with the per-path Doppler predicted from realistic sensors but wrong path angles; optionally
a narrow signal search around the prediction (±2e-5, 9 steps; the same search the signal-only baseline runs ±6e-5
around chirp estimates). Three real buoy–buoy OOI packets, 30 dB SNR; decoded packets (of 3), no search / search:
exact 3 / 3; random 0.5° 3 / 3; random 1° 1 / **3**; random 2° 1 / 1; random 5° 0 / 0; wrong geometry (range +10%,
water depth +5 m: angle 9.8° instead of 10.2°, **delay 10.86 instead of 10.71 ms**) 2 / **0**.
- The bottom bounce's heave term scales with sin θ: 1° at 10° is ≈ 1.5e-5 of Doppler, three times what cancellation
  tolerates. Sensors alone need the angles within ≈ 0.5°; the narrow search extends this to ≈ 1°.
- A delay error of 0.15 ms (5 samples) is not absorbed by a Doppler search and made the search pick wrong offsets.
  Delays should come from the signal (preamble peaks, < 0.1 ms) and angles from the geometry: test running.
- With delays from the signal (preamble peaks: 10.72–10.78 ms) and angles from the geometry: wrong geometry 2 / 3
  decoded with the ±2e-5 or ±5e-5 search (was 0 / 3 with geometric delays); random 2° (a different realisation from
  the line above) 3 / 3 with either grid. Default for the realistic-sensor receiver: measured delays, geometric angles,
  narrow search (`imu_sic_search` in `scripts/e2_multipath.py`). Requirement to state: path angles within ≈ 0.5–1°.

## Chain SNR sweep and header experiment, full runs (13 Oct 2026; fixed chirps, all receivers; 20 packets per point)
SINR [dB] (genie on used subcarriers) at per-subcarrier SNR 20 / 30 dB — Li/Lee, signal-only Kalman, Cho–Kim,
sensor-aided (32 bits), sensor fusion (32 bits), prior alone:

| Link, packet | Li/Lee | Signal Kalman | Cho–Kim | Aided | Fused | Prior only |
|---|---|---|---|---|---|---|
| seabed CDIP, 1 s | 19.8 / 28.3 | 20.0 / 30.0 | 18.6 / 23.1 | 20.0 / 30.0 | 20.0 / 30.0 | 19.2 / 25.4 |
| seabed CDIP, 4 s | 19.2 / 25.3 | 20.0 / 29.9 | 12.5 / 13.3 | 19.9 / 29.7 | 20.0 / 30.0 | 18.4 / 22.7 |
| seabed OOI, 1 s | 16.9 / 19.5 | 19.9 / 29.4 | 5.8 / 6.2 | 19.9 / 29.8 | 19.9 / 29.8 | 19.6 / 27.0 |
| seabed OOI, 4 s | 12.1 / 12.8 | 19.3 / 25.8 | −2.6 / −2.6 | 19.7 / 27.8 | **19.9 / 29.7** | 13.6 / 14.6 |
| buoy–buoy OOI, 1 s | 13.4 / 14.4 | 18.8 / 23.8 | 0.6 / 0.6 | 20.0 / 29.8 | **20.0 / 29.8** | 19.6 / 26.8 |
| buoy–buoy OOI, 4 s | 9.5 / 10.0 | 14.0 / 15.1 | −2.9 / −2.8 | 19.7 / 28.1 | **19.9 / 29.7** | 14.4 / 15.6 |

At 5 and 10 dB all receivers but Cho–Kim are noise-limited (within 0.5 dB of SNR), except buoy–buoy 4 s (Li/Lee
3.4 / 6.7, signal Kalman 4.4 / 8.6, fused 5.0 / 10.0).
- **Against the best signal-only receiver (signal Kalman), sensor fusion gains 0 dB in the moderate sea (CDIP),
  +0.4 / +3.9 dB on seabed OOI (1 / 4 s, 30 dB), +6.0 / +14.6 dB on buoy–buoy OOI (1 / 4 s, 30 dB), +1.2 / +5.9 dB at
  20 dB.** Against Li/Lee: up to +19.7 dB. The gain appears with rough sea, both ends moving, long packets, high SNR.
- Kalman fusion matters on long packets: aided 27.8 → fused 29.7 (seabed OOI 4 s), 28.1 → 29.7 (buoy–buoy 4 s).
- Sensors alone are not enough on long packets (prior only 14–16 dB at 30 dB): the gain is the fusion of sensors
  and signal, not sensors replacing the signal.
- Cho–Kim: within 1–7 dB of the others on CDIP 1 s, collapses with long packets or the rough sea (−3 to 6 dB).

Header (16 / 32 / 64 bits): read in 100% of packets at ≥ 5 dB on every link; at 0 dB 95–100% (16, 32 bits) and
55–100% (64 bits); fails at −5 dB, where every receiver is at ≈ −6.5 dB anyway. Whenever it is read, the full chain
equals ideally delivered side information (e.g. buoy–buoy 30 dB: 29.7 vs 29.7, signal-only 15.2). 32 bits: the
robust choice for 1 s packets.

## Coded PER, 1.05 s packets (13 Oct 2026; `results/e2_chain/per.json`, `per_16qam_7.json`)
Single path, real motion, rate-1/2 K=7 code; 100 (QPSK) / 40 (16-QAM) packets per point. PER Li/Lee / signal-only
Kalman / Cho–Kim / full sensor chain (real header) / sensor fusion:
- QPSK: all receivers but Cho–Kim reach PER ≤ 0.04 at 12 dB on CDIP; **Li/Lee keeps an error floor on the rough
  links** (buoy–buoy 0.04–0.06 at 15–25 dB); Cho–Kim fails in the rough sea (0.39–0.61 at 20–25 dB). Signal-only
  Kalman and the sensor chain are equal (0.00 from 15 dB; at 9 dB 0.31–0.33 vs 0.27–0.31).
- 16-QAM: Li/Lee floors at 0.23–0.38 (seabed OOI) and 0.42–0.55 (buoy–buoy) up to 30 dB; signal-only Kalman and the
  sensor chain both reach 0.00 at 22 dB everywhere; Cho–Kim 0.62–0.95 on OOI.
- Reading: on 1 s packets the sensors match the best signal-only receiver in PER (as in SINR, where they gained
  0–6 dB only at 30 dB); they beat Li/Lee and Cho–Kim, which floor. The 4 s variants (running) test where the SINR
  gain was large (+15 dB buoy–buoy).

## Multipath statistics (13 Oct 2026; `results/e2_chain/multipath.json`, 20 packets per point)
Bellhop coherent paths with per-path Doppler, optional scattered surface power; SINR (pilot estimate) / PER, QPSK
1 s. Buoy–buoy OOI (direct + bottom bounce 10.7 ms, differential heave Doppler), SNR 30 dB, ρ none / −30 / −20 dB:
- Li/Lee 13.1/0.40, 13.4/0.55, 11.3/0.60; signal Kalman 18.1/0.30, 16.9/0.30, 13.3/0.60; Tu grid MRC 16.8/0.30,
  15.5/0.30, 12.0/0.50; signal-only cancellation 18.1/0.20, 17.1/0.10, 13.9/0.35; sensor fusion (one resampling)
  19.1/0.30, 17.4/0.35, 13.5/0.55; **cancellation with ideal sensors 23.3/0.20, 21.4/0.10, 16.5/0.25; with realistic
  sensors (IMU SIC) 22.0/0.10, 20.4/0.10, 16.0/0.30**; IMU SIC with measured delays + search 21.6/0.15, 20.2/0.15,
  16.0/0.40. At 20 dB: IMU SIC PER 0.05 / 0.25 / 0.25 vs best signal-only 0.25 / 0.25 / 0.55.
- Buoy → seabed (direct + bottom 0.07 ms apart): per-path cancellation still adds 3–4 dB (OOI 30 dB: 25.6–28.1 vs
  21.0 dB signal Kalman), PER 0.00 vs 0.10.
- Reading: in multipath with heave-dependent per-path Doppler, sensor-predicted cancellation gives +3–4 dB SINR over
  the best signal-only receiver (including signal-only cancellation) and halves the PER; scattered surface power at
  −20 dB caps everything (≈ 16 dB). PER statistics from 20 packets are coarse (±0.1).

## Coded PER, 4 s packets (13 Oct 2026; `per_qpsk_27.json`, `per_16qam_27.json`; 40 packets per point, 96 bits)
PER Li/Lee / signal-only Kalman / Cho–Kim / full sensor chain (real header) / sensor fusion:
- **Buoy–buoy OOI, QPSK**: 15 dB 0.78 / 0.10 / 1.00 / 0.03 / 0.00; 20–25 dB 0.55–0.62 / 0.05 / 0.97–1.00 / 0.00 /
  0.00. **16-QAM**: 22 dB 0.97 / 0.23 / 1.00 / 0.05 / 0.05; 26–30 dB 0.95–0.97 / 0.10 / 1.00 / 0.00 / 0.00.
  The best signal-only receiver keeps an error floor (5–10%); the sensor chain has none.
- Seabed OOI: signal-only Kalman and sensor chain both 0.00 from 15 dB (QPSK) / 26 dB (16-QAM); Li/Lee floors at
  0.30–0.38 (QPSK) and 0.90–0.93 (16-QAM); Cho–Kim fails (≈ 1.00).
- CDIP: all but Cho–Kim reach 0.00 at 15 dB (QPSK) / 26 dB (16-QAM; Li/Lee 0.05–0.07).
- Header with 96 bits: read in 90–100% of packets at ≥ 3 dB, 35–55% at 0 dB (fewer repetitions per bit).
- Reading: the PER confirms the SINR regime map — sensors matter in a rough sea with both ends moving and long
  packets, where they remove the error floor that even the best signal-only receiver keeps.

## Channel-estimator loss fixed (12 Oct 2026; `receiver.channel_estimates`, `tests/test_coding.py`)
The pilot channel estimate of the coded chain (per symbol, regularised LS on ~390 delay taps of the twice-oversampled
grid from 248 pilots) had an error as large as the noise: on a **static AWGN channel** rate-1/2 QPSK gave PER 1.00 at
6 dB and 0.33 at 9 dB per-subcarrier SNR, against error-free at 5 dB with a genie channel (≈ 4 dB lost by every
receiver). Replaced by a sparse estimate shared by the packet: simultaneous OMP over the packet's symbols (common delay
support, per-symbol gains; taps kept while they remove more than 4x the noise energy they add; ≤ 40 taps).
- Static AWGN: PER genie / old LS / SOMP 0.23 / 1.00 / 0.30 at 3 dB, 0.00 / 1.00 / 0.00 at 5 dB, 0.00 / 0.33 / 0.00 at
  9 dB; estimate error / noise 0.99 (LS) vs 0.004 (SOMP). 5-path channel, 15 ms spread: 0.23 / 1.00 / 0.33 at 5 dB,
  0.07 / 0.87 / 0.07 at 7 dB; error / noise 0.98 vs 0.05-0.08. At high SNR (5 paths) the error relative to the channel
  is −29 / −36 / −44 dB at 20 / 30 / 40 dB SNR (LS −20 / −30 / −37): off-grid delays leave a small floor.
- Consequence: **all PER and multipath results above (`results/e2_chain/per*.json`, `multipath.json`, and the
  `*_hi.json` runs started before the fix) used the old estimate and must be rerun.** The SINR results with the genie
  metric (`points.json`, E1, C1/C2) are not affected. The synthetic two-path test now decodes with a single
  resampling (7.5 dB SINR, was 5.5 dB and failed); cancellation still gains ≈ 20 dB (27.8 dB). Part of the
  "only cancellation decodes" result on real multipath may have come from the weak estimate: to recheck.

## Realistic motion sensors (12 Oct 2026; `imu.py`, `scripts/e2_imu_grades.py`, `results/e2_imu_grades/`)
The original IMU model (true velocity read 50 ms late, bridged with the true acceleration, + Gauss-Markov drift
0.01 m/s, 20 s) has almost no error at wave frequencies. `imu.RealisticIMU` builds the estimate like a motion sensor:
accelerometer white noise, bias instability, scale factor, attitude errors rotating the specific force (gravity leaks
into the horizontal axes), complementary filter with GNSS (or DVL) velocity, or a 1st-order high-pass without aiding;
IMU-to-modem timing error; lever-arm length error and gyro noise. Grades nav / industrial / consumer (representative
class values, to check against datasheets), each with and without aiding.
- Wave-band (> 0.04 Hz) velocity error, OOI Irminger: old model 0.005 m/s; nav 0.005-0.006 (horizontal) / 0.003
  (vertical); industrial 0.05 / 0.008; consumer 0.20 / 0.025; unaided: nav 0.03-0.05, industrial 0.07-0.10 / 0.10.
  The old model was in effect a GNSS-aided navigation-grade sensor. Tilt leakage dominates, on the horizontal axes,
  which carry the Doppler of near-horizontal links.
- Chain (32-bit side information, Kalman fusion, 20 packets, same packets and noise for every grade), SINR at 30 dB:

| Link, packet | Li/Lee | signal Kalman | old | nav | industrial | consumer | nav unaided | industrial unaided |
|---|---|---|---|---|---|---|---|---|
| seabed CDIP, 1 / 4 s | 28.2 / 24.2 | 29.9 / 29.8 | 30.0 / 30.0 | 30.0 / 30.0 | 29.9 / 29.9 | 29.6 / 29.7 | 29.9 / 29.9 | 29.6 / 29.6 |
| seabed OOI, 1 / 4 s | 21.1 / 14.2 | 29.6 / 28.3 | 29.9 / 29.8 | 29.9 / 29.7 | 29.8 / 29.6 | 29.6 / 28.8 | 29.8 / 29.7 | 24.6 / 9.7 |
| buoy-buoy OOI, 1 / 4 s | 16.9 / 9.0 | 25.9 / 12.0 | 29.8 / 29.7 | 29.8 / 29.7 | 29.7 / 29.6 | 24.1 / 22.1 | 29.7 / 29.5 | 15.3 / 8.9 |

  Prior rms Doppler error (buoy-buoy 4 s): old 4.4e-5, nav 4.3e-5, industrial 9.9e-5, consumer 2.5e-4, nav unaided
  1.0e-4, industrial unaided 4.4e-4.
- **The gain survives GNSS-aided industrial MEMS** (+17.6 dB over the best signal-only receiver on buoy-buoy 4 s,
  although the prior is 2-6x worse than with the old model: fusion with the signal absorbs it); consumer grade keeps
  +10 dB there but falls below signal-only on buoy-buoy 1 s (24.1 vs 25.9).
- **Defect found: fusion can be much worse than signal-only** (industrial unaided: 8.9 vs 12.0, 9.7 vs 28.3 dB): the
  prior error exceeds the fixed +-3e-4 residual window and the Kalman prior variance tuned for the old model; the
  receiver trusts the prior blindly. Fix: prior-quality-aware fusion with a signal-only guard (next).

### Prior-quality-aware fusion (`chain.receive_fused_safe`, 12 Oct 2026)
Prior quality per packet from the chirps (mean scale over the full ±3e-3 range against the prior's mean; s = 2 |diff|),
residual window max(3e-4, 4 s), Kalman settings moved from the sensor ones to the signal-only ones as s → 2e-4, and a
guard that keeps the signal-only Kalman output when its null-subcarrier SINR is higher. Real OOI motion, 30 dB, 8
packets (same packets for all; choice per packet f = fused, s = signal):

| Link, packet | signal Kalman | industrial: fixed / safe | consumer: fixed / safe | industrial unaided: fixed / safe |
|---|---|---|---|---|
| buoy-buoy, 1 s | 23.9 | 29.7 / 29.7 | 21.8 / 28.0 | 27.3 / 28.8 |
| buoy-buoy, 4 s | 13.5 | 29.5 / 29.6 | 22.3 / 27.4 | 17.2 / 28.5 |
| seabed OOI, 4 s | 27.6 | 29.7 / 29.8 | 29.4 / 29.5 | 19.8 / 28.8 |

- Never below the signal-only receiver, no loss with good sensors; most of the recovery comes from the adaptation
  itself (industrial unaided, buoy-buoy 4 s: fused output chosen in 7 of 8 packets). Poor sensors still gain
  +4 to +15 dB on the rough links. To use as the sensor receiver in the reruns (PER, multipath, high statistics).

## Fair baselines (12 Oct 2026; `scripts/b1_tune_signal_kalman.py`, `scripts/b2_baselines.py`, `pfft.py`)
- **Signal-only Kalman tuned:** grid sigma_dr0 {5e-4, 2e-3, 8e-3} x q_std {1e-4 ... 6.4e-2}, buoy-buoy OOI 1 / 4 s,
  seabed OOI 4 s, seabed CDIP 1 s, 20 / 30 dB, 8 packets. sigma_dr0 hardly matters; q_std 5e-4 ... 1e-3 best (mean
  23.75-23.77 dB), 2.5e-4: 23.55, 1e-4: 22.0, 4e-3 (hand-set): 23.64, 6.4e-2: 22.4. New default (5e-4, (1e-3)^2):
  +0.13 dB on average, up to +0.5 dB on 4 s packets — the earlier comparisons were not unfair on this point. (The
  high-statistics runs started at 13:41 use the hand-set values.)
- **Partial-FFT demodulation** (Yerramalli, Stojanovic, Mitra 2012; segment weights and sparse taps by LS on the
  pilots): synthetic 6e-4 slow wave + 3e-4 at 0.8 Hz, 40 dB: signal Kalman 29.2 -> 32.4 / 35.8 / 37.9 dB with 2 / 4 / 8
  segments (no fast part: 35.9 -> 39.6); after the Li/Lee front end < 1 dB. A strong signal-only baseline; as a
  demodulator it can follow sensor fusion too.
- **Statistics warning:** power-averaged SINR over 8-20 packets is dominated by the worst packets: the signal-only
  Kalman on buoy-buoy 4 s gave 12-13.5 dB on the packet sets of e2_chain / e2_imu_grades and ~24 dB on the sweep's
  8 packets. Large SINR gains must be re-measured with more packets and reported with median and percentiles.
- **ICI-aware equaliser** (`ici.py`, `chain._demod_ici`: per-segment sparse taps from pilots + decisions, joint MMSE
  over the active band, initialised by partial FFT): synthetic, signal Kalman / pFFT-8 / ICI-aware: no fast part
  36.2 / 39.7 / 39.6; fast 3e-4 at 40 dB 29.1 / 37.9 / 37.0, at 30 dB 26.7 / 29.6 / 29.9; fast 6e-4 at 40 dB
  23.8 / 34.9 / 33.5 (16 segments: pFFT 36.3, ICI-aware 34.7; 4 iterations no better). Gains much over the
  Kalman alone but stays ~1.4 dB below partial FFT: **partial FFT is the strongest signal-only baseline we have**;
  ICI-aware receivers with per-path Doppler or polynomial bases (literature) are not implemented.
- **Comparison with the strongest signal-only receiver** (`scripts/b2_baselines.py`, 30 packets per case, same packets
  for all; power average / median / 10th percentile SINR at 30 dB):

| Case | signal Kalman | signal Kalman + pFFT-8 | sensor fusion, old IMU | sensor fusion, industrial MEMS (+ pFFT-8) |
|---|---|---|---|---|
| buoy-buoy OOI 1 s | 29.1 / 29.4 / 28.3 | 29.8 / 29.9 / 29.6 | 29.8 / 29.8 / 29.5 | 29.7 / 29.8 / 29.4 (29.8 / 29.9 / 29.6) |
| buoy-buoy OOI 4 s | 14.1 / 27.7 / 11.5 | 23.6 / 29.7 / 24.0 | 29.6 / 29.7 / 29.3 | 29.6 / 29.6 / 29.3 (29.8 / 29.8 / 29.5) |
| seabed OOI 4 s | 24.8 / 28.2 / 22.4 | 29.5 / 29.8 / 29.1 | 29.8 / 29.8 / 29.6 | 29.7 / 29.7 / 29.5 (29.8 / 29.9 / 29.7) |
| seabed CDIP 1 s | 29.9 / 29.9 / 29.8 | 29.9 / 29.9 / 29.8 | 29.9 / 30.0 / 29.8 | 29.9 / 29.9 / 29.8 (29.9 / 29.9 / 29.8) |

  At 20 dB, buoy-buoy 4 s: signal Kalman 13.2 / 19.7 / 11.0, + pFFT-8 18.3 / 19.6 / 18.3, sensor fusion (old)
  19.9 / 19.9 / 19.9. **Against the best signal-only receiver the sensor gain remains only with both ends moving,
  rough sea and long packets, and there only in the tail (worst packets): +5.5 dB at the 10th percentile, +6 dB
  power average, 0 dB median.** Partial FFT closes the seabed and short-packet cases. The earlier +14.6 dB was
  against a weaker baseline on 20 packets. Claim to reframe: robustness / outage, not average SINR.

## T3: Bayesian Cramer-Rao bound (12 Oct 2026; `bcrb.py`, `scripts/b7_bcrb.py`, `results/b7_bcrb/`)
Linear-Gaussian model: residual Doppler on a 10 ms grid over the frame; prior covariance from the auto- and
cross-correlations of up to 1 h of real motion (signal-only: the link Doppler; sensor-aided: the link Doppler
conditioned on the sensor prior, side information + own IMU, grades old / industrial / consumer); measurements =
per-symbol null-subcarrier offsets + postamble timing (+ pilot carrier-phase increments); bound = MMSE error
covariance, mapped to interference as in the simulations. (First attempt with packet-wise sample covariances
overfitted the 435 x 435 matrices and put consumer above industrial; prior from the error covariance alone was also
wrong: it discards the motion statistics.) SIR bound at 30 dB, cfo+timing / +pilots:

| Case | signal only | old IMU model | industrial MEMS | consumer MEMS | sensor prior alone (industrial) |
|---|---|---|---|---|---|
| buoy-buoy 1 s | 43.4 / 47.2 | 50.2 / 51.9 | 48.5 / 50.3 | 43.6 / 47.5 | 8.6 |
| buoy-buoy 4 s | 43.5 / 47.8 | 50.2 / 52.7 | 49.4 / 52.2 | 43.9 / 48.8 | 9.3 |
| seabed OOI 4 s | 44.9 / 48.3 | 51.4 / 54.0 | 49.7 / 52.6 | 45.4 / 49.4 | 13.8 |
| seabed CDIP 1 s | 51.6 / 59.0 | 55.7 / 64.5 | 53.9 / 61.7 | 53.1 / 60.9 | 13.9 |

- **In the linear regime every receiver could be noise-limited:** the signal-only bound is >= 43 dB SIR at 30 dB SNR,
  so SINR ~ SNR. The sensor prior raises the bound by +5-7 dB (industrial), +0.2-1.5 (consumer) — invisible at
  the noise level. The prior alone is far from enough (9-14 dB): sensors must be fused with the signal.
- **The prior replaces pilot tracking:** industrial prior without pilot phases (48.5-49.7) >= signal-only with them
  (47.2-48.3). Pilot count barely matters in the bound (31 vs 248 pilots: 0.2-0.7 dB on OOI).
- **Reading with the simulations:** typical packets reach ~29.7-29.9 dB SINR at 30 dB (near the bound); the
  simulated sensor gain lives in the tail of rough long packets, where signal-only receivers leave the linear regime
  (ICI corrupts the very measurements, wrapping/ambiguities, wrong locks). The mean-square bound cannot see this;
  the right theory for the claim is a threshold analysis (Ziv-Zakai bound, or the probability that the uncorrected
  residual exceeds the measurement's linear range), where the sensor prior moves the receiver away from the
  threshold. Next for T3.

## Threshold analysis, step 1 (12 Oct 2026; `scripts/b8_threshold.py calibrate`) — and a fairness correction
Per packet of the b2 comparison (120 packets, 30 dB), x = largest |mean residual Doppler| over the data FFT windows,
in subcarrier spacings, before fine estimation (signal-only: after the chirps' mean scale; sensor: after the
industrial-MEMS prior). Failure (SINR > 3 dB below SNR): signal Kalman 1% for x <= 1.5, 100% for x > 2 (14 packets);
+ pFFT-8 0% / 64%. The sensor prior keeps x <= 0.6 on every packet.
- **The threshold at x ~ 2 was the +-2-subcarrier search range of our null-subcarrier estimator (CFO_SPAN), not a
  property of signal-only estimation.** With +-4, the four worst buoy-buoy 4 s packets (x 2.8-3.7) go from 10.4 /
  11.6 / 26.6 / 24.8 dB to 28.1 / 28.9 / 29.4 / 29.4 dB (signal Kalman + pFFT-8; sensor fusion 29.3-29.7), with no
  change at 0 / 5 / 10 dB SNR. CFO_SPAN is now 4.
- **Consequence: every chain result with CFO_SPAN = 2 understates the signal-only receivers in rough seas** (e2_chain
  SINR and PER tables, e2_imu_grades, b2 comparison, and the runs started today: high-statistics PER / multipath,
  sea states, numerology, pilot tracking). The remaining sensor gain in the tail of buoy-buoy 4 s packets is likely
  ~0.3-1.5 dB rather than ~5-6 dB; to be confirmed by reruns.

## Acquisition with the sensor prior (12 Oct 2026; `scripts/b6_acquisition.py`, `results/b6_acquisition/`)
Postamble search over the full +-3e-3 Doppler range (signal-only) vs a window centred on the sensor prior (+-4 sigma),
200 packets per point, real motion. Wrong acquisition = |scale error| > 1e-4 among packets whose preamble was found.
buoy-buoy 4 s, -10 / -9 / -8 / -7 dB: preamble miss 85 / 67 / 44 / 15%; wrong lock full search 71 / 36 / 11 / 5%,
industrial prior 13 / 3 / 4 / 0.6%, old IMU model 3 / 0 / 1 / 0%. Seabed OOI 4 s similar (full 59 / 27 / 19 / 5%,
industrial 9 / 6 / 4 / 1%); buoy-buoy 1 s smaller gains. Search cost: 829 -> 98 / 26 lags (4 s), 253 -> 34 / 12 (1 s).
- The prior cuts wrong postamble locks 3-10x and the search 8-30x, but preamble detection fails in the same SNR band,
  so packet acquisition improves by ~1 dB of SNR only (the preamble is detected without the prior). A Doppler-
  sensitive preamble (searched over a Doppler grid) would gain more; with chirps the gain is mostly computational.

## Realistic multipath channel, code (12 Oct 2026; Phase 1; `waveform.warp_delayed`, `channel.py`, `multipath.py`)
- **Fractional path delays** in the channel and in the cancellation's regeneration (both rounded to whole samples
  before, which flattered per-path cancellation).
- **Path clusters** (`LinkMultipath(cluster=(n_sub, angle_spread_deg, delay_spread_s))`): each Bellhop path becomes
  n_sub sub-paths with random angle / delay offsets and complex Gaussian gains drawn per packet. With (5, 0.3 deg,
  0.05 ms) on buoy-buoy OOI: path power varies ~2 dB (std) between packets; Doppler spread within the bottom-bounce
  cluster ~3e-6 (~1.3 rad of relative phase over 4 s: fading within long packets).
- **Sound-speed profile** option (`SUMMER_SSP`, downward refracting) and **receiver-side path prediction from a
  nominal geometry** (`nominal_angles`). Buoy-buoy at 10 m depth: isovelocity gives direct + bottom bounce (10.2 deg,
  10.71 ms); the summer profile with the measured surface roughness leaves a single coherent path (the two-path heave
  problem disappears). Receiver errors of +3 m water depth and +2 % range: bottom-bounce angle 10.3 vs 10.2 deg (within
  the +-0.5-1 deg cancellation tolerance) but delay 11.21 vs 10.71 ms (delays must come from the signal); a wrong
  profile (summer assumed, isovelocity true) predicts one path instead of two.
- Not yet evaluated end to end (multipath reruns on hold, PLAN.md "Decision pending").

## Baseline comparison rerun with the +-4 CFO search (12 Oct 2026; `results/b1_baselines/compare.json`)
Same 30 packets per case as before (old run kept as `compare_span2.json`). Power average / median / 10th percentile
SINR at 30 dB:

| Case | signal Kalman | signal Kalman + pFFT-8 | sensor fusion, industrial | + pFFT-8 |
|---|---|---|---|---|
| buoy-buoy OOI 1 s | 29.1 / 29.4 / 28.3 | 29.8 / 29.9 / 29.6 | 29.7 / 29.8 / 29.4 | 29.8 / 29.9 / 29.6 |
| buoy-buoy OOI 4 s | 26.1 / 27.8 / 24.1 | 29.6 / 29.7 / 29.4 | 29.6 / 29.6 / 29.3 | 29.8 / 29.8 / 29.5 |
| seabed OOI 4 s | 28.2 / 28.2 / 27.1 | 29.8 / 29.8 / 29.7 | 29.7 / 29.7 / 29.5 | 29.8 / 29.9 / 29.7 |
| seabed CDIP 1 s | 29.9 / 29.9 / 29.8 | 29.9 / 29.9 / 29.8 | 29.9 / 29.9 / 29.8 | 29.9 / 29.9 / 29.8 |

At 20 dB, buoy-buoy 4 s: signal Kalman + pFFT-8 19.6 / 19.6 / 19.5, sensor fusion 19.9 / 19.9 / 19.9.
**With fair signal-only baselines the single-path SINR gain of sensor fusion is 0-0.3 dB, also in the tail.** The
sensor receiver reaches it without partial FFT; the signal-only one needs Kalman + partial FFT. Reruns on hold
(PLAN.md, "Decision pending").

## Wave-state Kalman model on real motion (12 Oct 2026; `scripts/b9_wave_kalman.py`) — negative result
Same packets. SINR at 30 dB, power avg / median / p10 (random-walk smoother in brackets, power avg): signal-only
buoy-buoy 4 s 23.7 / 26.6 / 19.4 (26.1); + pFFT-8 29.3 / 29.6 / 28.5 (29.6); sensor fusion industrial 24.9 / 26.6 /
22.4 (29.6); seabed OOI 4 s signal 27.4 (28.2), sensor 27.4 (29.7); buoy-buoy 1 s signal 28.0 (29.1), sensor 29.2
(29.7). **The single-oscillator wave model is worse than the tuned random-walk smoother everywhere** (up to -4.7 dB
for sensor fusion on buoy-buoy 4 s). Likely cause: one oscillator with the half-power bandwidth of the spectral peak
is too rigid for broad-band residuals (several wave components, the sensor residual is not wave-shaped). Not pursued:
the tuned random walk stays the default; a multi-oscillator model is possible but has no remaining gain to recover.

## T2: closed form for the side-information bits (12 Oct 2026; Phase 4; `sideinfo.predicted_interference`, `rd_bits`, `scripts/b10_t2_bits.py`)
I(B) = I_floor + sum_k w_k D_k(b_k): std_k of the transmitter's Legendre coefficients from its motion spectrum, w_k
the interference of a unit coefficient error after the receiver's stage (exact), D_k = std_k^2 if not sent, else
(2 LOAD std_k)^2 / 12 / 4^b_k; I_floor = the "transmitter applies its estimate" level. Against the E1 simulations
(198 points: 3 links x 1 / 4 s x 3 receiver stages x 4-192 bits, 200 packets each): **prediction - simulation
median -0.05 dB, |.| p90 1.1 dB, max 3.1 dB** (the largest misses: per-symbol stage at 12-48 bits, closed form
~1 dB pessimistic; line2 at 4 s, 1-1.7 dB pessimistic at high bits).
Bits to come within 1 dB of applying the estimate, rate-distortion limit (reverse water-filling, ideal quantiser) /
uniform quantiser closed form / simulated:

| Link | 1 s: none / line2 / per symbol | 4 s: none / line2 / per symbol |
|---|---|---|
| seabed CDIP | 16 / 21 / 24; 8 / 11 / 12; 4 / 7 / 8 | 42 / 56 / 64; 45 / 57 / 64; 16 / 27 / 32 |
| seabed OOI | 27 / 34 / 32; 32 / 48 / 48; 20 / 35 / 16 | 77 / 99 / 96; 115 / 162 / >192; 55 / 85 / 64 |
| buoy-buoy OOI | 23 / 30 / 32; 27 / 43 / 64; 21 / 35 / 16 | 67 / 85 / 96; 98 / 139 / 192; 54 / 83 / 48 |

- The closed form reproduces the bit counts within one budget step in most cases; it overestimates for the
  per-symbol stage (35 vs 16 bits), where it is ~1 dB pessimistic at mid budgets.
- The uniform quantiser needs ~1.3-1.7x the rate-distortion minimum (granular loss of a +-4 sigma uniform quantiser,
  ~7 dB per coefficient vs Gaussian R(D)): entropy coding or Lloyd-Max would save ~25-40 % of the bits.
- **Scaling:** rate-distortion minimum ~ 15-30 bits per second of packet on the rough OOI links (4-16 on CDIP),
  growing with packet length roughly linearly (more wave content per packet) and with sea state.

## Side-information ranges per sea-state class (12 Oct 2026; Phase 4; `scripts/b11_sea_classes.py`)
Ranges and allocation from the tested record (in-sample, as before) vs from the other days of its Hs class
(leave-one-out; calm < 1.6 m, moderate 1.6-3 m, rough > 3 m; 15 days, buoy -> seabed, industrial MEMS), mean or
maximum coefficient spread; quantisation interference with the T2 structure (Monte Carlo coefficient errors, clipping
included), 32 bits on 1 s and 96 on 4 s packets, combined with the E1 floors (32 dB no stage, 38.5 dB per symbol).
Total-SIR loss vs in-sample, median / p90: class maximum 0.15 / 0.45 (1 s, none), 0.28 / 0.45 (1 s, per symbol),
0.44 / 1.73 (4 s, none), 0.18 / 0.36 dB (4 s, per symbol); class mean worse in the tail (p90 up to 5.5 dB).
- **Class ranges work if they are conservative (class maximum) and per platform type:** the one outlier (11.8 /
  12.7 dB loss) is the large OOI Irminger buoy on its moderate day, the only OOI buoy in a class of small CDIP
  Waveriders, whose class ranges clip its coefficients. Classes by Hs alone are not enough; per-platform classes, or
  a few bits per packet carrying a range scale, fix it.

## Side information vs transmit-side pre-compensation (12 Oct 2026; Phase 4)
For the intended receiver, E1 / T2: ~15-30 bits per second of packet bring side information within 1 dB of the
transmitter applying its estimate. For other receivers (broadcast, multi-user), pre-compensation along the intended
direction u1 leaves v . (u2 - u1) instead of v . u2. OOI Irminger buoy (fast motion + 2 m lever arm), rms Doppler of
the transmitter share seen by a receiver at azimuth a from the intended one, no pre-compensation / with it:
0 deg 4.5e-4 / 0; 15 deg 4.4e-4 / 1.8e-4; 30 deg 4.7e-4 / 3.6e-4; 60 deg 6.0e-4 / 6.7e-4; 90 deg 7.0e-4 / 8.9e-4;
180 deg 4.5e-4 / 8.9e-4. Pre-compensation helps receivers within ~45 deg of the intended one and hurts beyond ~60 deg
(up to 2x the Doppler at 180 deg). Side information carrying the transmitter's 3-D velocity serves every receiver
(each projects it on its own path direction) at ~1.5-3x the bits of one component; it also needs no real-time
resampling at the transmitter.

## Header in multipath and cost of a wrong header (12 Oct 2026; Phase 4; `scripts/b12_header.py`)
32 bits + CRC-8 in the header symbol (DBPSK across subcarriers, repetition), buoy-buoy OOI 1 s, industrial MEMS,
100 packets per SNR; multipath = Bellhop isovelocity, rough surface, path clusters, fractional delays (direct +
bottom bounce 10.7 ms). Accepted at -3 / 0 / 3 / 6 / 10 dB: single path 0.40 / 0.90 / 1.00 / 1.00 / 1.00;
**multipath 0.21 / 0.46 / 0.80 / 0.91 / 0.97**; accepted-but-wrong 0 in 1000 reads.
- Multipath costs the header ~4-6 dB: DBPSK between adjacent subcarriers suffers from the frequency selectivity of a
  10.7 ms delay (0.5 rad between neighbours, deep fades). A coherent header (channel from the preamble or pilots) or
  stronger coding would recover it. The receiver falls back to signal-only when the header fails.
- **Wrong header accepted** (random quantiser indices; 20 packets, 30 dB; power avg / median / p10): 1 s: safe
  fusion 28.6 / 29.4 / 26.7 = signal Kalman 28.5 / 29.3 / 26.7 (correct header 29.6 / 29.8 / 29.3); fixed-setting
  fusion 0.4 / 0.9 / -18.1. 4 s: safe 27.1 / 27.5 / 25.6 vs signal 26.8 / 27.5 / 25.2; fixed 4.7 / 5.9 / -2.1.
  **The prior-quality guard turns a wrong header into a signal-only packet; without it the packet is lost.**
  With CRC-8 about 1 in 256 corrupted headers passes, so the guard is required.

## Phase 1 remainder (14 Oct 2026): sensor grades from datasheets, tilt / crossover sweep, scattered power
- **Grades anchored on datasheets** (`imu.GRADES`): navigation = Kongsberg MRU 5+ (roll / pitch 0.008 deg RMS, heave
  velocity 0.01 m/s RMS); industrial = Xsens MTi-680G (60 ug/sqrt(Hz), 10-15 ug, 0.2 deg RMS, velocity 0.05 m/s RMS);
  consumer = Bosch BMI088 (160-190 ug/sqrt(Hz)), ~1 deg AHRS tilt (typical, no datasheet). Previous values were
  close (navigation slightly pessimistic).
- **Tilt x crossover sweep** (`scripts/b15_imu_sweep.py`, industrial grade otherwise): wave-band horizontal error on
  OOI Irminger, by crossover 0.01 / 0.03 / 0.1 / 0.3 Hz: tilt 0.01 deg 0.011 / 0.012 / 0.015 / 0.022 m/s; 0.2 deg
  0.038 / 0.037 / 0.022 / 0.022; 1 deg 0.180 / 0.172 / 0.079 / 0.030 (CDIP similar, slightly lower). **The best
  crossover grows with the tilt error** (GNSS should carry more of the wave band when attitude is poor): with it
  tuned, industrial MEMS reach 0.022 m/s and consumer MEMS 0.030 m/s instead of 0.037 / 0.17. The industrial and
  consumer grades were pessimistic; crossovers now 0.1 / 0.3 Hz. Key assumption left: GNSS velocity accuracy at the
  buoy (0.03-0.05 m/s per sample at 5 Hz) and its transfer to the transducer below.
- **Scattered surface power (rho):** no measured coherent-to-incoherent ratio at 10-30 kHz in rough shallow water was
  found. Context: log(CTIR) falls linearly with time after the first arrival, slope proportional to Hs (3.5 kHz,
  JASA 2006); at 17 kHz surface motion reduces acoustic coherence by 20-30 % (Sci. Rep. 2020); Doppler spread grows
  with frequency (JASA EL 2010, 10-18 kHz). rho stays a swept parameter (-30, -20 dB; 0 dB worst case).

## E3 reframed: evaluator with the fair receiver stages and realistic sensors (14 Oct 2026; Phase 3)
`placement.py`: new receiver stages "smooth" / "smooth_s" (the Kalman smoother as a linear estimator, weights from
fusion.smooth on unit impulses; signal-only / sensor settings) and "+pfft8" (partial FFT: interference counted within
the 8 segments of each FFT window); `NodeSensors(grade=...)` takes the node's estimate from imu.RealisticIMU.
**Validation against the fair end-to-end comparison** (`scripts/b16_e3_validate.py`; 4 cases x 4 receivers x 20 / 30
dB): prediction - simulated median SINR, median +0.13 dB, max 1.9 dB (signal-only Kalman, buoy-buoy 4 s, 30 dB:
29.7 vs 27.8; power average 26.1 — the tail of bad packets that the mean-square model cannot see).
The old Phase 3 headline (+4 to +15 dB "at equal budget") compared sensors with a per-symbol piecewise stage (22.9 dB
SIR on buoy-buoy); against the Kalman smoother the node stage adds 1-2 dB of SIR at 38-45 dB, i.e. nothing at the
noise level. **Cost ladder** (SIR at 30 dB; signal-only / with industrial MEMS at both moving ends):

| Receiver machinery | buoy-buoy 1 s | buoy-buoy 4 s | seabed OOI 4 s | seabed CDIP 1 s |
|---|---|---|---|---|
| none (sensors alone) | - / 21.1 | - / 21.1 | - / 24.2 | - / 26.2 |
| two chirp estimates (line) | 6.5 / 33.0 | -7.5 / 23.3 | -4.4 / 25.8 | 19.8 / 43.6 |
| one estimate per symbol | 22.9 / 41.1 | 22.9 / 41.1 | 26.1 / 42.8 | 36.7 / 44.3 |
| Kalman smoother | 39.6 / 41.3 | 40.8 / 41.6 | 42.2 / 43.8 | 44.5 / 45.4 |
| Kalman + partial FFT | 53.2 / 55.6 | 54.9 / 55.9 | 57.1 / 59.4 | 62.0 / 63.3 |

**New E3 rule: sensors substitute for receiver complexity, they do not raise the ceiling.** With the cheapest
receiver (two chirp estimates, no per-symbol tracking) sensors add +17 to +30 dB and make 1 s packets noise-limited up
to ~30 dB SNR — what a signal-only receiver needs a Kalman tracker for; on 4 s packets sensors + per-symbol estimates
match the signal-only Kalman; with Kalman + partial FFT the difference vanishes.
