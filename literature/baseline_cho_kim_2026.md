# Cho and Kim 2026 compared with our strategy

**Paper:** Y.-H. Cho, S.-G. Kim, "Acceleration-Aware Estimation and Compensation of Doppler Scale for Mobile UWA
OFDM Systems", *IET Communications* 20:e70136, 2026, [doi:10.1049/cmu2.70136](https://doi.org/10.1049/cmu2.70136)
(local copy: `~/Underwater/baseline.pdf`). The deck already cites it as an acceleration-aware baseline
("Cho–Kim 2026"). Read in full on 9 Oct 2026.

## Verdict
**The methodology is not the same.** Both papers deal with Doppler that changes during a packet. Beyond that, they
solve different problems:
- **Cho–Kim** improve the receiver's estimate of a *constant acceleration within one packet*, with no memory between
  packets.
- **We** predict a *time-varying* motion *across* packets, from both ends of the link. The predictions are used for
  decisions taken before the packet is sent.

The paper is a strong baseline and a natural component of our receiver. It does not compete with the forecast. It
does narrow the regime where our receiver block (Block 1) can show a gain, and that regime has to be stated in
advance.

## What the paper does
- **Model:** the delay along the path is quadratic in time, τ_d(t) = t₀ + a₁t + ½a₂t², meaning constant velocity v₀
  plus constant acceleration v̇ over the frame (a₁ = v₀/c, a₂ = v̇/c).
- **Velocity term (a₁):** estimated from a preamble with two identical halves (Mason et al., JSAC 2008), then removed
  by resampling and frequency correction.
- **Acceleration term (a₂):** a new estimator, the paper's contribution.
  - In each OFDM symbol, it correlates the cyclic prefix (CP) with its copy at the end of the symbol, over a small
    search of timing offsets.
  - The phase difference of this correlation between consecutive symbols is proportional to a₂, which gives a
    closed-form estimate (their Eq. 13).
  - Symbols whose correlation falls below a threshold (0.4) are discarded.
- **Compensation:** a frequency correction quadratic in time, then resampling along the inverse quadratic warp, all in
  baseband. It costs no extra overhead and few multiplications. It beats the multibranch autocorrelation method
  (Li–Zakharov–Henson, JOE 2017) in both accuracy and complexity.
- **Evaluation:** AWGN, a synthetic 10-path channel, and the Waymark simulator (Bellhop-based; Liu–Zakharov–Chen,
  TVT 2012) with v = 2 m/s and v̇ = 0.4 m/s², in a 1 km, 100 m deep geometry.
  - Parameters: 16 kHz carrier, 8 kHz bandwidth, 1024 subcarriers, frames of 6 OFDM symbols (about 0.9 s).
  - Metrics: RMSE of a₂, and uncoded BER for QPSK and 16-QAM.
  - No sea trial, no coding, and only constant accelerations in [−0.5, 0.5] m/s² (plus 1 m/s² in one figure).
- **Main numbers:**
  - Without acceleration compensation, BER degrades from about 0.06 m/s² (QPSK) and 0.04 m/s² (16-QAM).
  - With it, BER is close to the no-Doppler curve.
  - The a₂ RMSE is under 5% of the true value at 0.5 m/s² for SNR ≥ 5 dB.
  - The estimator degrades below 0 dB SNR, and when the residual velocity error exceeds about 0.2 m/s.
- **Framing:** the paper targets AUV manoeuvres. It states that compared with surface waves or currents, "the motion
  of AUVs … is typically more pronounced".

## Side-by-side comparison

| | Cho–Kim 2026 | Our project (README + proposed revision) |
|---|---|---|
| Question | Estimate and remove the acceleration within a received frame | Predict the Doppler drift before the packet exists, and use the prediction to adapt the link |
| Motion model | Constant velocity + constant acceleration, per frame | Drift + damped oscillators (swell), carried across packets |
| Memory | None: every frame starts from zero | State-space filter over the whole exchange |
| Measurements | One end (receiver), one frame: preamble + CP | Both ends, every data packet and ACK |
| Where it acts | Receiver only, after reception | Transmitter decision (length, modulation, pilots); receiver pre-warp; proposed: transmitter pre-compensation |
| Uncertainty | Point estimate; threshold on correlation quality | Forecast with σ_e, which drives the decision rule and the safety margin |
| Overhead | None (uses the CP) | About 20 header bits per ACK, plus the filter |
| Validation | AWGN, synthetic multipath, Waymark; constant acceleration | Planned: Channel Library replay; simulation for the two-way part |
| Venue level | IET Communications (letter-like scope) | JSAC: needs system-level contribution and stronger evidence |

## Where it is strong, and what that means for us
1. **For packets of about 1 s, it already solves the receiver problem.** With 0.9 s frames and accelerations up to
   0.5 m/s², it restores BER to near ideal at no overhead and low complexity. A forecast-aided receiver (Block 1)
   will not beat it on PER in that regime. This agrees with the surface-buoy trial (Lee et al. 2026, at most 1.15×).
2. **Reviewers will compare our overhead with its zero overhead.** Our ACK field and filter must buy something this
   method cannot deliver.
3. **It confirms our gap G2 as worded.** The motion model is confined to one frame and re-estimated from scratch every
   time. That is exactly the "kinematic models stay inside one packet" point; cite it there.

## Where it stops
1. **Constant acceleration fails for swell as packets get longer.** For a swell-like relative velocity of 0.5 m/s
   amplitude and 8 s period, the best constant-acceleration fit over one frame leaves, at the worst swell phase:

   | Frame length | Residual velocity | Offset at 16 kHz | Share of subcarrier spacing (7.8 Hz) |
   |---|---|---|---|
   | 0.9 s (their frame) | 0.02 m/s | 0.22 Hz | 3% |
   | 2 s | 0.10 m/s | 1.0 Hz | 13% |
   | 3 s | 0.21 m/s | 2.2 Hz | 28% |
   | 4 s | 0.34 m/s | 3.6 Hz | 46% |

   These are my own numbers from an assumed swell, not measured data; the Channel Library analysis should replace
   them. Beyond about 2 s, the model they assume no longer holds for swell motion. A tracker that re-estimates a₂ on
   sliding windows would partly recover, at the cost of fewer symbols per estimate.
2. **The estimate degrades towards the end of the frame.** Their own Fig. 1 shows that at 1 m/s², only the first two
   symbols keep a correlation above 0.7. The distortion grows with the sample index because a₁ comes only from the
   preamble. Pre-warping along a forecast would remove most of the warp before the CP correlation. That keeps the
   correlation high in later symbols, which is what long packets need.
3. **At low SNR (below 0 dB), both stages degrade** (their Figs. 3 and 9). A prior from the forecast is most useful
   exactly there, i.e. at long range.
4. **It cannot act before transmission.** It has nothing to say about packet length, modulation or pilot layout, and
   it cannot pre-compensate at the transmitter.
5. **It does not extend directly to a multi-user uplink.** The CP correlation of a received sum mixes users with
   different Doppler. Separating them needs per-user subband filtering, and the interference between users is created
   before any receiver can separate them. This supports the proposed shift to transmitter-side pre-compensation.

## Consequences for our strategy
- **Use it as the receiver baseline, and as our tracker.** The baseline configuration is Mason 2008 preamble for a₁
  plus Cho–Kim for a₂, re-estimated every packet. In Block 1, the component that "tracks only the forecast error" can
  be the Cho–Kim estimator applied after the forecast pre-warp. The claim then becomes "forecast + Cho–Kim vs Cho–Kim
  alone": a complement, not a rival, which is easier to defend.
- **Declare the winning regime in advance.** Packets of 2 s or more, swell-dominated motion and SNR around or below
  0 dB. The table above gives the first argument; R1 on Channel Library data must confirm it.
- **Turn their BER thresholds into a forecast-accuracy target.** BER starts to degrade at 0.04–0.06 m/s² of
  uncompensated acceleration. A forecast that let the receiver skip in-packet acceleration estimation would need an
  acceleration error below about 0.05 m/s² at a 2–3 s horizon, about 13% of a 0.39 m/s² swell amplitude. That is
  demanding. It supports using the forecast as a prior that a tracker refines, not as a replacement.
- **Check their claim that waves are minor, early.** It is the opposite of our scope assumption, and reviewers may
  share it. The Doppler series from the Channel Library (the go/no-go check) should show how large the swell
  component really is.
- **Their results strengthen the case for the transmitter-side revision.** Receiver-side compensation of
  acceleration is close to solved for short packets. What no receiver can do is act before transmission, or undo
  interference between users with different Doppler. That is where a forecast is necessary rather than merely
  helpful.
- **Reuse their simulation tool.** Waymark accepts arbitrary trajectories, including oscillating ones, and uses
  Bellhop (already in the DESERT build). It is a credible way to simulate swell-driven motion and multiple users,
  which the one-way Channel Library cannot provide.

## Where to cite it
- **G2:** within-packet kinematic models (with Hu 2024 and Abdelkareem 2016).
- **Related work (2):** "Acceleration-aware compensation" row (already there).
- **Evaluation:** the receiver baseline.
- **Block 1:** the residual tracker.
