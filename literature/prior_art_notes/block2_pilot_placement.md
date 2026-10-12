# Block 2 prior art: forecast-driven pilot placement and packet length (UWA and terrestrial)

Scope: works that adapt pilot density or positions, training, pre/mid/postambles, or packet/frame length to Doppler,
mobility or a channel-state forecast, judged against Block 2. Block 2 is: a uniform grid P0 at the K_max set by tap
coherence, plus *extra* training symbols placed to minimise the integrated forecast-error variance of a kinematic
(drift plus oscillator) Kalman forecast under a pilot budget, solved greedily, with extras expected near the zero
crossings of the swell drift (jerk peaks); packet length N_sym chosen per packet.

Where I searched (8 Oct 2026):
- Local library (`~/Underwater/papers`, `tools/search.py`): "pilot adaptive", "pilot placement", "pilot pattern",
  "pilot density", "pilot spacing", "pilot overhead", "pilot insertion", "pilot allocation", "pilot interval",
  "variable pilot", "pilot Kalman", "midamble" (0 hits), "superimposed pilot", "pilot design Doppler", "training
  sequence Doppler", "training length adaptive", "training overhead", "OTFS pilot", "packet length adaptive", "packet
  duration", "packet size optimization", "frame length", "adaptive frame", "coherence time packet", "reconfigurable
  OFDM", "channel prediction pilot", "Doppler adaptive", "mobility adaptive", "preamble postamble".
- OpenAlex API: cited-by lists of Zakharov 2019 (26 citing works), Hong 2025 (0), Van der Werf UComms 2024 (1) and
  Dong–Tong–Sadler TSP 2004 (222), plus 16 keyword queries (adaptive pilot underwater; Doppler-adaptive pilot
  spacing; nonuniform pilot placement with prediction; pilot placement Kalman; adaptive frame/packet length for UWA;
  OTFS pilot adaptation; adaptive superimposed pilot power; DMRS adaptation; pilot scheduling with channel aging).
- WebSearch: 15 queries covering UWA, LTE/NR DMRS, HSR, LEO, radar revisit, sensor scheduling, submodular selection
  and swell-induced Doppler.
- Claude in Chrome was **not available** (extension not connected), so I could not use Google Scholar or IEEE Xplore
  cited-by. I used the OpenAlex cited-by graph instead. Its coverage is good for IEEE/JASA but not complete.

## Q1. UWA: is there adaptive pilot/training design or adaptive packet length driven by Doppler or motion, online or per packet?

### Takeaway
No. I found no UWA work that sets pilot positions or packet length per packet from a *predicted* Doppler or motion
state, and none that places training where a Doppler forecast is uncertain. UWA works fall into four groups:
(a) offline packet designs for a worst-case motion range (Zakharov 2019, Hong 2025); (b) offline or statistics-driven
pilot designs (OTFS: Van der Werf 2024/2025; compressed-sensing pilot position optimisation); (c) adaptive pilot
*allocation across OFDMA users* by channel condition (Yang et al. GLOBECOM 2025); (d) modems that adapt CP length and
subcarrier count to the measured delay and Doppler spread (Mangione et al. 2021). Group (d) is the closest UWA
example of "frame adapts to Doppler", but it is reactive, works on spreads, and leaves pilots alone.

### Cited Findings
- **Zakharov, Henson, Diamant, Fei, Mitchell, Morozs, Shen, Tozer, "Data Packet Structure and Modem Design for Dynamic
  Underwater Acoustic Channels," IEEE JOE, 2019, DOI 10.1109/JOE.2019.2932650.** Superimposed data and periodic pilots.
  The packet parameters are sized from a *maximum* speed and acceleration: "The parameter Na is chosen to guarantee
  Na > Θ fc aM/(2cΔf), where aM is the maximum acceleration" (local PDF, Sec. IV). Example design: packet duration
  Θ ≈ 1.0 s, search over v ∈ [−5, 5] m/s and a ∈ [−0.25, 0.25] m/s²; simulations use a ∈ [−1, 1] m/s². The design is
  fixed offline and has no online adaptation. — [IEEE Xplore](https://ieeexplore.ieee.org/document/8815825/); local
  `papers/pdf/02_doppler_sync_estimation/2019_Zakharov_DataPacketStructureAndModemDesign.pdf`. **Closeness: partial** (it
  names the right levers, pilot period and packet length against acceleration, but chooses them once).
- **Of the 26 works OpenAlex lists as citing Zakharov 2019**, none adapts pilots or packet length online. They are
  acceleration-aware estimators (Cho and Kim, IET Commun. 2026, DOI 10.1049/cmu2.70136: CP-based acceleration
  estimation "with no additional signaling overhead"), random-access interference cancellation (JOE 2025, DOI
  10.1109/JOE.2024.3507813), MAC, cyclostationarity and channel modelling. — [OpenAlex work record](https://api.openalex.org/works/doi:10.1109/joe.2019.2932650)
  (cited-by list via the `cites:` filter on its OpenAlex ID).
- **Hong, Jeong, Lee, Jung, "An optimal direct sequence spread spectrum packet design for underwater acoustic
  channels with multipath and Doppler effect," JASA (meeting abstract), 2025, DOI 10.1121/10.0037819.** The abstract says
  the spread data format must "consider channel state information (CSI) such as coherent time and coherent bandwidth"
  and that the authors propose, "through simulations," an "optimal DSSS packet design model." This is an offline
  design rule, not per-packet adaptation. OpenAlex lists 0 citations. — [DOI](https://doi.org/10.1121/10.0037819). **Closeness: partial/tangential.**
- **Van der Werf, Pelekanakis, Hendriks, Heusdens, Leus, "Optimized Pilot Design for OTFS Modulation in Underwater
  Acoustic Communications," UComms 2024, DOI 10.1109/UCOMMS64662.2024.10847980.** A pilot allocation and power split
  that "can be adjusted in a simple way to the delay and Doppler spread in the channel." This adapts to spread
  *statistics*, not to a forecast of bulk drift. Journal version: **van der Werf, Heusdens, Hendriks, Leus, "Optimal
  pilot design for OTFS in linear time-varying channels," Signal Processing, 2025, DOI 10.1016/j.sigpro.2025.110223**
  (arXiv 2403.19379): pilot placements that minimise the MSE of the channel taps, saving about 50% of pilot overhead. Its
  one citing work (Jia, Peng, Ai, Cui, WUWNet 2025, DOI 10.1145/3784941.3784943) is a DQN modulation selector using SNR
  and Doppler spread as state, with no pilots. — [arXiv](https://arxiv.org/abs/2403.19379); [OpenAlex](https://api.openalex.org/works/doi:10.1145/3784941.3784943). **Closeness: partial.**
- **Yang, Fan, Boukerche, Huang, Su, "Optimal Allocation of Scarce Pilot Resources for OFDMA-Based Underwater
  Acoustic Sensor Networks," GLOBECOM 2025, DOI 10.1109/GLOBECOM59602.2025.11432537.** "we propose an adaptive pilot
  allocation method that dynamically optimizes the number of pilots assigned to each user," followed by joint pilot
  position and power optimisation that minimises mutual coherence for compressed sensing. The adaptation is across
  users and in frequency (CS measurement matrix). It is not temporal and not Doppler-forecast driven. — local DB
  abstract; [DOI](https://doi.org/10.1109/globecom59602.2025.11432537). **Closeness: tangential.**
- **Mangione, Galioto, Croce, Tinnirello, Petrioli, "A Channel-Aware Adaptive Modem for Underwater Acoustic
  Communications," IEEE Access, 2021, DOI 10.1109/ACCESS.2021.3082766.** "The temporal variability of the channel
  behavior is summarized in terms of maximum delay spread and Doppler spread … we also calibrate the prefix length and
  the number of sub-carriers," and the modem can fall back to JANUS/FSK. The adaptation is reactive (measured spreads),
  changes symbol numerology rather than pilot positions, and has no forecast. — [OpenAlex abstract](https://api.openalex.org/works/doi:10.1109/access.2021.3082766). **Closeness: partial** (the closest UWA frame-adapts-to-Doppler example).
- **Nguyen Thi Thanh Nga, "Impact of Doppler effect on pilot augmentation techniques for channel estimation in
  UWA-OFDM systems," J. Military Sci. Technol., 2025, DOI 10.54939/1859-1043.j.mst.iite.2025.99-106.** A comparative
  study whose results support "the selection of appropriate pilot structures to mitigate Doppler effects." This is
  offline selection, not adaptation. — local DB abstract. **Closeness: tangential.**
- **Berger, Huang, Moura, "Study of pilot overhead for iterative OFDM receivers on time-varying and sparse underwater
  acoustic channels," OCEANS'11, DOI 10.23919/OCEANS.2011.6107108.** Sweeps pilot overhead on recorded data; "the
  performance saturates at some point," and this point depends on ICI modelling and iterative receivers. It is an
  offline overhead study, relevant to K_max and to treating decision-directed pilots as observations. — local DB
  abstract. **Closeness: tangential.**
- Other UWA pilot-design works found are all offline compressed-sensing or position optimisations with no Doppler or
  motion adaptation: Cho et al., "Pilot allocation design and experimental results for underwater OFDM systems," OCEANS
  2017 Aberdeen, DOI 10.1109/OCEANSE.2017.8084627; Jiang et al., IEEE Access 2019, DOI 10.1109/ACCESS.2019.2929305
  (whale-optimisation pilot allocation); Li et al., Applied Acoustics 2024, DOI 10.1016/j.apacoust.2024.109933
  (non-orthogonal pilot pattern); Ma et al. 2015 (Acta Phys. Sin.); Yu et al. ICSIDP 2019. — local DB.
- UWA packet-length works are MAC-level: energy or lifetime packet-size optimisation (Yıldız et al., IEEE TII 2018,
  DOI 10.1109/TII.2018.2841830; Yiğit et al. survey, IJCS 2018, DOI 10.1002/dac.3572) and variable-duration scheduling
  (Anjangi and Chitre, JOE 2017, DOI 10.1109/JOE.2016.2637098). None ties packet length to Doppler or coherence time. —
  local DB.
- The user's own survey, **Busacca, Galluccio, Palazzo, Panebianco, Qi, Pompili, "Adaptive versus predictive techniques
  in underwater acoustic communication networks," Computer Networks, 2024, DOI 10.1016/j.comnet.2024.110679**, classifies
  reactive and proactive UWA techniques. It should be checked for any predictive pilot or packet-length entry; I could
  not read the full text (403). — [OpenAlex](https://api.openalex.org/works/doi:10.1016/j.comnet.2024.110679).
- I found no UWA midamble work: "midamble" returns 0 hits in the local library, and web queries returned only
  superimposed-training receivers (e.g., [Sensors 2022](https://doi.org/10.3390/s22062142)), none adapting the training.

### Inferences
- The G1 statement ("pilot layout and packet length are fixed offline for an assumed motion range") holds across
  everything found. The strongest counter-examples are Mangione 2021 (reactive numerology vs Doppler spread) and Yang
  2025 (adaptive per-user pilot count). Neither is temporal, kinematic or forecast-driven.
- The deck's "4 papers, none forecast-driven" count is consistent with the local DB. A fuller list for a related-work
  paragraph would add Yang 2025, Mangione 2021, Nga 2025 and Hong 2025.

### Gaps
- No Google Scholar or IEEE Xplore cited-by lists (browser unavailable). The Zakharov 2019 cited-by list comes from
  OpenAlex only. Chinese-language venues (Acta Acustica, J. Harbin Eng. Univ.) are covered only through the local DB.
- I did not read full texts of Hong 2025 (meeting abstract only), Yang 2025 or Mangione 2021. Mangione should be read
  to confirm that its pilot layout is fixed.

## Q2. Terrestrial analogues: adaptive pilot patterns, Doppler-adaptive pilot spacing, DMRS adaptation, HSR, V2X, LEO, OTFS

### Takeaway
Adapting pilot *density* or *spacing* to Doppler spread, velocity or coherence time is well established terrestrially
and is standardised in 5G NR (dmrs-AdditionalPosition 0–3). Adaptive *non-uniform* pilot locations by feedback
also exist (Panah–Vaughan–Heath 2008; MDP-based V2X placement 2018). In every case the driver is a second-order
statistic (Doppler spread, coherence time, UE speed, fading profile) or the current channel estimate. None is a
forecast of a deterministic kinematic Doppler trajectory, and none places pilots by the time-profile of forecast-error
variance within a packet.

### Cited Findings
- **Rao, Marojevic, Reed, "Adaptive Pilot Patterns for CA-OFDM Systems in Nonstationary Wireless Channels," IEEE TVT,
  2017 (arXiv 1709.03176).** Codebook-based: the transmitter and receiver "exchange information about the fading
  characteristics of the channel over a short period of time," stored "as entries in a channel profile codebook," and
  a heuristic picks pilot spacing and power from candidates to maximise rate. Time spacing follows coherence time. —
  [arXiv](https://arxiv.org/abs/1709.03176). **Closeness: partial** (online, closed-loop spacing adaptation, but uniform
  spacing driven by a fading statistic).
- **Panah, Vaughan, Heath, "Optimizing Pilot Locations Using Feedback in OFDM Systems," IEEE TVT, 2008, DOI
  10.1109/TVT.2008.2008655.** "in closed-loop OFDM systems … uniform pilots may no longer be optimal … we propose a
  feedback technique in which the pilot allocation mechanism is adapted to the mobile channel." It optimises
  non-uniform positions (in frequency) for SER or capacity, with vector-quantised feedback. — [OpenAlex abstract](https://api.openalex.org/works/doi:10.1109/tvt.2008.2008655).
  **Closeness: partial** (non-uniform pilots adapted by feedback; frequency domain, driven by current CSI, no forecast).
- **Yang, Dang, He, Guizani, "Markov Decision-Based Pilot Optimization for 5G V2X Vehicular Communications," IEEE IoT-J,
  2018, DOI 10.1109/JIOT.2018.2867334.** "a finite MDP so as to determine an appropriate pilot pattern from a set of
  candidate pilot configurations" for fast time-varying vehicular channels, with mutual-information criteria. —
  [OpenAlex](https://api.openalex.org/works/doi:10.1109/jiot.2018.2867334). **Closeness: partial.**
- **Savazzi, Spagnolini, "Optimizing Training Lengths and Training Intervals in Time-Varying Fading Channels," IEEE
  TSP, 2008, DOI 10.1109/TSP.2008.2009270.** Optimal training length and interval depend "on the combination of Doppler
  frequency and frame length". For fast fading it is better to fragment "the frame with dispersed segments of training
  symbols" than to put all training at the start. This is the frame-length plus midamble-spacing trade-off against Doppler
  spread. Offline. Also Savazzi and Spagnolini, "On the pilot spacing constraints for continuous time-varying fading
  channels," IEEE TCOM 2009, DOI 10.1109/TCOMM.2009.11.080024. — [OpenAlex](https://api.openalex.org/works/doi:10.1109/tsp.2008.2009270). **Closeness: partial** (the frame-design part, statistic-driven).
- **Šimko, Wang, Rupp, "Optimal pilot pattern for time variant channels," ICC 2013, DOI 10.1109/ICC.2013.6655451.** For LTE:
  near-optimal pilot distance and pilot/data power under time variation (post-equalisation SINR). — [OpenAlex](https://api.openalex.org/works/doi:10.1109/icc.2013.6655451). **Tangential/partial.**
- **5G NR DMRS:** "dmrs-AdditionalPosition … allows setting values of zero, one, two or three" extra DMRS symbols per
  slot, mainly for high-speed UEs. Studies optimise the density per speed (Tomić et al., "Optimized DM-RS Configuration
  for Improved 5G New Radio Network Capacity and Performance," 2024, tests 40–500 km/h), and patents describe UE-assisted
  dynamic DMRS signalling ("Demodulation reference signal configuration and adaptation," US 11368268). —
  [ResearchGate (Tomić)](https://www.researchgate.net/publication/380828748_Optimized_DM-RS_Configuration_for_Improved_5G_New_Radio_Network_Capacity_and_Performance); [ACM 2023 adaptive DMRS count](https://dl.acm.org/doi/fullHtml/10.1145/3641584.3641703); [USPTO 11368268](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11368268).
  **Closeness: partial** (an industrial precedent for "extra pilots on top of a base grid, chosen by mobility"; driven by
  speed or Doppler-spread class, uniform positions inside the slot).
- **HSR:** Nguyen, Dao, Ha, "Adaptive Pilot-Assisted Channel Estimation for OFDM-Based High-Speed Railway
  Communications," Electronics (MDPI), 2026, DOI 10.3390/electronics15101991: a "channel-aware adaptive pilot insertion
  (CA-API) mechanism … to adjust pilot density based on temporal channel variations," with 38% and 30% overhead
  reduction in two scenarios. The adaptation is reactive (density), not forecast or position-optimal. MDPI is excluded
  from the owner's library, but it is cited here as evidence. — [OpenAlex](https://api.openalex.org/works/doi:10.3390/electronics15101991). **Partial/tangential.**
- **Velocity-aware pilot interval:** Hashida, Kawamoto, Kato, "Adaptive Pilot Interval Optimization for IRS-Aided
  Communication Systems," IEEE TVT 2022, DOI 10.1109/TVT.2022.3166639: "The channel coherent time depends on UE
  velocity … the proposed method aims to find a pilot interval … by considering UE velocity." — [OpenAlex](https://api.openalex.org/works/doi:10.1109/tvt.2022.3166639).
  **Partial** (uniform interval set by speed).
- **Mobility-aware CSI update or pilot spacing (channel aging):** Deng, Jiang, Zhou, Niu, "Intermittent CSI Update for
  Massive MIMO Systems With Heterogeneous User Mobility," IEEE TCOM 2019, DOI 10.1109/TCOMM.2019.2911575 (MDP/convex
  per-user CSI update pattern from temporal correlation); the Globecom 2017 precursor finds update intervals are
  "quasi-periodic"; Fodor et al., "Optimizing Pilot Spacing in MU-MIMO Systems Operating Over Aging Channels," arXiv
  2204.00213. — [OpenAlex](https://api.openalex.org/works/doi:10.1109/tcomm.2019.2911575); [arXiv](https://arxiv.org/abs/2204.00213). **Partial** (when to send pilots, from a statistical aging model).
- **LEO:** systems pre-compensate Doppler with ephemeris, and pilots estimate the residual. "leverage ephemeris
  information to pre-compensate Doppler shifts, with the residual Doppler effect estimated by pilots" (arXiv 2601.01956,
  affine-frequency-domain pilot plus LSTM predictor). This is a Block-1 analogue (pre-warp along a known trajectory). I
  found no LEO work that moves pilots according to ephemeris uncertainty. — [arXiv 2601.01956](https://arxiv.org/html/2601.01956v1). **Tangential for Block 2.**
- **Adaptive *training power* driven by Kalman state:** Agarwal, Honig, Ata, "Adaptive Training for Correlated Fading
  Channels With Feedback," IEEE Trans. IT 2012, DOI 10.1109/TIT.2012.2201345. The channel is an OU process tracked by a
  Kalman filter, and the optimal pilot policy is bang-bang with a "switching boundary … defined in terms of the system
  state (estimated channel mean and associated error variance)." — [OpenAlex](https://api.openalex.org/works/doi:10.1109/tit.2012.2201345).
  **Large conceptual overlap with the mechanism** (training decided online from the Kalman error variance, which
  amounts to placing pilot energy in time where it is needed). Differences: fading-gain state (OU), not a kinematic Doppler
  oscillator; power control, not discrete training-symbol positions in a packet; no base grid; not UWA.
- OTFS terrestrial pilot design (embedded or guard pilots sized to the maximum delay and Doppler) is offline, e.g.,
  [arXiv 1808.08360](https://arxiv.org/abs/1808.08360) and [LWC 2022](https://doi.org/10.1109/lwc.2022.3160657). — **Tangential.**

### Inferences
- The idea "more pilots when mobility is higher" is old and standardised, so Block 2 must not claim it. The novelty
  has to sit in (i) the driver: a forecast of a deterministic, oscillatory bulk drift and its error variance, shared
  by both ends; (ii) the *positions* inside a packet set by the time-profile of σ_e²(u) rather than a uniform spacing;
  (iii) the base grid plus extras structure under a budget.
- 5G NR "additional DMRS" is a close precedent for the base-grid-plus-extras structure. The deck should cite it to
  pre-empt reviewers, noting that NR's extras are uniform and chosen by speed class.

### Gaps
- I did not check 3GPP NR-V2X sidelink DMRS (2/3/4 symbols, UE-selected by speed) or 3GPP NTN contributions in primary
  documents. Both are likely further speed-class precedents, not forecast-driven ones.
- I found no HSR work that uses the deterministic train trajectory to place pilots, although HSR uses position for
  channel estimation ([arXiv 2003.02698](https://arxiv.org/pdf/2003.02698)). I cannot rule out such work without
  Scholar.

## Q3. Is there pilot (or measurement) placement that minimises the forecast or Kalman error variance of a kinematic or oscillatory process?

### Takeaway
In communications, the classical results (Dong–Tong–Sadler) optimise pilot placement for Kalman tracking of a
*stationary* Gauss–Markov fading process and find **periodic, single-pilot placement optimal**. This is what Block 2's
uniform grid reproduces, and it shows that non-uniform extras need a non-stationary, deterministic-mean component
such as a phase-uncertain swell. The direct methodological analogues of Block 2 are outside communications: radar
revisit-time scheduling from the predicted covariance, sensor scheduling for Kalman filters (greedy with
submodularity caveats), and AoI/remote-estimation sampling (threshold on the estimation error). None of them is
applied to Doppler-scale forecasts or packet pilots.

### Cited Findings
- **Dong, Tong, Sadler, "Optimal pilot placement for channel tracking in OFDM," MILCOM 2002, DOI
  10.1109/MILCOM.2002.1180512.** Gauss–Markov channel, Kalman tracker, minimise steady-state MSE: "single pilot periodic
  placement achieves the minimum MSE." Also **Dong, Tong, Sadler, "Optimal insertion of pilot symbols for transmissions
  over time-varying flat fading channels," IEEE TSP 52(5):1403–1418, 2004, DOI 10.1109/TSP.2004.826182** (222 citations
  in OpenAlex). — [IEEE Xplore](https://ieeexplore.ieee.org/document/1180512/); [OpenAlex](https://api.openalex.org/works/doi:10.1109/TSP.2004.826182).
  **Partial** (same objective, Kalman MSE vs pilot positions, but stationary channel taps, so the answer is uniform).
- **Radar adaptive revisit, the closest structural analogue:** "The revisit interval is selected adaptively, based on
  the predicted angular innovation standard deviations … the latest time at which the target can be expected to be in
  the predicted track gate" (summary of the van Keuk–Blackman line of work). **Cheng, Zou, He, "Adaptive waveform and
  sampling interval tracking based on estimation accuracy for Doppler radar," IET Int. Radar Conf. 2013, DOI
  10.1049/cp.2013.0265**: "both the sampling interval and transmitting waveform can be adjusted according to the
  predicted error covariance output from IMM filter adaptively." Also Zhang and Tian, EURASIP JASP 2016, DOI
  10.1186/s13634-016-0309-3 (sampling interval, power and waveform shape the predicted covariance; MDP scheduling). —
  [OpenAlex](https://api.openalex.org/works/doi:10.1049/cp.2013.0265); [EURASIP](https://asp-eurasipjournals.springeropen.com/articles/10.1186/s13634-016-0309-3).
  **Large methodological overlap, different domain**: measurement times chosen from a kinematic Kalman prediction
  covariance under a resource budget. This is what Block 2 does with training symbols. It is not communications and
  has no packet, pilot grid or two-way fusion.
- **Sensor scheduling and submodularity:** Krause, Singh, Guestrin, "Near-Optimal Sensor Placements in Gaussian
  Processes," JMLR 9, 2008 (the deck's cited bound). However, **Jawaid and Smith, "Submodularity and greedy algorithms in
  sensor scheduling for linear dynamical systems," Automatica, 2015, DOI 10.1016/j.automatica.2015.08.022** show "the
  sensor schedule cost is not, in general, a submodular function, contradicting an established result," and give
  checkable sufficient conditions for submodularity. — [PDF](https://ece.uwaterloo.ca/~sl2smith/papers/2015Aut_Greedy_Submodularity.pdf); [OpenAlex](https://api.openalex.org/works/doi:10.1016/j.automatica.2015.08.022).
  **This bears directly on R3**: a greedy bound for integrated Kalman error variance cannot be assumed. It needs
  Jawaid–Smith-type conditions or a weak-submodularity ratio (cf. randomized greedy with weak submodularity,
  [NSF PAR](https://par.nsf.gov/servlets/purl/10440126)).
- **Remote-estimation sampling (AoI):** Sun, Polyanskiy, Uysal-Biyikoglu, "Remote estimation of the Wiener process over a
  channel with random delay," ISIT 2017, DOI 10.1109/ISIT.2017.8006542 (journal version IEEE Trans. IT 2020): "the optimal
  sampling strategy is a threshold policy." Ornee and Sun, "Sampling and Remote Estimation for the Ornstein-Uhlenbeck
  Process Through Queues," IEEE/ACM ToN 2021, DOI 10.1109/TNET.2021.3078137. — [OpenAlex](https://api.openalex.org/works/doi:10.1109/isit.2017.8006542); [OpenAlex](https://api.openalex.org/works/doi:10.1109/tnet.2021.3078137).
  **Tangential/partial**: signal-aware sampling where the error grows, but for Wiener or OU processes and network
  updates, not in-packet pilots. Relevant to Block 3 / R2 (delayed, irregular samples) as well.
- Continuous-discrete Kalman sensor scheduling with measurement-rate optimisation (ICML 2025 poster, "Optimal Sensor
  Scheduling and Selection for Continuous-Discrete Kalman Filtering with Auxiliary Dynamics"). — [ICML](https://icml.cc/virtual/2025/poster/46077). **Tangential.**

### Inferences
- Dong–Tong–Sadler gives Block 2 a clean null result to build on: for stationary taps, uniform is optimal, which
  justifies P0. The extras can only help because the drift has a deterministic oscillatory mean with uncertain phase
  and amplitude, which makes σ_e²(u) non-stationary within the packet. This framing differentiates the contribution
  and should be stated explicitly.
- The radar revisit literature is the nearest prior art for the optimisation idea. It should be cited, and the
  contribution framed as bringing covariance-driven measurement scheduling into packet design for a two-way-fused
  Doppler forecast, not as inventing it.
- The deck's reliance on Krause 2008 for a greedy bound is fragile. Jawaid–Smith 2015 must be addressed.

### Gaps
- I did not find closed-form optimal measurement times for a Kalman-tracked *sinusoid with uncertain phase*. The
  question is classical enough (optimal sampling for frequency/phase estimation, CRB-optimal sampling of sinusoids) that
  results may exist under "optimal sampling design sinusoid parameter estimation" or "optimal experiment design harmonic
  oscillator." I did not search this specifically, and it is the most likely place for a prior closed form for R3.

## Q4. Is pilot placement at zero crossings or high-jerk instants of a periodic Doppler studied anywhere?

### Takeaway
No. I found nothing that places pilots, training or measurements at the zero crossings of a periodic Doppler or
drift, or at jerk peaks. Swell-induced Doppler is studied as an impairment or for receiver tracking, not for pilot
timing.

### Cited Findings
- No hits for zero-crossing or jerk-driven pilot placement in the local DB (pilot*/training queries above), in OpenAlex
  keyword queries, or in WebSearch ("wave-induced motion periodic Doppler prediction pilot surface buoy swell").
  Results were about coherence loss from surface waves, e.g., "Predicting the Effects of Random Ocean Dynamic Processes
  on Underwater Acoustic Sensing and Communication," Sci. Rep. 2020 (surface-wave Doppler spread "reduce[s] acoustic
  coherence by more than 20% within seconds"), and moored-buoy motion affecting handovers (Yun and Kwon, Sensors 2024,
  DOI 10.3390/s24123797). — [Sci. Rep.](https://www.nature.com/articles/s41598-020-61043-w); [OpenAlex](https://api.openalex.org/works/doi:10.3390/s24123797).
- Acceleration-aware UWA receivers (Zakharov 2019; Cho and Kim 2026, DOI 10.1049/cmu2.70136) model constant
  acceleration within a packet, so jerk is their residual. None turns this into pilot timing. — local PDF;
  [OpenAlex](https://api.openalex.org/works/doi:10.1049/cmu2.70136).

### Inferences
- The zero-crossing/jerk placement result (R3) appears to be new. The underlying fact (a phase error δφ gives drift
  error ∝ cos(ωt+φ)) is elementary, so the claim should rest on deriving the optimal set and showing overhead savings,
  not on the observation itself.

### Gaps
- Limited by the lack of Scholar full-text search; patents were searched only incidentally (USPTO hits on DMRS and
  adaptive pilots, none on periodic-motion-timed pilots).

## Q5. Closeness verdict per candidate

### Takeaway
No candidate is **identical** to Block 2. No UWA candidate has **large overlap**. Two non-UWA works have large overlap
in *mechanism*: Agarwal–Honig–Ata 2012 (training decided by Kalman error variance) and radar adaptive revisit from the
predicted covariance (Cheng et al. 2013; van Keuk–Blackman line). The combination (forecast of kinematic Doppler →
in-packet pilot positions over a coherence grid, plus packet length, in UWA, from two-way fused samples) was not found.

### Cited Findings
| Work | What it adapts / from what | Rating |
|---|---|---|
| Zakharov et al., JOE 2019, 10.1109/JOE.2019.2932650 | pilot period, packet length from *max* v, a; offline | partial |
| Hong et al., JASA 2025, 10.1121/10.0037819 | DSSS packet format from coherence time/bandwidth; offline | partial/tangential |
| Van der Werf et al., UComms 2024 / Signal Process. 2025, 10.1016/j.sigpro.2025.110223 | OTFS pilot positions/power from delay–Doppler spread; MSE-optimal | partial |
| Yang et al., GLOBECOM 2025, 10.1109/GLOBECOM59602.2025.11432537 | per-user pilot count/positions (CS coherence), UWA OFDMA | tangential |
| Mangione et al., IEEE Access 2021, 10.1109/ACCESS.2021.3082766 | CP length, #subcarriers from measured delay/Doppler spread, UWA | partial |
| Rao–Marojevic–Reed, TVT 2017 | pilot spacing/power from fading-profile codebook, closed loop | partial |
| Panah–Vaughan–Heath, TVT 2008, 10.1109/TVT.2008.2008655 | non-uniform pilot locations via feedback | partial |
| Yang–Dang–He–Guizani, IoT-J 2018, 10.1109/JIOT.2018.2867334 | MDP pilot pattern selection, V2X | partial |
| Savazzi–Spagnolini, TSP 2008, 10.1109/TSP.2008.2009270 | training length + interval vs Doppler and frame length; offline | partial |
| 5G NR dmrs-AdditionalPosition (+ Tomić 2024; US 11368268) | 0–3 extra DMRS by speed | partial |
| Hashida–Kawamoto–Kato, TVT 2022, 10.1109/TVT.2022.3166639 | pilot interval from UE velocity | partial |
| Deng et al., TCOM 2019, 10.1109/TCOMM.2019.2911575 | per-user CSI update pattern from mobility | partial |
| Nguyen–Dao–Ha, Electronics 2026, 10.3390/electronics15101991 | HSR pilot density from temporal variation (reactive) | partial/tangential |
| Dong–Tong–Sadler, MILCOM 2002 / TSP 2004, 10.1109/TSP.2004.826182 | pilot positions minimising Kalman MSE; uniform optimal | partial (theory baseline) |
| Agarwal–Honig–Ata, IT 2012, 10.1109/TIT.2012.2201345 | pilot power on/off from Kalman mean + error variance | large overlap (mechanism), different object |
| Cheng–Zou–He, IET Radar 2013, 10.1049/cp.2013.0265 (and radar revisit literature) | sampling interval from predicted IMM covariance | large overlap (method), different domain |
| Jawaid–Smith, Automatica 2015, 10.1016/j.automatica.2015.08.022 | greedy Kalman sensor scheduling; non-submodularity | tangential (but critical for R3) |
| Sun–Polyanskiy–Uysal 2017; Ornee–Sun 2021 | MSE-optimal sampling of Wiener/OU (threshold) | tangential |

### Inferences
- Defensible novelty statement for Block 2: "Pilot density adapted to mobility statistics is standard (NR DMRS,
  [Rao 2017], [Savazzi 2008]); measurement times chosen from a predicted covariance are standard in radar tracking.
  We are not aware of work that places training symbols *within a packet* from the error variance of a
  *kinematic Doppler forecast*, nor in UWA, nor with pilots clustered at the drift's zero crossings."
- Risks for R3: (1) Jawaid–Smith non-submodularity; (2) Dong–Tong–Sadler shows uniform placement is optimal unless
  σ_e²(u) is non-stationary, so the gain depends on the swell phase uncertainty being large relative to the
  tap-coherence grid; with dense P0 the extras may be few.

### Gaps
- No Scholar or IEEE Xplore cited-by verification (browser unavailable). Before submission, check the Scholar cited-by
  lists of Zakharov 2019, Dong–Tong–Sadler 2004, Panah–Vaughan–Heath 2008 and Agarwal–Honig–Ata 2012 for any later
  forecast-driven variant, and search "optimal sampling sinusoid phase estimation" for a prior closed form for R3.
