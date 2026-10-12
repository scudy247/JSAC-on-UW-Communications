# Block 1 prior art: receiver pre-warped along a kinematic Doppler forecast carried across packets

Scope: works in which an underwater acoustic (UWA) receiver, or a close terrestrial, radar or sonar analogue, compensates Doppler using a prediction taken from earlier packets, frames or blocks, from a motion model (swell, platform kinematics), from navigation data (INS/DVL/GPS/ephemeris), or from a Kalman or particle-filter track. Each work is rated against Block 1: node B resamples along a forecast ε̂(t) carried across packets (and ACKs), and the in-packet tracker sees only the forecast error.

Where I searched (8 Oct 2026):
- **Local library** (`~/Underwater/papers`, FTS5 over title, abstract, authors and venue): about 25 queries combining doppler with predict*, kalman, track*, "time-varying doppler", "particle filter", navigation/inertial/DVL/INS, swell/heave/"surface wave"/"wave-induced", "doppler rate", acceleration, sinusoidal/periodic/oscillat*, prior, previous/history/successive/consecutive packets/frames, resampl*, motion/kinematic aided.
- **Local PDFs read** (pdftotext): Ceballos Carrascosa–Stojanovic OCEANS 2008 (the conference version of JOE 2010), Choi et al. RA-L 2022, Roudsari 2017 thesis, Freitag et al. 2015 (WHOI Arctic).
- **Semantic Scholar API** "cited-by" lists: Ceballos Carrascosa–Stojanovic JOE 2010 (107 citing works), Zakharov et al. JOE 2019 (17), Sharif et al. JOE 2000 (423). I filtered them by keyword and read the titles. Keyword search on Semantic Scholar was mostly rate-limited (HTTP 429).
- **OpenAlex API**: 8 keyword searches.
- **WebSearch**: about 15 queries, including Google Patents.
- **WebFetch** of patents US9608738B2, EP1378078A2, CN108880699A and US8023596B2, and of the UNICT IRIS record.
- Not used: Claude in Chrome / Google Scholar / IEEE Xplore browsing. Paywalled pages (ScienceDirect, MDPI, ACM DL, DOAJ) returned 403, so the full text of some candidates was not checked.

---

## Q1. Which UWA works predict Doppler across blocks or packets instead of re-estimating it per packet? Has anyone extended this across inter-packet gaps?

### Takeaway
Every UWA receiver I found that predicts or tracks Doppler does so inside one packet: block to block (Ceballos Carrascosa–Stojanovic), symbol to symbol (Kalman or PLL trackers), or with a velocity-plus-acceleration model fitted to one packet (Sharif 2000, Zakharov 2019, Cho 2026). I found no UWA receiver that carries a Doppler state across the inter-packet gap and uses it to pre-warp the next packet. The cross-packet Doppler tracks I found feed navigation, time sync or authentication, never the demodulator.

### Cited Findings
- **Ceballos Carrascosa & Stojanovic, "Adaptive Channel Estimation and Data Detection for Underwater Acoustic MIMO–OFDM Systems", IEEE JOE 35(3), 2010, DOI 10.1109/JOE.2010.2052326.** The abstract says the algorithm "is coupled with nonuniform Doppler prediction and tracking, which enable decision-directed operation and reduces the overhead" — [JOE 2010](https://doi.org/10.1109/JOE.2010.2052326)
  - In the OCEANS 2008 version (same authors, DOI 10.1109/OCEANS.2008.5151953, local PDF) the "prediction" is a block-to-block phase recursion inside the packet: "θkt(n) = θkt(n−1) + at(n)2πfk T′ … where at(n) is the residual Doppler factor after initial signal resampling. It is assumed to be constant during one OFDM block, but may change from one block to another … initial resampling is performed over several blocks using an estimate a0." — [OCEANS 2008](https://doi.org/10.1109/OCEANS.2008.5151953)
  - So the prior is the previous OFDM block of the same packet, with no kinematic model and nothing carried across packets.
- **Sharif, Neasham, Hinton & Adams, "A computationally efficient Doppler compensation system for underwater acoustic communications", IEEE JOE 25(1), 2000, DOI 10.1109/48.820736.** A preprocessor "employs a novel Doppler estimation technique and efficient sample rate conversion to remove Doppler shift induced by platform velocity and acceleration". Estimation is within the packet — [JOE 2000](https://doi.org/10.1109/48.820736)
  - Companion paper: Sharif et al., "Closed loop Doppler tracking and compensation for non-stationary underwater platforms", OCEANS 2000, DOI 10.1109/OCEANS.2000.881287. Only the title was seen (abstract withheld by the publisher) — [OCEANS 2000](https://doi.org/10.1109/OCEANS.2000.881287)
- **Zakharov et al., "Data Packet Structure and Modem Design for Dynamic Underwater Acoustic Channels", IEEE JOE 2019, DOI 10.1109/JOE.2019.2932650.** Channel dynamics are "modeled by velocity and acceleration between the transmitter and the receiver". Periodic superimposed pilots allow a "multibranch autocorrelation method for coarse estimation of Doppler parameters related to the velocity and acceleration", refined by dichotomous iterations. Estimation is per packet — [JOE 2019](https://doi.org/10.1109/JOE.2019.2932650)
- **Cho & Kim, "Acceleration-Aware Estimation and Compensation of Doppler Scale for Mobile UWA OFDM Systems", IET Communications 2026, DOI 10.1049/cmu2.70136.** Uses "the inherent redundancy between the cyclic prefix (CP) and its replica in each OFDM symbol to extract phase-based measurements" of acceleration-induced Doppler. Estimation is per symbol and per packet — [IET Comm 2026](https://doi.org/10.1049/cmu2.70136)
- **Parrish, Roy & Arabshahi, "Symbol by symbol Doppler rate estimation for highly mobile underwater OFDM", WUWNet 2009, DOI 10.1145/1654130.1654131.** Within the packet (title only, no abstract in the database) — [WUWNet 2009](https://doi.org/10.1145/1654130.1654131)
- **Patent EP1378078A2 (QinetiQ; priority 2 April 2001), "Underwater communication system".** Doppler is taken from two chirp pulses and then updated "from immediately preceding sequence pairs (i.e. a closed loop system)" and "for use in connection with a following sequence". This is block-to-block within one transmission — [EP1378078A2](https://patents.google.com/patent/EP1378078A2/en)
- **Patent US9608738B2 (Riedl & Singer, Univ. of Illinois; priority 2012), "System and method for broadband doppler compensation".** A recursive sample-by-sample estimate of the inverse time-warp, motivated by the statement that assuming constant velocity "is prone to instabilities, since an estimation error in the Doppler Effect may be increasingly amplified with time". Within the packet — [US9608738B2](https://patents.google.com/patent/US9608738B2/en)
- **Patent CN108880699A**: Doppler from two LFM signals around each frame, then linear interpolation. The Google Patents summary states no sensors, Kalman filtering or prediction from previous frames — [CN108880699A](https://patents.google.com/patent/CN108880699A/en)
- **Lee, Park, Kim & Im, "Diagnosing Packet-Reliability Limits in a Surface-Buoy Underwater Acoustic OFDM Link: A Two-Deployment Field Study", SSRN preprint 2026, DOI 10.2139/ssrn.7134057.** Over a 10.5 h, 26,170-frame surface-buoy trial, "adding per-symbol Doppler/CFO refinement to the coarse compensation lowers the PER by at most a factor of 1.15 … Per-symbol Doppler tracking is therefore not what limits reliability here." Relevant as counter-evidence for the benefit Block 1 claims — [SSRN 2026](https://doi.org/10.2139/ssrn.7134057)
- **Cross-packet Doppler tracks used for something other than demodulation:**
  - Liu et al., "DA-Sync: A Doppler-Assisted Time-Synchronization Scheme for Mobile Underwater Sensor Networks", IEEE TMC 2013/14, DOI 10.1109/TMC.2013.13. Doppler from exchanged packets; "To refine the relative velocity estimation … the Kalman filter is employed". Used for clock skew and offset — [TMC](https://doi.org/10.1109/TMC.2013.13)
  - Ventura, Ardizzon & Tomasin, "Authentication by Location Tracking in Underwater Acoustic Networks", IEEE Access 2026, DOI 10.1109/ACCESS.2026.3651968. Predicts the transmitter's "future position based on the previously estimated locations" with "a Kalman filter or a long short term memory (LSTM)", and uses the prediction error for authentication — [Access 2026](https://doi.org/10.1109/ACCESS.2026.3651968)
  - Diamant, Wolff & Lampe, "Location Tracking of Ocean-Current-Related Underwater Drifting Nodes Using Doppler Shift Measurements", IEEE JOE 40(4), 2015, DOI 10.1109/JOE.2014.2370911. Doppler measurements across packets are used for location tracking. Title only; the abstract was not in the database — [JOE 2015](https://doi.org/10.1109/JOE.2014.2370911)

### Inferences
- The README's gap G2 ("Receivers re-estimate Doppler on every packet, without a kinematic prior") holds for everything found. The closest in-receiver work is still Ceballos Carrascosa–Stojanovic, and its "prediction" is a first-order block-to-block phase recursion, not a forecast across packets.
- The ingredients of Block 1 already exist separately:
  - in-packet Doppler models with velocity and acceleration (Sharif 2000, Zakharov 2019, Cho 2026);
  - cross-packet Kalman tracks of Doppler-derived velocity or position (DA-Sync, Ventura 2026, Diamant 2015).
  - Nobody has been found connecting the second to the first.
- Reviewers will probably cite DA-Sync and Ventura 2026 as "Kalman across packets already exists". The answer is that they track kinematics for time sync or authentication, not for resampling, and they use no swell model.
- The Lee 2026 buoy study suggests that on some surface-buoy links the residual after frame-level coarse Doppler correction does not dominate the PER. Block 1 needs to show a regime (longer packets, larger drift, fewer pilots) where it does.

### Gaps
- Full texts of Ceballos Carrascosa–Stojanovic JOE 2010 and Sharif OCEANS 2000 were not read. The OCEANS 2008 text was used as a proxy for the 2010 mechanism.
- I did not verify whether a deployed commercial modem (WHOI Micro-Modem, EvoLogics, Sonardyne, Teledyne Benthos) seeds Doppler search from the previous packet. I found no public documentation either way.
- Google Scholar "cited by" of Ceballos Carrascosa–Stojanovic was not browsed. Semantic Scholar lists 107 citing works and none has a title suggesting cross-packet Doppler prediction, but abstracts were not read for all of them.

---

## Q2. Which works use navigation/INS/DVL/acoustic-ranging data or wave models as a Doppler prior for the modem?

### Takeaway
In UWA, the information flows the other way: Doppler and timing from the modem are fed into the vehicle's navigation filter (Aubry 2019, Choi/Singer 2022, Randeni 2022). I found no UWA paper or patent that feeds INS/DVL/GPS data, or a wave model, back into the modem as a Doppler prior. The clear analogue outside UWA is 3GPP NTN, where the UE pre-compensates Doppler from its GNSS position and the satellite ephemeris. That is a deterministic kinematic prior, not a forecast learned from link measurements.

### Cited Findings
- **Aubry et al., "On the use of Doppler-shift estimation for simultaneous underwater acoustic localization and communication", OCEANS 2019 Marseille, DOI 10.1109/OCEANSE.2019.8867265.** The AUV "uses UWA communication decoding process to estimate both range and relative speed … to improve its navigation via a conventional Kalman filtering". The direction is modem to navigation — [OCEANS 2019](https://doi.org/10.1109/OCEANSE.2019.8867265)
- **Choi, Borkar, Singer & Chowdhary, "Broadband Acoustic Communication Aided Underwater Inertial Navigation System", IEEE RA-L 2022, DOI 10.1109/LRA.2022.3154004.** "Timing and Doppler information extracted from this process is then used to estimate biases and drift from an inertial navigation system (INS) using a Bayesian framework", with transmissions every 30 s. The conclusion describes "a Bayesian filtering framework that uses the recovered Doppler factor … and fuses them with the inertial measurement". There is no feedback from INS to Doppler compensation (local PDF read) — [RA-L 2022](https://doi.org/10.1109/LRA.2022.3154004)
- **Randeni P. et al., "A high-resolution AUV navigation framework with integrated communication and tracking for under-ice deployments", J. Field Robotics 2022, DOI 10.1002/rob.22133.** Modem one-way travel times aid a model-aided navigation engine. Again modem to navigation — [JFR 2022](https://doi.org/10.1002/rob.22133)
- **Liu et al., "Geo-Doppler Hybrid Mobile Ranging Method for Underwater Acoustic Sensor Networks", OCEANS 2026 Sanya, DOI 10.1109/OCEANS66983.2026.11617170.** Uses "radial velocity measurements from Doppler factor estimation or node velocity from a Doppler Velocity Log (DVL)" for ranging, not for compensation — [OCEANS 2026](https://doi.org/10.1109/OCEANS66983.2026.11617170)
- **Freitag, Ball, Partan, Koski & Singh, "Long Range Acoustic Communications and Navigation in the Arctic", WHOI, 2015.** Navigation and communication in one system. A text search of the PDF found no use of vehicle navigation as a Doppler prior — [WHOI PDF](https://acomms.whoi.edu/wp-content/uploads/sites/20/2016/11/FreitagBallPartanKoskiSingh_Arctic_2015.pdf)
- **3GPP NR NTN (TS 38.300 §16.14.2.2).** "The UE shall compute the frequency Doppler shift of the service link, and autonomously pre-compensate for it in the uplink transmissions, by considering UE position and the ephemeris", and in connected mode it shall "continuously update the Timing Advance and frequency pre-compensation" — [TS 38.300 §16.14.2.2](https://itecspec.com/3gpp/38.300/s/16.14.2.2)
- **Patent US8023596B2 (Thales SA, inventor P.-A. Laurent; priority 25 Nov 2005), "Doppler tracking method and device for a wide band modem"** (a radio modem, not UWA). A position predictor stores "the last T+1 … position pairs" and, "since, over a limited period, the Doppler variation may be assimilated locally to a parabola, the value of the position at the moment t is predicted by means of a weighted quadratic regression". This is extrapolation of Doppler/delay from past measurements with a polynomial motion model. Whether the past samples span frames or symbols of one continuous transmission was not verified — [US8023596B2](https://patents.google.com/patent/US8023596)
- **High-speed rail (radio).** The literature proposes "estimation and pre-compensation of high Doppler shifts" for OFDM, with estimates from a preamble — [eudl 2017](https://eudl.eu/doi/10.1007/978-3-319-74176-5_7). There is also a patent family on "Doppler shift reporting in a single frequency network" — [US12647916B2](https://eureka.patsnap.com/patent-US12647916B2)
- **Web search for "IMU-aided / sensor-aided Doppler compensation" in UWA.** The search summary itself stated the results "do not specifically address IMU … or sensor-aided approaches to Doppler compensation in underwater acoustic receivers" — [search result example](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4789520/)

### Inferences
- "Navigation-aided Doppler compensation" for UWA modems appears to be unpublished, or at least not visible in the library, OpenAlex, web search or patents. Reviewers may still say it is obvious, given the NTN precedent and the Choi/Aubry loop that only needs closing.
- Block 1 differs from NTN on four points:
  - its prior is learned from the link's own Doppler measurements (data and ACKs, from both ends), with no GNSS or ephemeris;
  - it has a stochastic swell model with uncertainty σ_e;
  - the receiver tracks the residual;
  - it accepts irregular, delayed samples.
- The paper should cite NTN pre-compensation, and the Thales parabolic predictor, as the "known kinematic prior" analogue.

### Gaps
- Not checked: vendor manuals that may accept external velocity (NMEA) to narrow the Doppler search, such as WHOI Micro-Modem, Sonardyne 6G/Wideband 2 and Kongsberg cNODE. Not checked for want of time and access.
- Full texts not checked: Hu et al. 2024 and Deng et al. 2026, for whether their Kalman initialisation uses external velocity.

---

## Q3. Kalman or particle-filter Doppler tracking in UWA receivers: does any of them carry state across packets?

### Takeaway
Kalman-based Doppler trackers in UWA receivers exist (Sun et al. 2022, Hu et al. 2024, Deng et al. 2026, and adaptive or PLL-type trackers since Johnson–Freitag–Stojanovic 1997 and Sharif 2000). All of them track within one frame, and several explicitly justify short-horizon kinematic models by the short duration of a packet. None of those found initialises from a previous packet or carries the filter state across the inter-packet gap.

### Cited Findings
- **Sun et al., "A Kalman-based Doppler tracking algorithm for underwater acoustic spread spectrum communications", Applied Acoustics 2022, DOI 10.1016/j.apacoust.2021.108374.** "The time difference of the two adjacent symbols is used as the input of the Kalman filter based on the delay-Doppler equation to estimate the instant Doppler factor sequentially." Symbol by symbol, within the frame — [Appl. Acoust. 2022](https://doi.org/10.1016/j.apacoust.2021.108374)
- **Hu, Wan, Zhao, Liu, Chen & Cheng, "Iterative Doppler Tracking based on Kalman Filter for Underwater Acoustic FH-FSK Communication in High Mobility", IEEE MSN 2024, DOI 10.1109/MSN63567.2024.00031.** "Since the duration of a UWA communication signal is usually short, the change in relative motion … is not too drastic. Therefore the motion during this period is modeled by constant acceleration model or constant jerk model." This is a kinematic Kalman model, scoped explicitly to one signal — [MSN 2024](https://doi.org/10.1109/MSN63567.2024.00031)
- **Deng, Hong, Wu, Cui, Sun & Mu, "Time-Varying Multipath Tracking for Direct-Sequence Spread-Spectrum Receiver in Mobile Underwater Acoustic Networks", IEEE IoT-J 2026, DOI 10.1109/JIOT.2026.3681800.** Addresses "time-varying Doppler effect and fast-fading multipath structure within a single frame" and "employs Kalman filtering for channel parameter estimation" — [IoT-J 2026](https://doi.org/10.1109/JIOT.2026.3681800)
- **Sun et al., "Iterative double-differential DSSS reception … with time-varying Doppler shifts", JASA 2023, DOI 10.1121/10.0017116.** Per-symbol cross-correlation Doppler tracking with "a dynamic linear prediction model … to estimate and track the channel phase variation". Within the frame — [JASA 2023](https://doi.org/10.1121/10.0017116)
- **Yang et al., "Doppler tracking and symbol synchronization method for mobile M-ary spread spectrum UWA communications", COA 2021, DOI 10.1109/COA50123.2021.9519885.** "The next symbol is synchronously corrected in the time domain according to the estimation results." Symbol to symbol — [COA 2021](https://doi.org/10.1109/COA50123.2021.9519885)
- **Choi, Chowdhary, Singer, Vishnu, Weiss, Wornell & Deane, "Online Segmented Recursive Least-Squares for Multipath Doppler Tracking", UComms 2022, DOI 10.1109/UComms56954.2022.9905690.** "Approximating the nonlinear time distortion as a piece-wise-linear Markov model" to track per-path Doppler sequentially. Within the received signal — [UComms 2022](https://doi.org/10.1109/UComms56954.2022.9905690) / [arXiv 2305.19127](https://arxiv.org/abs/2305.19127)
- **Deguchi, Kida, Watanabe & Shimura, "Equalization for mitigating nonlinear Doppler shifts and Doppler spreading in UWA communication", JASA 2025, DOI 10.1121/10.0040631.** Adaptive digital down-conversion (ADDC) plus a per-tap DPLL in the DFE. Within the packet — [JASA 2025](https://doi.org/10.1121/10.0040631)
- **Kaddouri, Beaujean, Bouvet & Real, "Least Square and Trended Doppler Estimation in Fading Channel for High-Frequency UWA Communications", IEEE JOE 39(1), 2014, DOI 10.1109/JOE.2013.2282065.** EMD-based trend estimation on LS channel estimates for the Hermes modem — [JOE 2014](https://doi.org/10.1109/JOE.2013.2282065)
  - A search summary states that "the Hermes modem currently uses moving average to estimate and compensate for Doppler shift" and that an autoregressive (IIR) approach was proposed. Whether that average spans packets is not stated in what I could access — [FAU record](https://fau.isle.flvc.org/taxonomy/term/27902)
- **Learning-based Doppler handling** (within the frame; no cross-packet state):
  - Hassan, Chen, Rong & Chan, OCEANS 2023 Limerick, DOI 10.1109/OCEANSLimerick52467.2023.10244541: LSTM receiver — [OCEANS 2023](https://doi.org/10.1109/OCEANSLimerick52467.2023.10244541)
  - Zeng et al., Sci. Rep. 2026, DOI 10.1038/s41598-026-38112-7: LFM block estimate, then an attention-LSTM predicts the residual CFO — [Sci. Rep. 2026](https://doi.org/10.1038/s41598-026-38112-7)
- **Background and context:**
  - Song, "Time reversal communication with a mobile source", JASA 2013, DOI 10.1121/1.4819115: "resampling may not be necessary when a channel-estimate-based time reversal approach is applied with frequent channel updates" — [JASA 2013](https://doi.org/10.1121/1.4819115)
  - Survey by the user's group (Busacca, Galluccio, Palazzo, Panebianco, Qi, Pompili), "Adaptive versus predictive techniques in underwater acoustic communication networks", Computer Networks 2024, DOI 10.1016/j.comnet.2024.110679: a qualitative survey of adaptive (reactive) and predictive (proactive) methods; Doppler-forecast receivers are not described in the abstract — [IRIS UNICT](https://www.iris.unict.it/handle/20.500.11769/640590)

### Inferences
- The UWA Kalman trackers all use a kinematic state model (constant acceleration or jerk), so "Kalman Doppler tracking with a kinematic model" is not new. What Block 1 adds:
  - the horizon is extended across packets and ACKs;
  - a swell oscillator is added to the state;
  - the forecast is used to pre-warp before any in-packet measurement;
  - the in-packet tracker sees only the residual.
- The quote from Hu et al. 2024 ("the duration of a UWA communication signal is usually short …") is useful when writing the gap: the literature restricts kinematic models to the packet duration on purpose.
- The authors named in the brief (Sharif, Zakharov, Li/Zhou/Willett, Yerramalli & Mitra, Wang/Zhou, Johnson/Sozer, Chitre, Song/Hodgkiss, Ma/Liu, Wei/Li) appeared in the local library only with per-packet or within-packet methods. Their work found here is estimation within the packet: multi-resampling, nonuniform Doppler, CP-based or ambiguity-function estimators.

### Gaps
- Not found or verified: UWA particle-filter Doppler trackers for communication. Particle-filter hits were navigation or target tracking only.
- OTFS: no UWA OTFS paper found that tracks delay-Doppler parameters across frames with a kinematic model. OTFS channel prediction papers exist but predict channel taps:
  - Wang 2022, OCEANS, RL-based prediction in the delay-Doppler domain, DOI 10.1109/OCEANS47191.2022.9977194;
  - Jing 2025, IoT-J, ConvLSTM prediction, DOI 10.1109/JIOT.2025.3639115.
  - Their abstracts were not read in full; they are tangential at best.
- Not read in full: Johnson, Freitag & Stojanovic 1997 (ICASSP, "Improved Doppler tracking and correction"), cited by Choi 2022 as ref. [2]. It is known to be a within-packet tracker.

---

## Q4. Wave-induced or swell periodic Doppler: modelling and prediction for UWA communication

### Takeaway
Wave-induced Doppler is well characterised:
- as a channel or scattering phenomenon (Walstead & Deane);
- in simulators (Hursky 2020);
- as a nuisance that receivers must absorb (Yoshizawa: parallel resampling for fluctuating Doppler; Lee 2026: buoy field study).

I found no receiver that fits a periodic, swell-oscillator model to Doppler measurements and forecasts it forward to the next packet.

### Cited Findings
- **Walstead & Deane, "Statistics of near surface underwater scattered sound including wave induced Doppler", JASA 2016, DOI 10.1121/1.4950090** — channel physics, no receiver prediction (title only from the database) — [JASA 2016](https://doi.org/10.1121/1.4950090)
- **Walstead & Deane, "Delay-Doppler characteristics of surface interacting underwater sound", JASA 2017, DOI 10.1121/1.5014113** — [JASA 2017](https://doi.org/10.1121/1.5014113)
- **Hursky, "Ray tracing for modeling underwater acoustic communications channel between moving platforms, with moving ocean surface and changing bathymetry", JASA 2020, DOI 10.1121/1.5147113.** "The signal processing to equalize such channel fading typically consists of a number of tracking loops, whose testing requires time varying realizations of such channels." A simulator, not a forecaster — [JASA 2020](https://doi.org/10.1121/1.5147113)
- **Yoshizawa et al., "Parallel Resampling of OFDM Signals for Fluctuating Doppler Shifts in Underwater Acoustic Communication", J. Electrical and Computer Engineering 2018, DOI 10.1155/2018/3579619.** The search summary says it "extends the resampling range by measured Doppler standard deviation", because "the conventional method assumes a constant Doppler shift during a communication frame and cannot cope with Doppler fluctuation". Whether the standard deviation comes from earlier frames is not verified — [DOAJ record](https://doaj.org/article/579a536b8d8449748e2666aff5a09f44)
- **"An Acoustic OFDM System with Symbol-by-Symbol Doppler Compensation for Underwater Communication", The Scientific World Journal 2016, DOI 10.1155/2016/7528353.** Vertical ship-to-bottom link where "transducers attached to the bottom of the ship swing due to ocean wave". It tracks with continual pilots symbol by symbol and does not predict — [PMC4789520](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4789520/)
- **Lee et al. 2026 (SSRN), surface-buoy OFDM field study.** "Wave-driven buoy motion is widely assumed to stress the … OFDM link", but per-symbol tracking gave at most 1.15× lower PER — [SSRN 2026](https://doi.org/10.2139/ssrn.7134057)
- **Ship heave prediction outside communications** (echo-state networks for heave compensation of offshore operations) shows that wave-motion forecasting over seconds is an established technique — [PMC6563959](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6563959/)

### Inferences
- Swell-driven Doppler is recognised as a problem, and wave-motion forecasting is mature in offshore and ship control. As far as these searches go, putting the two together (a swell-oscillator Doppler forecast driving UWA receiver compensation) is unoccupied.
- Heave-prediction literature (ship motion forecasting with AR models, Kalman filters, ESNs) is a natural citation to support claim R1 (how long Doppler stays predictable).

### Gaps
- Full texts not read: Yoshizawa 2018 and its OCEANS 2017 precursor ("Parallel resampling receiver for UWA communication with non-uniform Doppler shifts"). They are the most likely to contain a cross-frame Doppler statistic.
- No search was made of Japanese JAMSTEC vertical-link papers (Kida, Deguchi, Shimura) for use of ship motion-reference-unit data. A web query returned nothing specific.

---

## Q5. Candidate-by-candidate closeness to Block 1

### Takeaway
No candidate is identical to Block 1 or overlaps it largely. The closest works are partial:
- (a) in-receiver Doppler prediction within the packet: Ceballos Carrascosa–Stojanovic 2010; QinetiQ EP1378078;
- (b) kinematic Kalman Doppler trackers within the packet: Hu 2024, Sun 2022;
- (c) cross-packet Doppler/kinematic tracks used outside the receiver: DA-Sync, Ventura 2026, Aubry 2019, Choi 2022;
- (d) a deterministic kinematic pre-compensation analogue outside UWA: 3GPP NTN, and Thales US8023596 (parabolic Doppler extrapolation).

### Cited Findings
| Work | What it does | Difference from Block 1 | Rating |
|---|---|---|---|
| Ceballos Carrascosa & Stojanovic, JOE 2010 ([DOI](https://doi.org/10.1109/JOE.2010.2052326)) | Initial resampling, then block-to-block residual Doppler/phase prediction inside an OFDM packet | No cross-packet state, no kinematic or swell model, no forecast uncertainty | Partial (closest in-receiver) |
| QinetiQ EP1378078A2 ([link](https://patents.google.com/patent/EP1378078A2/en)) | Chirp-pair Doppler, then closed-loop update from preceding sequences for the following ones | Within one transmission; no model | Partial / tangential |
| Riedl–Singer US9608738B2 ([link](https://patents.google.com/patent/US9608738B2/en)) | Recursive sample-by-sample tracking of the time-warp | Within the packet; no prior | Tangential |
| Sharif et al., JOE 2000 and OCEANS 2000 ([DOI](https://doi.org/10.1109/48.820736)) | Velocity and acceleration Doppler estimation, closed-loop tracking, resampling | Within the packet | Tangential |
| Zakharov et al., JOE 2019 ([DOI](https://doi.org/10.1109/JOE.2019.2932650)); Cho & Kim 2026 ([DOI](https://doi.org/10.1049/cmu2.70136)) | Acceleration-aware per-packet Doppler | Within the packet; these are the "acceleration-aware" baselines Block 1 must beat | Tangential (baseline) |
| Hu et al., MSN 2024 ([DOI](https://doi.org/10.1109/MSN63567.2024.00031)); Sun et al. 2022 ([DOI](https://doi.org/10.1016/j.apacoust.2021.108374)); Deng et al. 2026 ([DOI](https://doi.org/10.1109/JIOT.2026.3681800)) | Kalman Doppler tracking with constant-acceleration or jerk models, symbol by symbol | Within the frame, explicitly; no cross-packet forecast or swell | Partial (method), tangential (scope) |
| DA-Sync, TMC ([DOI](https://doi.org/10.1109/TMC.2013.13)) | Kalman on Doppler-derived velocity across a two-way exchange | Used for clock sync, not compensation | Partial (cross-packet Kalman, other use) |
| Ventura, Ardizzon & Tomasin, Access 2026 ([DOI](https://doi.org/10.1109/ACCESS.2026.3651968)) | Kalman or LSTM position prediction across packets | Used for authentication | Tangential |
| Aubry et al., OCEANS 2019 ([DOI](https://doi.org/10.1109/OCEANSE.2019.8867265)); Choi et al., RA-L 2022 ([DOI](https://doi.org/10.1109/LRA.2022.3154004)); Randeni et al., JFR 2022 ([DOI](https://doi.org/10.1002/rob.22133)) | Modem Doppler and timing fused into a navigation Kalman or Bayesian filter across packets | Opposite direction: no feedback to the receiver | Partial (the missing reverse link is exactly Block 1 plus Block 3) |
| Geo-Doppler, OCEANS 2026 ([DOI](https://doi.org/10.1109/OCEANS66983.2026.11617170)); UWGS 2026 ([DOI](https://doi.org/10.52953/xqts2777)) | Doppler/DVL for ranging and time sync | Not compensation | Tangential |
| Yoshizawa et al. 2018 ([DOAJ](https://doaj.org/article/579a536b8d8449748e2666aff5a09f44)) | Parallel resampling spanning a measured Doppler spread for fluctuating Doppler | A statistic, not a forecast; unclear whether it is cross-frame | Tangential (to verify) |
| Lee et al., SSRN 2026 ([DOI](https://doi.org/10.2139/ssrn.7134057)) | Buoy field study: per-symbol Doppler tracking gives ≤1.15× PER gain | Evidence against large gains on that link | Context or risk |
| 3GPP NTN TS 38.300 §16.14.2.2 ([link](https://itecspec.com/3gpp/38.300/s/16.14.2.2)) | UE pre-compensates Doppler from its GNSS position and the satellite ephemeris, updated continuously | Deterministic known kinematics; not learned from link measurements; terrestrial/satellite RF | Partial (conceptual analogue of "pre-warp along a kinematic prediction") |
| Thales US8023596B2 ([link](https://patents.google.com/patent/US8023596)) | Weighted quadratic (parabolic) extrapolation of Doppler/delay from the last T+1 measurements, for a wideband radio modem | Polynomial, not oscillatory; RF; span across frames unverified | Partial (analogue) |
| Chude-Okonkwo et al., GLOBECOM Wkshps 2013 ([DOI](https://doi.org/10.1109/GLOCOMW.2013.6825184)) | Derives channel parameters "with respect to the receiver's movement" to adapt subcarrier spacing and guard interval (search summary) | Transmitter adaptation (closer to Block 2/3), not receiver pre-warp; prediction details unknown | Tangential (to verify) |

### Inferences
- Verdict for Block 1: not anticipated in whole or in large part by anything found. The novelty rests on four points together:
  - (i) a Doppler forecast carried across packets and ACKs;
  - (ii) a kinematic model with swell oscillators and uncertainty;
  - (iii) receiver pre-warping along the forecast before in-packet measurement;
  - (iv) residual-only tracking, so that PER depends on σ_e.
- Each piece alone has precedent: (iii) has an RF precedent in NTN, and (ii) has an in-packet precedent in Hu 2024 and Zakharov 2019.
- Main risk: a reviewer frames Block 1 as "NTN-style pre-compensation plus a standard Kalman tracker". The defence should stress (a) and (b):
  - (a) there is no external kinematics: the forecast comes from the link's own sparse, delayed, two-way Doppler samples, which is the part linked to Block 3;
  - (b) the swell forecast horizon (R1) and the σ_e-to-PER mapping are the contribution.
- Block 1 should be benchmarked against acceleration-aware per-packet receivers (Zakharov 2019, Cho 2026) and against symbol-level Kalman tracking (Hu 2024), because the deck's open question ("does a forecast prior still help acceleration-aware receivers?") is exactly what a reviewer will ask.

### Gaps
- Certainty is limited by:
  - no browsing of Google Scholar or IEEE Xplore;
  - paywalled full texts not read: Hu 2024, Sun 2022, Deng 2026, Yoshizawa 2018, Chude-Okonkwo 2013, Sharif OCEANS 2000;
  - Semantic Scholar keyword search mostly rate-limited;
  - no check of Chinese-language patents beyond CN108880699A and CN119254274B (the latter was seen only as a title in the search results).
- Recommended follow-ups:
  - Google Scholar "cited by" of Ceballos Carrascosa–Stojanovic 2010 and of Hu 2024;
  - Google Patents queries in Chinese ("多普勒 预测 前一帧 水声");
  - full-text check of Yoshizawa 2018 for whether its Doppler-spread prior comes from earlier frames.
