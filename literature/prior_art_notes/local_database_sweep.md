# Local database sweep: does any paper in ~/Underwater/papers implement the _nomab idea?

Scope: the _nomab idea (README_nomab.md and project_idea/doppler_link_adaptation_nomab.tex). Every packet, data or ACK, gives a Doppler-scale measurement. Both ends fuse them (B's estimate goes back in the ACK header) in a Kalman filter on a drift + damped-oscillator (swell) model. The result is one kinematic forecast of the Doppler drift d and its uncertainty σ_e. The forecast drives three blocks: (1) receiver resampling pre-warped along the forecast; (2) placement of extra training symbols where the forecast is uncertain, plus packet length and modulation; (3) a threshold rule on the forecast-predicted PER, with a margin set from delayed ACKs.
Source: only the local database `~/Underwater/papers/db/papers.sqlite` (no web), swept on 8 Oct 2026. "Source" links below are DOIs (https://doi.org/...) or local file paths.

## Q1. Does any paper in the local database implement the idea, in whole or in large part?

### Takeaway
No paper in the database implements the idea in whole or in large part. The sweep covered all 34,745 records, with regexes over titles and abstracts plus full-text reading of the closest local PDFs. Every block has partial precedents, but always alone. No record combines a cross-packet kinematic Doppler forecast with any transmitter-side decision (pilots, length, modulation). No record fuses Doppler measured at both ends of a link for communication. No record models swell-periodic Doppler for prediction. The closest work is Radosevic et al. (JOE 2014): a linear Doppler-rate motion model stabilises channel prediction one travel time ahead for adaptive OFDM with feedback. Next come Abdelkareem et al. (Ad Hoc Netw. 2015), with a receiver prior taken from the previous packet's acceleration, and Zakharov et al. (JOE 2019), with a packet structure designed offline for a velocity and acceleration range. One record could not be checked: Zhang et al., OCEANS 2015, "Closed loop bi-directional broadband Doppler compensation...". Its title matches block G3, but the database has neither its abstract nor its PDF, so it must be checked on the web.

### Cited Findings

**Closeness scale:** identical / large overlap / partial / tangential. Blocks: B1 = forecast-aided receiver (pre-warp); B2 = pilot / training placement + packet length / modulation; B3 = two-way Kalman fusion with a swell model; B4 = threshold rule on predicted PER with a delayed-ACK margin. Also: K = kinematic Doppler model or prediction across packets.

**Partial overlap (closest 3)**

1. **Radosevic, Ahmed, Duman, Proakis, Stojanovic, "Adaptive OFDM Modulation for Underwater Acoustic Communications: Design Considerations and Experimental Results", IEEE JOE 39(2), 2014 (online 2013), DOI 10.1109/JOE.2013.2253212.** Local PDF: `pdf/05_adaptive_link_feedback/2013_Radosevic_AdaptiveOfdmModulationForUnderwaterAcoustic.pdf`. Blocks: K + B4 (AMC with feedback, prediction one travel time ahead) + partly B1 (motion compensation). Rating: **partial**. From the PDF: "The system motion during a period of time corresponding to a few seconds (or several data packets) is modeled by velocity and acceleration terms which lead to a linear Doppler rate"; "proper Doppler compensation ensures stability over intervals of time that are long enough to support channel prediction several seconds ahead"; "For both schemes we linearly predict the channel one travel time ahead"; footnote 2 notes that predicting one travel time ahead versus the round trip "correspond to different feedback implementation strategies, i.e., different functions performed by the two ends of a link". Key difference: the Doppler model is used only to stabilise the phases of the channel taps for an RLS tap predictor, at the receiver. The Doppler rate is linear, with no oscillatory (swell) term and no uncertainty used for decisions. There is no fusion of measurements from both ends, no pilot or packet-length adaptation, and no PER-margin safety layer. What is adapted is the bit and power loading per subcarrier. — [Source](https://doi.org/10.1109/JOE.2013.2253212). Companion: Radosevic et al., "Channel prediction for adaptive modulation in underwater acoustic communications", OCEANS 2011 Spain, DOI 10.1109/OCEANS-Spain.2011.6003438 (local PDF), which predicts the channel "several seconds ahead" with the same time-varying path-delay model — [Source](https://doi.org/10.1109/OCEANS-Spain.2011.6003438).

2. **Abdelkareem, Sharif, Tsimenidis, "Adaptive time varying Doppler shift compensation algorithm for OFDM-based underwater acoustic communication systems", Ad Hoc Networks, 2015, DOI 10.1016/j.adhoc.2015.05.011.** Local PDF: `pdf/02_doppler_sync_estimation/2015_Abdelkareem_...pdf`. Blocks: B1 (a receiver prior taken from the previous packet) + K. Rating: **partial**. From the PDF: "In learning mode (LM), the acceleration of the previous packet is observed to designate an adaptive expectation range (ER), whereas the punishment mode (PM) drops an out of range estimation"; "the system is capable of predicting the drift in samples in the next symbol"; "accommodates an acceleration up to ±1 m/s2 during the packet time". Key difference: the prior from one packet to the next is only a search window (expectation range) for the receiver's own estimator, with no state-space or swell model and no two-way fusion. Nothing changes at the transmitter (frame, length, modulation, PER). — [Source](https://doi.org/10.1016/j.adhoc.2015.05.011)

3. **Zakharov, Henson, Diamant, Fei, Mitchell, Morozs, Shen, Tozer, "Data Packet Structure and Modem Design for Dynamic Underwater Acoustic Channels", IEEE JOE, 2019, DOI 10.1109/JOE.2019.2932650.** Local PDF. Blocks: B2 (packet and pilot structure for motion) + K (velocity and acceleration). Rating: **partial**. From the abstract and PDF: "we propose a data packet structure for high data rate transmission in time-varying UWA channels with the channel dynamic modeled by velocity and acceleration between the transmitter and the receiver. The data packet consists of superimposed data and periodic pilot sequences"; "In the MBA method, the number of processing branches is defined by the maximum acceleration"; example design "N = 6000, so that the packet duration is Θ = 1.00167 s". Key difference: the design is fixed offline for an assumed motion range, the pilots are uniform and superimposed, and nothing is predicted across packets or fed back. This matches gap G1 in the deck. — [Source](https://doi.org/10.1109/JOE.2019.2932650)

**Partial / tangential, by block**

- *K / B1, Doppler prediction or tracking inside the receiver:*
  - Ceballos Carrascosa & Stojanovic, "Adaptive Channel Estimation and Data Detection for Underwater Acoustic MIMO–OFDM Systems", IEEE JOE 2010, DOI 10.1109/JOE.2010.2052326. Abstract: "coupled with nonuniform Doppler prediction and tracking, which enable decision-directed operation and reduces the overhead". The earlier 2008 version (local PDF `2008_Carrascosa_AdaptiveMimoDetection...`) says: "Phase prediction, based on adaptive tracking of the Doppler factor, enables decision-directed operation". The prediction runs across OFDM blocks inside one receiver, with no use at the transmitter. **Partial (B1 only).** — [Source](https://doi.org/10.1109/JOE.2010.2052326)
  - Hu, Wan, Zhao, Liu, Chen, Cheng, "Iterative Doppler Tracking based on Kalman Filter for Underwater Acoustic FH-FSK Communication in High Mobility", MSN 2024, DOI 10.1109/MSN63567.2024.00031. Abstract: "the motion during this period is modeled by constant acceleration model or constant jerk model ... the refinement is carried out based on Kalman filter". The Kalman filter works within a packet, one way, for FH-FSK. **Partial (B1).** — [Source](https://doi.org/10.1109/MSN63567.2024.00031)
  - Sun, Hong, Cui, "A Kalman-based Doppler tracking algorithm for underwater acoustic spread spectrum communications", Applied Acoustics 2022, DOI 10.1016/j.apacoust.2021.108374. A Kalman filter on the delay-Doppler equation, symbol by symbol, within a packet. **Tangential.** — [Source](https://doi.org/10.1016/j.apacoust.2021.108374)
  - Cho & Kim, "Acceleration-Aware Estimation and Compensation of Doppler Scale for Mobile UWA OFDM Systems", IET Commun. 2026, DOI 10.1049/cmu2.70136 (local PDF). Acceleration is estimated per packet from CP redundancy, "no additional signaling overhead". There is no prior across packets. **Tangential.** — [Source](https://doi.org/10.1049/cmu2.70136)
  - Parrish, Roy, Arabshahi, "Symbol by symbol Doppler rate estimation for highly mobile underwater OFDM", WUWNet 2009, DOI 10.1145/1654130.1654131. No abstract or PDF in the database, so it could not be checked. Judging by the title, it estimates within a packet. — [Source](https://doi.org/10.1145/1654130.1654131)
  - Others in this class, all receiver-only and within a packet, all **tangential**: Pu et al. 2018 (DOI 10.3233/JIFS-169654); Cuji, Stojanovic, Singer, "Path-Specific Angle and Doppler Tracking in Dynamic Underwater Channels", OCEANS 2025 (DOI 10.1109/OCEANS58557.2025.11104574, no abstract); Liang et al. JASA 2023 (DOI 10.1121/10.0023448); Kida et al. OMAE 2026 (DOI 10.1115/OMAE2026-181615; resampler with "time-varying Doppler shifts exceeding 15 Hz caused by ship heave").
- *B3, two-way Doppler, fusion and Kalman filters on Doppler for sync, ranging or navigation (not the link):*
  - Liu, Wang, Zuba, Peng, Cui, Zhou, "DA-Sync: A Doppler-Assisted Time-Synchronization Scheme for Mobile Underwater Sensor Networks", IEEE TMC 2013, DOI 10.1109/TMC.2013.13: "To refine the relative velocity estimation ... the Kalman filter is employed". It serves time sync only. **Tangential (B3 machinery, different purpose).** — [Source](https://doi.org/10.1109/TMC.2013.13)
  - Ghalkhani, Zhang, Campagnaro, Zorzi, UWGS (Asilomar 2024, DOI 10.1109/IEEECONF60004.2024.10942777) and its extension (ITU J-FET 2026, DOI 10.52953/XQTS2777, local PDF). These replace Doppler-based radial velocity with one-way geometry, noting that Doppler-based protocols "often involve two-way message exchanges, which contribute to increased signaling overhead". Time sync only. **Tangential.** — [Source](https://doi.org/10.52953/XQTS2777)
  - Liu et al., "Geo-Doppler Hybrid Mobile Ranging Method for Underwater Acoustic Sensor Networks", OCEANS 2026 Sanya, DOI 10.1109/OCEANS66983.2026.11617170: "geometric-kinematic coupling model" from Doppler, DVL and surface reflection. Ranging only. **Tangential.** — [Source](https://doi.org/10.1109/OCEANS66983.2026.11617170)
  - Lu, Mirza, Schurgers, "D-sync", WUWNet 2010 (DOI 10.1145/1868812.1868815); Aubry et al., OCEANS 2019 (DOI 10.1109/OCEANSE.2019.8867265): "uses UWA communication decoding process to estimate both range and relative speed ... via a conventional Kalman filtering"; Garin et al., OCEANS 2021 (DOI 10.23919/OCEANS44145.2021.9705890), an EKF on modem Doppler for localization; Song et al., IEEE IoT-J 2021 (DOI 10.1109/JIOT.2021.3094305), velocity from PHY Doppler for location prediction. All **tangential**: the Doppler from communication feeds navigation, never link decisions.
  - **Not verifiable, flagged:** Youwen Zhang, Lu Liu, Dajun Sun, Xin Liu, Hongyu Cui, Zhisheng Li, "Closed loop bi-directional broadband Doppler compensation and channel equalization for single carrier high-speed underwater acoustic communication", OCEANS 2015 Washington, DOI 10.23919/OCEANS.2015.7401933. **No abstract and no PDF in the database.** Its title is the only one in the database that pairs "closed loop", "bi-directional" and "Doppler compensation" for a communication link (it is one of only 2 hits of query Q24 two-way+Doppler). It may implement a two-way Doppler loop (overlap with G3 / B3 + B1). Its closeness cannot be rated without the text. — [Source](https://doi.org/10.23919/OCEANS.2015.7401933)
- *B2, pilot design or packet design versus Doppler:*
  - Van der Werf, Pelekanakis, Hendriks, Heusdens, Leus, "Optimized Pilot Design for OTFS Modulation in Underwater Acoustic Communications", UComms 2024, DOI 10.1109/UCOMMS64662.2024.10847980: the pilot allocation "can be adjusted in a simple way to the delay and Doppler spread". It responds to the Doppler spread, not to a forecast. **Partial (B2).** — [Source](https://doi.org/10.1109/UCOMMS64662.2024.10847980). Related: van der Werf et al., Science Talks 2024, DOI 10.1016/j.sctalk.2024.100312: "a portion of the transmitted signal is allocated for training ... The optimization of this allocation, considering time, frequency, and space constraints ... can adapt to scenarios with time-varying channel distortions" (a PhD abstract, no method). — [Source](https://doi.org/10.1016/j.sctalk.2024.100312)
  - Hong, Jeong, Lee, Jung, "An optimal direct sequence spread spectrum packet design for underwater acoustic channels with multipath and Doppler effect", JASA 2025, DOI 10.1121/10.0037819: the packet is designed offline from coherence time and bandwidth. **Tangential (G1 baseline).** — [Source](https://doi.org/10.1121/10.0037819)
  - Berger, Gomes, Moura, "Study of pilot designs for cyclic-prefix OFDM on time-varying and sparse underwater acoustic channels", OCEANS 2011, DOI 10.1109/OCEANS-Spain.2011.6003627 (clustered versus random pilots under ICI); Nga 2025 (DOI 10.54939/1859-1043.j.mst.iite.2025.99-106); Kim & Lee 2025 (DOI 10.1016/j.apacoust.2025.110554, shorter packets for UUVs). All **tangential**, with no forecast.
- *B4, AMC and link adaptation with feedback delay or physics context:*
  - Wu, Chitre, Anjangi, "Adaptive Modulation and Coding With Feedback Scheduling for an Underwater Acoustic Link", IEEE JOE 2025, DOI 10.1109/JOE.2025.3585657: "extending our BER prediction model from a point prediction to an interval predictor ... Gaussian process regression", plus "feedback scheduling" with DQN. The predicted error interval drives the AMC, which is analogous to B4. There is no Doppler forecast, and the frame layout is fixed. **Partial (B4).** UComms 2022 precursor: DOI 10.1109/UCOMMS56954.2022.9905694. — [Source](https://doi.org/10.1109/JOE.2025.3585657)
  - Anjangi & Chitre, UComms 2018, DOI 10.1109/UCOMMS.2018.8493161: "channel physics such as delay spread, coherence time, doppler spread ... narrowing down the search space of modulation scheme parameters". Uses the Doppler spread as a feature, with no kinematic forecast. **Partial (B4).** — [Source](https://doi.org/10.1109/UCOMMS.2018.8493161)
  - Ranganath et al., ICDCECE 2026, DOI 10.1109/ICDCECE68961.2026.11645748: a DRL state including "Doppler characteristics ... and packet delivery feedback" that sets "modulation strategy, transmission power, packet size and retransmission interval". It adapts packet size and modulation from a Doppler feature, but with no forecast, pilot placement or PER safety. It is a low-tier venue. **Partial/tangential (B2-length + B4).** — [Source](https://doi.org/10.1109/ICDCECE68961.2026.11645748)
  - Jing et al., IEEE IoT-J 2025, DOI 10.1109/JIOT.2025.3639115: delay-Doppler channel prediction with ConvLSTM "for AMC systems". It predicts the channel, not the motion. **Tangential.** — [Source](https://doi.org/10.1109/JIOT.2025.3639115)
  - Busacca, Panebianco, Sun, GLOBECOM 2025, DOI 10.1109/GLOBECOM59602.2025.11432193 (curated PDF): an AoI-aware delayed contextual MAB for modulation and power, with no Doppler or kinematic model. **Tangential (G4, follow-up paper).** — [Source](https://doi.org/10.1109/GLOBECOM59602.2025.11432193)
  - Wang, Yuen, Guan, Ge, MATEC 2019, DOI 10.1051/matecconf/201928307001: RL link adaptation with "juggling-like ARQ" for "long-delayed reward feedback". The delayed-ACK issue is addressed without physics. **Tangential.** — [Source](https://doi.org/10.1051/matecconf/201928307001)
  - Tong, Fu, Wang, Han, IEEE TWC 2023, DOI 10.1109/TWC.2023.3247450: a model-based Thompson-sampling bandit for frequency and rate. **Tangential.**
  - Hasan et al., Ad Hoc Netw. 2025, DOI 10.1016/j.adhoc.2025.103978: "dynamically adjusts guard bands to effectively mitigate Doppler caused by mobile nodes". It adapts to Doppler without prediction. **Tangential.**
- *Wave or swell physics (motivation only):* Rawat, Lall, Srirangarajan, IEEE TAES 2021, DOI 10.1109/TAES.2021.3133323 (sea surface as "sum of sinusoids", Pierson–Moskowitz; "can be used to design an optimized system to enable Doppler compensation at the receiver"); Walstead, JASA 2016, DOI 10.1121/1.4950090 ("time-varying Doppler shift ... attributed to SRP motion associated with surface wave propagation"); Cho & Makris, Sci. Rep. 2020, DOI 10.1038/s41598-020-61043-w (local PDF): "Surface gravity waves and near-sea-surface air bubbles decorrelate acoustic signals over seconds or less". All **tangential**: they characterise the channel and none forecasts for link control.
- *Counter-evidence worth knowing:* Lee, Park, Kim, Im, "Diagnosing Packet-Reliability Limits in a Surface-Buoy Underwater Acoustic OFDM Link: A Two-Deployment Field Study", SSRN preprint 2026, DOI 10.2139/ssrn.7134057. Over 26,170 buoy frames, "adding per-symbol Doppler/CFO refinement to the coarse compensation lowers the PER by at most a factor of 1.15 ... Per-symbol Doppler tracking is therefore not what limits reliability here". This is directly relevant to the open question in the deck ("does a forecast prior still help?"). — [Source](https://doi.org/10.2139/ssrn.7134057)
- *Long-term prediction of channel parameters:* Sun & Wang, IEEE Access 2018, DOI 10.1109/ACCESS.2018.2882890. Recursive online prediction of "the average Doppler spread" and other slowly varying parameters, with a "multiplicative seasonal autoregressive process" for periodic dynamics. It predicts with periodic structure, but on time scales of hours to days and from environmental data, not packet-scale kinematics. **Tangential.** — [Source](https://doi.org/10.1109/ACCESS.2018.2882890)
- *Mobility prediction at the MAC (not PHY Doppler):* Zhu et al., IEEE IoT-J 2026, DOI 10.1109/JIOT.2026.3718797, "short-horizon AUV prediction with motion-induced safety margins" for conflict-graph scheduling (a safety margin from prediction uncertainty, used for scheduling). Basavaraj et al., ICIICS 2026, DOI 10.1109/ICIICS67880.2026.11483582. Both **tangential**.

### Inferences
- The idea's distinctive combination does not appear in any record of this database: a kinematic forecast across packets shared by both ends, driving both the transmit-side frame and receiver pre-warping. Each ingredient exists alone: Doppler-rate models (Radosevic 2014), priors from the previous packet (Abdelkareem 2015), Kalman filters on Doppler (DA-Sync, Hu 2024, Sun 2022), motion-range packet design (Zakharov 2019), and uncertainty-aware AMC with feedback scheduling (Wu–Chitre–Anjangi 2025).
- Radosevic et al. 2014 should be added to the deck's related work as the closest "K + AMC" precedent. It already models Doppler rate across "several data packets" and predicts one travel time ahead for adaptive modulation. Its footnote about the two ends of the link touches G3. The differentiation should stress the swell model, two-way fusion, frame and pilot control, and the use of σ_e.
- Abdelkareem 2015 weakens the novelty claim of G2 ("receivers re-estimate Doppler on every packet without a kinematic prior"). It uses the previous packet's acceleration as a prior range, but not as a kinematic forecast. G2 should be worded as "no state-space forecast with uncertainty".
- The SSRN 2026 buoy study is evidence that, on one buoy link, residual Doppler after coarse compensation did not limit PER. This supports the deck's caution that the gain must be shown against acceleration-aware receivers.

### Gaps
- Zhang et al., OCEANS 2015 ("Closed loop bi-directional broadband Doppler compensation…", DOI 10.23919/OCEANS.2015.7401933) has neither abstract nor PDF locally. It is the main unresolved candidate for G3 and needs checking on the web or by institutional access.
- Also without abstract or PDF: Parrish et al., WUWNet 2009 (Doppler rate); Cuji–Stojanovic–Singer, OCEANS 2025 (path-specific Doppler tracking); Ping Wang et al., MILCOM 2013, "Doppler Compensation Based Optimal Resource Allocation for QoS Guarantees in Underwater MIMO-OFDM Acoustic Wireless Relay Networks" (DOI 10.1109/MILCOM.2013.95). The last pairs Doppler compensation with resource allocation and should be checked.
- The local PDFs could not be read for Wu–Chitre–Anjangi 2025, Ceballos Carrascosa–Stojanovic 2010 (JOE; only the 2008 conference version is local), Van der Werf 2024, DA-Sync 2013 or Hu 2024 (all paywalled). Their ratings rest on abstracts.
- Excluded publishers are not in the database by policy (MDPI, Hindawi), e.g. Park et al., Remote Sens. 2026, which the deck cites. Any prior art in Sensors, JMSE or Remote Sensing is invisible to this sweep.

## Q2. How much does the absence of hits mean? (database coverage)

### Takeaway
The database is large and harvested systematically, but its recall is only moderate: 82 % in the publishers it covers, 73 % overall. A third of the records have no abstract and only about 12 % have a PDF. A local "no hit" is good evidence against well-known IEEE, JASA or Elsevier prior art. It is weaker evidence for conference papers without abstracts and for excluded publishers.

### Cited Findings
- Records: 34,745 in total (years 1946–2027); 33,415 in 2006–2026; 2,842 dated 2026. Computed from `db/papers.sqlite` (8 Oct 2026). — [Source](file:///home/fabio/Underwater/papers/db/papers.sqlite)
- Abstracts: 21,961 of 34,745 records (≈ 63 %), or 21,312 of 33,415 in 2006–2026. The README warns that "Abstracts come from OpenAlex and are missing for some publishers (notably Elsevier)". — [Source](file:///home/fabio/Underwater/papers/README.md)
- Local PDFs (pdf/ or curated/): 4,093 records overall, 4,029 in 2006–2026 (≈ 12 %). Of the 1,424 records that mention "Doppler" in the title or abstract, 209 (≈ 15 %) have a PDF. Paywalled IEEE JOE, TWC and JSAC papers are "in the database with DOI and abstract but without PDF". — [Source](file:///home/fabio/Underwater/papers/README.md)
- Category 02 (Doppler, sync, estimation) holds 2,805 distinct papers from 2006–2026. Category 05 (adaptive link, feedback) holds 2,788 papers in total. Category assignment uses keywords and is multi-label; this sweep found many off-topic AUV-control papers in category 05. — [Source](file:///home/fabio/Underwater/papers/db/papers.sqlite)
- Sources: OpenAlex core harvest (the README said 20 of ≈ 69 pages were done on 6 Oct 2026; `raw/core/COMPLETE` now reads "69 pages"), Crossref (all IEEE JOE since 2001, plus the main journals and conferences), DBLP and Semantic Scholar enrichment. MDPI and Hindawi are excluded. The README's earlier estimate was "≈ 90 % or more" once the core harvest is complete. — [Source](file:///home/fabio/Underwater/papers/README.md)
- Measured recall (`raw/recall/report_20261008.json`, `tools/measure_recall.py`): against the 2006–2026 underwater references of 9 surveys, **609 of 833 found (73.1 %)**. 93 of the 224 missing items are MDPI or Hindawi (excluded by policy), so recall within the covered publishers is ≈ 609/740 = **82.3 %**. The surveys lean towards networking and routing (COMST 2020/2021, IoT-J 2021, JNCA 2024, Access 2025, JSEN 2022, …), so recall for the PHY and Doppler literature has not been measured separately. — [Source](file:///home/fabio/Underwater/papers/raw/recall/report_20261008.json)

### Inferences
- The absence of hits is fairly strong evidence against a prior implementation in IEEE JOE, JASA, TWC or JSAC, which Crossref harvests completely or by relevance. It is weaker evidence for OCEANS, WUWNet and UComms papers whose abstracts are missing, as the unverified Zhang et al. 2015 shows. It is weakest for MDPI, Chinese-language and very recent work.
- Even at 82 % recall, roughly one relevant paper in five may be missing. The web sweeps by the other agents remain necessary.

### Gaps
- Recall has not been measured for the PHY, Doppler or adaptive-modulation subfield. A recall check against the reference lists of a Doppler or AMC survey (e.g. Cooper-Baldock et al. 2026, the AMC survey in the database) would firm this up.

---

### Appendix A: sweep method
1. Read the README and CLAUDE.md of `papers/`, and the schema (`papers`, `paper_category`, `papers_fts`).
2. **Battery 1:** 75 regex queries (case-insensitive) over title + abstract of all 34,745 records (Python `re`; AND-queries as anchored lookaheads).
3. **Battery 2:** 36 FTS5 queries for unusual wording.
4. **Scored sweep:** Doppler records that also hit forecast terms (predict|kalman|track|state-space|kinematic|…) and ≥ 2 of {two-way/ACK/feedback, pilot/frame/packet, AMC/link adaptation, motion/swell/AUV}: 26 records. Looser variant (Doppler + tracking or motion terms): 378 records, titles read manually.
5. **Category sweeps:** category 02 × (transmitter-side / feedback / AMC / pilot / packet design / predict / two-way) = 139 records, titles read. Categories 05/06/07 × (Doppler / motion / AUV / kinematic) plus Doppler × (AMC / packet / feedback / ACK / two-way) = 437 records, titles read.
6. **Union of narrow-query hits** (17 broad queries left out): 771 records; 720 left after dropping records whose categories are only 14/15 (sonar, vehicles). Read by title; top ~50 abstracts read in full.
7. **PDFs read** with pdftotext and passage grep: Zakharov 2019, Abdelkareem 2015, Cho & Kim 2026, Ghalkhani 2026, Radosevic 2011, Radosevic 2014 (JOE; two local copies), Ceballos Carrascosa 2008, Busacca 2025, Cho & Makris 2020.

### Appendix B: queries and hit counts (title + abstract, all 34,745 records)

| ID | Query (regex intent) | Hits |
|---|---|---|
| Q01 | Doppler … predict | 8 |
| Q02 | Doppler … forecast | 0 |
| Q03 | Doppler tracking | 30 |
| Q04 | Kalman AND Doppler | 90 |
| Q05 | time-varying Doppler | 43 |
| Q06 | Doppler rate | 5 |
| Q07 | acceleration AND Doppler | 28 |
| Q08 | sinusoidal/periodic/oscillating Doppler | 1 |
| Q09 | wave-induced / surface-wave motion | 23 |
| Q10 | swell | 82 |
| Q11 | heave | 60 |
| Q12 | platform/transducer/buoy motion | 31 |
| Q13 | (sea-)surface motion AND Doppler | 8 |
| Q14 | AUV AND Doppler | 193 |
| Q15 | navigation/INS/DVL-aided comm | 7 |
| Q16 | INS/DVL AND comm/modem | 78 |
| Q17 | motion-compensated/-aware/-induced | 54 |
| Q18 | kinematic | 433 |
| Q19 | Doppler prior/side information | 1 |
| Q20 | Doppler extrapolation/interpolation | 4 |
| Q21 | resampling AND Doppler | 58 |
| Q22 | time-varying/adaptive resampling | 3 |
| Q23 | Doppler scale / dilation / time-scaling | 182 |
| Q24 | two-way/bidirectional/round-trip Doppler | 2 |
| Q25 | ACK/acknowledgement AND Doppler | 1 |
| Q26 | feedback AND Doppler | 73 |
| Q27 | handshake/RTS/CTS AND Doppler (inflated: "rts" substring) | 378 |
| Q28 | reciprocity … Doppler | 1 |
| Q29 | two-way time / time sync … Doppler | 19 |
| Q30 | fusion of Doppler estimates | 2 |
| Q31 | cooperative Doppler | 1 |
| Q32 | pilot placement/pattern/design/allocation/density | 34 |
| Q33 | adaptive pilot / adaptive training | 2 |
| Q34 | midamble/postamble | 3 |
| Q35 | packet/frame length adaptation | 9 |
| Q36 | packet length AND Doppler | 7 |
| Q37 | training symbols/sequence AND Doppler | 15 |
| Q38 | pilot AND mobility AND adapt/optimise | 18 |
| Q39 | superimposed pilot/training | 13 |
| Q40 | adaptive modulation / AMC | 106 |
| Q41 | link adaptation | 23 |
| Q42 | rate adaptation | 14 |
| Q43 | AMC/link adaptation AND Doppler | 15 |
| Q44 | AMC AND mobility/AUV | 11 |
| Q45 | outer loop / OLLA | 37 (mostly control) |
| Q46 | PER/BLER/BER target | 19 |
| Q47 | channel prediction | 60 |
| Q48 | feedback delay / outdated CSI | 23 |
| Q49 | channel prediction AND Doppler | 7 |
| Q50 | AMC AND prediction | 31 |
| Q51 | Doppler-aware/-adaptive/-aided/-tolerant | 32 |
| Q52 | mobility/motion/velocity-aware | 26 |
| Q53 | ISAC / joint sensing and comm | 49 |
| Q54 | ISAC AND Doppler/velocity | 8 |
| Q55 | sensing/localization/position-aided, location-aware | 26 |
| Q56 | ranging AND Doppler AND communication | 16 |
| Q57 | OTFS / delay-Doppler | 105 |
| Q58 | OTFS AND pilot | 9 |
| Q59 | OFDM AND Doppler | 286 |
| Q60 | non-uniform / multi-scale Doppler | 263 |
| Q61 | ICI AND time-varying/acceleration | 30 |
| Q62 | Doppler AND sea trial AND mobile | 232 |
| Q63 | PLL/phase tracking AND Doppler | 24 |
| Q64 | Doppler spectrum/spread … surface/wave | 9 |
| Q65 | state-space/particle/EKF/UKF AND Doppler | 82 |
| Q66 | predictive/proactive link/transmission | 6 |
| Q67 | Doppler AND MAC/ARQ | 20 |
| Q68 | piggyback / header … Doppler | 3 |
| Q69 | delay-tolerant AND Doppler | 1 |
| Q70 | sea-state adaptive | 1 |
| Q71 | environment-aware link | 3 |
| Q72 | relative velocity/speed | 100 |
| Q73 | pre-compensation / pre-warp | 8 |
| Q74 | adaptive OFDM parameters (CP, guard, subcarrier) | 20 |
| Q75 | coherence time AND adapt/packet/pilot | 15 |

FTS5 battery (hits): doppler+"previous packet" 0; doppler+"previous frame" 0; doppler+history 9; doppler+"next packet" 0; doppler+"next frame" 0; doppler+transmitter+adjust* 3; doppler+"pre-compensat*" 0; doppler+precompensat* 1; "wave motion"+communication 5; "drifting buoy"+doppler 0; buoy+doppler+(ofdm|modem|communication) 9; "surface buoy"+(modem|ofdm) 6; doppler+"rate selection" 0; doppler+scheduling+transmission 3; doppler+"transmission time" 4; Doppler+kalman+(ofdm|modem) 5; "interacting multiple model"+doppler 2; doppler+acceleration+(ofdm|modem|packet) 5; "jerk" 6; "mobility prediction" OR "trajectory prediction" 39; "velocity prediction" 6; pilot+mobility+adaptive 0; "channel aging" 0; "age of information"+(feedback|adaptation) 1; "closed loop"+doppler 5; "closed-loop"+(modulation|adaptation|doppler) 16; "feedback channel"+doppler 1; bidirectional+(link|communication)+doppler 3; duplex+doppler 3; ranging+communication+doppler+joint 1; sensing+communication+doppler+velocity 2; heave+(communication|modem) 4; "ship motion"+(communication|modem) 1; swell+(communication|modem|doppler) 3; "vessel motion" OR "boat motion" 2; periodic+doppler+(communication|modem) 5.

Notable zeros: "Doppler forecast" (Q02), "previous/next packet" with Doppler, "channel aging", and adaptive pilots tied to mobility. Two-way Doppler (Q24) has only 2 hits: UWGS 2024 (sync) and the unverified Zhang et al. 2015.
