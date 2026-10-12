# Gap check: unchecked candidate papers, citation chains and Chinese-language search (8 Oct 2026)

Scope: ten candidates that were still unchecked after the block notes, citation-chain checks on nine base papers, and a
Chinese-language web search. The question each time is whether the work implements part of the idea in
`README_nomab.md`: per-packet Doppler from data and ACKs, fused at both ends (B's estimate returned in the ACK) by a
Kalman filter on a drift-plus-swell model, with the forecast driving receiver pre-warping, extra pilots, packet
length and modulation, and a PER-threshold rule with an ACK-calibrated margin.

Method: OpenAlex API (abstracts from `abstract_inverted_index`; `filter=cites:` for the citation chains), the
Semantic Scholar citations API, Google Patents, open PDFs (arXiv, KoreaScience, the Newcastle thesis repository),
and WebSearch/WebFetch. IEEE Xplore, ScienceDirect, Wiley/Hindawi and ResearchGate full texts returned 403 or a
Cloudflare challenge, so for paywalled items only the abstract was read (noted for each).

**Bottom line.** None of the ten candidates, none of the citing works screened, and nothing found in the Chinese
search implements the idea or any two of its blocks together. Two earlier attributions are corrected below:
(a) the "previous packet's acceleration sets the expectation range" quote comes from Abdelkareem's 2012 PhD thesis,
not from Unal et al. ICC 2023; (b) Thales US 8,023,596 is an **underwater acoustic** modem, not a radio modem.

## Q1. Zhang et al., OCEANS 2015, "Closed loop bi-directional ..." (highest priority): two-way Doppler exchange, or a bidirectional equalizer?

### Takeaway
Here "bi-directional" means a **bidirectional decision-feedback equalizer** (forward and time-reversed DFE) inside one
receiver, and "closed loop" means the in-receiver loop in which equalizer/phase-tracking output updates the
resampling (interpolation) factor symbol by symbol. Nothing is exchanged over the link. Closeness: **tangential**
(a Block 1-type receiver only, within the packet). G3 is not affected.

### Cited Findings
- Exact record: Youwen Zhang, Lu Liu, Dajun Sun, Xin Liu, Hongyu Cui, Zhisheng Li, "Closed loop bi-directional broadband Doppler compensation and channel equalization for single carrier high-speed underwater acoustic communication," OCEANS 2015 MTS/IEEE Washington, DOI 10.23919/OCEANS.2015.7401933; OpenAlex W2289450062; 5 citations. — [OpenAlex](https://api.openalex.org/works/doi:10.23919/OCEANS.2015.7401933)
- Abstract, verbatim: "we propose a broadband Doppler estimation and compensation receiver, which combines a bidirectional decision feedback equalizer (DFE). Compared with the existing receivers, the proposed closed loop Doppler estimation and compensation receiver combined with a bi-directional DFE can not only estimate and compensate the broadband Doppler symbol-by-symbol, but also reduce the error propagation effect caused by DFE ... the performance of the proposed receiver can be improved by 1.17 dB in terms of output SNR at 1 kbps with the platform moving at up to 6 knots." — [OpenAlex](https://api.openalex.org/works/doi:10.23919/OCEANS.2015.7401933)
- Same group (Harbin Engineering University), Chinese journal paper on the same receiver family: the closed loop is internal to the receiver ("通过估计插值因子实现闭环多普勒补偿", closed-loop Doppler compensation by estimating the interpolation factor). The phase error from symbol decisions updates the interpolation factor. The bidirectional DFE it cites is Han, Sheng, Yin et al., "基于双向判决反馈均衡器的水声通信海试试验研究", 兵工学报 37(3):553–558, 2016, DOI 10.3969/j.issn.1000-1093.2016.03.023. It does not mention feedback to the transmitter or prediction for later packets (WebFetch summary). — [移动单载波水声通信中的有效空时处理技术, J. Harbin Eng. Univ. 2018](https://html.rhhz.net/HEBGCDXXB/html/201805021.htm)
- Citing works (5, OpenAlex): joint Doppler-scale estimation and timing sync (ICSPCC 2019), ultrasonic metal-liquid links (Sensors 2023; a PUC-Rio thesis 2022), a preamble for integrated communication and positioning (OCEANS 2024), and two-step inter-frequency interference cancellation (ICSPCC 2023). None is a link-level or two-way scheme (titles). — [OpenAlex cites:W2289450062](https://api.openalex.org/works?filter=cites:W2289450062)

### Inferences
- Can be removed from the list of possible G3 threats. At most it is a within-packet single-carrier tracker to cite with Block 1 baselines.

### Gaps
- The full OCEANS 2015 text was not read (IEEE paywall). The abstract plus the group's 2018 journal description are unambiguous.

## Q2. Li S. et al., Applied Mathematical Modelling 2022: what does it do?

### Takeaway
It is a mechanical and control paper: a multi-body model of an underwater glider plus wave glider, and a 2-DoF motion
platform that keeps the acoustic transducer pointing at the USBL. It has no Doppler estimation, prediction or link
adaptation. Closeness: **tangential**.

### Cited Findings
- Exact record: Shuai Li, Yanhui Wang, Shangshang Wu, Wendong Niu, Shaoqiong Yang, Shiquan Lan, "Multi-body modelling and analysis of the motion platform for underwater acoustic dynamic communication," Applied Mathematical Modelling 109:455–472, Sept 2022, DOI 10.1016/j.apm.2022.05.005. — [OpenAlex search](https://api.openalex.org/works?filter=title.search:Multi-body%20modelling%20motion%20platform%20underwater%20acoustic%20dynamic%20communication); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0307904X22002153)
- Content (search summary of the ScienceDirect abstract): it "constructs a multi-body mathematical model of the network integrating an underwater glider and a wave glider", a platform "requiring at least two rotational degrees of freedom", angles "modified based on the bend of the sound ray", and "the underwater acoustic transducer keeps tracing and pointing to the USBL positioning system." — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0307904X22002153)

### Inferences
- It could be cited only as evidence that wave-coupled platform motion disrupts UWA links (motivation).

### Gaps
- Full text not read (paywall). The abstract summary came via WebSearch, not a direct fetch.

## Q3. Cuji, Stojanovic, Singer, OCEANS 2025: is Doppler predicted across packets?

### Takeaway
No. It does path-specific beamforming plus a PLL/DLL per beamformed path inside a multichannel DFE, a within-packet SIMO
single-carrier receiver, in simulation. Closeness: **tangential** (Block 1 receiver class; relevant only to the
README's "per-direction locked path" bias b_j, as evidence that different paths carry different Doppler).

### Cited Findings
- Exact record: Diego A. Cuji, Milica Stojanovic, Andrew C. Singer, "Path-Specific Angle and Doppler Tracking in Dynamic Underwater Channels," OCEANS 2025, DOI 10.1109/OCEANS58557.2025.11104574; 0 citations. — [OpenAlex](https://api.openalex.org/works/doi:10.1109/OCEANS58557.2025.11104574)
- Abstract, verbatim: "We address scenarios in which the multipath channel exhibits different Doppler distortions across different propagation paths ... a multi-channel fractionally-spaced decision-feedback equalizer (DFE), which couples a phase-locked loop (PLL) with a delay-locked loop (DLL) per beamformed path to perform adaptive resampling ... simulations of highly mobile scenarios with speeds on the order of 20 m/s". — [OpenAlex](https://api.openalex.org/works/doi:10.1109/OCEANS58557.2025.11104574)

### Inferences
- Supports the README's measurement model (direction-dependent path lock, path-specific Doppler), but it is not prior art for the forecast or the two-way fusion.

### Gaps
- Full text not read. Nothing in the abstract suggests cross-packet state, but the receiver initialisation was not verified.

## Q4. P. Wang et al., MILCOM 2013: Doppler compensation combined with resource allocation

### Takeaway
It allocates power for throughput under BER-QoS in a MIMO-OFDM relay network, with a Doppler-compensated ICI model.
It does not forecast, adapt pilots or packet length, or exchange two-way Doppler. Closeness: **tangential** (a
Block 4-adjacent "Doppler-aware resource allocation" analogue, in simulation).

### Cited Findings
- Exact record: Ping Wang, Xi Zhang, Mei Na Song, "Doppler Compensation Based Optimal Resource Allocation for QoS Guarantees in Underwater MIMO-OFDM Acoustic Wireless Relay Networks," MILCOM 2013, DOI 10.1109/MILCOM.2013.95; 10 citations. — [OpenAlex](https://api.openalex.org/works/doi:10.1109/MILCOM.2013.95)
- Abstract, verbatim: "due to the harsh underwater environments, dramatic water-wave dynamics, and the network nodes' mobility, there is serious Doppler shift problem ... we propose the Doppler compensation resource allocation scheme for QoS guarantee ... our proposed Doppler compensation based optimal power allocation scheme ... outperforms the other existing scheme without Doppler compensation in terms of maximizing the capacity while satisfying all wireless links' bit-error-rate QoS requirements." — [OpenAlex](https://api.openalex.org/works/doi:10.1109/MILCOM.2013.95)

### Inferences
- The knob is power and the Doppler input is a compensated-ICI term. Nothing is predicted or fed back. It can be cited as a "Doppler-aware allocation" related work.

### Gaps
- Full text not read. How the Doppler term is obtained (assumed known or estimated) is unverified.

## Q5. Parrish, Roy, Arabshahi, WUWNet 2009: Doppler rate

### Takeaway
A symbol-by-symbol (within-packet) Doppler-rate estimator for OFDM (marginal ML plus an ad-hoc estimator, CRLB), tested
in simulation at 0.29 m/s². Closeness: **tangential** (Block 1 baseline class).

### Cited Findings
- Exact record: Nathan H. Parrish, Sumit Roy, Payman Arabshahi, "Symbol by symbol Doppler rate estimation for highly mobile underwater OFDM," WUWNet 2009, DOI 10.1145/1654130.1654131; 23 citations. — [OpenAlex](https://api.openalex.org/works/doi:10.1145/1654130.1654131)
- Abstract, verbatim: "We propose an OFDM receiver capable of estimating and correcting, on a symbol-by-symbol basis, the subcarrier dependent Doppler shifting due to the movement of source and receiver ... We show through simulation that the proposed receiver design performs well for a source that is accelerating at 0.29 m/s2." — [OpenAlex](https://api.openalex.org/works/doi:10.1145/1654130.1654131)
- Citing works screened (23): the only "two-way" hit is UWGS (Asilomar 2024), already in the deck as a time-sync use. The rest are receivers. — [OpenAlex cites:W2144071016](https://api.openalex.org/works?filter=cites:W2144071016)

### Gaps
- Full text not read (ACM paywall).

## Q6. Unal et al., ICC 2023: does the previous packet's acceleration set the Doppler search range? (Attribution corrected)

### Takeaway
The quote does **not** come from Unal et al. It is from **Abdelkareem's 2012 Newcastle PhD thesis** (Ch. 5, published as
Abdelkareem–Sharif–Tsimenidis, Ad Hoc Networks 2016). The search snippet credited to the ICC 2023 PDF merged the two;
Unal et al. may reuse the idea, but this is unverified. In Abdelkareem the previous packet's acceleration sets an
"expectation range" that gates the receiver's per-packet timing and Doppler estimates. This is a one-ended
inter-packet prior inside the receiver: no state model, no Kalman filter, no feedback, no transmitter adaptation.
Closeness: **partial for Block 1** (the nearest UWA "inter-packet Doppler prior"), **tangential** otherwise. Unal et
al. ICC 2023 itself, from its abstract: a field characterisation of high-frequency Doppler plus a multicarrier
estimator and compensator, **tangential**.

### Cited Findings
- Abdelkareem thesis, verbatim: "the estimation errors are subject to penalization by enabling a learning and punishment (LP) action to fine tune ϕ iteratively ... In learning mode (LM), the acceleration of the previous packet is observed to designate an adaptive expectation range (ER), whereas the punishment mode (PM) drops an out of range estimation." — [A. Abdelkareem, "Doppler Compensation Algorithms for DSP-based Implementation of OFDM Underwater Acoustic Communication Systems," PhD thesis, Newcastle University, 2012](https://theses.ncl.ac.uk/jspui/bitstream/10443/1443/1/Abdelkareem12.pdf)
- Same thesis, Ch. 3: "the error in the OFDM symbol of index 1 comes from the Doppler variation of the last symbol in the previous packet". This motivates including packet boundaries in the smoothing. — [Abdelkareem thesis](https://theses.ncl.ac.uk/jspui/bitstream/10443/1443/1/Abdelkareem12.pdf)
- Same thesis, packet design is fixed offline: "required Doppler resolution and acceleration contribute in determining the OFDM frame and packet length, respectively." — [Abdelkareem thesis](https://theses.ncl.ac.uk/jspui/bitstream/10443/1443/1/Abdelkareem12.pdf)
- Journal version, abstract verbatim: it addresses "the effect of a sudden change in the velocity direction between packets on the entire orthogonal frequency division multiplexing (OFDM) symbols ... a penalization algorithm is applied in order to drop the anomalous parameter among them ... the proposed system accommodates an acceleration up to ± 1 m/s2 during the packet time." Abdelkareem, Sharif, Tsimenidis, "Adaptive time varying doppler shift compensation algorithm for OFDM-based underwater acoustic communication systems," Ad Hoc Networks 45 (2016), DOI 10.1016/j.adhoc.2015.05.011. — [Khalifa University repository](https://khazna.ku.ac.ae/en/publications/adaptive-time-varying-doppler-shift-compensation-algorithm-for-of/)
- Unal, Enhos, Demirors, Melodia, "Field Experiments with Doppler Compensation in High-Frequency Underwater Acoustic Communication System," ICC 2023, DOI 10.1109/ICC45041.2023.10279776. Abstract, verbatim: "we present a detailed set of field experiments characterizing the Doppler effect in a high-frequency underwater acoustic channel. We also introduce a multicarrier communication system incorporating a Doppler estimation and compensation method." — [OpenAlex W4387869504](https://api.openalex.org/works?filter=title.search:Field%20Experiments%20Doppler%20Compensation%20High-Frequency%20Underwater%20Acoustic)

### Inferences
- `block3_two_way_fusion.md` (line 65) and the Block 1 table should credit the "previous-packet acceleration → expectation range" idea to Abdelkareem 2012/2016, and keep Unal 2023 only as a field Doppler characterisation.
- The paper can say that inter-packet Doppler information has been used only as a heuristic gating window at one receiver. The idea replaces that with a two-ended state-space forecast.

### Gaps
- The Unal ICC 2023 full text could not be obtained: the WiNES URL redirects to the lab homepage, S2 lists it as CLOSED, and there is no NSF-PAR copy. Whether it reuses Abdelkareem's LP gating is unknown.

## Q7. Yoshizawa et al. 2018: does the Doppler spread that widens the resampling range come from earlier frames?

### Takeaway
Still unresolved at full-text level. The abstract says the range is "extended by measured Doppler standard deviation", and
a search summary says the number of parallel resamplers is "controlled on frame-by-frame basis according to the
measured Doppler shift mean and deviation". This most likely means statistics measured on the current frame (its
preamble/postamble/pilots). Either way it is a spread statistic that widens a receiver search, not a forecast, and has
no transmitter-side use. Closeness: **tangential**.

### Cited Findings
- Exact record: Shingo Yoshizawa, Takashi Saito, Yusaku Mabuchi, Tomoya Tsukui, Shin-ichi Sawada, "Parallel Resampling of OFDM Signals for Fluctuating Doppler Shifts in Underwater Acoustic Communication," J. Electrical and Computer Engineering 2018, art. 3579619, DOI 10.1155/2018/3579619 (Hindawi, CC-BY). — [OpenAlex](https://api.openalex.org/works/doi:10.1155/2018/3579619)
- Abstract, verbatim: "The conventional method assumes a constant Doppler shift during a communication frame. It cannot cope with Doppler fluctuation ... We propose a parallel resampling technique that a resampling range is extended by measured Doppler standard deviation." — [OpenAlex](https://api.openalex.org/works/doi:10.1155/2018/3579619)
- Search summary (not a verbatim fetch): "The parallel numbers of resampling units are controlled on frame-by-frame basis according to the measured Doppler shift mean and deviation." — [WebSearch result pointing to Wiley](https://onlinelibrary.wiley.com/doi/10.1155/2018/3579619)
- OCEANS 2017 precursor (local DB abstract): "a resampling ratio is assumed to be roughly constant and estimated on average in the measured Doppler shifts ... We propose a parallel resampling receiver using multiple resampling ratios by taking account into Doppler shift dispersion." — local `~/Underwater/papers/db/papers.sqlite`
- Citing works (6): surveys and Doppler-tolerant modulation papers; none adds a cross-frame forecast. — [OpenAlex cites:W2899086529](https://api.openalex.org/works?filter=cites:W2899086529)

### Gaps
- The PDF could not be fetched: downloads.hindawi.com sends a Cloudflare challenge, Wiley/ResearchGate/Gale return 403, and web.archive.org is blocked for WebFetch. Open the CC-BY PDF in a browser to settle "current vs previous frame"; the closeness rating would not change.

## Q8. Kim et al., KICS 2013 / SMP-Sync ICNC 2014: sync messages timed to sea motion

### Takeaway
Confirmed from the full text (KoreaScience PDF). Moored nodes are assumed to move periodically under sea motion. Each
node tracks its own trajectory with **gyro and accelerometer** (not Doppler), finds its **turning point** (TP, where
the direction reverses), and transmits sync messages near the TP so that the propagation delay is the same across
exchanges, which improves the linear-regression skew estimate. It uses no Doppler, no forecast filter and no
link/PHY adaptation. Closeness: **tangential** to the core idea. It is the nearest precedent for the README's
"quiet windows" extension, i.e. timing transmissions to the swell phase.

### Cited Findings
- Exact record: Sungryul Kim, Seong-Jin Park, Younghwan Yoo, "Time Synchronization with Oceanic Movement Pattern in Underwater Wireless Networks" (해수운동의 특성을 활용한 수중 무선 네트워크 시각 동기화), J. KICS 38C(5):486, 2013, DOI 10.7840/kics.2013.38c.5.486. — [OpenAlex](https://api.openalex.org/works/doi:10.7840/kics.2013.38c.5.486); [PDF](http://koreascience.or.kr/article/JAKO201317748766307.pdf)
- English abstract, verbatim: "It controls the instant of transmission by exploiting the feature of an oceanic movement and node deployment." — [KICS](https://journal.kics.or.kr/digital-library/34812)
- Full text (Korean), verbatim: "본 연구에서는 수중에 배치된 센서 노드가 일정한 주기성을 가지고 움직이는 것을 가정한다" (we assume the deployed sensor nodes move with a certain periodicity). "센서노드는 자이로 센서와 가속도 센서를 통해 스스로의 움직임을 실시간으로 관측" (nodes observe their own motion in real time with gyro and acceleration sensors). "참조 노드는 방향이 전환되는 지점, TP(Turning Point)를 파악하고 ... TP 지점에서 타임스탬프가 포함된 메시지를 전송한다" (the reference node identifies the turning point and sends the timestamped message at the TP). The simulation uses movement periods T = 4 s and 3 s. — [KoreaScience PDF](http://koreascience.or.kr/article/JAKO201317748766307.pdf)
- SMP-Sync (Kim, Yoo, ICNC 2014) is the conference version of the same idea. A later SLSMP journal version (sync plus localization on the "semiperiodic property of seawater movement") is in Int. J. Distributed Sensor Networks 2014, DOI 10.1155/2014/172043 (search summary). — [IDEAS/RePEc](https://ideas.repec.org/a/sae/intdis/v10y2014i1p172043.html)
- Citing works (2): both are KICS surveys of time synchronization. — [OpenAlex cites:W2108408866](https://api.openalex.org/works?filter=cites:W2108408866)

### Inferences
- Worth one sentence in related work: "transmission timing exploiting periodic sea motion has been used for time sync (IMU-based, Kim et al. 2013); we infer the oscillation from the link's own Doppler and use it for PHY/link decisions."

### Gaps
- No DOI was found for SMP-Sync ICNC 2014, and its text was not read.

## Q9. Yuan, Wei, Li, Yuan, Ng, IEEE JSTSP 2021: what does the prediction drive?

### Takeaway
Full text read (arXiv 2105.03125). The RSU estimates vehicle position and speed from radar echoes and predicts the next
frame with a **constant-velocity** extrapolation, not an oscillator. The prediction drives (i) downlink predictive
beamforming plus transmitter pre-equalization of the predicted delay/Doppler, so vehicles detect with single-tap ML
and no channel estimation, and (ii) an uplink **symbol placement without guard space** (pilot superimposed on data),
plus an SPA detector that accounts for prediction uncertainty. It adapts no frame length or modulation and has no
PER/ACK control. Closeness: **partial (non-UWA analogue)** for "one kinematic prediction drives frame design and
receiver". It is monostatic radar sensing, not two-way communication Doppler fusion.

### Cited Findings
- Exact record: W. Yuan, Z. Wei, S. Li, J. Yuan, D. W. K. Ng, "Integrated Sensing and Communication-Assisted Orthogonal Time Frequency Space Transmission for Vehicular Networks," IEEE JSTSP 2021, DOI 10.1109/jstsp.2021.3117404. — [arXiv PDF](https://arxiv.org/pdf/2105.03125.pdf)
- Prediction model, verbatim: "Based on the estimated vehicle locations and speeds, we can predict the motion parameters for vehicles in the following OTFS frame transmission. Specifically, for the OTFS frame duration of ∆T ... q̄i = q̂i + ∆T · ŝi · [1, 0]^T" (eq. 29); the predicted Doppler is ν̄i = s̄i cos θ̄i fc / c (eq. 32). — [arXiv](https://arxiv.org/pdf/2105.03125.pdf)
- Transmitter use, verbatim: "Using h̄i and ν̄i, the RSU can pre-equalize the transmitted signal to compensate the channel attenuation at the transmitter side and the vehicles do not have to estimate the channel". — [arXiv](https://arxiv.org/pdf/2105.03125.pdf)
- Frame use, verbatim: "We propose to predict the delay and Doppler shifts associated with the paths by exploiting the slow time-varying property of the DD domain channel. Based on the prediction of the channel, we design a new symbol placement scheme that does not require the guard space ... the pilot and data symbols are superimposed while the guard space can be absent. Accordingly, the entire uplink OTFS frame can be used for carrying information". — [arXiv](https://arxiv.org/pdf/2105.03125.pdf)
- Receiver use, verbatim: "we propose a factor graph and the SPA-based detector while taking the channel estimation uncertainty into account." — [arXiv](https://arxiv.org/pdf/2105.03125.pdf)
- Citing works on OpenAlex (6) include OTFS-chirp JSAC and ISAC interference papers; none is UWA. — [OpenAlex cites:W3163822817](https://api.openalex.org/works?filter=cites:W3163822817)

### Inferences
- The deck should keep presenting Yuan 2021 as the strongest non-UWA analogue. The differences are a constant-velocity vs oscillatory model, radar echoes vs two-way communication Doppler, pilot removal vs extra-pilot placement under a coherence grid, and no PER control.
- OpenAlex shows only 6 citations, which is far below its real count (the arXiv record is probably split), so this citation chain is not exhaustive.

### Gaps
- No Kalman filter appears in the JSTSP text: the grep for "Kalman"/"EKF" returned nothing, so the "EKF" wording in earlier notes probably comes from Yuan's related ISAC predictive-beamforming papers, not this one.

## Q10. US 8,023,596 B2 (Thales): do the extrapolation measurements span frames/packets? (Domain corrected)

### Takeaway
It is a **submarine acoustic modem** (not RF, contrary to `block1_forecast_receiver.md` line 67). The predictor runs
inside one continuous message, which carries "throughout the whole message" a set of unmodulated pilot tones. A new
Doppler (filter-bin "position") measurement arrives every 80 ms symbol, and a weighted quadratic regression over the
last T+1 measurements predicts the next one. It does not span packets, has no oscillator model, no feedback and no
transmitter-side adaptation. Closeness: **partial for Block 1 only** (a within-message parabolic Doppler predictor in a
UWA modem). It is prior art for "predict Doppler from past measurements" at symbol scale, not at packet scale.

### Cited Findings
- Bibliographic: US 8,023,596 B2, "Doppler tracking method and device for a wide band modem," inventor Pierre-André Laurent, assignee Thales SA, priority 2005-11-25, filed 2006-11-24, granted 2011-09-20 (also US20090067562A1); status "Expired - Fee Related". — [Google Patents](https://patents.google.com/patent/US8023596B2/en)
- Domain, verbatim: "the modem (modulator-demodulator) therefore uses acoustic waves to transport the messages transmitted. the submarine environment has an extremely complex behavior for acoustic waves." — [Google Patents](https://patents.google.com/patent/US8023596B2/en)
- Within-message, verbatim: "This system requires the presence, throughout the whole message and even a little longer, of a set of N unmodulated fixed carriers in addition to the payload signal." The Doppler estimate is made on these carriers, "an offset that is capable of varying rapidly". — [Google Patents](https://patents.google.com/patent/US8023596B2/en)
- Predictor, verbatim: "It is constituted notably of a delay line containing the last T+1 M/position_M pairs ... the Doppler variation may be assimilated locally to a parabola, the value of the position at the moment t is predicted by means of a weighted quadratic regression." The measurement rate is "a new symbol appears every 80 ms and then a new frequency offset value is needed" (WebFetch extraction). — [Google Patents](https://patents.google.com/patent/US8023596B2/en)

### Inferences
- Correct the earlier note: Thales is a UWA in-receiver Doppler extrapolator over a continuous pilot-tone signal. It strengthens the claim that predicting Doppler from past measurements exists at symbol scale within one message, and that nobody carries the prediction across packets or into frame design.

### Gaps
- None material.

## Q11. Citation-chain checks: is any citing work closer than its base paper?

### Takeaway
No citing work implements cross-packet Doppler forecasting, two-way Doppler fusion for link use, or forecast-driven
pilots, length or modulation. Three citing works are new to the notes and worth adding as partial or tangential
related work: **Petroni et al. 2023** (Kalman-like CSI prediction → cyclic-prefix length, delay spread rather than
Doppler), **Kim, Park, Im, JMSE 2026** (prediction-driven per-transmission segment-size and FEC selection), and the
**VELP/LSVP** Doppler-based velocity prediction papers (used for localization, not for the link).

### Cited Findings
- **DE-Sync (10.3390/s18061710), 28 citers; APE-Sync (10.1109/ACCESS.2019.2912229), 21; DA-Sync (10.1109/TMC.2013.13), 89.** Keyword-filtered hits are all time sync or localization. Examples: APE-Sync (Kalman, sync); "Underwater Localization Using Differential Doppler Scale and TDOA" (WCMC 2022); "Residual Asymmetry Modeling ... High-Dynamic Two-Way Microwave Links" (Sensors 2026, non-UWA two-way time/frequency); and an EKF clock-tracking TCOM 2021 paper. — [OpenAlex cites:W2804929802](https://api.openalex.org/works?filter=cites:W2804929802); [cites:W2941329658](https://api.openalex.org/works?filter=cites:W2941329658); [cites:W2141290731](https://api.openalex.org/works?filter=cites:W2141290731)
  - Song et al., "Efficient Velocity Estimation and Location Prediction in Underwater Acoustic Sensor Networks" (VELP), IEEE IoT-J 2021, DOI 10.1109/jiot.2021.3094305: "we apply an AUV-assisted velocity estimation algorithm with Doppler shift estimation in the physical layer ... we build a belief propagation-neural network-based location prediction model". **Tangential**: Doppler-derived velocity plus prediction, used for localization only. — [OpenAlex](https://api.openalex.org/works/doi:10.1109/jiot.2021.3094305)
  - Wang et al., "An Efficient Localization Scheme With Velocity Prediction for Large-Scale UASNs" (LSVP), IEEE IoT-J 2023, DOI 10.1109/jiot.2023.3311791: "we first design a Doppler-assisted velocity prediction (DVP) algorithm to decrease energy consumption, which can solve the excessive communications caused by node mobility under ocean currents." **Tangential** (localization; current-driven, no swell, no link use). — [OpenAlex](https://api.openalex.org/works/doi:10.1109/jiot.2023.3311791)
- **Radosevic et al. JOE 2014 (10.1109/JOE.2013.2253212), 240 citers screened.** The hits are the known adaptive-modulation and CSI-prediction literature (Wan et al. JOE 2014; Qarabaqi–Stojanovic; DRL/Q-learning AMC; Wu–Chitre–Anjangi JOE 2025; Park et al. 2026; Holt-Winters SNR prediction ICITBS 2018; Markov CSI prediction OCEANS 2018). Two are new and relevant:
  - Petroni, Scarano, Cusani, Biagi, "On the Effect of Channel Knowledge in Underwater Acoustic Communications: Estimation, Prediction and Protocol," Electronics 12(7):1552, 2023, DOI 10.3390/electronics12071552 (MDPI). Verbatim: "we present an adaptive OFDM scheme where channel state information is predicted through a Kalman-like filter so as to optimize communication parameters, including the cyclic prefix length. This mechanism aims to mitigate the variability of channel delay spread. This is cast in a protocol where channel estimation/prediction are jointly considered". **Partial** (Kalman-type prediction drives a frame parameter, inside a protocol), but it predicts CSI/delay spread, not Doppler kinematics, and is one-ended. — [OpenAlex](https://api.openalex.org/works/doi:10.3390/electronics12071552)
  - Fu, Song, Tao, Zhang, Pan, "Virtual MIMO Transmissions for Underwater Acoustic Communications with Moving Platforms," WUWNet 2019, DOI 10.1145/3366486.3366532: "The user selection and modulation choice are decided by predicted data rates derived from the communication range rather than the channel state information". **Tangential** (mobility → predicted rate → modulation, range not Doppler). — [OpenAlex](https://api.openalex.org/works/doi:10.1145/3366486.3366532)
  - Roudsari, Bousquet, Sensors 19(1):105, 2018/19 (DOI 10.3390/s19010105): pilot tones in every OFDM symbol estimate the time-scaling, followed by time-varying resampling within the frame. **Tangential**. — [OpenAlex](https://api.openalex.org/works/doi:10.3390/s19010105)
  — [OpenAlex cites:W2024977294](https://api.openalex.org/works?filter=cites:W2024977294)
- **Abdelkareem et al. Ad Hoc Networks 2015/16 (10.1016/j.adhoc.2015.05.011), 35 citers.** All are receivers: multibranch autocorrelation (JOE 2017), Bousquet's data packet structure for dynamic channels (JOE 2019, already known as the Zakharov/Bousquet-type offline design), iterative DD-DSSS (JASA 2023/2025), ZC pilots (Electronics 2024). — [OpenAlex cites:W2218383705](https://api.openalex.org/works?filter=cites:W2218383705)
- **Park et al. Remote Sensing 2026 (10.3390/rs18111747), 2 citers** (OpenAlex and S2 agree):
  - Seunggyu Kim, Saeyong Park, Taeho Im, "Channel-Adaptive Joint Selection of FEC Scheme, Rate, and Segment Size for Short-Block Underwater Acoustic Communication," JMSE 14(13):1239, 2026, DOI 10.3390/jmse14131239 (MDPI). Verbatim: "the payload segment (block) size ... is seldom adapted per transmission ... We propose a per-transmission controller that jointly selects the FEC scheme, code rate, and segment size from a prediction of the near-term channel state ... the segment-size degree of freedom is the dominant throughput lever ... a lightweight persistence predictor matches a learned LSTM for the first-order channel-state model studied ... recorded-channel replay over the public Watermark benchmark". **Partial for Block 4** (prediction-driven packet-length choice under ARQ goodput, Watermark replay), but the predicted state is SNR/fading, with no Doppler, kinematics, pilots or two-way fusion. It does not beat Park 2026 on closeness, but it pre-empts any claim that "per-packet length adaptation" in UWA is new. — [OpenAlex](https://api.openalex.org/works/doi:10.3390/jmse14131239)
  - The other citer is an FPGA LT decoder (ISSET 2026): irrelevant.
- **Simeone & Spagnolini ICC 2004 (10.1109/icc.2004.1312647), 31 citers.** All are terrestrial OFDM pilot-pattern or Kalman channel-estimation papers. The closest is Šimko, Diniz, Rupp, "Design requirements of adaptive pilot-symbol patterns," ICC Workshops 2013, DOI 10.1109/iccw.2013.6649217: "only four bits of additional wide-band feedback per user are required in order to optimally support RMS delay spreads up to 800 ns and Doppler frequencies up to 1200 Hz". This is an RF, Doppler-spread-indexed pilot pattern chosen from a feedback index, **tangential** (and a useful number for the "≈ 20 header bits" budget). No UWA citer. — [OpenAlex cites:W2132924662](https://api.openalex.org/works?filter=cites:W2132924662); [Šimko 2013](https://api.openalex.org/works/doi:10.1109/iccw.2013.6649217)
- **Hu et al. MSN 2024 (10.1109/MSN63567.2024.00031)**: 0 citers on both OpenAlex and S2. — [OpenAlex](https://api.openalex.org/works?filter=cites:W4411584749)
- **Wu–Chitre–Anjangi JOE 2025 (10.1109/JOE.2025.3585657)**: citers are a shipborne-relay covert-communication paper (TNSE 2026), a survey of adaptive multi-mode UWA communication (Ocean Eng. 2026, DOI 10.1016/j.oceaneng.2026.126617), and a DRL MAC under mobility (arXiv 2605.06536, "MobiU-MAC", a MAC protocol with no Doppler-PHY link). None is closer. — [OpenAlex cites:W4413847041](https://api.openalex.org/works?filter=cites:W4413847041); [S2 citations](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1109/JOE.2025.3585657/citations)
- **Extra chains run on the candidates themselves**: Zhang 2015 (5), Kim KICS 2013 (2), Unal ICC 2023 (9: reviews, FEC evaluations, and Donnelly et al. "Comparison of Doppler Estimation Methods in Mobile UWA," OCEANS 2024, preamble/postamble displacement vs FrFT vs dilation, all within-packet), Yoshizawa 2018 (6), Parrish 2009 (23): none closer than its base paper. — OpenAlex `filter=cites:` on W2289450062, W2108408866, W4387869504, W2899086529, W2144071016

### Inferences
- Add Petroni 2023 and Kim–Park–Im 2026 to the Block 2/4 related-work rows ("prediction → frame/packet parameter in UWA, without Doppler kinematics"). Both are MDPI, so they are outside the local library by policy.
- The Ocean Engineering 2026 survey of adaptive multi-mode UWA communication should be skimmed. A survey that recent would probably list a Doppler-forecast-driven adaptive scheme if one existed.

### Gaps
- OpenAlex citation counts for conference and arXiv-split records are incomplete (e.g. Yuan 2021 shows 6). Google Scholar "cited by" was not accessible.
- The Ocean Engineering 2026 survey (DOI 10.1016/j.oceaneng.2026.126617) was not read.

## Q12. Chinese-language search (papers and patents)

### Takeaway
Five Chinese queries found nothing that forecasts Doppler across packets, fuses two-way Doppler, or drives
pilots/length/modulation from a Doppler forecast. The hits are in-receiver compensators (symbol-by-symbol PTLL, closed-loop
interpolators, space-time variable-sampling beamforming), channel prediction (Arctic ice-water, sparse channel
MMSE/LMS/RLS), Kalman equalizers, and patents for Doppler measurement or estimation.

### Cited Findings
- Queries: "水声通信 多普勒 预测 双向"; "水声 多普勒 卡尔曼 跨包 预测 自适应 导频"; "水声 双向 多普勒 估计 融合 应答 链路自适应"; "水声通信 多普勒 涌浪 预测 发射端 预补偿 专利"; "水声通信 波浪运动 多普勒 卡尔曼滤波 下一帧 预测 浮标"; plus an author-targeted query for Zhang/Sun (Q1).
- Representative hits, all receiver-side or channel-prediction:
  - Arctic channel prediction (信号处理 2019) — [link](https://signal.ejournal.org.cn/fileXHCL/journal/article/xhcl/HTML/2019/9/2019-9-1496.html)
  - IOA-CAS symbol-by-symbol motion compensation for 256QAM (PTLL, up to 2 m/s²) — [IOA news, Dec 2024](http://ioa.cas.cn/kychg/kyjz/202412/t20241212_7454651.html)
  - space-time joint variable-sampling beamforming Doppler compensation (声学学报 2024) — [link](https://www.jac.ac.cn/cn/article/pdf/preview/10.12395/0371-0025.2024096.pdf)
  - Kalman-based hybrid bidirectional iterative equalization (电子与信息学报, JEIT211343). Note that "双向" again means a bidirectional equalizer — [link](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT211343)
  - patent CN104168232A, multipath delay and Doppler measurement — [Google Patents](https://patents.google.com/patent/CN104168232A/zh)
  - patent WO2022110409A1, UWA communication-detection signal processing — [Google Patents](https://patents.google.com/patent/WO2022110409A1/zh)

### Inferences
- In the Chinese UWA literature "双向" (bidirectional) consistently refers to bidirectional equalizers, not two-way links. A search for two-way Doppler fusion needs terms like "往返/双程 多普勒" or "应答 携带 多普勒". That was tried only indirectly via "应答"; CNKI itself was not searched.

### Gaps
- CNKI, Wanfang and CNIPA full-text patent search were not accessible. Google Patents was queried only through WebSearch, so Chinese patent coverage is shallow.
