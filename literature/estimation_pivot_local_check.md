# Estimation pivot: quick local check (9 Oct 2026, 10-minute timebox)

Question: are E1 (the transmitter sends its own measured motion inside the packet as Doppler side information) and
E2 (the receiver fuses that side information, its own IMU and pilot-based estimates) open?

Searched: local library (`papers/tools/search.py`, 19 queries: Doppler side information / inertial / IMU /
accelerometer / INS / velocity feedback / sensor aided / Kalman / fusion / tracking OFDM / prediction /
pre-compensation transmitter / motion sensor communication / buoy motion / platform motion compensation / pilot
overhead / tracking adaptive / cooperative Doppler estimation / relative velocity estimation), plus our earlier
prior-art notes (`prior_art_summary.md`, `prior_art_notes/block3_two_way_fusion.md`). No web search, no full texts.

## Findings
- **Closest to E1 (sensor-aided Doppler for communication):** Oiwa et al., IEICE Tech. Rep. 2020, accelerometer-aided
  higher-order Doppler compensation, at one end, not peer reviewed (already in the deck). Nothing else in the library
  uses an IMU/accelerometer to estimate Doppler for the communication link.
- **Doppler values carried in messages:** DE-Sync 2018 / APE-Sync 2019 return the far end's measured Doppler in the
  reply, for clock synchronisation only (earlier check). Earlier finding still holds: "no UWA work puts a Doppler or
  velocity estimate into an ACK or control header for the communication link".
- **Sensors + Doppler elsewhere:** the large DVL/INS navigation literature (Hegrenaes 2009, Troni 2012, Pan 2015,
  Karmozdi 2021, ...) and modem Doppler into navigation filters (Aubry 2019, Garin 2021): communication feeds
  navigation, never the reverse.
- **Signal-based Doppler tracking (crowded):** Kalman Doppler tracking for spread spectrum (Sun 2022) and FH-FSK
  (Hu 2024), OFDM tracking (Pu 2018), cyclostationary estimation (Fan 2025), pilot overhead for iterative OFDM
  receivers (Berger 2011).
- **To read:** Hyosong Lee, "Diagnosing Packet-Reliability Limits in a Surface-Buoy Underwater Acoustic OFDM Link"
  (SSRN 2026, 10.2139/ssrn.7134057) — same setting as ours; check whether motion/Doppler is the diagnosed limit.
- **Outside UWA (to cite as analogues):** 3GPP NTN, where the terminal estimates and pre-compensates Doppler from
  ephemeris and its own GNSS position (sensor/side-information estimation at the transmitter).

## Verdict (provisional)
E1 (in-packet motion side information, sized in bits for a target accuracy) and E2 (fusion of transmitter side
information, receiver IMU and pilots) were **not found** in the local library. Note that the archived forecast /
two-way-fusion version of this project (Block 3) was killed by Phase 0 because it predicted across a round trip;
E1/E2 avoid that by sending motion measured during the same packet. Needs a web / OpenAlex check before it is
claimed as a gap.

## Lee, Park, Kim, Im 2026 (SSRN 10.2139/ssrn.7134057, preprint, not peer reviewed) — read 9 Oct 2026
Local copy: `/home/dario/Underwater/ssrn-7134057.pdf` (29 pages).
- Setting: bottom-mounted transmitter (120 m) → surface-buoy receiver (5 m), 10.5 h, 26,170 frames, sea state 3
  (Hs 0.5–1 m, wind 4–8 m/s); control: static rope-suspended link 250 m–1 km (5,130 frames).
- Waveform: fc 20 kHz, B 20 kHz, N 2048, subcarrier spacing 9.77 Hz, symbol 125 ms, frames of 7 symbols (0.875 s),
  QPSK + turbo code; field median pilot SNR ≈ 9 dB; receiver on the buoy only, no motion sensors.
- Findings: raw Doppler up to ≈ 6 Hz (0.61 Δf); after the preamble-based front end (frame-averaged scale + resampling)
  the residual has median ≈ 0 and IQR ±2 Hz (0.2 Δf). Per-symbol Doppler/CFO refinement improves PER by at most
  ×1.15 (×1.03 on the static link); failures track pilot SNR, not Doppler. Reliability comes from per-subcarrier LLR
  weighting and channel-estimate smoothing (PER 0.044 buoy, 0.047 static). Simulation (Bellhop, 0.3 Hz heave, SNR 6 dB):
  all tiers fine up to A = 4 Hz (0.41 Δf); collapse at 6 Hz; everything fails at 8 Hz (0.82 Δf).
- No IMU / side information / transmitter-side processing.
- Relevance: strongest counter-evidence to "Doppler limits buoy links" — but in a noise-limited regime (≈ 9 dB SNR,
  QPSK, 0.9 s frames, sea state 3, one moving end, small buoy). A 0.2 Δf residual is ≈ 9 dB SIR: harmless at 9 dB SNR,
  binding for high-order modulation / high SNR, longer frames, rough seas, both ends moving. Must be cited and the
  paper's regime stated against it; it also supports the user's premise that frame-level compensators work once the
  estimate is right.

## Web / OpenAlex check (12 Oct 2026)
Searched: 11 web searches (IMU / accelerometer / INS / DVL / navigation- or motion-aided Doppler estimation for UWA
communication; velocity side information in packet headers; Doppler prior to narrow the preamble search; surface-buoy
motion sensors; GNSS-aided pre-compensation in NTN; IMU-aided CFO/Doppler in UAV and vehicular radio) and 4 OpenAlex
queries. ACM page of "Doppler Correction of Mobile Acoustic Communication via ..." (10.1145/3291940.3291978) returned
403 — title and content still unchecked.
- **UWA: no work found** that uses IMU / DVL / INS / navigation velocity as an input or prior to a communication
  Doppler estimator, nor that sends transmitter motion as side information. All sensor + Doppler work runs the other
  way (communication Doppler → navigation filters: Aubry 2019, Garin 2021, Choi et al. RA-L 2022 "Broadband acoustic
  communication aided underwater INS"). Oiwa 2020 (accelerometer, IEICE tech. rep.) is the only sensor-for-
  communication item, and is not indexed beyond OpenAlex (no DOI).
- **Radio analogues (must cite):** 3GPP NTN / GNSS-assisted patents (US 10,826,595 "GNSS-assisted wireless
  communication": velocity from GNSS → Doppler → transmit pre-compensation); Hong Gan UAV patents (US 10,015,762;
  10,687,297; 11,234,202): IMU → vibration/rotation Doppler → transmitter predistortion, noting that reference signals
  consume bandwidth; inertial-assisted mmWave beam tracking (Polimi EuCNC 2019).
- **Counterpoint (must address):** AsyMov (arXiv 2412.10387, ISAC): channel-based Doppler beats IMU when enough
  multipath components are available; the IMU helps only when the channel lacks them. Our analogue: the IMU should
  be used where signal estimation is weak or costly (low SNR, short preambles, per-symbol tracking without pilots,
  large Doppler search ranges), not as a replacement.
- **Verdict:** E1/E2 not found in UWA; the mechanism has radio precedents (GNSS/IMU-aided pre-compensation, mostly
  patents), so the novelty claim is "first for UWA, with real wave-motion data, the side-information sizing, and the
  fusion with signal estimates", not "first sensor-aided Doppler" in general.
