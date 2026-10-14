# Public real data for validation (14 Oct 2026; PLAN Phase 2 / 8)

**Question.** Is there a public sea-trial dataset with acoustic recordings (or measured channels) and the platforms'
motion-sensor logs, to validate sensor-aided Doppler estimation on real data before 23 Oct?

## Verdict
No public dataset pairs acoustic recordings with IMU / heave logs. The best available option is **semi-real**:
measured time-varying channels whose Doppler trajectory is kept (Underwater Acoustic Channel Library, 2026). Use the
measured delay / phase drift as the truth, and derive the sensor prior from it through our IMU error model (imu.py). That
validates the signal-side estimators on real channels and gives the sensor branch a realistic error, but not a
*measured* sensor error.

## Candidates
| Dataset | What it has | Motion / sensor data | Access | Use for us |
|---|---|---|---|---|
| **Underwater Acoustic Channel Library** (Li, Chitre, Cuji, Preisig, Singer, Stojanovic, van Walree; arXiv 2609.03207, Sept 2026); site uwa-channels.github.io, data [Zenodo 10.5281/zenodo.21287414](https://doi.org/10.5281/zenodo.21287414) | Measured time-varying impulse responses + noise models + replay code, 8 channels. Delay drift and phase are tracked and stored (`phi_hat` / `theta_hat`) and reinserted on replay, so the **Doppler is preserved** | None (mobility class only). **Blue**: MACE'10, transmitter from a moving vessel, racetrack up to 1.5 m/s, 3–7 km, fc 13 kHz, 12-element array. **Red**: Singapore 2024, transmitter from a **drifting boat** (wave-driven motion), 100–400 m, fc 25 kHz, 3 elements | Open source (paper CC BY 4.0; check the Zenodo licence) | Best option. Red = wave-driven drift at short range; Blue = vessel motion (slow, mostly predictable). Cite it in any case |
| Watermark (FFI; van Walree et al., IEEE JOE 2017) | Measured channels NOF1, NCS1, BCH1, KAU1, KAU2 (KAU: towed source) | None | Free download (MATLAB) | Superseded by the library above (same group); not checked whether KAU keeps the drift |
| SFI Smart Ocean (IEEE DataPort, DOI 10.21227/3aa6-4k33, 2025) | Raw hydrophone recordings, several modulations, 4–8 / 9–14 / 24–32 kHz, Norwegian fjord | None (SSP, bathymetry, a sea-surface photo) | IEEE account | Fixed nodes: no motion to estimate |
| ACommSet (Rutgers, ACM WUWNet 2024) | CSI estimated from OFDM, static and dynamic (steered dinghy) | None stated | Release agreement | CSI only, no raw signal |
| MACE'10 raw data (WHOI) | Raw recordings; ship GPS every 1 min | GPS at 1-min intervals (useless at wave frequencies) | Not public | Blue channel above is its public part |

## How to use the library (proposal, not done)
1. Download Red (and Blue); reconstruct the channel with drift; pass our OFDM frames through it (band-limit to the
   channel's band: Red 20–30 kHz, Blue 10.5–15.5 kHz, so fc / B change from our 16 / 8 kHz; frequency-scaling rule
   of b18 says the conclusions carry over).
2. Truth: the stored drift trajectory (delay drift -> Doppler scale). Sensor prior: truth + imu.py error for a
   chosen grade (projected along the path); side information quantised as in Phase 4.
3. Compare the same receivers as E2 (signal Kalman, pFFT, pilot tracking, safe fusion). Claim wording: "real
   channels, modelled sensor error".
Cost: download size unknown (Zenodo record), ~1 day of work; runs light (single packets).
