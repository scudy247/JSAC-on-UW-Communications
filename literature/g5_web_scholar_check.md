# G5 check: transmit-side Doppler pre-compensation and multi-user uplinks (9 Oct 2026)

**Gap under test (G5):** no UWA transmitter pre-compensates Doppler, and multi-user links with a different Doppler
per user are handled only at the receiver, with guard bands following mobility rather than a forecast.

## Verdict
**G5 holds for transmit-side pre-compensation, but its wording on the receiver side must change.**
- No search found an underwater work where a node pre-compensates its own Doppler (from a measurement, a prediction
  or navigation data) before transmitting, single-user or multi-user.
- The receiver side is stronger than the deck said. Tu, Duman, Stojanovic and Proakis already run OFDMA with a
  different Doppler per user, using **one resampling branch per user** at the receiver (Allerton 2011; JOE 2013).
  The deck's "no receiver undoes the interference" is therefore wrong as worded. The defensible statement is that
  receiver-side separation needs K resampling branches and still leaves inter-user interference, which
  pre-compensation at the source avoids.
- **Terminology trap.** In the UWA literature, "Doppler pre-compensation" usually means the coarse *receiver-side*
  correction applied before the equaliser (Johnson–Freitag–Stojanovic 1997; Eynard–Laot 2008; Gomes et al. 2008).
  "Transmitter precompensation" (Berger et al. 2010) means a filter for the transducer response, not Doppler. The
  paper must say "transmit-side Doppler pre-compensation" every time and define it.

## What was searched
- **Web search:** 7 queries in English and 1 in Chinese (e.g. transmitter-side Doppler pre-compensation underwater;
  uplink OFDMA with per-user Doppler; nodes predicting Doppler and pre-compensating; modem resampling from its own
  navigation data; gateway/buoy and AUV-swarm concurrent uplink; 水声通信 多普勒预补偿 发射端).
- **Google Scholar via Chrome:** two queries completed. Scholar then showed a "verify you're not a robot" CAPTCHA
  for the network (IP 151.97.13.95). It was not solved, so the cited-by lists were not read on Scholar.
- **OpenAlex citation lists instead:** all works citing Tu et al. 2011 (OFDMA), Tu et al. JOE 2013
  (multiple resampling), Daoud et al. 2015 and Hasan et al. 2025. That is 159 unique works with titles and
  abstracts, filtered for Doppler + transmitter, pre-compensation, feedback or prediction (15 hits), and for
  multi-user / uplink / OFDMA (27 hits). All were read.
- **One thesis read in full metadata:** Cuji Dután 2025 (abstract from the Northeastern repository; the PDF could not
  be read through the browser).

## Works to cite (new since the library search)

| Work | What it does | Relation to G5 |
|---|---|---|
| K. Tu, T. M. Duman, M. Stojanovic, J. G. Proakis, "OFDMA for underwater acoustic communications", Allerton 2011, [doi:10.1109/allerton.2011.6120227](https://doi.org/10.1109/allerton.2011.6120227) | OFDMA uplink; per-user Doppler causes ICI and inter-user interference; receiver front end with one resampling branch per user; contiguous vs interleaved subcarrier allocation | **Closest receiver-side precedent.** Same problem, opposite side of the link. Guest editor co-author |
| K. Tu, T. M. Duman, M. Stojanovic, J. G. Proakis, "Multiple-resampling receiver design for OFDM over Doppler-distorted underwater acoustic channels", IEEE JOE 2013, [doi:10.1109/joe.2012.2221812](https://doi.org/10.1109/joe.2012.2221812) | Multiple-resampling branches per user or path; MACE10 and KAM08 sea data | The baseline Block 4 must beat, in complexity and residual interference |
| A. Pottier et al., "Data rate performance of mobile multiuser MIMO UWA systems", IEEE JOE 2024 | Multi-user MIMO; each user's Doppler estimated and its signal resampled at the receiver | Receiver-side, recent |
| D. A. Cuji Dután, "Multipath beamforming in time-varying channels with applications to underwater acoustics", PhD thesis, Northeastern Univ., 2025 (advisor M. Stojanovic), [doi:10.17760/D20794306](https://doi.org/10.17760/D20794306) | Transmit beamforming suffers because the channel estimate formed at the transmitter from receiver feedback is stale; solved by using the slowly varying angle of arrival. Multi-user SDMA uplink and downlink to a base station | **Same stale-feedback argument at the transmitter**, solved by choosing a slow parameter instead of predicting a fast one. Supports our framing; cite it |
| Downlink UWA OFDMA with outdated CSI: OCEANS 2018 [doi:10.1109/oceanskobe.2018.8559440](https://doi.org/10.1109/oceanskobe.2018.8559440); IEEE Access 2019 [doi:10.1109/access.2019.2892812](https://doi.org/10.1109/access.2019.2892812); Appl. Ocean Res. 2025 (kernel RLS channel prediction) [doi:10.1016/j.apor.2025.104586](https://doi.org/10.1016/j.apor.2025.104586) | Resource allocation at the transmitter with outdated or predicted channel gains | **Prediction-driven OFDMA allocation exists** for channel gains in the downlink, not for Doppler in the uplink. G5's allocation part must be worded as forecast of *Doppler residuals*, not prediction in general |
| Johnson, Freitag, Stojanovic 1997; Eynard, Laot 2008; Gomes et al. 2008 | "Doppler pre-compensation" = coarse receiver-side correction | Terminology only |

Previously known and still valid: Daoud et al. 2015 (multiple resampling, multiple-access channel); Li et al. 2020;
Yang et al. 2018; Zhang et al. 2026; Hasan et al. 2025; Oiwa et al. 2020; 3GPP NTN.

## Consequences for the plan
1. **Reword G5:** "No UWA transmitter pre-compensates its own Doppler; per-user Doppler in multi-user links is
   separated at the receiver, with one resampling branch per user (Tu et al. 2011, 2013), and transmitter-side
   allocation uses predicted channel gains, not Doppler."
2. **Block 4 claim:** replace "no receiver undoes the interference" with "receiver-side separation needs one
   resampling branch per user and leaves inter-user interference; pre-compensation at the source removes the bulk
   part before it arises". The comparison against the multiple-resampling receiver of Tu et al. becomes a required
   baseline, both for interference and for receiver complexity (K branches vs one FFT).
3. **R4** should compare the pre-compensated uplink with the multiple-resampling receiver in the interleaved and
   contiguous allocations used by Tu et al. 2011.
4. **Framing for the editors:** two guest-editor groups have worked on this problem from the receiver side (Tu et
   al., Stojanovic) and on stale transmitter knowledge (Cuji Dután, Stojanovic). Present Block 4 as the
   transmitter-side complement made possible by the forecast, and cite both.

## Not covered
- Google Scholar cited-by lists (blocked by the CAPTCHA); OpenAlex citation coverage is good but not complete.
- IEEE Xplore full-text search (needs institutional access).
- Full text of Tu et al. 2011 and of the Cuji Dután thesis; whether either discusses transmit-side pre-compensation
  as an option or future work is still unknown. It is worth reading both before submission.

## Sources
- Web: [IEEE 6120227](https://ieeexplore.ieee.org/document/6120227/); [Bilkent repository, Tu et al. JOE 2013](https://repository.bilkent.edu.tr/items/987f5cda-c489-4c3c-962c-e0f0b7ec9f56); [Northeastern repository, Cuji Dután 2025](https://repository.library.northeastern.edu/files/neu:ms39s8036); [USPTO 7844006](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7844006); [Ozyegin repository, multi-resampling in cooperative OFDM](https://eresearch.ozyegin.edu.tr/handle/10679/2454); [MU-CSS for AUV fleets, PMC7085585](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7085585/); [NWPU, Doppler compensation under acceleration](https://pure.nwpu.edu.cn/en/publications/effective-doppler-compensation-for-underwater-acoustic-communicat).
- Google Scholar result pages for "underwater acoustic" "Doppler" "pre-compensation" and for transmitter-side
  multi-user Doppler (9 Oct 2026).
- OpenAlex API: works citing W2550131768, W1996444993, W1996205685, W4412559047.
