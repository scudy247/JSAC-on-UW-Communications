# Study guide: where to start

Reading order for someone new to the project (12 Oct 2026). Paths are relative to this folder.

## Documents
| File | What it is for |
|---|---|
| [`README.md`](README.md) | The idea, candidate claim, literature gaps, evidence and status: the one-page overview |
| [`project_idea/doppler_estimation.pdf`](project_idea/doppler_estimation.pdf) | Group-discussion deck: the same story with figures |
| [`code/PLAN.md`](code/PLAN.md) | The phases (0-8): the question each answers, which code does what, status |
| [`code/RESULTS.md`](code/RESULTS.md) | Chronological log of every result and correction; for the current work read from "End-to-end chain" onward |
| [`literature/estimation_pivot_local_check.md`](literature/estimation_pivot_local_check.md) | Gap check for E1/E2, notes on Lee et al. 2026 (the main counter-evidence), WUWNet'18 check |

## Code (`code/src/uwofdm/`), in the order of the simulation chain
| File | What it is for |
|---|---|
| `motion.py`, `buoy.py` | Real buoy records (CDIP, OOI) -> node velocities -> link Doppler (chain stages 1-2) |
| `waveform.py`, `frame.py` | OFDM numerology, wideband Doppler warp / unwarp, packet structure (stage 3) |
| `channel.py`, `multipath.py`, `bellhop.py` | Single-path and Bellhop multipath channels with per-path Doppler (stage 4) |
| `imu.py`, `sideinfo.py`, `header.py` | Motion-sensor error models; the transmitter's side information and its header (stage 5, E1) |
| `chain.py` (with `estimators.py`, `fusion.py`, `pfft.py`, `ici.py`) | The core: every receiver (Li/Lee, Cho-Kim, signal Kalman, pilot tracking, partial FFT, ICI-aware, sensor fusion, multipath cancellation) (stage 6, E2) |
| `receiver.py`, `coding.py` | Sparse channel estimate, soft bits, Viterbi decoding (stage 7) |
| `theory.py`, `placement.py`, `bcrb.py` | Closed forms: estimation error -> interference (T1), placement evaluator (E3), Bayesian Cramer-Rao bound (T3) |

## Running things
| File | What it is for |
|---|---|
| `code/scripts/e2_chain.py`, `code/scripts/b2_baselines.py` | Templates of a full experiment and of the current fair comparison (one script per analysis or figure) |
| `code/tests/` | Small runnable examples of every block (`pytest`) |
| `code/run_job.sh` | How experiments are launched on the shared machine: tmux session `doppler`, nice 19, single thread, max 4 jobs, logs in `code/results/logs/` |
