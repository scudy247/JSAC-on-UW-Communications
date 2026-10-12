# Follow-up paper: physics-informed contextual bandit

Moved out of the JSAC paper after the internal review (8 Oct 2026); in the JSAC paper a threshold rule with an
ACK-calibrated margin takes its place (see [`README.md`](README.md), decision rule). The old deck that still has the
bandit (as its Block 4; the current Block 4 is transmitter pre-compensation) is [`project_idea/archive/doppler_link_adaptation.pdf`](project_idea/archive/doppler_link_adaptation.pdf); the
current deck keeps a one-slide summary under "Follow-up".

## Gap (G4)
Learning link adaptation ignores kinematic context and arm-dependent feedback delay. Closest work: contextual bandits
for band/relay selection (Schutz et al. 2023); Wu–Chitre–Anjangi 2025 (GP interval BER, already quantile-calibrated
bounds); AoI bandit (Busacca et al. 2025); Park et al. 2026. Delayed ACI theory covers fixed delay only
(El Halabi–Brandt 2026).

## Design
- **Context:** forecast over the candidate receive interval (d̂, σ_e), estimated SNR, range.
- **Arms:** (M, N_sym); for each arm, the pilot set 𝒫 comes from Block 2.
- **Reward:** delivered bits per unit airtime, B_a 1{success} / T_a.
- **Model:** p_fail(a, x) = g(θ_aᵀ ψ(x)), generalised linear in physical features, monotone in forecast error and
  action; θ_a initialised from the receiver calibration curves and learned online (UCB or Thompson sampling).
- **Safety layer:** the bandit plays only arms whose upper-bound PER, plus a margin m_t updated from delayed ACKs
  (outer-loop / ACI style), meets the PER target α.

## Theory (F1)
Regret of the physics-informed bandit when feedback for arm a arrives D_a ≥ h + 2τ + T_a later, so the delay depends
on the arm. Short packets return feedback sooner and can bias a naive learner towards them. Target: Õ(√T) plus a
delay term, extending fixed-delay results (Joulani et al., ICML 2013).
