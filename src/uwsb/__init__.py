"""uwsb — staleness-aware bandits for UWA link adaptation.

Design source of truth: ../../THEORY.md (math) and ../../project.MD (conventions).
Locked decisions used throughout this package:
  D1  R(dt) = exp(-|dt|/Tc), Tc defined by R(Tc)=1/e (Tc = kernel time constant).
  D2  tau := round-trip; Kalman predicts FORWARD to channel-time t + tau_ow.
"""

__version__ = "0.1.0"
