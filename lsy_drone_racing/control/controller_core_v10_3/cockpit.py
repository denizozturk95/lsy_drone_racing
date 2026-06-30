"""Tunable constants (the "cockpit") for the controller_v10_3 controller.

v10.3 = v10.1 + REPLAN CONTINUITY (the shipped mechanism, no knob -- see controller_core_v10_3.mpcc) plus
an optional GATE-WINDOW SPEED CAP (OFF by default on level2, see below). Every other knob is
inherited unchanged from v10.1 (see controller_core_v10_1.cockpit), including the speed budget: the win
over v10.1 comes entirely from not cold-starting the solver at every replan, not from speed.

Measured on level2, paired 20-run track sequences (scripts/compare_v10_3.py):
  seed 42: v10.1 14/20 at 8.30 s  ->  v10.3 (this config) 19/20 at 8.10 s
  3 seeds (42/7/123) combined: v10.1 49/60 at 8.22 s -> v10.3 54/60 at 8.20 s

AGGRESSION PROBE LEDGER (all on seed 42 vs this config's 19/20 at 8.098 s; every variant
regressed -- this config is the speed/robustness Pareto knee, do not re-derive):
  V 3.3 uniform ................................. 16/20  8.13  (gate-0 + gate-2 fails)
  A_LAT 9.0 ..................................... 16/20  8.20  (gate-2 reversal fails return)
  A_LAT 9.0 + V 3.4 + Gaussian cap 2.9/0.35 ..... 17/20  8.03  (gate-3 exit overdriven)
  V 3.4 + Gaussian cap 2.9/0.35 ................. 14/20  7.89  (4x gate-0: hotter ramp-in)
  V 3.4 + Gaussian cap 2.6/0.45 ................. 18/20  8.29  (cap too broad, slow)
  V 3.4 + asym cap 2.8 pre0.5/post0.1 + ramp 2.2  15/20  8.29  (scattered fails)
  ramp 0.12/1.4 (v9.2's, at V 3.2) .............. 13/20  7.93  (5x gate-0: hand-off is a
                                                  dynamic transient, not a cold-solver issue)
  A_THETA 10 .................................... 19/20  8.098 (byte-identical: bound inactive)
The pattern: ANY knob that raises gate-crossing or hand-off speed trades finish 1-for-1
against the +/-0.15 m reveal-correction ceiling; the dvth bound is the only orthogonal knob
and it is already inactive.

About the speed cap: the curvature profile only brakes where the PATH bends, so a
straight-approach gate is crossed at full V_MAX while the true gate pose (revealed at the
0.7 m sensor range, up to ~0.21 m off nominal) still has to be threaded -- the correction
demand scales ~v^2. Capping speed to V_GATE on the approach side of each gate buys that
margin deterministically. On level2 it costs more than it pays (gates too closely spaced);
keep it for long-straight tracks or real-flight margin.
"""

from __future__ import annotations

# Re-export every v10.1 knob unchanged (HORIZON, contouring weights, MU, solver, ramp...).
from lsy_drone_racing.control.controller_core_v10_1.cockpit import *  # noqa: F401,F403

# --- Gate-window speed cap (available, OFF by default on level2) ---
# v(s) = min(v_curv(s), V_GATE) inside [s_gate - GATE_V_PRE, s_gate + GATE_V_POST], then a
# backward/forward longitudinal pass keeps the braking into / acceleration out of each window
# dynamically feasible (the entry ramp forms upstream automatically). The window is approach-
# heavy because the reveal correction must be absorbed BEFORE the gate plane. Set
# V_GATE >= V_MAX to disable (recovers v10.1's profile exactly). Every cap variant measured
# on level2 regressed (see the probe ledger above) -- the gates are so closely spaced that
# the cap costs more than a raised straight budget recovers. Keep it OFF here; it is the
# knob to reach for on tracks with long straights or for real-flight margin, not on level2.
V_GATE = 999.0      # m/s, speed cap inside the gate window; >= V_MAX disables the cap
GATE_V_PRE = 0.5    # m, cap window upstream of the gate plane (covers the 0.7 m reveal run)
GATE_V_POST = 0.1   # m, cap window past the gate plane (exit takes speed back immediately)

# --- Speed budget (= v10.1's; every raise traded finish 1-for-1, see module docstring) ---
V_MAX = 3.2         # m/s, hard velocity cap (straight-line top speed)
V_THETA_MAX = 3.2   # m/s, max progress rate on straights (= top of the curvature profile)

# --- Navigation-start ramp (v10.1's -- BOTH directions were probed and regress) ---
# Hotter (v9.2's 0.12/1.4): 13/20 at 7.93 with FIVE gate-0 crashes -- the hand-off transient
# is a genuine dynamic problem (hover -> acceleration onto the plan), not a cold-solver
# artifact, so the warm rebase does not buy ramp headroom. Gentler (0.08/2.4): 18/20 at
# 8.31 -- slower and the marginal cases just moved. 0.08/2.0 is the knee; do not re-derive.
RAMP_START = 0.08
RAMP_S = 2.0

# --- Progress-rate recovery (= v10.1's 8.0) ---
# 10.0 was probed and produced byte-identical results (the dvth bound is never active);
# keep v10.1's value so the cached solver build is shared.
A_THETA_MAX = 8.0

# --- Corner speed budget (= v10.1's 8.5) ---
# 9.0 was re-probed on the warm (rebased) solver and STILL regressed: alone 16/20 at 8.20
# (gate-2 reversal fails return); with the narrow gate cap + V 3.4 it ran 17/20 at 8.03
# (fails moved to gate 3 -- the reversal exit overdriven). Corner authority stays at 8.5.
A_LAT_MAX = 8.5
