"""Tunable constants (the "cockpit") for the controller_v10_4 controller.

v10.4 = v10.3 + a fast LAUNCH (mini-takeoff, hot launch ramp, honest cold start, gate-0
approach cap). Every racing knob -- speed budget, contouring weights, corner authority -- is
inherited unchanged from v10.3. The profiled start phase cost ~2.6 s to gate 0 (PID climb
0.54 s + ramp-throttled run 2.08 s) against a curvature-limited floor of ~1.45 s; the launch
knobs close most of that gap, worth ~0.4-0.5 s of lap time.

SPEED/ROBUSTNESS FRONTIER (paired-seed evals, 3 x 20 runs on seeds 42/7/123 vs the SAME
tracks for every config; v10.3 baseline 54/60 at 8.20 s). ~20 probed configurations collapse
onto one line -- every knob that raises any gate's REVEAL-window speed trades finish for
time at a steep rate:
  v10.4 SHIP (this file) ............... 52/60 at 8.01           (finish within paired noise
                                          of v10.3, -0.19 s, launch crashes 2/60)
  hot ramp (0.30, 1.6) + g0 cap ........ 48/60 at 7.80           (speed-leaning; the gate-0
                                          reveal corridor fails pile up at 1.5-1.9 s)
  + clearance trim + horizon 24 ........ 45/60 at 7.50, best 6.7  (max-speed: CLR_EXT_MIN=0,
                                          HORIZON=24, RAMP 0.30/1.6 -- gate-3 U-turn fails)
  full trims + hot everything .......... 13-15/20 range           (condemned)
CONSISTENT SUB-7 IS NOT REACHABLE on level2 under +/-0.15 m gate randomisation at 0.7 m
sensor range with this architecture: the reveal correction's ~v^2 ceiling binds at every
gate, and EIGHT distinct reconciliation mechanisms (global/static/reactive gate caps, wider
obstacle keep-out, wider contour window, longer horizon, ramp shapes, hand-off altitudes)
all landed ON the frontier, not above it. The gate-0 corridor specifically tolerates only
~2.4 m/s however it is approached (cap/altitude/keep-out variants at hot ramp: 44-50/60).
Sub-7 needs the reveal constraint changed (different sensing or trajectory law) -- a v11,
not a cockpit value.

IMPORTANT: v10.3's ramp ledger ("hotter ramps regress") does NOT transfer here. Those probes
were measured against the 0.5 m near-hover hand-off transient, which v10.4 removes (hand-off
at ~0.3 s with upward momentum along the plan tangent + an honest cold-start linearisation).
"""

from __future__ import annotations

# Re-export every v10.3 knob unchanged (budget 3.2/8.5, gate cap off, contouring, solver...).
from lsy_drone_racing.control.controller_core_v10_3.cockpit import *  # noqa: F401,F403

# --- Mini-takeoff (XY-held vertical climb; the env's floor-touch carve-out guard) ---
# The climb exists to hold the start XY inside the +/-0.02 m carve-out while the rotors spin up
# (~40 ms of free-fall every episode start). Both 0.22 m and 0.42 m hand-offs were flown; at
# the shipped (slow) ramp both are fine, and at hot ramps NEITHER survives the gate-0 reveal
# corridor -- so altitude is not the gate-0 lever (the ramp is). 0.42 m / 0.55 s ships: it
# stays close to v8's proven geometry while still handing off ~0.15 s earlier than v8's
# 0.54 s. Peak vertical accel 6*h/T^2 ~ 8 m/s^2 keeps worst-case thrust under the 0.8 N cap.
TAKEOFF_ALT = 0.42        # m, climb target before the MPCC takes over (v8: 0.5)
TAKEOFF_CLIMB_TIME = 0.55  # s, fixed climb-spline duration (replaces v8's speed/0.6 s floor)

# --- Launch ramp (replaces v10.3's hand-off ramp 0.08/2.0) ---
# vth cap = V_THETA_MAX * (RAMP_START + (1-RAMP_START) * t/RAMP_S), anchored at the hand-off.
# SHIPPED at (0.25, 2.4): the hot ladder was condemned by exhaustive paired measurement.
# (0.30, 1.6) at any takeoff altitude (0.22/0.42), any gate-0 cap width (0.6/1.4 m), and any
# obstacle keep-out (0.20/0.26) lands at 44-50/60 with the failures piled in the gate-0
# reveal corridor at 1.5-1.9 s -- the corridor tolerates ~2.4 m/s under +/-0.15 m
# randomisation and a hot ramp simply arrives above it. (0.25, 2.4) measured 52/60 at 8.06
# before the gate-0 cap; the cap (below) targets its two residual gate-0 fails.
RAMP_START = 0.25
RAMP_S = 2.4

# --- Horizon (CANDIDATE B: v10.x's 18; 24 measured -0.14 s but commits harder at gates) ---
HORIZON = 18

# --- Gate speed caps ---
# PERMANENT global cap: OFF -- condemned at both paces (v10.3 ledger; 15/20 at the v10.4
# pace): it taxes every gate on every lap, costing more than it saves.
V_GATE = 999.0       # global cap (all gates): disabled
GATE_V_PRE = 0.5     # (unused while the global cap is disabled)
GATE_V_POST = 0.15   # (unused while the global cap is disabled)
# REACTIVE per-gate cap: ON. The reveal correction is only expensive when the gate actually
# MOVED, and the controller knows each gate's revealed delta the moment it replans. Gates
# whose revealed pose differs from nominal by more than REACT_DELTA_M get an approach-window
# cap of V_GATE_REACT; the rest fly the full curvature profile. Lucky draws keep full pace;
# unlucky ones brake for the one gate that needs it. (12 of 23 paired-eval failures across
# the launch configs were reveal corrections; a gate-0-only static version of this cut seed
# 123's launch crashes 5 -> 2 at zero time cost.)
V_GATE_REACT = 2.5   # m/s, approach cap: ALWAYS applied to gate 0 while it is the target
                     # (launch-window protection: 16/20 vs 14/20 on the hostile seed at zero
                     # time cost -- it also slows the obstacle-0 passage, whose reveal swerve
                     # was the actual pre-2 s killer), and to any other gate whose revealed
                     # pose moved more than REACT_DELTA_M.
REACT_DELTA_M = 9.9  # m, reactive flagging DISABLED (gate 0 stays always-capped). Measured:
                     # 2.4/0.10 cost -0.2 s with no finish gain; 2.5/0.12 + r_obs 0.26 killed
                     # the launch crashes but moved the fails to gates 2/3 (15/20 at 8.14).
                     # The mid-race reveal ceiling does not yield to braking-on-detection.
REACT_V_PRE = 1.4    # m, window upstream of the gate plane. WIDE on purpose: at 0.6 the cap
                     # started AT gate 0's reveal run but left the obstacle-0 passage (arc
                     # ~1.5-1.8 m, 0.21-0.26 m clearance, +/-0.15 m shift revealed at 0.7 m)
                     # at the hot ramp's ~3.1 m/s -- 7 of 12 ship-validation failures were
                     # launch-window crashes there at 1.56-1.70 s. 1.4 pins the whole launch
                     # corridor at V_GATE_REACT (v10.3's effective speed there) for ~0.1 s.
REACT_V_POST = 0.3   # m, window past the plane (re-accelerate beyond the frame, not at it)

# --- Obstacle keep-out (= v8's 0.20; 0.26 REGRESSED) ---
# 0.26 killed the launch-window obstacle swerve crashes but re-routes the late-race
# approaches (obstacle at [-0.5,-0.75] sits on the gate-3 approach): 15/20 at 8.14 on the
# hostile seed, slower AND less reliable late-race. The gate-0 cap above already slows the
# obstacle-0 passage; keep the keep-out at v8's value.
R_OBS = 0.20

# --- Gate contour window (= v10.1's 20/0.5; retuning for pace REGRESSED) ---
# 24/0.65 at the v10.4 pace measured 15/20 at 7.61 s -- wider/stronger over-brakes and
# over-constrains, the same lesson v10.1's sweep learned at 8.3 s pace. Do not re-derive.
W_CONTOUR_GATE = 20.0
GATE_SIGMA = 0.5

# --- Clearance run-out trim (the audited geometry trim, as a dial) ---
# extension = 0.60 * max(cos_to_next, CLR_EXT_MIN): 0.0 = full cos-scaling (saves ~1.05 m
# = ~0.38 s on level2), 1.0 = v8's fixed 0.60 m (trim off). The trim's measured cost is
# ~1-2 finishes per 20 on seed 42 (gate-2 fails on the resharpened g1->g2 leg) for ~0.15 s.
# CANDIDATE B: trim off while the gate-3 U-turn fails of the trimmed/h24 config are open.
CLR_EXT_MIN = 1.0
