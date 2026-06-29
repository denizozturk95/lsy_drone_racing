"""Tunable constants (the "cockpit") for the controller_v10_1 controller.

v10.1 = v10's real-time time-optimal MPCC with a GATE-AWARE CONTOURING WEIGHT. v10 uses one
constant contouring weight, so raising the speed budget trades contour error for progress
everywhere -- including at the gate -- and the drone overshoots; that is the level2 ceiling
(lap time is capped by gate-passing precision under +/-0.15 m randomisation, not by actuator
authority). v10.1 makes the contouring weight a per-stage value that spikes in a window around
each gate's arc-position and relaxes on the straights, so the drone hugs the line exactly where
a gate must be threaded but stays free between gates. That precision-where-it-matters is what
lets v10.1 carry a HIGHER speed budget (V_MAX, A_LAT_MAX, V_THETA_MAX) than v10 at the same
finish rate.

(An earlier v10.1 attempt added a thrust-vector slew limit by making acceleration a state and
jerk the control. It was abandoned: the plant takes acceleration as a zero-order-hold command,
so a jerk-as-state model mismatches it and tracked WORSE than v10 -- and v10's plans turned out
to already be well-tracked in sim, so attitude lag was not the real limiter. The real limiter
is gate precision, which this gate-aware weight targets directly.)

Everything else is v10: the time-optimal -mu*vth progress reward, the friction-circle curvature
cap, the planner, and the takeoff. REAL-TIME via acados SQP-RTI -- run under ``pixi run``; the C
solver is generated/compiled once per process.
"""

from __future__ import annotations

# --- Horizon (look-ahead) ---
HORIZON = 18        # number of prediction steps
STEP_DT = 0.05      # s, prediction step (HORIZON * STEP_DT = look-ahead, ~0.9 s)

# --- Gate-aware contouring weight (the v10.1 mechanism) ---
# The contouring weight at progress theta is W_CONTOUR_BASE + W_CONTOUR_GATE * sum_g
# exp(-(s(theta)-s_gate)^2 / (2*GATE_SIGMA^2)). BASE is held at v10's uniform 20 -- a 20-run sweep
# showed dropping it below 20 tracks the gate APPROACHES looser than v10 and HURTS finish (the
# approach is where precision builds up), so the gate weight is stacked ON TOP of 20, never below.
# GATE is the extra weight peaking at each gate (BASE+GATE ~ 40 there) so the drone threads it
# precisely at the raised speed; SIGMA is the half-width (m) of that window -- ~0.5 m brackets a
# gate's approach+exit without spanning the short gate-2/3 link. In the sweep, GATE 20 beat 22+
# (more weight over-brakes through gates and drops finish). Tune GATE up only if gates are still
# overshot; otherwise leave it.
W_CONTOUR_BASE = 20.0   # baseline contouring weight (= v10's; do NOT drop below this)
W_CONTOUR_GATE = 20.0   # extra contouring weight peak at each gate (BASE+GATE ~ 40 at the gate)
GATE_SIGMA = 0.5        # m, half-width of the gate weighting window

# --- Actuator limits ---
# Raised above v10's 3.0: gate-aware contouring lets >3.0 hold the gates. A 20-run sweep picked
# 3.2 as the fastest point whose finish stays within run-to-run noise of v10 (16/20 vs 17/20)
# while cutting the average lap ~0.2 s and the best lap ~0.4 s. 3.3+ trades finish for little
# extra speed. v10-SAFE FALLBACK: 3.0. SPEED-LEANING: 3.3 (accepts ~70-75% finish).
V_MAX = 3.2         # m/s, hard velocity cap (the straight-line top speed)
TILT_RATIO = 0.85   # tan(max tilt), ~40 deg (the v9.x cornering-authority optimum)
A_Z_MIN = 7.0       # m/s^2, minimum vertical thrust acceleration

# --- Progress (the time-optimality driver) ---
MU = 1.5            # progress reward weight (drives speed up to the curvature cap)
V_THETA_MAX = 3.2   # m/s, max progress rate on straights (= top of the curvature profile = V_MAX)
A_THETA_MAX = 8.0   # m/s^2, max progress acceleration
R_DV = 0.01         # progress-acceleration regulariser

# --- Curvature speed limit (friction circle; caps progress rate AND actual speed per stage) ---
# v_curv(theta) = clip(sqrt(A_LAT_MAX / curvature), V_MIN, V_THETA_MAX). v10 sat at 8.0; with the
# gate-aware weight holding the line through the gate-2/3 reversal, 8.5 held the gate in the sweep
# (paired with V_MAX 3.2). 9.0+ started dropping finish. v10-SAFE FALLBACK: 8.0.
A_LAT_MAX = 8.5     # m/s^2, lateral-acceleration budget that sets corner speed
V_MIN = 1.3         # m/s, floor on the curvature-limited speed (creep through hairpins)

# --- Contouring lag / smoothness weights ---
W_LAG = 1.0         # keeps the progress state synced to the foot-point
W_ACCEL = 0.02      # command smoothness regulariser

# --- Navigation-start ramp (gentler, as the faster v10.1 needs) ---
# The higher budget re-exposes the gate-0 hand-off transient, so ease the progress-rate cap in
# more slowly than v10. Start at v10's 0.08 / 2.0; if gate-0 crashes appear, lengthen RAMP_S.
RAMP_START = 0.08
RAMP_S = 2.0

# --- Solver ---
MAX_ITER = 20       # HPIPM QP iteration cap per control step
GRAVITY = 9.81
