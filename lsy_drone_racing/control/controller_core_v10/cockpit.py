"""Tunable constants (the "cockpit") for the controller_v10 controller.

v10 keeps v9's gate-aware planner and vertical takeoff but replaces v9/v9.1's
fixed-recede-rate contouring MPCC with a *time-optimal* MPCC: the path progress (arc length)
and its rate are decision variables, and the cost pays the optimiser to make progress
(``-MU * v_theta``), so traversal speed is an OUTPUT of the optimisation. v9.1's external
projection and 5-knob stall governor are gone -- progress-as-a-state cannot freeze.

The progress rate is bounded by a friction-circle CURVATURE LIMIT (v_theta <= v_curv(theta),
reusing controller_core_v9_1.speed_profile): the optimiser maximises progress only up to what each turn
can hold, so it brakes the sharp gate-2/3 reversal instead of cutting it. The takeoff->gate-0
hand-off uses the gentler ramp v9.2 found necessary at these speeds.

REAL-TIME via acados SQP-RTI (see controller_core_v10.mpcc). It is solved with one real-time iteration
(one QP) per step, and the reference is LINEARISED out of the solver each step (the path
point/tangent/curvature-speed at the predicted progress are passed as per-stage parameters,
computed in numpy), so a solve costs a few ms -- within the 20 ms / 50 Hz budget. (An earlier
IPOPT version that embedded the spline cost ~1.3 s/solve and was not real-time.) Requires the
acados environment: run under ``pixi run``; the C solver is generated/compiled once per process.

Only the MPCC knobs live here; the planner/takeoff tuning is inherited from v8 via v9.
"""

from __future__ import annotations

# --- Horizon (look-ahead) ---
# ~0.9 s of look-ahead; v10 needs this much to anticipate the gates. With acados SQP-RTI the
# per-step cost is roughly linear in the horizon and still a few ms, so 18 is affordable.
HORIZON = 18        # number of prediction steps
STEP_DT = 0.05      # s, prediction step (HORIZON * STEP_DT = look-ahead, ~0.9 s)

# --- Actuator limits ---
# 3.0 is the reliable straight-line top speed (the curvature cap brakes the corners). Pushing it
# to 3.5+ is a few % quicker but re-exposes the gate-0 takeoff transient (finish ~94% -> ~80%
# over 20 runs), so 3.0 is the reliability-first default.
V_MAX = 3.0         # m/s, hard velocity cap (the straight-line top speed)
TILT_RATIO = 0.85   # tan(max tilt), ~40 deg (the v9.x cornering-authority optimum)
A_Z_MIN = 7.0       # m/s^2, minimum vertical thrust acceleration

# --- Progress (the time-optimality driver) ---
# MU rewards advancing along the path. Because v_theta is now hard-capped by the curvature
# limit below, MU no longer trades against corner-cutting -- it just has to be strong enough to
# push v_theta up to that cap, so it can be large. V_THETA_MAX is the straight-line cap (= the
# top of the curvature profile); A_THETA_MAX caps progress acceleration; R_DV smooths it.
MU = 1.5            # progress reward weight (drives speed up to the curvature cap)
V_THETA_MAX = 3.0   # m/s, max progress rate on straights (top of the curvature profile)
A_THETA_MAX = 8.0   # m/s^2, max progress acceleration
R_DV = 0.01         # progress-acceleration regulariser

# --- Curvature speed limit (friction circle; caps progress rate AND actual speed per stage) ---
# v_curv(theta) = clip(sqrt(A_LAT_MAX / curvature), V_MIN, V_THETA_MAX), evaluated per stage at
# the predicted progress and passed as a parameter. It caps both v_theta and |vel| (a soft
# constraint), so the optimiser brakes for turns. A_LAT_MAX ~ tilt_ratio*g (~8 at tilt 0.85) is
# the physical ceiling; V_MIN keeps it creeping at hairpins. Same values as v9.2's profile.
A_LAT_MAX = 8.0     # m/s^2, lateral-acceleration budget that sets corner speed
V_MIN = 1.3         # m/s, floor on the curvature-limited speed

# --- Contouring weights ---
# W_CONTOUR keeps the drone on the path (gate centring); with the curvature bound holding the
# corner speed down it no longer has to fight overspeed, so a moderate value tracks gates.
# W_LAG keeps the progress state synced to the foot-point. W_ACCEL smooths the command.
W_CONTOUR = 20.0
W_LAG = 1.0
W_ACCEL = 0.02

# --- Navigation-start ramp (gentler than v9.1's, as the faster v10 needs) ---
# Ease the progress-rate cap in from RAMP_START over RAMP_S after the takeoff hand-off, or the
# fast reference lunges at gate 0 (~2 s crash). A sweep showed v10 needs MORE easing than v9.2
# (0.12/1.4 left ~70% finish from gate-0 transients); 0.08 / 2.0 s took it to 100% at full speed.
RAMP_START = 0.08
RAMP_S = 2.0

# --- Solver ---
# acados SQP-RTI does ONE QP per control step; MAX_ITER caps the inner HPIPM QP iterations.
# ~20 is plenty for the warm-started QP and keeps the per-step cost a few ms (see mpcc.py).
MAX_ITER = 20       # HPIPM QP iteration cap per control step
GRAVITY = 9.81
