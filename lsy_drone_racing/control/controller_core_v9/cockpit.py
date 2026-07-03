"""Tunable constants (the "cockpit") for the controller_v9 (MPCC) controller.

v9 reuses controller v8's gate-aware global planner (as the path geometry) and vertical takeoff,
and replaces v8's hand-tuned speed caps and cascaded-PID tracker with a model-predictive
contouring controller (MPCC, see controller_core_v9.mpcc). Speed isn't a constant here: the MPCC
chases a reference receding along the path at V_REF and goes as fast as the thrust, tilt,
and velocity limits allow, so it adapts to whatever geometry the planner hands it. The
planner and takeoff tuning is inherited from v8 (edit v8's cockpit to reshape the path);
this file holds only the MPCC knobs.
"""

from __future__ import annotations

# --- Horizon ---
HORIZON = 14        # number of prediction steps
STEP_DT = 0.05      # s, prediction step (HORIZON * STEP_DT = look-ahead, ~0.7 s)

# --- Speed (comes out of the limits, not a cruise constant) ---
# V_REF is how fast the reference recedes along the path; the drone chases it and is capped
# by the actuator limits below, so it flies as fast as is dynamically feasible. At 1.8 the
# MPCC already laps both tracks ~40% faster than v8's tuned PID; pushing it higher trades
# finish rate for time, since the point-mass model gets less forgiving.
# V_REF = 1.8         # m/s, reference recede speed (set above what the drone can sustain)
# V_MAX = 2.4         # m/s, hard velocity cap for safety

# -----results -----
# INFO:__main__:Average Time: successful_times_avg (s): 9.547912087912088
# INFO:__main__:Success Rate: 91.0%

V_REF = 2
V_MAX = 3

# -----results -----
# INFO:__main__:Average Time: successful_times_avg (s): 8.661587301587302
# INFO:__main__:Success Rate: 63.0%

# --- Actuator limits (these, not tuning, set the achievable speed) ---
# TILT_RATIO caps |horizontal thrust| / |vertical thrust| = tan(max tilt). 0.55 is about
# 29 deg, well below the 80-90 deg saturation that killed v8's PID in climb-turns. A_Z_MIN
# bounds how far the optimiser may cut vertical thrust to descend: too low lets it sag
# toward the floor on the first-gate approach, too high (above ~9) starves real descents.
TILT_RATIO = 0.55   # tan(max tilt), about 29 deg
A_Z_MIN = 7.0       # m/s^2, minimum vertical thrust acceleration

# --- Navigation-start ramp (settle onto the path before accelerating) ---
# After the takeoff handoff the drone is slightly off the path (the climb settles with a
# brief transient). Without a ramp the MPCC charges from that perturbed state straight at
# the first gate and can sag toward the floor. Instead the reference recede speed ramps
# from V_REF*RAMP_START up to full V_REF over RAMP_S seconds, so it settles first.
RAMP_S = 1.1
RAMP_START = 0.2

# --- Contouring weights ---
# W_CONTOUR (stay on the path laterally) is much larger than W_LAG (catch the receding
# reference, which is what drives forward speed). W_ACCEL regularises the command for
# smoothness.
W_CONTOUR = 14.0
W_LAG = 1.0
W_ACCEL = 0.02

# --- Solver ---
MAX_ITER = 40       # ipopt iteration cap per control step
GRAVITY = 9.81
