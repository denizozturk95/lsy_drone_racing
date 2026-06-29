"""Tunable constants (the "cockpit") for the controller_v9.1 controller.

v9.1 owns its MPCC knobs (it no longer inherits them from v9) so it can be tuned without
touching v9, plus the progress governor that fixes the stall. Reference progress is tracked
in spline curve-time (the plan is time-parameterized at roughly cruise speed, so 1 s of
curve-time is about 1 s of nominal flight).

The MPCC values below were chosen by a parameter sweep over level2 (speed) and
stress_synth_01 (the loop/sharp-turn track where v9 stalled). Versus v9's defaults
(horizon 14, w_lag 1.0, tilt 0.55) this config is about 6% faster on level2 with no real
finish-rate cost, and is actually more robust on stress_synth_01 (finish 85% -> 92%),
because the longer look-ahead and extra tilt let it carry the higher forward drive (w_lag)
through tight geometry instead of overshooting. Raising v_ref turned out to hurt: it just
makes the point-mass model overshoot the gates, so speed is bought through w_lag and
look-ahead, not v_ref.
"""

from __future__ import annotations

# --- Horizon (look-ahead) ---
# Bumped from v9's 14 to 18 (~0.9 s look-ahead). More anticipation lets the MPCC carry
# speed through turns; past ~20 the per-step solve cost grows for little gain.
HORIZON = 18        # number of prediction steps
STEP_DT = 0.05      # s, prediction step (HORIZON * STEP_DT = look-ahead, ~0.9 s)

# --- Speed (comes out of the limits and the lag pull, not a cruise constant) ---
# V_REF is how fast the reference recedes along the path. Sweeps showed raising it just
# causes gate overshoot (finish rate collapses), so it stays at v9's safe value; the actual
# speed-up comes from W_LAG below, which lets the drone close on the reference faster (up
# toward V_MAX) without moving the reference faster.
V_REF = 1.8         # m/s, reference recede speed
V_MAX = 2.4         # m/s, hard velocity cap for safety

# --- Actuator limits ---
# TILT_RATIO caps |horizontal thrust| / |vertical thrust| = tan(max tilt). Raised from
# v9's 0.55 (~29 deg) to 0.65 (~33 deg): at v9's tilt the drone could not corner at the
# higher drive and fell off the path, so the extra authority is what keeps the finish rate
# up while going faster. A_Z_MIN bounds how far vertical thrust may be cut to descend.
TILT_RATIO = 0.65   # tan(max tilt), about 33 deg
A_Z_MIN = 7.0       # m/s^2, minimum vertical thrust acceleration

# --- Navigation-start ramp (settle onto the path before accelerating) ---
RAMP_S = 1.1        # s, ramp the reference recede speed up over this long after takeoff
RAMP_START = 0.2    # fraction of V_REF the reference recedes at right after the handoff

# --- Contouring weights ---
# W_CONTOUR (stay on the path laterally) is much larger than W_LAG (catch the receding
# reference, which drives forward speed). W_LAG raised from v9's 1.0 to 1.5: this is the
# main speed lever, it makes the drone close the lag harder and fly faster up toward V_MAX.
# Higher (2.0+) is faster still but starts overshooting gates, so 1.5 is the robust knee.
# W_ACCEL regularises the command for smoothness.
W_CONTOUR = 14.0
W_LAG = 1.5
W_ACCEL = 0.02

# --- Solver ---
MAX_ITER = 40       # ipopt iteration cap per control step
GRAVITY = 9.81

# --- Progress governor (the v9.1 stall fix) ---
# In normal flight the geometric projection advances faster than MIN_PROGRESS_RATE, so the
# governor is a no-op and behaviour matches v9. It only bites when the projection freezes:
# then progress creeps forward at MIN_PROGRESS_RATE until the reference leads the projection
# by MAX_LEAD_T, which grows the lag error and pulls the drone out of the stall. The lead is
# capped so a truly blocked drone (against a pole) gets a steady bounded pull, not a runaway
# reference.
MIN_PROGRESS_RATE = 0.5   # curve-s advanced per real second while the projection is frozen
MAX_LEAD_T = 0.30         # s, max curve-time the reference may lead the projection

# Stall watchdog: if the drone stays slower than V_STALL for T_STALL, treat it as stuck and
# scale the creep and lead by STALL_BOOST so the pull is strong enough to escape.
V_STALL = 0.25            # m/s, below this the drone counts as not moving
T_STALL = 0.20            # s, stalled once it's been slow for this long
STALL_BOOST = 2.0         # creep/lead multiplier while stalled
