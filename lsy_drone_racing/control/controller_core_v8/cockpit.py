"""Tunable constants (the "cockpit") for the controller_v8 controller.

This is the one place to tune v8, PID included. Every value here is read only through
settings.py, where it becomes a dataclass field default; nothing imports this file
directly except settings. v8 does not use the shared controller_cockpit and no longer
inherits anything from common_controller_v2.settings, so tuning v6/v7 can't affect v8 or vice
versa.

We assume the whole track (gate and obstacle positions and orientations) is known at
t=0, so we use a global planner with full look-ahead. Start conservative on speed and
sweep up from there (see the velocity block).
"""

from __future__ import annotations

import math

# --- NAVIGATE velocity constants ---
# These tracks are tight and dominated by corners and obstacles, so the peri-gate
# V_CRUISE (used within PERI_GATE_RADIUS of every gate) sets most of the lap time;
# V_CRUISE_INTER and VMAX rarely bind because the inter-gate straights are short.
# 1.7 / 2.3 was way too hot for the tracker on the high-speed first-gate approach
# (60-75 deg commanded tilt, early crashes). Raising V_CRUISE 1.2 -> 1.25 was the most
# the first-gate approach and the gate planes would take: faster on every common seed of
# both tracks with no drop in finish rate. 1.3 tips real_track's climb-turn over (loses 2
# finishes), so 1.25 is the ceiling. Pushing V_CRUISE_INTER above 1.3 helps level2's
# longer straights but crashes a real_track gate approach, so it stays conservative.
V_CRUISE = 1.25         # m/s, cruise speed near gates (peri-gate; dominates lap time here)
V_CRUISE_INTER = 1.3    # m/s, cruise speed between gates (short straights; rarely binds)
VMAX = 1.6              # m/s, peak-velocity cap

# --- Spline timing and stability tuning ---
# Short-segment time floor (s). Any segment shorter than speed * T_MIN_SEG gets floored,
# i.e. flown slower than cruise regardless of the speeds above. Lower means short
# straights (and the gate run-in/run-out) can go faster, but too low risks jerk at the
# gate plane. Move this on its own when sweeping speeds.
T_MIN_SEG = 0.4
# turn_slowdown: corners sharper than TURN_MIN_SHARPNESS (0 straight ... 1 reversal) get
# their duration stretched by (1 + TURN_SLOW_GAIN * sharpness). This is the lever for
# settling the high-speed gate-entry corner without giving back straight-line speed:
# lower TURN_MIN_SHARPNESS to slow more corners, raise TURN_SLOW_GAIN to slow the
# qualifying corners harder.
# Bumped from v6's 0.8 to 1.0: a steep post-gate climb-turn (>0.5 m climb onto the next
# gate over a short run) otherwise pushed commanded tilt to ~56 deg with collective thrust
# saturated, clipping the frame we'd just passed. 1.0 stretches that climb-turn enough to
# stay within thrust authority, and it recovers the one real_track seed v8 had lost vs v6
# with no real cost to the gentle gate approaches.
TURN_MIN_SHARPNESS = 0.25
TURN_SLOW_GAIN = 1.0
# Tracker authority. LATERAL_ACCEL_LIMIT caps the reference lateral accel we feed forward;
# at these speeds it isn't what binds (the saturating climb-turn tilt comes from
# position-error feedback, not feedforward), so leave it at 8.0. FEEDFORWARD_SCALE blends
# model feedforward into the command: bumped 0.6 -> 0.8 so the tracker anticipates the
# climb-turn acceleration instead of lagging it. That lag was the root cause of the
# gate-2 -> gate-3 crashes (the drone fell behind the climbing+turning spline, position
# error built up, and commanded tilt saturated at 80-90 deg). 0.8 lifts the finish rate on
# both tracks (real_track 0.90 -> 0.95, level2 0.70 -> 0.75) at no time cost; 0.9-1.0
# over-feedforwards and destabilises the first-gate approach, so 0.8 is the sweet spot.
LATERAL_ACCEL_LIMIT = 8.0
FEEDFORWARD_SCALE = 0.8
# Reference look-ahead time (s). Smaller means the drone tracks the spline point closer to
# its current progress instead of aiming ahead, which cuts down corner-cutting (and the
# body tilt that clips a frame) on the curved gate approach.
LOOKAHEAD_S = 0.20

# --- PID / feedback profile (v8-owned; moved here from common_controller_v2) ---
# The cascaded position/velocity controller's gains and clamps, per world axis [x,y,z].
# These used to be inherited from common_controller_v2.settings, which was a trap: tuning them for
# v8 would silently change v6. They live here now so this cockpit is the one tuning
# surface for v8. Raise the gains/clamps to give the tracker more authority when sweeping
# speed up; too high amplifies tracking noise and can oscillate.
KP = (0.60, 0.60, 1.65)             # outer-loop proportional gain (legacy PID table)
KI = (0.05, 0.05, 0.05)             # outer-loop integral gain
KD = (0.35, 0.35, 0.50)             # inner-loop derivative gain
OUTER_I_LIMIT = (1.5, 1.5, 0.4)     # outer-loop integral clamp (anti-windup)
OUTER_CLAMP = (2.4, 2.35, 1.8)      # outer-loop velocity-request clamp
INNER_I_LIMIT = (0.75, 0.75, 0.45)  # inner-loop integral clamp
OUTPUT_CLAMP = (3.2, 3.2, 4.2)      # final force-like command clamp (z higher for lift authority)
DERIVATIVE_TAU = (0.05, 0.05, 0.06)  # inner-loop derivative low-pass time constant (s)
FEEDBACK_EPS = 1e-9                 # numerical floor, not a tuning knob

# --- Gate-approach geometry ---
# d_pre / d_post set the length of the straight run-in / run-out aligned with the gate
# normal; a longer run-in cuts down corner-cutting at the gate plane. d_stop extends the
# final run-out past the last gate. r_obs is the obstacle keep-out radius used by the
# planner's 2-D avoidance and the spline-repair pass.
D_PRE = 0.40
D_POST = 0.30
D_STOP = 0.30
R_OBS = 0.20

# --- Planner internals (v8-owned; surfaced from PlannerSettings) ---
# liftoff_* govern the cold-start ramp off the ground; cold_start_min_seg floors the
# duration of the first two segments from rest. peri_gate_radius is the XY radius around a
# gate within which the slower peri-gate V_CRUISE applies instead of V_CRUISE_INTER;
# shrink it to let more of the track run at the faster inter-gate speed. clearance_* is
# the gate-to-gate height change that triggers an extra clearance/turn waypoint.
LIFTOFF_Z_THRESHOLD = 0.15    # m, below this altitude the state counts as on-ground
LIFTOFF_HEIGHT = 0.55         # m, first airborne waypoint height on a cold start
COLD_START_MIN_SEG = 0.45     # s, min duration of the first two segments from rest
PERI_GATE_RADIUS = 0.55       # m, XY radius around a gate that flies at V_CRUISE
CLEARANCE_HEIGHT_DELTA = 0.15  # m, gate-to-gate height change that adds a clearance waypoint

# --- Gate-post funnels (from v7) ---
# For each remaining gate we inject two virtual columns at +/-GATE_POST_OFFSET along the
# gate lateral axis, so the global spline gets funnelled through the opening centre instead
# of clipping a frame bar. 0.30 m is tighter than the 0.36 m outer frame half-width and
# wider than R_OBS, so the gate-centre waypoint never gets flagged by the repair pass.
FUNNEL_ENABLED = True
GATE_POST_OFFSET = 0.30  # m, lateral offset from gate centre to each virtual column

# --- Gate crossing direction ---
# False means cross every gate along its canonical +x axis. This matters: the env only
# counts a gate when it's crossed from gate-local -x to +x (see envs/utils.py:gate_passed).
# On a correctly oriented (flyable) track True and False give identical trajectories, but
# False is the safe choice and also survives a gate whose +x opposes the path flow.
ORIENT_GATES_TO_TRAVEL = False

# --- Takeoff phase (dedicated vertical climb before gate tracking) ---
# A clean rest-to-altitude vertical climb that holds the start x/y, decoupling lift-off
# from cruise speed. Without it the global spline ramps vertical acceleration to whatever
# the downstream cruise speed demands and whips the drone off the ground at high speed.
# The climb duration comes from TAKEOFF_CLIMB_SPEED, not cruise speed, so lift-off
# aggressiveness is independent of race speed. Peak vertical accel is about 6 * height /
# t_climb^2, so 0.9 m/s over ~0.5 m stays well within thrust authority.
TAKEOFF_ALT = 0.5            # m, climb target before gate tracking begins
TAKEOFF_CLIMB_SPEED = 0.9    # m/s, average vertical climb speed (peak ~1.5x)
TAKEOFF_Z_TOL = 0.05         # m, hand off to tracking within this of the target altitude
TAKEOFF_TIME_MARGIN = 1.0    # s, fallback handoff once the climb spline overruns by this

# --- Command / actuation limits (v8-owned; moved here from common_controller_v2) ---
# The final attitude+thrust action limits and numerical floors. THRUST_MIN/MAX are the
# crazyflie's physical collective-thrust bounds and EULER_LIMIT caps the commanded tilt.
# These are not race-speed knobs, so don't relax them to "go faster". NORM_EPS is a
# numerical floor for thrust-vector normalisation.
NORM_EPS = 1e-6             # numerical floor for thrust-vector normalisation, not tuning
CLIP_ACTIONS = True         # clip the final attitude+thrust action to the limits below
EULER_LIMIT = math.pi / 2   # rad, max commanded roll/pitch/yaw magnitude
THRUST_MIN = 0.0854505226   # collective-thrust floor (crazyflie physical limit)
THRUST_MAX = 0.8            # collective-thrust ceiling (crazyflie physical limit)

# --- Runtime ---
# GRAVITY is the physical constant used for thrust feedforward. PROJECTION_WINDOW_S is the
# spline-progress search window. REPLAN_*_DELTA_M are the gate/obstacle pose shifts that
# trigger a global replan when the scanned layout gets corrected in flight.
TIMEOUT_S = 30.0               # s, episode time budget
GRAVITY = 9.81                 # m/s^2, physical constant, not a tuning knob
PROJECTION_WINDOW_S = 0.6      # s, window for projecting the current pose onto the spline
REPLAN_GATE_DELTA_M = 0.05     # m, gate pose shift that triggers a global replan
REPLAN_OBSTACLE_DELTA_M = 0.05  # m, obstacle pose shift that triggers a global replan
