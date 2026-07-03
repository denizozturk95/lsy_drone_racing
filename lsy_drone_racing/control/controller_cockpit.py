"""Shared tunable constants ("cockpit") for the controller v6 and v7 controllers.

One place to tweak the controllers. The NAVIGATE velocity constants
(:data:`V_CRUISE`, :data:`V_CRUISE_INTER`, :data:`VMAX`) are SHARED by both
``ControllerV6`` and ``ControllerV7`` — edit them once and both controllers pick up
the change. All remaining constants are v7-specific (SEARCH / spiral / takeoff /
gate-approach geometry); v6 keeps every other planner value at the defaults in
``common_controller_v2/settings.py``.

Values here carry no leading underscore because they are imported by the
controller modules; the controllers re-bind them to their historical
``_NAME`` form on import so their bodies stay unchanged.
"""

from __future__ import annotations

import numpy as np

# ── Shared NAVIGATE velocity constants (used by BOTH v6 and v7) ─────────────────
# These are v6's tuned values. v7 NAVIGATE shares them (its previous 1.0/1.0/1.4
# is superseded). v7's SEARCH speeds remain separate (see below).
V_CRUISE = 1.3              # m/s — cruise speed near gates (peri-gate; keep low for precision)
V_CRUISE_INTER = 1.95        # m/s — cruise speed BETWEEN gates (raise for faster traversal)
VMAX = 2.3                  # m/s — peak-velocity cap

# ── Shared spline-timing & stability tuning (used by BOTH v6 and v7 NAVIGATE) ───
# These were previously buried in common_controller_v2/{settings,timing}.py. They are the
# "go faster while staying stable" levers that pair with the velocity constants
# above. Defaults equal the historical hard-coded values (pure refactor).
#
# Short-segment time floor (s). Any segment shorter than speed × T_MIN_SEG is
# time-floored — i.e. flown SLOWER than cruise regardless of the speeds above.
# Lower => short straights (and the gate run-in/run-out) may go faster; too low
# risks jerk at the gate plane. Move this in isolation when sweeping speeds.
T_MIN_SEG = 0.4
# turn_slowdown: corners sharper than TURN_MIN_SHARPNESS (0 straight … 1 reversal)
# get their duration stretched by (1 + TURN_SLOW_GAIN × sharpness). This is the
# auto-protection that lets VMAX rise without clipping frames on bends — raise the
# gain (or lower the threshold) as speeds go up; lower it to carry more speed
# through corners. Per the v6 tuning history, turn_slowdown is load-bearing for
# the Level-2 finish rate, so change it carefully.
TURN_MIN_SHARPNESS = 0.25
TURN_SLOW_GAIN = 0.8
# Tracker authority — how hard the feedforward may push to follow a fast, curved
# reference. If raising the speeds produces corner-cutting / frame clips (the
# reference outrunning the tracker), raise LATERAL_ACCEL_LIMIT; too high amplifies
# tracking noise. FEEDFORWARD_SCALE blends model feedforward into the command.
LATERAL_ACCEL_LIMIT = 8.0
FEEDFORWARD_SCALE = 0.6

# ── v6 Takeoff phase (v6 only — dedicated vertical climb before gate tracking) ──
# v6 leaves the ground with a clean rest-to-altitude VERTICAL climb (holding the
# start x/y), then hands off to gate tracking from a stable near-hover. This
# decouples lift-off from the global gate spline: without it the C²-continuous
# spline ramps vertical acceleration to whatever the downstream cruise speed
# demands, so raising the speeds whips the drone off the ground (large vertical
# thrust swing at low altitude → instability). With the phase, takeoff aggressiveness
# is independent of cruise speed. (v7 already has its own TAKEOFF mode below.)
V6_TAKEOFF_ALT = 0.5            # m — climb target before gate tracking begins
V6_TAKEOFF_CLIMB_SPEED = 0.9   # m/s — average vertical climb speed (peak ≈ 1.5×).
# Lower => gentler lift-off (peak vertical accel ≈ 6 × height / t_climb²). 0.9 m/s
# over ~0.9 m gives ~5 m/s² peak — well within thrust authority and no thrust cutoff.
V6_TAKEOFF_Z_TOL = 0.05        # m — hand off to tracking within this of the target altitude
V6_TAKEOFF_TIME_MARGIN = 1.0   # s — fallback handoff after the climb spline overruns by this

# ── v7 SEARCH velocity constants (v7 only — kept separate from the shared set) ──
V_CRUISE_SEARCH = 2.5   # SEARCH cruise speed near (spiral) waypoints
VMAX_SEARCH = 3.0       # SEARCH peak-velocity cap

# ── v7 Spiral / search constants ────────────────────────────────────────────────
# Arena safety limits from level3.toml: x ∈ [-2.5, 2.5], y ∈ [-1.5, 1.5].
# We keep 0.3 m inside the hard limits on each axis.
SEARCH_ALT = 1.8           # m — between gate heights 0.7 m and 1.2 m
SPIRAL_RADIAL_STEP = 0.6   # m radial gap per revolution; < 2 × 0.7 m sensor range
SPIRAL_ANGLE_STEP = np.pi / 6   # 30° step = 12 waypoints per revolution
SPIRAL_ADVANCE_RADIUS = 0.6     # m (2-D) — advance to next spiral point when this close
SPIRAL_HORIZON = 3              # waypoints ahead to include in each SEARCH plan
# False (legacy, much better): outermost-first — fly out to SEARCH_RADIUS then spiral inward,
# ending the sweep near the arena centre, a good hub from which any gate-0 location is
# reachable. True (spiral outward from centre) ends the sweep at the edge and tested far worse
# (7.5% vs 40% on 40 seeds), so it is disabled.
SPIRAL_OUTWARD = False
# Gates are only ever placed within border_margin (0.5 m) of the hard limits, i.e. inside
# ±2.0 m (x) × ±1.0 m (y) (see envs/randomize.py:build_random_track_fn). The search box is
# pulled in to those bounds: full coverage is still guaranteed (sensor range is 0.7 m) while
# keeping the fast spiral well clear of the ±2.5/±1.5 m hard boundary, where the uncapped
# outbound dash used to overshoot and disable the drone.
SEARCH_RADIUS = 2.2             # m — outer radius the search starts from, spiralling inward
ARENA_X_LIM = 1.9          # m from centre — search boundary (gates are within ±2.0 m)
ARENA_Y_LIM = 1.0          # m from centre — search boundary (gates are within ±1.0 m)
GATE_SKIP_RADIUS = 1.85    # m — skip spiral waypoints within this XY distance of a detected gate
# Virtual columns are placed ±GATE_POST_OFFSET along the gate lateral axis to funnel the
# spline through the opening (the gate frame outer half-width is 0.36 m).
GATE_POST_OFFSET = 0.30    # m — lateral offset from gate centre to each virtual column.
# Tighter than the 0.36 m outer frame edge: the virtual columns squeeze the approach/exit
# waypoints closer to the opening centre (a straighter, better-centred run-in => fewer frame
# clips), while staying clear of the gate-centre waypoint's obstacle-repair trigger (~0.24 m
# at r_obs=0.12). This was the change that pushed the Level-3 finish rate to 50%.

# ── v7 Takeoff constants ────────────────────────────────────────────────────────
TAKEOFF_ALT = 1          # m — straight-up climb target before SEARCH begins (tunable)
TAKEOFF_Z_TOL = 0.05       # m — switch to SEARCH when within this of TAKEOFF_ALT
TAKEOFF_TIME_MARGIN = 1.0  # s — fallback handoff after the takeoff spline overruns by this

# ── v7 NAVIGATE gate-approach geometry ──────────────────────────────────────────
# d_pre/d_post set the length of the straight run-in / run-out aligned with the gate
# normal; a longer run-in reduces corner-cutting at the gate plane.  r_obs is the
# obstacle keep-out radius: Level-3 obstacles are thin poles (0.015 m) placed in the
# approach corridor *close to* the gate, so a large keep-out shoves the path toward the
# far frame bar.  A tighter keep-out keeps the crossing near the opening centre while
# still clearing the (thin) pole + drone radius (~0.07 m needed).
NAV_D_PRE = 0.60
NAV_D_POST = 0.40
NAV_R_OBS = 0.12
# Scale applied to the drone's velocity when building the FIRST navigate plan after a
# search handoff. The drone carries cross-arena spiral momentum that, used as the spline
# start boundary condition, forces a wrong-way U-turn loop toward the first gate. Damping
# it (0.0 = plan from rest) makes the spline head straight at the gate; the real velocity
# is still used for tracking, so the drone simply decelerates onto the new plan. 0.5 gave
# the best average gates passed (it heads more directly at gate 0 without a hard transient).
NAV_START_VEL_SCALE = 0.5
# Reference look-ahead time (s). Smaller => the drone tracks the spline point closer to its
# current progress instead of aiming ahead, which reduces corner-cutting (and the body tilt
# that makes a rotor clip the gate frame) on the curved gate approach. Default v6 value 0.20.
NAV_LOOKAHEAD = 0.20

# ── v7 Search-strategy switches ─────────────────────────────────────────────────
# Search strategy: True = discover ALL gates before navigating; False = navigate each gate as found
DISCOVER_ALL_FIRST = True
# Search master switch. True = full Level-3 behavior: TAKEOFF -> SEARCH -> NAVIGATE, reverting
# to SEARCH whenever the next gate is still unknown. False = SEARCH is disabled entirely:
# TAKEOFF hands directly to NAVIGATE and the drone never reverts to SEARCH.
# Set False ONLY when the gate positions are known from t=0 — i.e. the deployment / Level-2
# case where the nominal positions handed to the planner are the real measured track. On a
# genuine Level-3 track (nominal gates all at the origin until sensed) disabling search would
# make NAVIGATE plan toward the origin placeholders and fail.
ALLOW_SEARCH = False
