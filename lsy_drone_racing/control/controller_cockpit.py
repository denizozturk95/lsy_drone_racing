"""Shared tunable constants ("cockpit") for controller v6 and v7.

NAVIGATE velocity constants are shared by both controllers. SEARCH/spiral/takeoff
constants are v7-specific.
"""

from __future__ import annotations

import numpy as np

# Shared NAVIGATE velocity (v6 + v7)
V_CRUISE = 1.3
V_CRUISE_INTER = 1.95
VMAX = 2.3

# Shared spline timing & stability (v6 + v7 NAVIGATE)
T_MIN_SEG = 0.4
TURN_MIN_SHARPNESS = 0.25
TURN_SLOW_GAIN = 0.8
LATERAL_ACCEL_LIMIT = 8.0
FEEDFORWARD_SCALE = 0.6

# v6 Takeoff phase
V6_TAKEOFF_ALT = 0.5
V6_TAKEOFF_CLIMB_SPEED = 0.9
V6_TAKEOFF_Z_TOL = 0.05
V6_TAKEOFF_TIME_MARGIN = 1.0

# v7 SEARCH velocity
V_CRUISE_SEARCH = 2.5
VMAX_SEARCH = 3.0

# v7 Spiral / search
SEARCH_ALT = 1.8
SPIRAL_RADIAL_STEP = 0.6
SPIRAL_ANGLE_STEP = np.pi / 6
SPIRAL_ADVANCE_RADIUS = 0.6
SPIRAL_HORIZON = 3
SPIRAL_OUTWARD = False
SEARCH_RADIUS = 2.2
ARENA_X_LIM = 1.9
ARENA_Y_LIM = 1.0
GATE_SKIP_RADIUS = 1.85
GATE_POST_OFFSET = 0.30

# v7 Takeoff
TAKEOFF_ALT = 1
TAKEOFF_Z_TOL = 0.05
TAKEOFF_TIME_MARGIN = 1.0

# v7 NAVIGATE gate-approach geometry
NAV_D_PRE = 0.60
NAV_D_POST = 0.40
NAV_R_OBS = 0.12
NAV_START_VEL_SCALE = 0.5
NAV_LOOKAHEAD = 0.20

# v7 Search-strategy switches
DISCOVER_ALL_FIRST = True
ALLOW_SEARCH = False
