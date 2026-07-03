"""Planner settings for the v10.6 guarded-smoothing reference.

MPCC/solver settings are inherited unchanged from v10.4/v10.5 (shared compiled solver). Only the
planner gains the smoothing and acceptance-guard knobs read by trajectory.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from lsy_drone_racing.control.controller_core_v10_4.settings import (
    LaunchPlannerSettings as _V104PlannerSettings,
)

# Waypoint smoothing: free waypoints move pull * (neighbour midpoint - self) per sweep; gate
# triplets, chain ends, the replan-continuity arc, and whole reversal legs never move.
SMOOTH_PULL = 0.5
SMOOTH_ITERS = 3

# Parity caps copied from the unsmoothed profile. Reveal window = sensor range upstream of each
# gate plane; obstacle tube = passage region where a revealed obstacle shift still needs a swerve.
REVEAL_WINDOW_M = 0.7
OBS_CAP_RADIUS = 0.45

# Acceptance guard (build 1 per episode): a smoothed candidate ships only if every gate-plane
# crossing inside the frame threads the opening within CROSS_TOL_M, real-obstacle clearance holds,
# and the capped profile beats the base by MIN_GAIN_S. The decision is episode-sticky.
CROSS_TOL_M = 0.15
MIN_GAIN_S = 0.05


@dataclass(frozen=True)
class SmoothPlannerSettings(_V104PlannerSettings):
    """v10.4's planner settings plus the v10.6 smoothing and acceptance-guard knobs."""

    smooth_pull: float = SMOOTH_PULL
    smooth_iters: int = SMOOTH_ITERS
    reveal_window_m: float = REVEAL_WINDOW_M
    obs_cap_radius: float = OBS_CAP_RADIUS
    cross_tol_m: float = CROSS_TOL_M
    min_gain_s: float = MIN_GAIN_S
