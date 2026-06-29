"""Generic, track-agnostic parameters for the online_planner controller."""
# ruff: noqa: TC002

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Array3 = NDArray[np.float64]


@dataclass(frozen=True)
class PlannerSettings:
    """Observation-driven spline planning settings (no track-specific constants)."""

    d_pre: float = 0.40
    d_post: float = 0.30
    d_stop: float = 0.30
    v_cruise: float = 0.9
    v_cruise_inter: float = 1.0
    max_speed: float = 1.3
    t_min_seg: float = 0.30
    r_obs: float = 0.20
    liftoff_z_threshold: float = 0.15
    liftoff_height: float = 0.55
    cold_start_min_seg: float = 0.45
    peri_gate_radius: float = 0.55
    clearance_height_delta: float = 0.15
    # When True (default, Level-2 behavior) gate forward axes are flipped to match the
    # drone's travel direction. When False the gate's canonical +x axis is used as the
    # crossing direction — required for Level 3, where the env only counts a gate as
    # passed when crossed from gate-local -x to +x (see envs/utils.py:gate_passed) and
    # the randomizer orients every gate's +x along the intended traversal direction.
    orient_gates_to_travel: bool = True
    # When True, an upward gate-to-gate transition climbs to the *full* next-gate height at the
    # clearance/turn-apex waypoints (instead of topping out ~0.05 m below it), so the drone
    # reaches tall-gate height early and flies level into the run-in rather than clipping the
    # bottom frame bar on a late climb. Default False keeps the original Level-2/3 behavior
    # (GateSearchV4+ opt in; GateSearchV2/V3 unaffected).
    early_climb: bool = False
    # U-turn (reversal) swing geometry, in metres along the lateral axis. The reversal waypoints
    # bow out to the next-gate side so the clamped spline rounds the reversal without a cusp;
    # smaller values keep the curve tighter (closer to the gates), which matters in tight arenas.
    # Defaults reproduce the original hard-coded reversal_turn() values, so GateSearchV2..V8 are
    # byte-equivalent; GateSearchV9 opts into smaller swings.
    reversal_swing_m: float = 0.55
    reversal_apex_m: float = 0.10
    # Arena geofence (track-agnostic). When ``geofence_margin`` > 0 and arena bounds are provided,
    # the planned path is clamped to stay at least ``geofence_margin`` m inside [arena_low,
    # arena_high] in X/Y, reserving room for downstream tracking overshoot so a wide curve can
    # never push the drone across a safety plane (the "leaves the zone -> abort" failure). Gate
    # crossings are protected from the clamp. Default 0.0 disables it (GateSearchV2..V8 unchanged).
    geofence_margin: float = 0.0
    arena_low: tuple[float, float, float] | None = None
    arena_high: tuple[float, float, float] | None = None
    # Cap on the reference's downward speed (m/s). When > 0, any segment that descends is stretched
    # so |dz|/dt <= max_descent_rate, keeping the planned descent gentle enough for the tracker to
    # follow. Without it, a steep descent (e.g. from the Level-3 search altitude down to a gate)
    # outruns the controller and the drone crosses the gate plane too high, clipping the top frame
    # bar. Default 0.0 disables it (GateSearchV2..V9 unchanged); GateSearchV10 opts in.
    max_descent_rate: float = 0.0
    # ── Time-optimal path parameterization (TOPP) ──────────────────────────────────────────────
    # When True, build_spline() replaces the heuristic time-allocation (fixed v_cruise + peri-gate /
    # turn / obstacle slowdowns) with a curvature-aware, acceleration-limited speed profile: the
    # straightaways run at ``max_speed`` and the path slows ONLY where curvature would exceed
    # ``topp_a_lat`` (and near gates/obstacles for precision). A forward/backward pass bounds the
    # tangential acceleration by ``topp_a_tang`` so the whole trajectory is dynamically feasible and
    # therefore trackable at speed. This is the GateSearchV12 navigate paradigm. Default False keeps
    # the classic time-allocation (every other controller/mode unchanged).
    use_topp: bool = False
    topp_a_lat: float = 8.0     # m/s² — max lateral (cornering) acceleration the speed profile allows
    topp_a_tang: float = 6.0    # m/s² — max tangential (accel/brake along path) acceleration
    topp_v_gate: float = 1.6    # m/s — speed cap within ``peri_gate_radius`` of a gate (crossing precision)
    topp_v_obs: float = 1.2     # m/s — speed cap near a detected obstacle
    topp_v_stop: float = 0.3    # m/s — speed at the final waypoint


@dataclass(frozen=True)
class FeedbackProfile:
    """Legacy-style gains resolved into the cascaded controller."""

    kp: Array3
    ki: Array3
    kd: Array3
    outer_i_limit: Array3


@dataclass(frozen=True)
class FeedbackSettings:
    """PID limits and a single resolved gain profile."""

    outer_clamp: Array3 = field(
        default_factory=lambda: np.array([2.4, 2.35, 1.8], dtype=np.float64)
    )
    inner_i_limit: Array3 = field(
        default_factory=lambda: np.array([0.75, 0.75, 0.45], dtype=np.float64)
    )
    output_clamp: Array3 = field(
        default_factory=lambda: np.array([3.2, 3.2, 4.2], dtype=np.float64)
    )
    derivative_tau: Array3 = field(
        default_factory=lambda: np.array([0.05, 0.05, 0.06], dtype=np.float64)
    )
    eps: float = 1e-9
    profile: FeedbackProfile = field(
        default_factory=lambda: FeedbackProfile(
            np.array([0.60, 0.60, 1.65], dtype=np.float64),
            np.array([0.05, 0.05, 0.05], dtype=np.float64),
            np.array([0.35, 0.35, 0.50], dtype=np.float64),
            np.array([1.5, 1.5, 0.4], dtype=np.float64),
        )
    )


@dataclass(frozen=True)
class CommandSettings:
    """Feedforward, attitude, and final action limits."""

    lateral_accel_limit: float = 8.0
    feedforward_scale: float = 0.6
    norm_eps: float = 1e-6
    clip_actions: bool = True
    euler_limit: float = np.pi / 2
    thrust_min: float = 0.0854505226
    thrust_max: float = 0.8


@dataclass(frozen=True)
class RuntimeSettings:
    """Episode timing and replanning policy."""

    timeout_s: float = 30.0
    gravity: float = 9.81
    lookahead_s: float = 0.20
    projection_window_s: float = 0.6
    replan_gate_delta_m: float = 0.05
    replan_obstacle_delta_m: float = 0.05


@dataclass(frozen=True)
class ControllerSettings:
    """All configurable values used by the controller."""

    planner: PlannerSettings = field(default_factory=PlannerSettings)
    feedback: FeedbackSettings = field(default_factory=FeedbackSettings)
    command: CommandSettings = field(default_factory=CommandSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
