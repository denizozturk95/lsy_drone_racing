"""Configuration dataclasses for the controller_v8 controller.

Every field default comes from controller_core_v8.cockpit, so the cockpit is the one tuning surface
for v8 (PID included) and v8 no longer inherits anything from common_controller_v2.settings. We
reuse the feedback/command/runtime dataclass shapes from common_controller_v2 since they're just
data containers with no v6 tuning of their own, but every field is filled in from the v8
cockpit, so editing v6 has no effect on v8.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lsy_drone_racing.control.common_controller_v2.settings import (
    CommandSettings,
    FeedbackProfile,
    FeedbackSettings,
    RuntimeSettings,
)
from lsy_drone_racing.control.controller_core_v8 import cockpit as cp


def _arr(values: tuple[float, ...]) -> np.ndarray:
    """Convert a per-axis cockpit tuple into a float64 array (PID profile/clamps)."""
    return np.array(values, dtype=np.float64)


@dataclass(frozen=True)
class PlannerSettings:
    """Global-spline planning settings, every default taken from the v8 cockpit."""

    d_pre: float = cp.D_PRE
    d_post: float = cp.D_POST
    d_stop: float = cp.D_STOP
    v_cruise: float = cp.V_CRUISE
    v_cruise_inter: float = cp.V_CRUISE_INTER
    max_speed: float = cp.VMAX
    t_min_seg: float = cp.T_MIN_SEG
    r_obs: float = cp.R_OBS
    liftoff_z_threshold: float = cp.LIFTOFF_Z_THRESHOLD
    liftoff_height: float = cp.LIFTOFF_HEIGHT
    cold_start_min_seg: float = cp.COLD_START_MIN_SEG
    peri_gate_radius: float = cp.PERI_GATE_RADIUS
    clearance_height_delta: float = cp.CLEARANCE_HEIGHT_DELTA
    # False means cross gates along canonical +x (the env's counted direction). See cockpit.
    orient_gates_to_travel: bool = cp.ORIENT_GATES_TO_TRAVEL
    # Turn-slowdown levers, read from settings rather than a module-level cockpit import.
    turn_min_sharpness: float = cp.TURN_MIN_SHARPNESS
    turn_slow_gain: float = cp.TURN_SLOW_GAIN
    # Gate-post funnel: virtual columns injected at +/-gate_post_offset (see trajectory).
    funnel_enabled: bool = cp.FUNNEL_ENABLED
    gate_post_offset: float = cp.GATE_POST_OFFSET


@dataclass(frozen=True)
class TakeoffSettings:
    """Dedicated vertical takeoff-climb parameters."""

    alt: float = cp.TAKEOFF_ALT
    climb_speed: float = cp.TAKEOFF_CLIMB_SPEED
    z_tol: float = cp.TAKEOFF_Z_TOL
    time_margin: float = cp.TAKEOFF_TIME_MARGIN


def _feedback_settings() -> FeedbackSettings:
    """Build the v8-owned PID profile + clamps from the cockpit (no v6 values)."""
    return FeedbackSettings(
        outer_clamp=_arr(cp.OUTER_CLAMP),
        inner_i_limit=_arr(cp.INNER_I_LIMIT),
        output_clamp=_arr(cp.OUTPUT_CLAMP),
        derivative_tau=_arr(cp.DERIVATIVE_TAU),
        eps=cp.FEEDBACK_EPS,
        profile=FeedbackProfile(_arr(cp.KP), _arr(cp.KI), _arr(cp.KD), _arr(cp.OUTER_I_LIMIT)),
    )


def _command_settings() -> CommandSettings:
    """Build the v8-owned feedforward/attitude/action limits from the cockpit."""
    return CommandSettings(
        lateral_accel_limit=cp.LATERAL_ACCEL_LIMIT,
        feedforward_scale=cp.FEEDFORWARD_SCALE,
        norm_eps=cp.NORM_EPS,
        clip_actions=cp.CLIP_ACTIONS,
        euler_limit=cp.EULER_LIMIT,
        thrust_min=cp.THRUST_MIN,
        thrust_max=cp.THRUST_MAX,
    )


def _runtime_settings() -> RuntimeSettings:
    """Build the v8-owned episode timing + replanning policy from the cockpit."""
    return RuntimeSettings(
        timeout_s=cp.TIMEOUT_S,
        gravity=cp.GRAVITY,
        lookahead_s=cp.LOOKAHEAD_S,
        projection_window_s=cp.PROJECTION_WINDOW_S,
        replan_gate_delta_m=cp.REPLAN_GATE_DELTA_M,
        replan_obstacle_delta_m=cp.REPLAN_OBSTACLE_DELTA_M,
    )


@dataclass(frozen=True)
class ControllerSettings:
    """All configurable values used by the v8 controller (every default from the cockpit)."""

    planner: PlannerSettings = field(default_factory=PlannerSettings)
    feedback: FeedbackSettings = field(default_factory=_feedback_settings)
    command: CommandSettings = field(default_factory=_command_settings)
    runtime: RuntimeSettings = field(default_factory=_runtime_settings)
    takeoff: TakeoffSettings = field(default_factory=TakeoffSettings)
