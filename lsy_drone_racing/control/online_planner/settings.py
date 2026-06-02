"""generic track-agnostic parameters for the controller."""
# ruff: noqa: TC002

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Array3 = NDArray[np.float64]


@dataclass(frozen=True)
class PlannerSettings:
    """observation-driven spline planning settings."""

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


@dataclass(frozen=True)
class FeedbackProfile:
    """gains resolved into the cascaded controller."""

    kp: Array3
    ki: Array3
    kd: Array3
    outer_i_limit: Array3


@dataclass(frozen=True)
class FeedbackSettings:
    """pid limits and a single resolved gain profile."""

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
    """feedforward attitude and final action limits."""

    lateral_accel_limit: float = 8.0
    feedforward_scale: float = 0.6
    norm_eps: float = 1e-6
    clip_actions: bool = True
    euler_limit: float = np.pi / 2
    thrust_min: float = 0.0854505226
    thrust_max: float = 0.8


@dataclass(frozen=True)
class RuntimeSettings:
    """episode timing and replanning policy."""

    timeout_s: float = 30.0
    gravity: float = 9.81
    lookahead_s: float = 0.20
    projection_window_s: float = 0.6
    replan_gate_delta_m: float = 0.05
    replan_obstacle_delta_m: float = 0.05


@dataclass(frozen=True)
class ControllerSettings:
    """all configurable values used by the controller."""

    planner: PlannerSettings = field(default_factory=PlannerSettings)
    feedback: FeedbackSettings = field(default_factory=FeedbackSettings)
    command: CommandSettings = field(default_factory=CommandSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
