"""Configuration for the controller_v9 (MPCC) controller.

The planner, takeoff, command, and runtime settings are reused verbatim from
controller_core_v8.settings.ControllerSettings (v9 flies v8's plan geometry and uses v8's attitude
conversion and takeoff), so this module only adds the MPCC settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsy_drone_racing.control.controller_core_v8.settings import ControllerSettings as _V8ControllerSettings
from lsy_drone_racing.control.controller_core_v9 import cockpit as cp


@dataclass(frozen=True)
class MPCCSettings:
    """MPCC horizon, speed, actuator limits, and contouring weights (from the cockpit)."""

    horizon: int = cp.HORIZON
    step_dt: float = cp.STEP_DT
    v_ref: float = cp.V_REF
    v_max: float = cp.V_MAX
    ramp_s: float = cp.RAMP_S
    ramp_start: float = cp.RAMP_START
    tilt_ratio: float = cp.TILT_RATIO
    a_z_min: float = cp.A_Z_MIN
    w_contour: float = cp.W_CONTOUR
    w_lag: float = cp.W_LAG
    w_accel: float = cp.W_ACCEL
    max_iter: int = cp.MAX_ITER
    gravity: float = cp.GRAVITY


@dataclass(frozen=True)
class ControllerSettings(_V8ControllerSettings):
    """v8's planner/takeoff/command/runtime settings plus the MPCC settings."""

    mpcc: MPCCSettings = field(default_factory=MPCCSettings)
