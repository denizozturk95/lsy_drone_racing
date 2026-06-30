"""Configuration for the controller_v10 controller.

Inherits v8's planner/takeoff/command/runtime settings (the same path geometry v9 flies) and
adds v10's own time-optimal MPCC settings, built from the v10 cockpit. Nothing here touches
v9/v9.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsy_drone_racing.control.controller_core_v8.settings import ControllerSettings as _V8ControllerSettings
from lsy_drone_racing.control.controller_core_v10 import cockpit as cp


@dataclass(frozen=True)
class MPCCSettings:
    """Time-optimal MPCC: horizon, limits, progress reward, weights, and the QP iteration cap."""

    horizon: int = cp.HORIZON
    step_dt: float = cp.STEP_DT
    v_max: float = cp.V_MAX
    tilt_ratio: float = cp.TILT_RATIO
    a_z_min: float = cp.A_Z_MIN
    mu: float = cp.MU
    v_theta_max: float = cp.V_THETA_MAX
    a_theta_max: float = cp.A_THETA_MAX
    r_dv: float = cp.R_DV
    a_lat_max: float = cp.A_LAT_MAX
    v_min: float = cp.V_MIN
    w_contour: float = cp.W_CONTOUR
    w_lag: float = cp.W_LAG
    w_accel: float = cp.W_ACCEL
    ramp_s: float = cp.RAMP_S
    ramp_start: float = cp.RAMP_START
    max_iter: int = cp.MAX_ITER
    gravity: float = cp.GRAVITY


@dataclass(frozen=True)
class ControllerSettings(_V8ControllerSettings):
    """v8's planner/takeoff/command/runtime settings plus v10's time-optimal MPCC."""

    mpcc: MPCCSettings = field(default_factory=MPCCSettings)
