"""Configuration for the controller_v10_1 controller.

Inherits v8's planner/takeoff/command/runtime settings (the same path geometry v9/v10 fly) and
adds v10.1's own gate-aware time-optimal MPCC settings, built from the v10.1 cockpit. The MPCC keeps
v10's 8-state structure (acceleration is the zero-order-hold control); the only additions over
v10 are the gate-aware contouring knobs (base/gate/sigma). Nothing here touches v9/v9.1/v10.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsy_drone_racing.control.controller_core_v8.settings import ControllerSettings as _V8ControllerSettings
from lsy_drone_racing.control.controller_core_v10_1 import cockpit as cp


@dataclass(frozen=True)
class MPCCSettings:
    """Gate-aware time-optimal MPCC: horizon, limits, progress reward, contouring weights."""

    horizon: int = cp.HORIZON
    step_dt: float = cp.STEP_DT
    w_contour_base: float = cp.W_CONTOUR_BASE
    w_contour_gate: float = cp.W_CONTOUR_GATE
    gate_sigma: float = cp.GATE_SIGMA
    v_max: float = cp.V_MAX
    tilt_ratio: float = cp.TILT_RATIO
    a_z_min: float = cp.A_Z_MIN
    mu: float = cp.MU
    v_theta_max: float = cp.V_THETA_MAX
    a_theta_max: float = cp.A_THETA_MAX
    r_dv: float = cp.R_DV
    a_lat_max: float = cp.A_LAT_MAX
    v_min: float = cp.V_MIN
    w_lag: float = cp.W_LAG
    w_accel: float = cp.W_ACCEL
    ramp_s: float = cp.RAMP_S
    ramp_start: float = cp.RAMP_START
    max_iter: int = cp.MAX_ITER
    gravity: float = cp.GRAVITY


@dataclass(frozen=True)
class ControllerSettings(_V8ControllerSettings):
    """v8's planner/takeoff/command/runtime settings plus v10.1's gate-aware time-optimal MPCC."""

    mpcc: MPCCSettings = field(default_factory=MPCCSettings)
