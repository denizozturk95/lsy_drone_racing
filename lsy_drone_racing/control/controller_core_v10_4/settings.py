"""Configuration for the controller_v10_4 controller.

Inherits v10.3's MPCC settings (same OCP/solver-cache key, so the compiled acados solver is
shared) and overrides only the launch knobs: the ramp (now a launch ramp) and the takeoff
(now a mini-takeoff with a fixed climb duration).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsy_drone_racing.control.controller_core_v10_3.settings import (
    ControllerSettings as _V103ControllerSettings,
)
from lsy_drone_racing.control.controller_core_v10_3.settings import MPCCSettings as _V103MPCCSettings
from lsy_drone_racing.control.controller_core_v10_4 import cockpit as cp
from lsy_drone_racing.control.controller_core_v8.settings import PlannerSettings as _V8PlannerSettings
from lsy_drone_racing.control.controller_core_v8.settings import TakeoffSettings as _V8TakeoffSettings


@dataclass(frozen=True)
class MPCCSettings(_V103MPCCSettings):
    """v10.3's MPCC settings with the launch ramp and the longer horizon."""

    horizon: int = cp.HORIZON
    ramp_start: float = cp.RAMP_START
    ramp_s: float = cp.RAMP_S
    v_gate: float = cp.V_GATE
    gate_v_pre: float = cp.GATE_V_PRE
    gate_v_post: float = cp.GATE_V_POST
    w_contour_gate: float = cp.W_CONTOUR_GATE
    gate_sigma: float = cp.GATE_SIGMA
    v_gate_react: float = cp.V_GATE_REACT
    react_delta_m: float = cp.REACT_DELTA_M
    react_v_pre: float = cp.REACT_V_PRE
    react_v_post: float = cp.REACT_V_POST


@dataclass(frozen=True)
class LaunchPlannerSettings(_V8PlannerSettings):
    """v8's planner settings plus the trim dial and the wider obstacle keep-out."""

    clr_ext_min: float = cp.CLR_EXT_MIN
    r_obs: float = cp.R_OBS


@dataclass(frozen=True)
class LaunchTakeoffSettings(_V8TakeoffSettings):
    """v8's takeoff settings plus a fixed climb duration, at the v10.4 mini-takeoff values."""

    alt: float = cp.TAKEOFF_ALT
    climb_time: float = cp.TAKEOFF_CLIMB_TIME


@dataclass(frozen=True)
class ControllerSettings(_V103ControllerSettings):
    """v10.3's settings with the v10.4 launch ramp, mini-takeoff, and planner trim dial."""

    mpcc: MPCCSettings = field(default_factory=MPCCSettings)
    takeoff: LaunchTakeoffSettings = field(default_factory=LaunchTakeoffSettings)
    planner: LaunchPlannerSettings = field(default_factory=LaunchPlannerSettings)
