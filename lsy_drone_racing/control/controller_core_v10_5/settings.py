"""Configuration for the controller_v10_5 controller.

Inherits v10.4's full settings stack -- the MPCC settings (same OCP/solver-cache key, so the
compiled acados solver is shared), the launch ramp, the mini-takeoff, and the trimmed planner --
and adds a single field: ``proj_band_m``, the half-width of the geometric correction applied
around the solver's predicted progress (the v10.2 dynamics-aware anchor, re-applied on the v10.4
base). ``proj_band_m`` is NOT part of the solver-cache key, so the solver stays shared with v10.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsy_drone_racing.control.controller_core_v10_4.settings import (
    ControllerSettings as _V104ControllerSettings,
)
from lsy_drone_racing.control.controller_core_v10_4.settings import MPCCSettings as _V104MPCCSettings
from lsy_drone_racing.control.controller_core_v10_5 import cockpit as cp


@dataclass(frozen=True)
class MPCCSettings(_V104MPCCSettings):
    """v10.4's launch/MPCC settings plus the v10.2 dynamics-aware anchor band."""

    proj_band_m: float = cp.PROJ_BAND_M


@dataclass(frozen=True)
class ControllerSettings(_V104ControllerSettings):
    """v10.4's settings (launch ramp, mini-takeoff, reactive caps) + the predicted-progress band."""

    mpcc: MPCCSettings = field(default_factory=MPCCSettings)
