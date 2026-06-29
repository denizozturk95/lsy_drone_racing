"""Configuration for the controller_v10_3 controller.

Inherits v10.1's gate-aware time-optimal MPCC settings and adds the v10.3 knobs: the gate-window
speed cap (``v_gate``, ``gate_v_sigma``) and the raised straight budget (``v_max`` /
``v_theta_max`` overridden from the v10.3 cockpit). The OCP/solver structure is identical to
v10.1; replan continuity (the other v10.3 change) is behavioural and needs no setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsy_drone_racing.control.controller_core_v10_1.settings import (
    ControllerSettings as _V101ControllerSettings,
)
from lsy_drone_racing.control.controller_core_v10_1.settings import MPCCSettings as _V101MPCCSettings
from lsy_drone_racing.control.controller_core_v10_3 import cockpit as cp


@dataclass(frozen=True)
class MPCCSettings(_V101MPCCSettings):
    """v10.1's MPCC settings plus the gate-window speed cap, at the raised v10.3 budget."""

    v_max: float = cp.V_MAX
    v_theta_max: float = cp.V_THETA_MAX
    v_gate: float = cp.V_GATE
    gate_v_pre: float = cp.GATE_V_PRE
    gate_v_post: float = cp.GATE_V_POST
    ramp_start: float = cp.RAMP_START
    ramp_s: float = cp.RAMP_S
    a_lat_max: float = cp.A_LAT_MAX
    a_theta_max: float = cp.A_THETA_MAX


@dataclass(frozen=True)
class ControllerSettings(_V101ControllerSettings):
    """v10.1's settings with the v10.3 MPCC settings (gate speed cap + raised budget)."""

    mpcc: MPCCSettings = field(default_factory=MPCCSettings)
