"""v16 settings: v11's tunnel-MPCC settings over the v10.6 guarded-smoothing planner.

v11's solver/tunnel knobs (same compiled solver) with the v10.6 smoother and parity-cap geometry
reused unchanged -- only the planner the tunnel flies over changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsy_drone_racing.control.controller_core_v10_6.settings import SmoothPlannerSettings
from lsy_drone_racing.control.controller_core_v11.settings import (
    ControllerSettings as _V11ControllerSettings,
)


@dataclass(frozen=True)
class ControllerSettings(_V11ControllerSettings):
    """v11's settings over the v10.6 guarded-smoothing planner."""

    planner: SmoothPlannerSettings = field(default_factory=SmoothPlannerSettings)
