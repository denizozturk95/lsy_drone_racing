"""Time-optimal MPCC drone racing controller for known tracks (v10).

Thin subclass of ControllerV9. Replaces v9's fixed-recede-rate contouring MPCC with a
time-optimal MPCC: path progress is a decision variable, the cost rewards progress,
so the optimiser picks traversal speed against actuator limits. REQUIRES acados (``pixi run``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.controller_v9 import ControllerV9
from lsy_drone_racing.control.controller_core_v10.arc_path import ArcPath
from lsy_drone_racing.control.controller_core_v10.mpcc import MPCC
from lsy_drone_racing.control.controller_core_v10.settings import ControllerSettings

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation


class ControllerV10(ControllerV9):
    """v9's planner/takeoff flown by a time-optimal (progress-maximising) MPCC."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build v9, then swap in the v10 settings and the time-optimal MPCC."""
        super().__init__(obs, info, config)
        self._settings = ControllerSettings()
        mpcc = self._settings.mpcc
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCC(mpcc, a_max)
        self._v_theta_max = mpcc.v_theta_max
        self._ramp_s, self._ramp_start = mpcc.ramp_s, mpcc.ramp_start
        # Friction-circle params for the curvature speed cap (v_cap = top straight speed).
        self._a_lat_max, self._v_min = mpcc.a_lat_max, mpcc.v_min
        self._path: ArcPath | None = None
        self._s = 0.0

    def reset(self) -> None:
        """Reset v9 state plus the arc-length path/progress anchor."""
        super().reset()
        self._path = None
        self._s = 0.0

    def _track_action(self, frame: DroneObservation) -> NDArray[np.floating]:
        """Fly the plan with the time-optimal MPCC; progress is anchored to the drone."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if rebuilt or self._path is None:  # new plan -> rebuild the arc-length view and reload
            self._path = ArcPath(plan.curve, plan.t_total, self._v_theta_max, self._a_lat_max,
                                 self._v_min)
            self._mpcc.set_path(self._path)
            self._s = self._path.project(frame.pos, 0.0)
        self._s = self._path.project(frame.pos, self._s)  # advance the foot-point anchor
        elapsed = (self._tick - self._nav_start_tick) * self._dt
        ramp = min(1.0, self._ramp_start + (1.0 - self._ramp_start) * elapsed / self._ramp_s)
        accel = self._mpcc.solve(frame.pos, frame.vel, self._s, self._v_theta_max * ramp)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)
