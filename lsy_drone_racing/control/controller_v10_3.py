"""Gate-aware MPCC with replan continuity: rebases the warm start instead of cold-resetting (v10.3).

Thin subclass of ControllerV101. On replan the solver keeps its world-frame warm start and
re-anchors progress onto the new path. Optional gate-window speed cap (off for level2).
REQUIRES the acados environment (``pixi run``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line

from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.controller_v10_1 import ControllerV101
from lsy_drone_racing.control.controller_core_v10_3.arc_path import GateArcPath
from lsy_drone_racing.control.controller_core_v10_3.mpcc import MPCC
from lsy_drone_racing.control.controller_core_v10_3.settings import ControllerSettings

if TYPE_CHECKING:
    from crazyflow import Sim

    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation


class ControllerV103(ControllerV101):
    """v10.1's gate-aware time-optimal MPCC with replan continuity and a gate speed cap."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v10.1, then swap in the v10.3 settings and the rebase-capable MPCC."""
        super().__init__(obs, info, config)
        self._settings = ControllerSettings()
        mpcc = self._settings.mpcc
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCC(mpcc, a_max)
        self._v_theta_max = mpcc.v_theta_max
        self._ramp_s, self._ramp_start = mpcc.ramp_s, mpcc.ramp_start
        self._a_lat_max, self._v_min = mpcc.a_lat_max, mpcc.v_min
        self._w_base, self._w_gate, self._gate_sigma = (
            mpcc.w_contour_base, mpcc.w_contour_gate, mpcc.gate_sigma,
        )
        self._v_gate = mpcc.v_gate
        self._gate_v_pre, self._gate_v_post = mpcc.gate_v_pre, mpcc.gate_v_post
        self._path: GateArcPath | None = None
        self._s = 0.0

    def _track_action(self, frame: DroneObservation) -> np.ndarray:
        """Fly the plan; mid-flight replans rebase the solver instead of resetting it."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if rebuilt or self._path is None:
            gates_ahead = plan.gate_pos_snapshot[max(frame.target_gate, 0):]
            path = GateArcPath(
                plan.curve, plan.t_total, self._v_theta_max, self._a_lat_max, self._v_min,
                gates_ahead, self._w_base, self._w_gate, self._gate_sigma,
                self._v_gate, self._gate_v_pre, self._gate_v_post,
            )
            self._s = path.project(frame.pos, 0.0)
            if self._path is None:
                self._mpcc.set_path(path)
            else:
                self._mpcc.rebase(path, self._s)
            self._path = path
        self._s = self._path.project(frame.pos, self._s)
        elapsed = (self._tick - self._nav_start_tick) * self._dt
        ramp = min(1.0, self._ramp_start + (1.0 - self._ramp_start) * elapsed / self._ramp_s)
        accel = self._mpcc.solve(frame.pos, frame.vel, self._s, self._v_theta_max * ramp)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)

    def render_callback(self, sim: Sim) -> None:
        """Draw the active plan (green) plus the MPCC's predicted horizon (cyan)."""
        super().render_callback(sim)
        pred = self._mpcc.predicted_positions()
        if pred is not None and self._mode == self._MODE_NAVIGATE:
            draw_line(sim, pred.astype(np.float32), rgba=(0.0, 0.9, 1.0, 1.0))
