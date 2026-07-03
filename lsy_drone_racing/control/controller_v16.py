"""Tunnel MPCC over the guarded-smoothed reference (controller_v16).

v16 flies v11's tunnel-constrained, de-paced MPCC over v10.6's shorter, parity-capped smoothed
reference: the de-paced solver flying the shortened route. Solver and tunnel knobs are v11's
(shared compiled solver); planner and parity caps are v10.6's. Built as the ready-made platform
for exact-pose tracks (e.g. the static real track). REQUIRES the acados environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.controller_v11 import ControllerV11
from lsy_drone_racing.control.controller_core_v8.trajectory import gate_post_obstacles
from lsy_drone_racing.control.controller_core_v10_6.trajectory import ReferenceManager
from lsy_drone_racing.control.controller_core_v16.arc_path import CappedTunnelArcPath
from lsy_drone_racing.control.controller_core_v16.settings import ControllerSettings

if TYPE_CHECKING:
    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation


class ControllerV16(ControllerV11):
    """v11's tunnel MPCC planning over v10.6's guarded-smoothed reference."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v11 (its tunnel MPCC/solver is reused), then swap in the v10.6 planner."""
        super().__init__(obs, info, config)
        self._settings = ControllerSettings()
        self._references = ReferenceManager(
            self._settings.planner,
            self._settings.runtime.replan_gate_delta_m,
            self._settings.runtime.replan_obstacle_delta_m,
            self._v_theta_max,
            self._a_lat_max,
            self._v_min,
        )

    def _track_action(self, frame: DroneObservation) -> np.ndarray:
        """v11's flow with the smoothed plan's parity caps folded into the path view."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if self._gate_nominal is None:
            self._gate_nominal = np.asarray(frame.gate_pos, dtype=np.float64).copy()
        new_plan = rebuilt or self._path is None
        if new_plan:
            first = max(frame.target_gate, 0)
            posts = gate_post_obstacles(
                frame.gate_pos, frame.gate_quat, 0, self._settings.planner.gate_post_offset
            )
            path = CappedTunnelArcPath(
                plan.curve,
                plan.t_total,
                self._settings.mpcc,
                plan.gate_pos_snapshot[first:],
                frame.obstacles_pos,
                posts,
                gate_is_target_zero=first == 0,
                gate_window_caps=plan.gate_window_caps,
                window_pre=self._settings.planner.reveal_window_m,
                obstacle_caps=plan.obstacle_caps,
            )
            self._s = path.project(frame.pos, 0.0)
            if self._path is None:
                self._mpcc.set_path(path)
            else:
                self._mpcc.rebase(path, self._s)
            self._path = path
        th_pred = self._mpcc.predicted_progress()
        if th_pred is None:
            self._s = self._path.project(frame.pos, self._s)
        else:
            self._s = self._path.project_near(frame.pos, th_pred, self._proj_band)
            self._band_calls += 1
            if abs(self._s - th_pred) >= self._proj_band - 1e-9:
                self._band_edge_hits += 1
        if not new_plan and self._anchor_prev_s is not None:
            self._anchor_jumps.append(abs(self._s - self._anchor_prev_s))
        self._anchor_prev_s = self._s
        elapsed = (self._tick - self._nav_start_tick) * self._dt
        ramp = min(1.0, self._ramp_start + (1.0 - self._ramp_start) * elapsed / self._ramp_s)
        accel = self._mpcc.solve(frame.pos, frame.vel, self._s, self._v_theta_max * ramp)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)
