"""Tunnel-constrained gate-aware MPCC (v11).

v10.5 with gate-weight pacing replaced by MPCC++-style tunnel constraints + explicit reveal caps.
Thin subclass of ControllerV105. REQUIRES acados (``pixi run``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.controller_v10_5 import ControllerV105
from lsy_drone_racing.control.controller_core_v8.trajectory import gate_post_obstacles
from lsy_drone_racing.control.controller_core_v11.arc_path import TunnelArcPath
from lsy_drone_racing.control.controller_core_v11.mpcc import MPCC
from lsy_drone_racing.control.controller_core_v11.settings import ControllerSettings

if TYPE_CHECKING:
    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation


class ControllerV11(ControllerV105):
    """v10.5's racing stack with tunnel constraints instead of gate-weight pacing."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v10.5, then swap in the v11 MPCC (tunnel OCP, own solver namespace)."""
        super().__init__(obs, info, config)
        self._settings = ControllerSettings()
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCC(self._settings.mpcc, a_max)

    def _track_action(self, frame: DroneObservation) -> np.ndarray:
        """v10.5's flow with the tunnel path view; reveal caps live inside the path view."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if self._gate_nominal is None:
            self._gate_nominal = np.asarray(frame.gate_pos, dtype=np.float64).copy()
        new_plan = rebuilt or self._path is None
        if new_plan:
            first = max(frame.target_gate, 0)
            # The tunnel must respect EVERY gate's frame posts (passed gates included --
            # the frames are physical whether or not a gate is still the target).
            posts = gate_post_obstacles(
                frame.gate_pos, frame.gate_quat, 0, self._settings.planner.gate_post_offset
            )
            path = TunnelArcPath(
                plan.curve,
                plan.t_total,
                self._settings.mpcc,
                plan.gate_pos_snapshot[first:],
                frame.obstacles_pos,
                posts,
                gate_is_target_zero=first == 0,
            )
            self._s = path.project(frame.pos, 0.0)
            if self._path is None:  # episode start: honest cold start (v10.4 mpcc flow)
                self._mpcc.set_path(path)
            else:  # mid-flight replan: keep the warm start, re-anchor onto the new path
                self._mpcc.rebase(path, self._s)
            self._path = path
        # v10.5's dynamics-aware anchor, including its telemetry.
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
