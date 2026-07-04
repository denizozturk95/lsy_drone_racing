"""Gate-aware time-optimal MPCC with a fast launch (v10.4).

Thin subclass of ControllerV103. Adds mini-takeoff, hotter launch ramp, honest cold start,
and reactive gate-0 approach cap. v10.3 measured 54/60 @ 8.20s → v10.4 52/60 @ 8.01s.
REQUIRES the acados environment (``pixi run``).
"""

from __future__ import annotations

import numpy as np

from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.controller_v10_3 import ControllerV103
from lsy_drone_racing.control.controller_core_v10_4.arc_path import GateArcPath
from lsy_drone_racing.control.controller_core_v10_4.mpcc import MPCC
from lsy_drone_racing.control.controller_core_v10_4.settings import ControllerSettings
from lsy_drone_racing.control.controller_core_v10_4.takeoff import TakeoffPhase
from lsy_drone_racing.control.controller_core_v10_4.trajectory import ReferenceManager


class ControllerV104(ControllerV103):
    """v10.3's racing MPCC behind a mini-takeoff, a launch ramp, and an honest cold start."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v10.3, then swap in the v10.4 launch components."""
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
        self._takeoff = TakeoffPhase(self._settings, self._settings.takeoff)
        # Plan through the trimmed clearance geometry
        self._references = ReferenceManager(
            self._settings.planner,
            self._settings.runtime.replan_gate_delta_m,
            self._settings.runtime.replan_obstacle_delta_m,
        )
        self._v_gate_react = mpcc.v_gate_react
        self._react_delta = mpcc.react_delta_m
        self._react_v_pre, self._react_v_post = mpcc.react_v_pre, mpcc.react_v_post
        self._gate_nominal: np.ndarray | None = None
        self._path = None
        self._s = 0.0

    def reset(self) -> None:
        """Reset v10.3 state plus the nominal gate-pose snapshot."""
        super().reset()
        self._gate_nominal = None

    def _track_action(self, frame):
        """v10.3's flow, flagging moved (revealed) gates with a reactive approach speed cap."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if self._gate_nominal is None:
            self._gate_nominal = np.asarray(frame.gate_pos, dtype=np.float64).copy()
        if rebuilt or self._path is None:
            first = max(frame.target_gate, 0)
            gates_ahead = plan.gate_pos_snapshot[first:]
            deltas = np.linalg.norm(
                plan.gate_pos_snapshot[first:] - self._gate_nominal[first:], axis=1
            )
            caps = np.where(deltas > self._react_delta, self._v_gate_react, np.inf)
            if first == 0:
                caps[0] = self._v_gate_react
            path = GateArcPath(
                plan.curve, plan.t_total, self._v_theta_max, self._a_lat_max, self._v_min,
                gates_ahead, self._w_base, self._w_gate, self._gate_sigma,
                caps, self._react_v_pre, self._react_v_post,
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
