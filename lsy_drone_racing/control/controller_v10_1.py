"""Time-optimal MPCC with a gate-aware contouring weight that spikes at each gate (v10.1).

Narrows v10 by making contouring weight per-stage: high near gates for precision under
+/-0.15 m randomisation, low on straights for speed. Thin subclass of ControllerV10.
REQUIRES the acados environment (``pixi run``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line

from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.controller_v10 import ControllerV10
from lsy_drone_racing.control.controller_core_v10_1.arc_path import GateArcPath
from lsy_drone_racing.control.controller_core_v10_1.mpcc import MPCC
from lsy_drone_racing.control.controller_core_v10_1.settings import ControllerSettings

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation


class ControllerV101(ControllerV10):
    """v10's time-optimal MPCC with a gate-aware contouring weight, flown at a raised budget."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v10, then swap in the v10.1 settings and the gate-aware time-optimal MPCC."""
        super().__init__(obs, info, config)
        self._settings = ControllerSettings()
        mpcc = self._settings.mpcc
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCC(mpcc, a_max)
        self._v_theta_max = mpcc.v_theta_max
        self._ramp_s, self._ramp_start = mpcc.ramp_s, mpcc.ramp_start
        self._a_lat_max, self._v_min = mpcc.a_lat_max, mpcc.v_min
        self._w_base = mpcc.w_contour_base
        self._w_gate = mpcc.w_contour_gate
        self._gate_sigma = mpcc.gate_sigma
        self._path: GateArcPath | None = None
        self._s = 0.0
        self._dbg_obs_pos = np.empty((0, 3), dtype=np.float64)
        self._r_obs = float(getattr(self._references._settings, "r_obs", 0.20))
        self._nominal_obs_pos = np.asarray(
            obs.get("obstacles_pos", []), dtype=np.float64
        ).reshape(-1, 3).copy()

    def reset(self) -> None:
        """Reset v10 state plus the detected-obstacle overlay markers."""
        super().reset()
        self._dbg_obs_pos = np.empty((0, 3), dtype=np.float64)

    def _track_action(self, frame: DroneObservation) -> np.ndarray:
        """Fly the plan with the gate-aware MPCC; the contouring weight spikes at each gate."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if rebuilt or self._path is None:
            gates_ahead = plan.gate_pos_snapshot[max(frame.target_gate, 0):]
            self._path = GateArcPath(
                plan.curve, plan.t_total, self._v_theta_max, self._a_lat_max, self._v_min,
                gates_ahead, self._w_base, self._w_gate, self._gate_sigma,
            )
            self._mpcc.set_path(self._path)
            self._s = self._path.project(frame.pos, 0.0)
        self._s = self._path.project(frame.pos, self._s)
        elapsed = (self._tick - self._nav_start_tick) * self._dt
        ramp = min(1.0, self._ramp_start + (1.0 - self._ramp_start) * elapsed / self._ramp_s)
        accel = self._mpcc.solve(frame.pos, frame.vel, self._s, self._v_theta_max * ramp)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance v10's clock and latch the sensor-confirmed obstacle positions for rendering."""
        obs_pos = np.asarray(obs.get("obstacles_pos", []), dtype=np.float64).reshape(-1, 3)
        visited = np.asarray(obs.get("obstacles_visited", []), dtype=bool).reshape(-1)
        if visited.any() and len(visited) <= len(obs_pos):
            self._dbg_obs_pos = obs_pos[: len(visited)][visited].copy()
        return super().step_callback(action, obs, reward, terminated, truncated, info)

    def render_callback(self, sim: Sim) -> None:
        """Overlay the path's knot points, nominal + detected obstacles, and the keep-out rings."""
        super().render_callback(sim)

        def _cross(pos: NDArray[np.floating], rgba: tuple, arm: float) -> None:
            """Draw a 3-axis cross at pos via three two-point segments."""
            p = np.asarray(pos, dtype=np.float32)
            for axis in range(3):
                seg = np.repeat(p[None, :], 2, axis=0)
                seg[0, axis] -= arm
                seg[1, axis] += arm
                draw_line(sim, seg, rgba=rgba)

        for op in self._nominal_obs_pos:
            _cross(op, rgba=(1.0, 1.0, 1.0, 0.6), arm=0.06)

        plan = self._references.plan
        if plan is not None and self._mode == self._MODE_NAVIGATE:
            for wp in np.asarray(plan.waypoints, dtype=np.float64).reshape(-1, 3):
                _cross(wp, rgba=(0.2, 0.4, 1.0, 1.0), arm=0.05)

        angles = np.linspace(0.0, 2.0 * np.pi, 32)
        unit_ring = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
        for op in self._dbg_obs_pos:
            _cross(op, rgba=(1.0, 0.0, 0.0, 1.0), arm=0.08)
            ring = (unit_ring * self._r_obs + np.asarray(op, dtype=np.float32)).astype(np.float32)
            draw_line(sim, ring, rgba=(1.0, 0.5, 0.0, 0.8))
