"""Gate-aware time-optimal MPCC drone racing controller for known tracks (controller_v10_1).

v10.1 is v10 made *precise where it matters*. v10 already lets the optimiser pick traversal speed
(progress is a state, the cost rewards progress, a friction-circle cap brakes corners), but it
uses ONE constant contouring weight, so raising the speed budget trades contour error for
progress everywhere -- including at the gates -- and the drone overshoots. On this track the lap
time is capped by gate-passing precision under +/-0.15 m gate randomisation, not by actuator
authority, so that overshoot is the ceiling. v10.1 makes the contouring weight a per-stage value
that spikes in a window around each gate's arc-position (see controller_core_v10_1.arc_path / controller_core_v10_1.mpcc):
the drone hugs the line exactly where a gate must be threaded and stays free on the straights,
which lets v10.1 carry a higher speed budget (V_MAX, A_LAT_MAX, V_THETA_MAX) than v10 without
losing gates.

Only the MPCC and how the path is built differ from v10, so this is a thin subclass of
ControllerV10: it reuses v10's takeoff, the foot-point progress anchor, and the NAVIGATE flow, and
overrides _track_action only to build the gate-aware path (GateArcPath) from the plan's gate
snapshot. Acceleration stays the zero-order-hold control, matching the plant -- the v10 solver
structure is unchanged apart from one extra per-stage parameter (the gate weight).

REQUIRES the acados environment -- run under ``pixi run`` (sets ACADOS_SOURCE_DIR + the tera
renderer). The C solver is code-generated/compiled once per process into c_generated_code/.

Measured on level2 (scripts/evaluate.py, 20 runs, vs v10 at the same call): v10.1 at the shipped
config (V_MAX = V_THETA_MAX = 3.2, A_LAT_MAX = 8.5, gate weight base 20 + peak 20) ran
19/20 (95%) at avg 8.27 s / best 7.66 s, against v10's 16/20 (80%) at avg 8.45 s -- faster and at
least as reliable, with a tighter worst case. The gain is modest because the binding limit is
gate-passing precision under randomisation, not actuator authority; the gate-aware weight buys a
little speed at equal-or-better finish, not a large jump.

Tuning recipe to push further (re-confirm over 20 runs each; the run-to-run finish noise is ~+/-2):
  1. Keep W_CONTOUR_BASE = 20 (v10's value). Dropping it below 20 tracks gate APPROACHES looser
     than v10 and HURTS finish -- the sweep showed this clearly. Stack the gate weight ON TOP.
  2. Speed-leaning: raise V_MAX = V_THETA_MAX (3.2 -> 3.3) and A_LAT_MAX (8.5 -> 9.0). This is a
     little faster but drops finish toward ~70-75%; only worth it if the score rewards speed over
     finish. 3.6 / 10.0 collapses finish -- do not ship it.
  3. If gate-0 crashes appear at a higher budget, lengthen RAMP_S (2.0 -> 2.4) before backing off.

Note: an earlier v10.1 tried a thrust-vector slew limit (acceleration as a state, jerk as the
control). It was dropped -- the plant takes acceleration as a zero-order-hold command, so a
jerk-as-state model mismatched it and tracked worse than v10, and v10's plans were already well
tracked in sim (attitude lag was not the limiter). Gate precision is, hence this design.
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
        # Refresh the cached limits v10's flow reads, from the v10.1 settings.
        self._v_theta_max = mpcc.v_theta_max
        self._ramp_s, self._ramp_start = mpcc.ramp_s, mpcc.ramp_start
        self._a_lat_max, self._v_min = mpcc.a_lat_max, mpcc.v_min
        # Gate-aware contouring knobs (read when (re)building the GateArcPath).
        self._w_base = mpcc.w_contour_base
        self._w_gate = mpcc.w_contour_gate
        self._gate_sigma = mpcc.gate_sigma
        self._path: GateArcPath | None = None
        self._s = 0.0
        # Detected-obstacle markers for the in-sim overlay (sensor-confirmed positions only).
        self._dbg_obs_pos = np.empty((0, 3), dtype=np.float64)
        # Keep-out radius the planner actually avoids each obstacle by (the 2-D XY ring radius).
        self._r_obs = float(getattr(self._references._settings, "r_obs", 0.20))
        # Nominal obstacle layout: at construction nothing is sensed yet, so the env reports
        # each obstacle's nominal (un-randomised) position -- latch it for the overlay.
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
        if rebuilt or self._path is None:  # new plan -> rebuild the gate-aware arc view and reload
            gates_ahead = plan.gate_pos_snapshot[max(frame.target_gate, 0):]
            self._path = GateArcPath(
                plan.curve, plan.t_total, self._v_theta_max, self._a_lat_max, self._v_min,
                gates_ahead, self._w_base, self._w_gate, self._gate_sigma,
            )
            self._mpcc.set_path(self._path)
            self._s = self._path.project(frame.pos, 0.0)
        self._s = self._path.project(frame.pos, self._s)  # advance the foot-point anchor
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

        # White crosses at the nominal (config) obstacle positions, before/independent of sensing.
        for op in self._nominal_obs_pos:
            _cross(op, rgba=(1.0, 1.0, 1.0, 0.6), arm=0.06)

        # Blue crosses at the generated spline's knot points (the planner's waypoint chain).
        plan = self._references.plan
        if plan is not None and self._mode == self._MODE_NAVIGATE:
            for wp in np.asarray(plan.waypoints, dtype=np.float64).reshape(-1, 3):
                _cross(wp, rgba=(0.2, 0.4, 1.0, 1.0), arm=0.05)

        # Red crosses + orange keep-out rings at the detected (sensor-confirmed) obstacles.
        angles = np.linspace(0.0, 2.0 * np.pi, 32)
        unit_ring = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], axis=1)
        for op in self._dbg_obs_pos:
            _cross(op, rgba=(1.0, 0.0, 0.0, 1.0), arm=0.08)
            # Keep-out radius as a horizontal ring at the obstacle's height (XY-plane avoidance).
            ring = (unit_ring * self._r_obs + np.asarray(op, dtype=np.float32)).astype(np.float32)
            draw_line(sim, ring, rgba=(1.0, 0.5, 0.0, 0.8))
