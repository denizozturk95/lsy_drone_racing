"""Arena-aware tunnel MPCC (controller_v15).

v11.2 = v11 (tunnel-constrained time-optimal MPCC) made to keep the drone inside the REAL flight
arena, which is slightly smaller than the sim safety box -- the run is aborted the instant the
drone leaves it. The measured cause (probe: scripts/_arena_probe.py) is that the reference SPLINE
itself routes to y ~ 1.4-1.6 / x ~ 1.55 on the gate-1 exit arc, at/past the real-arena edge, and
the drone tracks it; the worst excursions are also inertial (the drone arrives too hot to turn
inside the wall). v11.2 prices/keeps that in with three coordinated SOFT mechanisms:

1. SPLINE CLIP -- the reference point handed to the OCP each stage is clipped to the keep-in box
   (sim box shrunk by ARENA_INSET), so the contouring reference and tunnel centre never target a
   point outside the real arena (controller_core_v15.arc_path.eval).
2. ARENA SPEED CAP -- the curvature/reveal speed profile is capped to V_ARENA wherever the
   reference enters the border ramp band, so the drone arrives slow enough to turn in bounds
   (controller_core_v15.arc_path; reuses v11's reveal/obstacle cap machinery).
3. MPC POSITION BARRIER -- a quadratic-hinge keep-in penalty on the predicted world (x, y) in the
   OCP cost, the soft lateral margin on top of (1)+(2) (controller_core_v15.mpcc).

Only the OCP cost, the path view, and the solver namespace (``controller_v11_2``) change; the planner,
launch, anchor, replan rebase, and telemetry are v11's, inherited unchanged. The arena box is
static, so no new per-stage OCP parameters are needed -- the NAVIGATE flow below is v11's copied
with the arena path view and arena MPCC woven in (the house pattern). REQUIRES the acados
environment -- run under ``pixi run``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.controller_v11 import ControllerV11
from lsy_drone_racing.control.controller_core_v8.trajectory import gate_post_obstacles
from lsy_drone_racing.control.controller_core_v15.arc_path import ArenaTunnelArcPath
from lsy_drone_racing.control.controller_core_v15.mpcc import MPCC
from lsy_drone_racing.control.controller_core_v15.settings import ControllerSettings

if TYPE_CHECKING:
    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation


class ControllerV15(ControllerV11):
    """v11's tunnel MPCC with arena keep-in baked into the spline, speed profile, and OCP cost."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v11, then swap in the v11.2 settings and arena-aware MPCC (own namespace)."""
        super().__init__(obs, info, config)
        self._settings = ControllerSettings()
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCC(self._settings.mpcc, a_max)

    def _track_action(self, frame: DroneObservation) -> np.ndarray:
        """v11's flow with the arena-aware path view (spline clip + border speed cap)."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if self._gate_nominal is None:
            self._gate_nominal = np.asarray(frame.gate_pos, dtype=np.float64).copy()
        new_plan = rebuilt or self._path is None
        if new_plan:
            first = max(frame.target_gate, 0)
            posts = gate_post_obstacles(
                frame.gate_pos, frame.gate_quat, 0, self._settings.planner.gate_post_offset
            )
            path = ArenaTunnelArcPath(
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
