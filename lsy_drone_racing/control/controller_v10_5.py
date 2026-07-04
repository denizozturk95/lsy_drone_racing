"""Gate-aware MPCC merging v10.4's launch with v10.2's dynamics-aware progress anchor (v10.5).

Thin subclass of ControllerV104. Substitutes geometric projection for a dynamics-feasible
predicted-progress anchor, preventing fold-teleport failures on sharp slaloms.
REQUIRES the acados environment (``pixi run``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.controller_v10_4 import ControllerV104
from lsy_drone_racing.control.controller_core_v10_5.arc_path import GateArcPath
from lsy_drone_racing.control.controller_core_v10_5.mpcc import MPCC
from lsy_drone_racing.control.controller_core_v10_5.settings import ControllerSettings

if TYPE_CHECKING:
    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation


class ControllerV105(ControllerV104):
    """v10.4's launch-optimised racing MPCC with v10.2's dynamics-aware progress anchor."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v10.4, then swap in the v10.5 MPCC (predicted-progress) and the anchor band."""
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
        self._v_gate_react = mpcc.v_gate_react
        self._react_delta = mpcc.react_delta_m
        self._react_v_pre, self._react_v_post = mpcc.react_v_pre, mpcc.react_v_post
        self._proj_band = mpcc.proj_band_m
        self._gate_nominal: np.ndarray | None = None
        self._path: GateArcPath | None = None
        self._s = 0.0
        self._reset_anchor_telemetry()

    def _reset_anchor_telemetry(self) -> None:
        """Clear the per-episode anchor diagnostics (max jump, jump samples, band-edge rate)."""
        self._anchor_prev_s: float | None = None
        self._anchor_jumps: list[float] = []
        self._band_edge_hits = 0
        self._band_calls = 0

    def reset(self) -> None:
        """Reset v10.4 state (incl. the nominal gate snapshot) plus the anchor telemetry."""
        super().reset()
        self._reset_anchor_telemetry()

    def _track_action(self, frame: DroneObservation) -> np.ndarray:
        """v10.4's flow verbatim, but the geometric anchor becomes the predicted-progress anchor."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if self._gate_nominal is None:
            self._gate_nominal = np.asarray(frame.gate_pos, dtype=np.float64).copy()
        new_plan = rebuilt or self._path is None
        if new_plan:
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

    def anchor_telemetry(self) -> dict[str, float | int]:
        """Return per-episode progress-anchor diagnostics (max jump, band edge rate)."""
        jumps = np.asarray(self._anchor_jumps, dtype=np.float64)
        edge_rate = (self._band_edge_hits / self._band_calls) if self._band_calls else 0.0
        return {
            "n_steps": int(jumps.size),
            "max_jump_m": float(jumps.max()) if jumps.size else 0.0,
            "p99_jump_m": float(np.percentile(jumps, 99)) if jumps.size else 0.0,
            "n_jumps_gt_1m": int((jumps > 1.0).sum()),
            "band_calls": int(self._band_calls),
            "band_edge_rate": edge_rate,
        }
