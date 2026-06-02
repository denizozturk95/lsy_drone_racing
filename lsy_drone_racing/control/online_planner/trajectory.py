"""observation-driven reference planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.online_planner.avoidance import (
    nudge_lateral,
    push_off_obstacles,
    reversal_turn,
)
from lsy_drone_racing.control.online_planner.timing import repair_obstacles

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.online_planner.settings import PlannerSettings
    from lsy_drone_racing.control.online_planner.state import DroneObservation

ReferenceCurve = CubicSpline


@dataclass(frozen=True)
class ReferencePlan:
    """a global plan through all remaining gates from a fixed start state."""

    curve: CubicSpline
    t_total: float
    waypoints: NDArray[np.float64]
    gate_pos_snapshot: NDArray[np.float64]
    obstacle_pos_snapshot: NDArray[np.float64]
    built_target_gate: int


def _oriented_forward(
    quat: NDArray[np.float64], gate_pos: NDArray[np.float64], reference: NDArray[np.float64]
) -> NDArray[np.float64]:
    """return the gate forward axis flipped to point along the travel direction."""
    forward = Rotation.from_quat(quat).as_matrix()[:, 0]
    if float(np.dot(forward, gate_pos - reference)) < 0.0:
        forward = -forward
    return forward


def _clearance_points(
    prev_pos: NDArray[np.float64],
    prev_forward: NDArray[np.float64],
    next_pos: NDArray[np.float64],
    next_forward: NDArray[np.float64],
    settings: PlannerSettings,
) -> list[NDArray[np.float64]]:
    """return a clearance and turn-apex waypoint when the next gate height differs."""
    next_approach = next_pos - settings.d_pre * next_forward
    next_z = float(next_approach[2])
    prev_z = float(prev_pos[2])
    if abs(next_z - prev_z) <= settings.clearance_height_delta:
        return []
    to_next = (next_pos - prev_pos)[:2]
    pf = prev_forward[:2]
    denom = float(np.linalg.norm(pf) * np.linalg.norm(to_next))
    cos_to_next = float(np.dot(pf, to_next)) / denom if denom > 1e-9 else 1.0
    if cos_to_next < -0.3:
        exit_xy = (prev_pos + settings.d_post * prev_forward)[:2]
        return reversal_turn(exit_xy, prev_pos, prev_forward, next_approach, to_next, settings)
    clearance_xy = (prev_pos + (settings.d_post + 0.60) * prev_forward)[:2]
    if next_z > prev_z:
        clearance_z = max(prev_z + 0.55, next_z - 0.05)
        apex_z = next_z - 0.05
    else:
        clearance_z = max(prev_z - 0.30, next_z + 0.15)
        apex_z = next_z + 0.05
    mid_xy = 0.5 * (clearance_xy + next_approach[:2])
    away = mid_xy - prev_pos[:2]
    away_norm = float(np.linalg.norm(away))
    if away_norm > 1e-6:
        mid_xy = mid_xy + (away / away_norm) * 0.10
    return [
        np.array([clearance_xy[0], clearance_xy[1], clearance_z]),
        np.array([mid_xy[0], mid_xy[1], apex_z]),
    ]


def _insert_exited_clearance(
    waypoints: list[NDArray[np.float64]],
    start_pos: NDArray[np.float64],
    start_vel: NDArray[np.float64],
    gates_pos: NDArray[np.float64],
    gates_quat: NDArray[np.float64],
    target_gate: int,
    settings: PlannerSettings,
) -> None:
    """re-create the post-gate arc when replanning near the just-exited gate."""
    prev_pos = gates_pos[target_gate - 1]
    if float(np.linalg.norm(start_pos - prev_pos)) >= 0.6:
        return
    prev_forward = Rotation.from_quat(gates_quat[target_gate - 1]).as_matrix()[:, 0]
    travel = np.array([start_vel[0], start_vel[1], 0.0])
    if float(np.linalg.norm(travel)) > 0.1:
        reference = float(np.dot(prev_forward, travel))
    else:
        reference = float(np.dot(prev_forward, gates_pos[target_gate] - prev_pos))
    if reference < 0.0:
        prev_forward = -prev_forward
    prev_exit = prev_pos + settings.d_post * prev_forward
    next_forward = _oriented_forward(gates_quat[target_gate], gates_pos[target_gate], prev_exit)
    points = _clearance_points(
        prev_pos, prev_forward, gates_pos[target_gate], next_forward, settings
    )
    for offset, point in enumerate(points):
        waypoints.insert(1 + offset, point)


def build_waypoints(
    start_pos: NDArray[np.float64],
    start_vel: NDArray[np.float64],
    gates_pos: NDArray[np.float64],
    gates_quat: NDArray[np.float64],
    obstacles_pos: NDArray[np.float64],
    target_gate: int,
    settings: PlannerSettings,
) -> NDArray[np.float64]:
    """build an obstacle-aware waypoint chain through the remaining gates."""
    start_pos = np.asarray(start_pos, dtype=np.float64)
    start_vel = np.asarray(start_vel, dtype=np.float64)
    gates_pos = np.asarray(gates_pos, dtype=np.float64)
    gates_quat = np.asarray(gates_quat, dtype=np.float64)
    remaining_pos = gates_pos[target_gate:]
    remaining_quat = gates_quat[target_gate:]
    obstacles_pos = np.asarray(obstacles_pos, dtype=np.float64)

    waypoints: list[NDArray[np.float64]] = [start_pos.copy()]
    if start_pos[2] < settings.liftoff_z_threshold and len(remaining_pos) > 0:
        target_z = max(settings.liftoff_height, 0.85 * float(remaining_pos[0][2]))
        toward = remaining_pos[0][:2] - start_pos[:2]
        distance = float(np.linalg.norm(toward))
        offset = toward / distance * 0.40 if distance > 1e-6 else np.zeros(2)
        waypoints.append(np.array([start_pos[0] + offset[0], start_pos[1] + offset[1], target_z]))
    elif target_gate > 0 and len(remaining_pos) > 0:
        _insert_exited_clearance(
            waypoints, start_pos, start_vel, gates_pos, gates_quat, target_gate, settings
        )

    n_gates = len(remaining_pos)
    forwards: list[NDArray[np.float64]] = []
    reference = waypoints[-1]
    for index in range(n_gates):
        gate_pos = remaining_pos[index]
        fwd = _oriented_forward(remaining_quat[index], gate_pos, reference)
        forwards.append(fwd)
        reference = gate_pos + settings.d_post * fwd
    gate_indices: set[int] = set()
    for index in range(n_gates):
        gate_pos = remaining_pos[index]
        forward = forwards[index]
        lateral = Rotation.from_quat(remaining_quat[index]).as_matrix()[:, 1]
        previous = waypoints[-1]
        approach_raw = gate_pos - settings.d_pre * forward
        exit_raw = gate_pos + settings.d_post * forward
        lateral_bias = float(np.dot((previous - approach_raw)[:2], lateral[:2]))
        bias_sign = float(np.sign(lateral_bias)) if abs(lateral_bias) > 1e-3 else 0.0
        approach = nudge_lateral(approach_raw, lateral, obstacles_pos, settings.r_obs, bias_sign)
        exit_point = nudge_lateral(exit_raw, lateral, obstacles_pos, settings.r_obs)
        far_raw = gate_pos - (settings.d_pre + 0.30) * forward
        far_approach = nudge_lateral(far_raw, lateral, obstacles_pos, settings.r_obs, bias_sign)
        behind = float(np.dot(previous - far_approach, forward)) < 0.0
        if behind and float(np.linalg.norm((far_approach - previous)[:2])) > 0.20:
            waypoints.append(far_approach)
        waypoints.extend([approach, gate_pos.copy(), exit_point])
        gate_indices.add(len(waypoints) - 2)
        if index + 1 < n_gates:
            waypoints.extend(
                _clearance_points(
                    gate_pos, forward, remaining_pos[index + 1], forwards[index + 1], settings
                )
            )

    final_post = settings.d_post + settings.d_stop
    waypoints.append(remaining_pos[-1] + final_post * forwards[-1])
    return push_off_obstacles(
        np.asarray(waypoints), gate_indices, obstacles_pos, settings.r_obs + 0.06
    )


class ReferenceManager:
    """build and globally replan the reference spline from observed poses."""

    def __init__(
        self, settings: PlannerSettings, replan_gate_delta_m: float, replan_obstacle_delta_m: float
    ):
        """create a manager with the pose-change replanning thresholds."""
        self._settings = settings
        self._gate_delta = float(replan_gate_delta_m)
        self._obstacle_delta = float(replan_obstacle_delta_m)
        self._plan: ReferencePlan | None = None

    @property
    def plan(self) -> ReferencePlan | None:
        """the active plan if one exists."""
        return self._plan

    def reset(self) -> None:
        """forget the cached plan."""
        self._plan = None

    def build(
        self,
        start_pos: NDArray[np.float64],
        start_vel: NDArray[np.float64],
        frame: DroneObservation,
    ) -> ReferencePlan:
        """build a global plan from a fixed start state through all remaining gates."""
        waypoints = build_waypoints(
            start_pos,
            start_vel,
            frame.gate_pos,
            frame.gate_quat,
            frame.obstacles_pos,
            frame.target_gate,
            self._settings,
        )
        waypoints, knot_times, curve = repair_obstacles(
            waypoints, start_vel, frame.gate_pos, frame.obstacles_pos, self._settings
        )
        return ReferencePlan(
            curve=curve,
            t_total=float(knot_times[-1]),
            waypoints=waypoints,
            gate_pos_snapshot=frame.gate_pos.copy(),
            obstacle_pos_snapshot=frame.obstacles_pos.copy(),
            built_target_gate=frame.target_gate,
        )

    def ensure_plan(self, frame: DroneObservation) -> tuple[ReferencePlan, bool]:
        """return the active plan and whether it was rebuilt this call."""
        if self._needs_plan(frame):
            self._plan = self.build(frame.pos, frame.vel, frame)
            return self._plan, True
        return self._plan, False

    def _needs_plan(self, frame: DroneObservation) -> bool:
        plan = self._plan
        if plan is None or frame.target_gate != plan.built_target_gate:
            return True
        remaining = frame.gate_pos[frame.target_gate :]
        snapshot = plan.gate_pos_snapshot[frame.target_gate :]
        if len(remaining) and float(np.max(np.linalg.norm(remaining - snapshot, axis=1))) > (
            self._gate_delta
        ):
            return True
        if (
            len(frame.obstacles_pos)
            and float(
                np.max(np.linalg.norm(frame.obstacles_pos - plan.obstacle_pos_snapshot, axis=1))
            )
            > self._obstacle_delta
        ):
            return True
        return False
