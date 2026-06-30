"""v10.4-owned global planner: v8's gate-funnel planner with trimmed clearance geometry.

A geometry audit of the level2 plan (arc 11.27 m vs a 6.52 m straight gate chain) traced most
of the recoverable detour to the forward-clearance branch of ``_clearance_points``, whose
hardcoded +0.60 m run-out extension fires on ~70-90 deg climbing/descending turns and points
AWAY from the next gate (g0->g1 fires at cos ~ +0.05, g1->g2 at +0.32; the latter detours to
y=1.39 while gate 2 sits at y=-0.25 and forces an extra obstacle repair). Three audited trims,
re-validated on the full pipeline (gate crossings stay dead-centre, max|lat| <= 0.023 within
+/-0.15 m of every gate plane; min obstacle clearance 0.207 m vs r_obs 0.20):

1. The run-out extension is scaled by the turn alignment: 0.60 * max(cos_to_next, 0).
   Genuinely straight transitions keep the full straight climbing run-out; perpendicular turns
   get only d_post. Self-guarding, saves ~1.05 m (~0.38 s) on level2.
2. The turn-apex is no longer pushed 0.10 m outward (it only fattened an already-wide arc).
3. Climbs no longer overshoot to prev_z + 0.55 before dipping back (every level2 hop is 0.5 m;
   the overshoot was protection for >0.6 m hops, which the audit found nowhere on this track).

Everything else -- waypoint chain, gate-post funnels, obstacle nudges/repair, the reversal
U-turn branch (geometrically necessary, ~0.03 m recoverable), spline timing -- is v8's,
imported unchanged. The copies below exist because the v8 planner is shared by every v8+
controller and must not change behaviour under them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.common_controller_v2.avoidance import (
    nudge_lateral,
    push_off_obstacles,
    reversal_turn,
)
from lsy_drone_racing.control.controller_core_v8.timing import repair_obstacles
from lsy_drone_racing.control.controller_core_v8.trajectory import (
    ReferenceManager as _ReferenceManager,
)
from lsy_drone_racing.control.controller_core_v8.trajectory import (
    ReferencePlan,
    _oriented_forward,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import Rotation  # noqa: F401  (typing only)

    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation
    from lsy_drone_racing.control.controller_core_v8.settings import PlannerSettings

from scipy.spatial.transform import Rotation


def _clearance_points(
    prev_pos: NDArray[np.float64],
    prev_forward: NDArray[np.float64],
    next_pos: NDArray[np.float64],
    next_forward: NDArray[np.float64],
    settings: PlannerSettings,
) -> list[NDArray[np.float64]]:
    """v8's clearance/turn-apex insertion with the three audited trims (see module docstring)."""
    next_approach = next_pos - settings.d_pre * next_forward
    next_z = float(next_approach[2])
    prev_z = float(prev_pos[2])
    if abs(next_z - prev_z) <= settings.clearance_height_delta:
        return []
    to_next = (next_pos - prev_pos)[:2]
    pf = prev_forward[:2]
    denom = float(np.linalg.norm(pf) * np.linalg.norm(to_next))
    cos_to_next = float(np.dot(pf, to_next)) / denom if denom > 1e-9 else 1.0
    if cos_to_next < -0.3:  # clear reversal: keep v8's wide U-turn swing (necessary geometry)
        exit_xy = (prev_pos + settings.d_post * prev_forward)[:2]
        return reversal_turn(exit_xy, prev_pos, prev_forward, next_approach, to_next, settings)
    # Forward turn: the straight climbing run-out is only useful when the next gate actually
    # lies ahead -- scale the extension by the alignment instead of a fixed 0.60 m, dialled by
    # clr_ext_min (0 = full trim, 1 = v8's fixed 0.60). (The two other audited trims --
    # dropping the climb-z overshoot and the 0.10 m apex push -- were measured TOO HOT in
    # flight: paired eval 13/20 with gate-3 U-turn fails and extra launch crashes; the cubic
    # spline is global, so they also reshape neighbouring segments. Keep v8's values for both.)
    extension = 0.60 * max(cos_to_next, float(getattr(settings, "clr_ext_min", 1.0)))
    clearance_xy = (prev_pos + (settings.d_post + extension) * prev_forward)[:2]
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
    """v8's post-gate replan arc, byte-for-byte, but using the trimmed _clearance_points."""
    prev_pos = gates_pos[target_gate - 1]
    if float(np.linalg.norm(start_pos - prev_pos)) >= 0.6:
        return
    prev_forward = Rotation.from_quat(gates_quat[target_gate - 1]).as_matrix()[:, 0]
    if settings.orient_gates_to_travel:
        travel = np.array([start_vel[0], start_vel[1], 0.0])
        if float(np.linalg.norm(travel)) > 0.1:
            reference = float(np.dot(prev_forward, travel))
        else:
            reference = float(np.dot(prev_forward, gates_pos[target_gate] - prev_pos))
        if reference < 0.0:
            prev_forward = -prev_forward
    prev_exit = prev_pos + settings.d_post * prev_forward
    next_forward = _oriented_forward(
        gates_quat[target_gate], gates_pos[target_gate], prev_exit, settings.orient_gates_to_travel
    )
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
    """v8's obstacle-aware waypoint chain, byte-for-byte, over the trimmed clearance geometry."""
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
        fwd = _oriented_forward(
            remaining_quat[index], gate_pos, reference, settings.orient_gates_to_travel
        )
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


class ReferenceManager(_ReferenceManager):
    """v8's ReferenceManager building plans through the trimmed clearance geometry."""

    def build(
        self,
        start_pos: NDArray[np.float64],
        start_vel: NDArray[np.float64],
        frame: DroneObservation,
    ) -> ReferencePlan:
        """v8's build, byte-for-byte, calling the v10.4 build_waypoints."""
        planning_obstacles = self._planning_obstacles(frame)
        waypoints = build_waypoints(
            start_pos,
            start_vel,
            frame.gate_pos,
            frame.gate_quat,
            planning_obstacles,
            frame.target_gate,
            self._settings,
        )
        waypoints, knot_times, curve = repair_obstacles(
            waypoints, start_vel, frame.gate_pos, planning_obstacles, self._settings
        )
        return ReferencePlan(
            curve=curve,
            t_total=float(knot_times[-1]),
            waypoints=waypoints,
            gate_pos_snapshot=frame.gate_pos.copy(),
            obstacle_pos_snapshot=frame.obstacles_pos.copy(),
            built_target_gate=frame.target_gate,
        )
