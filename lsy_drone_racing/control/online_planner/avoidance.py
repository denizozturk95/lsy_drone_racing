"""obstacle-aware waypoint and timing helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.online_planner.settings import PlannerSettings


def nudge_lateral(
    point: NDArray[np.floating],
    lateral: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    r_obs: float,
    bias_sign: float = 0.0,
) -> NDArray[np.floating]:
    """shift a waypoint along the gate lateral axis until it clears all obstacles."""
    lateral_xy = lateral[:2]
    lateral_norm = float(np.linalg.norm(lateral_xy))
    if lateral_norm < 1e-6:
        return _nudge_radial(point, lateral, obstacles_pos, r_obs)
    lateral_unit = lateral_xy / lateral_norm
    nudged_point = point.copy()
    margin = r_obs + 0.02
    for _ in range(4):
        offenders = [
            obstacle
            for obstacle in obstacles_pos
            if np.linalg.norm(nudged_point[:2] - obstacle[:2]) < margin
        ]
        if not offenders:
            break
        closest = min(offenders, key=lambda o: np.linalg.norm(nudged_point[:2] - o[:2]))
        delta = nudged_point[:2] - closest[:2]
        delta_dot_lateral = float(np.dot(delta, lateral_unit))
        discriminant = delta_dot_lateral**2 + margin**2 - float(np.dot(delta, delta))
        if discriminant < 0:
            return _nudge_radial(point, lateral, obstacles_pos, r_obs)
        root = np.sqrt(discriminant)
        positive_offset = -delta_dot_lateral + root
        negative_offset = -delta_dot_lateral - root
        if (
            bias_sign > 0
            and positive_offset > 0
            and abs(positive_offset) <= 1.5 * abs(negative_offset)
        ):
            offset = positive_offset
        elif (
            bias_sign < 0
            and negative_offset < 0
            and abs(negative_offset) <= 1.5 * abs(positive_offset)
        ):
            offset = negative_offset
        else:
            offset = (
                positive_offset if abs(positive_offset) < abs(negative_offset) else negative_offset
            )
        nudged_point[:2] = nudged_point[:2] + offset * lateral_unit
    return nudged_point


def _nudge_radial(
    point: NDArray[np.floating],
    lateral: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    r_obs: float,
) -> NDArray[np.floating]:
    """push a waypoint radially away from the nearest obstacle as a fallback."""
    nudged_point = point.copy()
    for _ in range(4):
        distances = [float(np.linalg.norm(nudged_point[:2] - o[:2])) for o in obstacles_pos]
        if not distances or min(distances) >= r_obs:
            break
        nearest = obstacles_pos[int(np.argmin(distances))]
        delta = nudged_point[:2] - nearest[:2]
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm < 1e-6:
            fallback = lateral[:2]
            fallback_norm = float(np.linalg.norm(fallback))
            direction = fallback / fallback_norm if fallback_norm > 1e-6 else np.array([1.0, 0.0])
        else:
            direction = delta / delta_norm
        nudged_point[:2] = nearest[:2] + direction * (r_obs + 0.05)
    return nudged_point


def push_off_obstacles(
    waypoints: NDArray[np.floating],
    protected: set[int],
    obstacles_pos: NDArray[np.floating],
    clearance: float,
) -> NDArray[np.floating]:
    """push every non-protected waypoint radially out of each obstacle keep-out disk."""
    if len(obstacles_pos) == 0:
        return waypoints
    out = waypoints.copy()
    for index in range(len(out)):
        if index in protected:
            continue
        for _ in range(4):
            moved = False
            for obstacle in obstacles_pos:
                delta = out[index, :2] - obstacle[:2]
                norm = float(np.linalg.norm(delta))
                if norm < clearance:
                    direction = delta / norm if norm > 1e-6 else np.array([1.0, 0.0])
                    out[index, :2] = obstacle[:2] + direction * clearance
                    moved = True
            if not moved:
                break
    return out


def reversal_turn(
    base_xy: NDArray[np.floating],
    prev_pos: NDArray[np.floating],
    prev_forward: NDArray[np.floating],
    next_approach: NDArray[np.floating],
    to_next: NDArray[np.floating],
    settings: PlannerSettings,
) -> list[NDArray[np.floating]]:
    """swing wide to the next-gate side for a reversal transition."""
    prev_z = float(prev_pos[2])
    next_z = float(next_approach[2])
    perp = np.array([-prev_forward[1], prev_forward[0]])
    perp_norm = float(np.linalg.norm(perp))
    perp = perp / perp_norm if perp_norm > 1e-6 else np.array([0.0, 1.0])
    if float(np.dot(perp, to_next)) < 0.0:
        perp = -perp
    swing_xy = np.asarray(base_xy, dtype=np.float64) + 0.55 * perp
    apex_xy = 0.5 * (swing_xy + next_approach[:2]) + 0.10 * perp
    swing_z = prev_z + 0.30 * (next_z - prev_z)
    apex_z = prev_z + 0.70 * (next_z - prev_z)
    return [
        np.array([swing_xy[0], swing_xy[1], swing_z]),
        np.array([apex_xy[0], apex_xy[1], apex_z]),
    ]
