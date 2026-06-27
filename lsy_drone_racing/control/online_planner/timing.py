"""spline time-parameterization and obstacle-clearance repair."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.online_planner.settings import PlannerSettings


def obstacle_slowdown(
    waypoints: NDArray[np.floating],
    segment_times: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    slow_radius: float = 0.32,
) -> NDArray[np.floating]:
    """stretch segment durations that pass close to an obstacle for safer tracking."""
    if len(obstacles_pos) == 0:
        return segment_times
    segment_times = np.asarray(segment_times, dtype=np.float64).copy()
    for index in range(len(waypoints) - 1):
        start = waypoints[index, :2]
        end = waypoints[index + 1, :2]
        segment = end - start
        segment_sq = float(np.dot(segment, segment))
        if segment_sq < 1e-9:
            continue
        min_distance = np.inf
        for obstacle in obstacles_pos:
            ratio = float(np.clip(np.dot(obstacle[:2] - start, segment) / segment_sq, 0.0, 1.0))
            closest = start + ratio * segment
            min_distance = min(min_distance, float(np.linalg.norm(obstacle[:2] - closest)))
        if min_distance < slow_radius:
            segment_times[index] *= 1.0 + 0.6 * (slow_radius - min_distance) / slow_radius
    return segment_times


def turn_slowdown(
    waypoints: NDArray[np.floating],
    segment_times: NDArray[np.floating],
    min_sharpness: float = 0.4,
    slow_gain: float = 0.6,
) -> NDArray[np.floating]:
    """stretch the segments around sharp corners so reversals are flown slower."""
    segment_times = np.asarray(segment_times, dtype=np.float64).copy()
    diffs = np.diff(waypoints, axis=0)
    for index in range(1, len(waypoints) - 1):
        before, after = diffs[index - 1], diffs[index]
        n_before, n_after = float(np.linalg.norm(before)), float(np.linalg.norm(after))
        if n_before < 1e-6 or n_after < 1e-6:
            continue
        cos_angle = float(np.dot(before, after) / (n_before * n_after))
        sharpness = (1.0 - cos_angle) / 2.0
        if sharpness <= min_sharpness:
            continue
        factor = 1.0 + slow_gain * sharpness
        segment_times[index - 1] *= factor
        segment_times[index] *= factor
    return segment_times


def build_spline(
    waypoints: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    settings: PlannerSettings,
) -> tuple[NDArray[np.floating], CubicSpline]:
    """time-parameterize the waypoints into a clamped cubic spline."""
    start_vel = np.asarray(start_vel, dtype=np.float64)
    gates_pos = np.asarray(gates_pos, dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    inter_speed = settings.v_cruise_inter if settings.v_cruise_inter > 0 else settings.v_cruise
    cold_start = float(np.linalg.norm(start_vel)) < 0.3
    if cold_start and float(waypoints[0, 2]) <= settings.liftoff_z_threshold:
        start_vel = start_vel.copy()
        start_vel[2] = max(float(start_vel[2]), 0.0)
    segment_times = np.empty(len(segment_lengths), dtype=np.float64)
    for index in range(len(segment_lengths)):
        peri = any(
            float(np.linalg.norm(waypoints[index, :2] - gate[:2])) < settings.peri_gate_radius
            or float(np.linalg.norm(waypoints[index + 1, :2] - gate[:2]))
            < settings.peri_gate_radius
            for gate in gates_pos
        )
        speed = settings.v_cruise if peri else inter_speed
        segment_times[index] = max(segment_lengths[index] / speed, settings.t_min_seg)
        if cold_start and index < 2:
            segment_times[index] = max(segment_times[index], settings.cold_start_min_seg)
    segment_times = obstacle_slowdown(waypoints, segment_times, obstacles_pos)
    segment_times = turn_slowdown(waypoints, segment_times)
    segment_times = _cap_peak_velocity(
        waypoints, segment_times, start_vel, settings.max_speed, skip=2
    )
    knot_times = np.concatenate([[0.0], np.cumsum(segment_times)])
    bc = ((1, start_vel), (1, np.zeros(3)))
    return knot_times, CubicSpline(knot_times, waypoints, bc_type=bc)


def _cap_peak_velocity(
    waypoints: NDArray[np.floating],
    segment_times: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    max_speed: float,
    skip: int = 0,
) -> NDArray[np.floating]:
    """stretch segments whose spline velocity exceeds the max speed."""
    bc = ((1, np.asarray(start_vel, dtype=np.float64)), (1, np.zeros(3)))
    for _ in range(4):
        knot_times = np.concatenate([[0.0], np.cumsum(segment_times)])
        velocity = CubicSpline(knot_times, waypoints, bc_type=bc).derivative(1)
        worst = 1.0
        for index in range(skip, len(segment_times)):
            sample = np.linspace(knot_times[index], knot_times[index + 1], 8)
            peak = float(np.max(np.linalg.norm(velocity(sample), axis=1)))
            if peak > max_speed:
                ratio = peak / max_speed
                segment_times[index] *= ratio
                worst = max(worst, ratio)
        if worst <= 1.02:
            break
    return segment_times


def repair_obstacles(
    waypoints: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    settings: PlannerSettings,
) -> tuple[NDArray[np.floating], NDArray[np.floating], CubicSpline]:
    """insert push-out waypoints until the sampled spline clears every obstacle."""
    knot_times, curve = build_spline(waypoints, start_vel, gates_pos, obstacles_pos, settings)
    if len(obstacles_pos) == 0:
        return waypoints, knot_times, curve
    margin = settings.r_obs + 0.12
    for _ in range(6):
        sample_t = np.linspace(0.0, float(knot_times[-1]), max(60, 20 * len(waypoints)))
        points = np.asarray(curve(sample_t), dtype=np.float64)
        worst_index, worst_distance, worst_obstacle = -1, settings.r_obs, None
        for obstacle in obstacles_pos:
            distances = np.linalg.norm(points[:, :2] - obstacle[:2], axis=1)
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) < worst_distance:
                worst_index, worst_distance, worst_obstacle = (
                    nearest,
                    float(distances[nearest]),
                    obstacle,
                )
        if worst_obstacle is None:
            break
        violation = points[worst_index]
        direction = violation[:2] - worst_obstacle[:2]
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
        repair = np.array(
            [
                worst_obstacle[0] + direction[0] * margin,
                worst_obstacle[1] + direction[1] * margin,
                float(violation[2]),
            ]
        )
        segment = int(
            np.clip(np.searchsorted(knot_times, sample_t[worst_index]), 1, len(waypoints))
        )
        waypoints = np.insert(waypoints, segment, repair, axis=0)
        knot_times, curve = build_spline(waypoints, start_vel, gates_pos, obstacles_pos, settings)
    return waypoints, knot_times, curve


def enforce_geofence(
    waypoints: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    settings: PlannerSettings,
) -> tuple[NDArray[np.floating], NDArray[np.floating], CubicSpline]:
    """Clamp the planned path into the arena (X/Y) so a wide curve can never leave the zone.

    No-op unless ``settings.geofence_margin`` > 0 and both arena bounds are set. Otherwise the
    fence sits ``geofence_margin`` m inside [arena_low, arena_high] in X/Y -- the margin reserves
    room for downstream tracking overshoot. Two passes: (1) clamp every waypoint to the fence,
    then (2) pull in any residual cubic-overshoot bulge between knots by inserting a fence-side
    waypoint. The drone start (index 0) and every gate crossing are protected so the path still
    begins at the drone and flies through each gate; bulges that sit on a gate are left alone.
    Z is intentionally not fenced (the floor/ceiling are handled by gate heights and liftoff).
    """
    knot_times, curve = build_spline(waypoints, start_vel, gates_pos, obstacles_pos, settings)
    if (
        settings.geofence_margin <= 0.0
        or settings.arena_low is None
        or settings.arena_high is None
    ):
        return waypoints, knot_times, curve

    low = np.asarray(settings.arena_low, dtype=np.float64)[:2] + settings.geofence_margin
    high = np.asarray(settings.arena_high, dtype=np.float64)[:2] - settings.geofence_margin
    gates_pos = np.asarray(gates_pos, dtype=np.float64)
    gates_xy = gates_pos[:, :2] if len(gates_pos) else np.empty((0, 2), dtype=np.float64)
    protect_r = 0.15  # waypoints/samples within this of a gate centre are gate crossings

    def _on_gate(xy: NDArray[np.floating]) -> bool:
        return bool(
            len(gates_xy) and float(np.min(np.linalg.norm(gates_xy - xy, axis=1))) < protect_r
        )

    waypoints = np.asarray(waypoints, dtype=np.float64).copy()
    moved = False
    for index in range(1, len(waypoints)):  # index 0 is the drone start -> never move it
        xy = waypoints[index, :2]
        if _on_gate(xy):
            continue
        clamped = np.clip(xy, low, high)
        if not np.array_equal(clamped, xy):
            waypoints[index, :2] = clamped
            moved = True
    if moved:
        knot_times, curve = build_spline(waypoints, start_vel, gates_pos, obstacles_pos, settings)

    inner_low, inner_high = low + 0.03, high - 0.03  # insert just inside so the bulge stays in
    for _ in range(6):
        sample_t = np.linspace(0.0, float(knot_times[-1]), max(80, 20 * len(waypoints)))
        pts = np.asarray(curve(sample_t), dtype=np.float64)
        over = np.max(np.maximum(pts[:, :2] - high, low - pts[:, :2]), axis=1)
        if len(gates_xy):  # ignore bulges that sit on a gate crossing (must reach the gate)
            on_gate = np.min(
                np.linalg.norm(pts[:, None, :2] - gates_xy[None, :, :], axis=2), axis=1
            )
            over = np.where(on_gate < protect_r, 0.0, over)
        worst = int(np.argmax(over))
        if float(over[worst]) <= 0.01:
            break
        viol = pts[worst]
        repair = np.array(
            [
                float(np.clip(viol[0], inner_low[0], inner_high[0])),
                float(np.clip(viol[1], inner_low[1], inner_high[1])),
                float(viol[2]),
            ]
        )
        segment = int(np.clip(np.searchsorted(knot_times, sample_t[worst]), 1, len(waypoints)))
        waypoints = np.insert(waypoints, segment, repair, axis=0)
        knot_times, curve = build_spline(waypoints, start_vel, gates_pos, obstacles_pos, settings)
    return waypoints, knot_times, curve
