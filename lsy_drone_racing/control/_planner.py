"""Shared gate-aware cubic-spline planner.

Used by both the state-mode and attitude-MPC controllers. Given a drone state
and observed gate/obstacle poses, builds a clamped cubic spline threading the
approach/center/exit waypoints of each remaining gate, with lateral midpoints
inserted where a segment passes too close to an observed obstacle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class PlannerConfig:
    d_pre: float = 0.45
    d_post: float = 0.30
    d_stop: float = 0.30
    v_cruise: float = 0.8
    v_cruise_inter: float = 0.0  # inter-gate cruise (>v_cruise to speed up). 0 disables.
    t_min_seg: float = 0.4
    r_obs: float = 0.28
    # Time-optimal refinement caps (0 to disable). The heuristic refiner
    # iterates per-segment toward target utilization; the slsqp refiner
    # solves a small NLP (min sum(seg_t) s.t. peak vel/accel caps).
    max_vel: float = 0.0
    max_accel: float = 0.0
    use_slsqp: bool = False
    # M3-lite: replace the cubic interpolator with a quintic (k=5) spline.
    # Quintic has continuous jerk (C⁴) across knots — removing the cubic's
    # discontinuous-snap spikes that force the MPC to over-brake at waypoint
    # transitions. Time allocation and waypoint sequence are unchanged.
    use_quintic: bool = False
    # Path 1: drop conservative clearance + turn_apex waypoints between gates.
    # The MPC's wing soft constraints (current-gate L/R/T/B) prevent frame
    # clipping on the entry side; if we also trust them on the exit side,
    # the ~0.5–0.8 s per transition spent traversing clearance points vanishes.
    skip_clearance: bool = False


@dataclass
class Plan:
    waypoints: np.ndarray  # (n, 3)
    t_knots: np.ndarray  # (n,)
    pos_spline: CubicSpline
    vel_spline: CubicSpline
    t_total: float


def build_plan(
    start_pos: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    gates_quat: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    target_gate: int,
    cfg: PlannerConfig = PlannerConfig(),
) -> Plan:
    """Build a clamped cubic spline through gates ``[target_gate:]`` from ``start_pos``."""
    start_pos = np.asarray(start_pos, dtype=np.float64)
    start_vel = np.asarray(start_vel, dtype=np.float64)
    gates_pos = np.asarray(gates_pos, dtype=np.float64)
    gates_quat = np.asarray(gates_quat, dtype=np.float64)
    obstacles_pos = np.asarray(obstacles_pos, dtype=np.float64)

    remaining_pos = gates_pos[target_gate:]
    remaining_quat = gates_quat[target_gate:]
    wps = _build_waypoints(start_pos, remaining_pos, remaining_quat, obstacles_pos, cfg)
    # Replan guard: if the drone has just passed a gate (target_gate > 0) and
    # is still within ~0.6 m of that gate, the cubic spline built from the
    # current velocity can dip back through the exited gate's frame. Prepend
    # the same clearance + turn_apex waypoints that the forward build would
    # have produced for that transition, pinning the spline above (or below)
    # the exited frame.
    if target_gate > 0 and target_gate < len(gates_pos) and not cfg.skip_clearance:
        prev_gp = gates_pos[target_gate - 1]
        if float(np.linalg.norm(start_pos - prev_gp)) < 0.6:
            extras = _exited_gate_clearance(
                prev_gp,
                gates_quat[target_gate - 1],
                gates_pos[target_gate],
                gates_quat[target_gate],
                obstacles_pos,
                cfg,
            )
            if extras is not None:
                wps = np.vstack([wps[:1], extras, wps[1:]])
    t, spline = _build_spline(wps, start_vel, obstacles_pos, cfg, gates_pos=gates_pos)
    return Plan(
        waypoints=wps,
        t_knots=t,
        pos_spline=spline,
        vel_spline=spline.derivative(1),
        t_total=float(t[-1]),
    )


def _exit_axis_obstructed(
    gp: NDArray[np.floating],
    x_axis: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    r_obs: float,
    max_dist: float,
) -> bool:
    """True if any obstacle sits within ``r_obs`` of the gate's exit axis up to
    ``max_dist`` m past the gate — i.e. a straight clearance detour along the
    axis would graze the obstacle and need to be routed further out."""
    if len(obstacles_pos) == 0:
        return False
    x2 = x_axis[:2]
    xn = float(np.linalg.norm(x2))
    if xn < 1e-6:
        return False
    x_u = x2 / xn
    for o in obstacles_pos:
        rel = (o[:2] - gp[:2])
        along = float(np.dot(rel, x_u))
        if along < 0.0 or along > max_dist:
            continue
        lateral = float(np.linalg.norm(rel - along * x_u))
        if lateral < r_obs:
            return True
    return False


def _clearance_distance(
    gp: NDArray[np.floating],
    x_axis: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    r_obs: float,
) -> float:
    """Return the past-exit clearance distance. Short (0.35) when the exit axis
    is obstacle-free, long (0.60) when an obstacle sits along the axis and the
    clearance must push past it."""
    if _exit_axis_obstructed(gp, x_axis, obstacles_pos, r_obs + 0.08, 1.0):
        return 0.60
    return 0.35


def _exited_gate_clearance(
    prev_gp: NDArray[np.floating],
    prev_quat: NDArray[np.floating],
    next_gp: NDArray[np.floating],
    next_quat: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    cfg: PlannerConfig,
) -> NDArray[np.floating] | None:
    """Return clearance + turn_apex waypoints for a recently-exited gate.

    Mirrors the logic in ``_build_waypoints`` that runs between gate i's exit
    and gate i+1's approach, but callable standalone for use on replans where
    the start position is near the just-exited gate.
    """
    if abs(float(next_gp[2]) - float(prev_gp[2])) <= 0.15:
        return None
    prev_rot = R.from_quat(prev_quat).as_matrix()
    prev_x_axis = prev_rot[:, 0]
    next_rot = R.from_quat(next_quat).as_matrix()
    next_x_axis = next_rot[:, 0]
    next_approach = next_gp - cfg.d_pre * next_x_axis
    clearance_xy = (prev_gp + (cfg.d_post + 0.60) * prev_x_axis)[:2]
    if float(next_gp[2]) > float(prev_gp[2]):
        z_c = max(float(prev_gp[2]) + 0.55, float(next_gp[2]) - 0.05)
        z_apex = float(next_gp[2]) + 0.05
    else:
        z_c = min(float(prev_gp[2]) - 0.55, float(next_gp[2]) + 0.05)
        z_apex = float(next_gp[2]) - 0.05
    clearance = np.array([clearance_xy[0], clearance_xy[1], z_c])
    mid_xy = 0.5 * (clearance_xy + next_approach[:2])
    away = mid_xy - prev_gp[:2]
    away_n = float(np.linalg.norm(away))
    if away_n > 1e-6:
        mid_xy = mid_xy + (away / away_n) * 0.10
    turn_apex = np.array([mid_xy[0], mid_xy[1], z_apex])
    return np.stack([clearance, turn_apex])


def _build_waypoints(
    start_pos: NDArray[np.floating],
    gates_pos: NDArray[np.floating],
    gates_quat: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    cfg: PlannerConfig,
) -> NDArray[np.floating]:
    wps: list[NDArray[np.floating]] = [start_pos.copy()]
    # Lift-off waypoint when the drone is on the ground: climb in place to
    # half the first gate's height before starting horizontal motion.
    if start_pos[2] < 0.15 and len(gates_pos) > 0:
        first_gate_z = float(gates_pos[0][2])
        target_z = max(0.4, 0.5 * first_gate_z)
        toward_first = gates_pos[0][:2] - start_pos[:2]
        n = float(np.linalg.norm(toward_first))
        offset_xy = (toward_first / n * 0.15) if n > 1e-6 else np.zeros(2)
        liftoff = np.array([start_pos[0] + offset_xy[0], start_pos[1] + offset_xy[1], target_z])
        wps.append(liftoff)
    n_gates = len(gates_pos)
    for gi, (gp, gq) in enumerate(zip(gates_pos, gates_quat)):
        rot = R.from_quat(gq).as_matrix()
        x_axis, y_axis = rot[:, 0], rot[:, 1]
        approach_raw = gp - cfg.d_pre * x_axis
        exit_raw = gp + cfg.d_post * x_axis
        # Bias the approach nudge toward whichever side of the gate y-axis
        # the previous waypoint lies on, so near-tie nudges (obstacle exactly
        # on approach axis) keep the drone on its incoming side and avoid an
        # S-curve into the gate.
        prev_wp = wps[-1]
        bias = float(np.dot((prev_wp - approach_raw)[:2], y_axis[:2]))
        bias_sign = np.sign(bias) if abs(bias) > 1e-3 else 0.0
        approach = _nudge_lateral(
            approach_raw, y_axis, obstacles_pos, cfg.r_obs, bias_sign=bias_sign
        )
        exit_ = _nudge_lateral(exit_raw, y_axis, obstacles_pos, cfg.r_obs)
        # If the approach had to be nudged laterally, the spline tangent at
        # the gate blends an off-axis arc with the on-axis exit — giving a
        # diagonal entry that clips the gate frame. Insert a short on-axis
        # "near-gate" waypoint 0.15 m before the gate so the tangent at the
        # gate center is forced to align with the gate x-axis.
        nudge_dist = float(np.linalg.norm((approach - approach_raw)[:2]))
        if nudge_dist > 0.05:
            near_gate = gp - 0.15 * x_axis
            wps.extend([approach, near_gate, gp.copy(), exit_])
        else:
            wps.extend([approach, gp.copy(), exit_])
        # If the next gate is significantly above/below, the tangent at the
        # spline's exit points toward it — the drone climbs/descends while
        # returning through the just-passed gate's plane and its bounding
        # box clips the horizontal frame bars (frame opening is ±0.2, frame
        # outer is ±0.36 from center). Insert a post-exit "clearance"
        # waypoint further along the exit x-axis AT A Z THAT PLACES the
        # subsequent line-to-next-approach *above* the frame top. Concretely:
        # the clearance sits 0.35 m past the exit, with z chosen so linear
        # interpolation to the next approach crosses the gate plane at
        # gate_z + 0.40 m (frame outer-edge top is gate_z + 0.36 m).
        if gi + 1 < n_gates and not cfg.skip_clearance:
            next_gp = gates_pos[gi + 1]
            next_rot = R.from_quat(gates_quat[gi + 1]).as_matrix()
            next_x_axis = next_rot[:, 0]
            next_approach_raw = next_gp - cfg.d_pre * next_x_axis
            next_z = float(next_approach_raw[2])
            if abs(next_z - gp[2]) > 0.15:
                clearance_xy = (gp + (cfg.d_post + 0.60) * x_axis)[:2]
                # Place the clearance waypoint at a z that safely clears the
                # exited gate's frame top/bottom (frame half-extent is 0.36 m,
                # +0.10 m safety margin). The cubic spline through a climb/
                # descent turn tends to overshoot back toward the exited gate
                # plane; positioning clearance at safe-side-of-frame ensures
                # the dip still clears.
                if next_z > gp[2]:
                    z_c = max(gp[2] + 0.55, next_z - 0.05)
                else:
                    z_c = min(gp[2] - 0.55, next_z + 0.05)
                clearance = np.array([clearance_xy[0], clearance_xy[1], z_c])
                wps.append(clearance)
                # For sharp turns (e.g. gate 2 → gate 3, ~180°), the cubic
                # spline from clearance → next_approach curves through the
                # gate plane. Insert a turn_apex waypoint at a z above the
                # exited frame top (or below the bottom) with xy placed on
                # the clearance-to-approach line but pushed slightly away
                # from the exited gate center — pins the spline into the
                # safe altitude band during the turn.
                next_approach_xy = next_approach_raw[:2]
                mid_xy = 0.5 * (clearance_xy + next_approach_xy)
                away = mid_xy - gp[:2]
                away_n = float(np.linalg.norm(away))
                if away_n > 1e-6:
                    mid_xy = mid_xy + (away / away_n) * 0.10
                if next_z > gp[2]:
                    z_apex = next_z + 0.05
                else:
                    z_apex = next_z - 0.05
                turn_apex = np.array([mid_xy[0], mid_xy[1], z_apex])
                wps.append(turn_apex)
    last_x = R.from_quat(gates_quat[-1]).as_matrix()[:, 0]
    wps.append(gates_pos[-1] + (cfg.d_post + cfg.d_stop) * last_x)
    wps_arr = np.asarray(wps)
    return _insert_obstacle_midpoints(wps_arr, obstacles_pos, cfg.r_obs)


def _approach_swing(
    gate_pos: NDArray[np.floating],
    x_axis: NDArray[np.floating],
    y_axis: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    d_pre: float,
    r_obs: float,
) -> NDArray[np.floating] | None:
    """Return a swing waypoint that routes laterally around an obstacle blocking the
    gate's approach corridor, or ``None`` if no obstacle sits in the corridor.

    The corridor spans ``[-(d_pre + r_obs), r_obs]`` along the gate x-axis (from
    slightly past the approach point to slightly past the gate center) with width
    ``r_obs + 0.15`` in the gate y-axis. If blocked, the swing is placed at the
    blocker's x-position (along gate x-axis) with a lateral offset of
    ``r_obs + 0.15`` on the side away from the blocker, at gate z-height.
    """
    x2 = x_axis[:2]
    y2 = y_axis[:2]
    xn = float(np.linalg.norm(x2))
    yn = float(np.linalg.norm(y2))
    if xn < 1e-6 or yn < 1e-6:
        return None
    x_u, y_u = x2 / xn, y2 / yn
    blocker = None
    best_lateral = 0.0
    best_along = 0.0
    smallest_lateral_abs = r_obs
    for o in obstacles_pos:
        d_world = o[:2] - gate_pos[:2]
        along = float(np.dot(d_world, x_u))
        lateral = float(np.dot(d_world, y_u))
        # Obstacle blocks the approach if it sits between the approach waypoint
        # (along = -d_pre) and the gate center, roughly on the ray.
        if -(d_pre + 0.1) < along < 0.05 and abs(lateral) < smallest_lateral_abs:
            blocker = o
            best_along = along
            best_lateral = lateral
            smallest_lateral_abs = abs(lateral)
    if blocker is None:
        return None
    side = -1.0 if best_lateral > 0 else 1.0
    swing_xy = (
        gate_pos[:2] + best_along * x_u + side * (r_obs + 0.15) * y_u
    )
    swing = np.array([swing_xy[0], swing_xy[1], gate_pos[2]])
    return swing


def _nudge_lateral(
    point: NDArray[np.floating],
    lateral: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    r_obs: float,
    bias_sign: float = 0.0,
) -> NDArray[np.floating]:
    """Shift ``point`` along ``lateral`` (gate y-axis) until clear of all obstacles (2D).

    Preserves the along-gate-x-axis component so the drone still enters perpendicular.
    Falls back to radial nudging if lateral direction is degenerate (e.g. vertical gate).

    ``bias_sign`` (in the lateral direction's sign) breaks near-ties: when the
    obstacle sits on the approach axis both ``+y`` and ``-y`` shifts are valid,
    and picking the minimum-abs shift alone can route the drone to the wrong
    side of the gate (e.g. the drone coming from the north ends up on a
    south-side approach, forcing a 45° S-curve into the gate). When
    ``bias_sign`` is nonzero and the matching-side shift is at most ~1.5× the
    other, prefer it.
    """
    y2 = lateral[:2]
    yn = float(np.linalg.norm(y2))
    if yn < 1e-6:
        return _nudge(point, lateral, obstacles_pos, r_obs)
    y2 = y2 / yn
    p = point.copy()
    margin = r_obs + 0.02
    for _ in range(4):
        offenders = [o for o in obstacles_pos if np.linalg.norm(p[:2] - o[:2]) < margin]
        if not offenders:
            break
        closest = min(offenders, key=lambda o: np.linalg.norm(p[:2] - o[:2]))
        d = p[:2] - closest[:2]
        d_dot_y = float(np.dot(d, y2))
        d_dot_d = float(np.dot(d, d))
        disc = d_dot_y**2 + margin**2 - d_dot_d
        if disc < 0:
            # lateral direction is exactly perpendicular to d; need radial fallback
            return _nudge(point, lateral, obstacles_pos, r_obs)
        root = np.sqrt(disc)
        a_plus = -d_dot_y + root
        a_minus = -d_dot_y - root
        if bias_sign > 0 and a_plus > 0 and abs(a_plus) <= 1.5 * abs(a_minus):
            a = a_plus
        elif bias_sign < 0 and a_minus < 0 and abs(a_minus) <= 1.5 * abs(a_plus):
            a = a_minus
        else:
            a = a_plus if abs(a_plus) < abs(a_minus) else a_minus
        p[:2] = p[:2] + a * y2
    return p


def _nudge(
    point: NDArray[np.floating],
    lateral: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    r_obs: float,
) -> NDArray[np.floating]:
    p = point.copy()
    for _ in range(4):
        dists = [float(np.linalg.norm(p[:2] - o[:2])) for o in obstacles_pos]
        if not dists or min(dists) >= r_obs:
            break
        j = int(np.argmin(dists))
        o = obstacles_pos[j]
        diff = p[:2] - o[:2]
        n = float(np.linalg.norm(diff))
        if n < 1e-6:
            fallback = lateral[:2]
            ln = float(np.linalg.norm(fallback))
            dir2 = fallback / ln if ln > 1e-6 else np.array([1.0, 0.0])
        else:
            dir2 = diff / n
        p[:2] = o[:2] + dir2 * (r_obs + 0.05)
    return p


def _insert_obstacle_midpoints(
    waypoints: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    r_obs: float,
) -> NDArray[np.floating]:
    if len(obstacles_pos) == 0:
        return waypoints
    result: list[NDArray[np.floating]] = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        ab_xy = (b - a)[:2]
        denom = float(np.dot(ab_xy, ab_xy))
        if denom < 1e-9:
            result.append(b)
            continue
        worst_d = r_obs
        worst_t: float | None = None
        worst_o: NDArray[np.floating] | None = None
        for o in obstacles_pos:
            ao = o[:2] - a[:2]
            t_ = float(np.clip(np.dot(ao, ab_xy) / denom, 0.0, 1.0))
            closest = a[:2] + t_ * ab_xy
            d = float(np.linalg.norm(o[:2] - closest))
            if d < worst_d:
                worst_d, worst_t, worst_o = d, t_, o
        if worst_o is not None and worst_t is not None:
            closest_xy = a[:2] + worst_t * ab_xy
            diff = closest_xy - worst_o[:2]
            n = float(np.linalg.norm(diff))
            if n < 1e-6:
                perp = np.array([-ab_xy[1], ab_xy[0]])
                pn = float(np.linalg.norm(perp))
                push = perp / pn if pn > 1e-6 else np.array([1.0, 0.0])
            else:
                push = diff / n
            z_mid = float(a[2] + worst_t * (b[2] - a[2]))
            mid = np.array(
                [
                    worst_o[0] + push[0] * (r_obs + 0.08),
                    worst_o[1] + push[1] * (r_obs + 0.08),
                    z_mid,
                ]
            )
            result.append(mid)
        result.append(b)
    return np.asarray(result)


def _build_spline(
    waypoints: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    obstacles_pos: NDArray[np.floating],
    cfg: PlannerConfig,
    gates_pos: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.floating], CubicSpline]:
    diffs = np.diff(waypoints, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    # Per-segment cruise speed: peri-gate segments (either endpoint within
    # 0.55 m of a gate center) use the conservative v_cruise; inter-gate
    # segments run at v_cruise_inter for faster transitions.
    v_inter = cfg.v_cruise_inter if cfg.v_cruise_inter > 0 else cfg.v_cruise
    seg_t = np.empty(len(seg_len))
    for i in range(len(seg_len)):
        v = cfg.v_cruise
        if gates_pos is not None and v_inter > cfg.v_cruise:
            peri = False
            for gp in gates_pos:
                if float(np.linalg.norm(waypoints[i, :2] - gp[:2])) < 0.55:
                    peri = True
                    break
                if float(np.linalg.norm(waypoints[i + 1, :2] - gp[:2])) < 0.55:
                    peri = True
                    break
            if not peri:
                v = v_inter
        seg_t[i] = max(seg_len[i] / v, cfg.t_min_seg)
    # Stretch (slow) segments whose 2D path comes very close to an obstacle.
    # Tight threshold + capped stretch so we don't blow up cycle times.
    # Inversely, shrink (speed up) obstacle-free segments that are also well
    # clear of any gate center, letting the drone run faster on open stretches.
    if len(obstacles_pos) > 0:
        slow_radius = 0.32
        fast_radius = 0.70  # above this min-clearance, segment is "open"
        for i in range(len(waypoints) - 1):
            a, b = waypoints[i, :2], waypoints[i + 1, :2]
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom < 1e-9:
                continue
            min_d = np.inf
            for o in obstacles_pos:
                t_ = float(np.clip(np.dot(o[:2] - a, ab) / denom, 0.0, 1.0))
                closest = a + t_ * ab
                d = float(np.linalg.norm(o[:2] - closest))
                if d < min_d:
                    min_d = d
            if min_d < slow_radius:
                stretch = 1.0 + 0.6 * (slow_radius - min_d) / slow_radius
                seg_t[i] *= stretch
    if cfg.max_vel > 0 and cfg.max_accel > 0:
        if cfg.use_slsqp:
            seg_t = _slsqp_time_optimal(waypoints, start_vel, seg_t, cfg)
        else:
            seg_t = _time_optimal_refine(waypoints, start_vel, seg_t, cfg)
    t = np.concatenate([[0.0], np.cumsum(seg_t)])
    if cfg.use_quintic and len(waypoints) >= 6:
        # Quintic spline with clamped vel + zero-accel BCs at both ends.
        # make_interp_spline(k=5) needs 4 extra conditions (2 at each end).
        sv = np.asarray(start_vel, dtype=np.float64)
        bc_type = (
            [(1, sv), (2, np.zeros(3))],
            [(1, np.zeros(3)), (2, np.zeros(3))],
        )
        try:
            spline = make_interp_spline(t, waypoints, k=5, bc_type=bc_type)
            return t, spline
        except Exception:
            # Fall through to cubic on degenerate knot sequences.
            pass
    bc = ((1, np.asarray(start_vel, dtype=np.float64)), (1, np.zeros(3)))
    spline = CubicSpline(t, waypoints, bc_type=bc)
    return t, spline


def _slsqp_time_optimal(
    waypoints: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    seg_t0: NDArray[np.floating],
    cfg: PlannerConfig,
) -> NDArray[np.floating]:
    """Minimize total time subject to peak vel/accel caps via scipy SLSQP.

    Variables: segment times (one per segment). Objective: sum(seg_t).
    Constraints: seg_t >= t_min_seg (via bounds), peak vel <= max_vel,
    peak accel <= max_accel (sampled at 12 points per segment). Uses
    finite-difference gradients; fine for a small number of segments.
    """
    n_seg = len(seg_t0)
    bc = ((1, np.asarray(start_vel, dtype=np.float64)), (1, np.zeros(3)))

    def _peaks(seg_t: np.ndarray) -> tuple[float, float]:
        t = np.concatenate([[0.0], np.cumsum(np.maximum(seg_t, 1e-3))])
        sp = CubicSpline(t, waypoints, bc_type=bc)
        ts = np.linspace(0.0, float(t[-1]), 12 * n_seg)
        v_peak = float(np.max(np.linalg.norm(sp.derivative(1)(ts), axis=1)))
        a_peak = float(np.max(np.linalg.norm(sp.derivative(2)(ts), axis=1)))
        return v_peak, a_peak

    def _obj(seg_t):
        return float(np.sum(seg_t))

    def _v_constr(seg_t):
        return cfg.max_vel - _peaks(seg_t)[0]

    def _a_constr(seg_t):
        return cfg.max_accel - _peaks(seg_t)[1]

    bounds = [(cfg.t_min_seg, None)] * n_seg
    cons = [
        {"type": "ineq", "fun": _v_constr},
        {"type": "ineq", "fun": _a_constr},
    ]
    try:
        res = minimize(
            _obj,
            seg_t0.copy(),
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 25, "ftol": 1e-4},
        )
        if res.success and _peaks(res.x)[0] <= cfg.max_vel * 1.05 and _peaks(res.x)[1] <= cfg.max_accel * 1.1:
            return np.asarray(np.maximum(res.x, cfg.t_min_seg))
    except Exception:
        pass
    # Fall back to the heuristic refiner on any failure.
    return _time_optimal_refine(waypoints, start_vel, seg_t0, cfg)


def _time_optimal_refine(
    waypoints: NDArray[np.floating],
    start_vel: NDArray[np.floating],
    seg_t: NDArray[np.floating],
    cfg: PlannerConfig,
) -> NDArray[np.floating]:
    """Iteratively shrink segments with headroom under the peak vel/accel caps.

    This is a gradient-free, monotone heuristic: each pass computes per-segment
    peak velocity and acceleration on the current spline, then scales each
    segment's time by a factor in (0.85, 1.15) chosen so that the peak utilization
    approaches (but does not exceed) the cap. Converges in ~6 iterations.
    """
    seg_t = np.asarray(seg_t, dtype=np.float64).copy()
    bc = ((1, np.asarray(start_vel, dtype=np.float64)), (1, np.zeros(3)))
    for _ in range(8):
        t = np.concatenate([[0.0], np.cumsum(seg_t)])
        spline = CubicSpline(t, waypoints, bc_type=bc)
        dspline = spline.derivative(1)
        d2spline = spline.derivative(2)
        for i in range(len(seg_t)):
            ts = np.linspace(t[i], t[i + 1], 24)
            v_peak = float(np.max(np.linalg.norm(dspline(ts), axis=1)))
            a_peak = float(np.max(np.linalg.norm(d2spline(ts), axis=1)))
            util = max(v_peak / cfg.max_vel, np.sqrt(a_peak / cfg.max_accel))
            # Scale toward utilization = 0.75 (leave 25% margin for tracking lag).
            target = 0.75
            scale = float(np.clip(util / target, 0.85, 1.15))
            new_t = seg_t[i] * scale
            if cfg.t_min_seg > 0:
                new_t = max(new_t, cfg.t_min_seg)
            seg_t[i] = new_t
    return seg_t
