"""Planner-geometry validator: does the *planned* spline stay inside the arena, and how wide?

Why this exists
---------------
The real-lab failure is "the spline curves are so wide the drone flies far from the gate, leaves
the safety box, and the run aborts." That is a property of the PLANNED PATH, which is pure
numpy/scipy (build_waypoints -> repair_obstacles -> CubicSpline). This script builds exactly the
plan a GateSearch controller would, for every replan the controller makes (target_gate = 0..N-1),
and reports the worst-case approach to each ``safety_limits`` plane plus the planned traversal
time. No physics sim / MuJoCo / numpy-LAPACK needed -- so it runs even where the full sim can't.

It mirrors GateSearchV7/V8 planning: ``orient_gates_to_travel=False``, ``early_climb=True``, the
per-gate virtual frame-post columns, and (V8) the measured lateral pre-bias. Pass --bias to toggle
the V8 bias and --settings to sweep planner constants without editing a controller.

Run:
    cd <repo> && SCIPY_ARRAY_API=1 .pixi/envs/default/python.exe scripts/plan_geom.py \
        --config real_track.toml
"""
# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, E501, T201, C901  (diagnostic script)

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

import fire
import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.online_planner.settings import PlannerSettings
from lsy_drone_racing.control.online_planner.timing import enforce_geofence, repair_obstacles
from lsy_drone_racing.control.online_planner.trajectory import build_waypoints

# v8 NAVIGATE planner constants (gate_search_v8.py)
V8 = dict(
    d_pre=0.60, d_post=0.40, v_cruise=1.4, v_cruise_inter=1.6, max_speed=1.9,
    r_obs=0.20, orient_gates_to_travel=False, early_climb=True,
)
GATE_POST_OFFSET = 0.30
GATE_LATERAL_BIAS = {2: -0.073}


def _load_track(config: str):
    path = Path(__file__).parents[1] / "config" / config
    with open(path, "rb") as fh:
        cfg = tomllib.load(fh)
    track = cfg["env"]["track"]
    gates = track["gates"]
    gate_pos = np.array([g["pos"] for g in gates], dtype=np.float64)
    gate_rpy = np.array([g["rpy"] for g in gates], dtype=np.float64)
    gate_quat = Rotation.from_euler("xyz", gate_rpy).as_quat()
    obstacles = np.array([o["pos"] for o in track["obstacles"]], dtype=np.float64)
    drone = np.array(track["drones"][0]["pos"], dtype=np.float64)
    lim = track["safety_limits"]
    low = np.array(lim["pos_limit_low"], dtype=np.float64)
    high = np.array(lim["pos_limit_high"], dtype=np.float64)
    return gate_pos, gate_quat, obstacles, drone, low, high


def _post_obstacles(gate_pos, gate_quat):
    posts = []
    for gi in range(len(gate_pos)):
        lateral = Rotation.from_quat(gate_quat[gi]).as_matrix()[:, 1]
        posts.append(gate_pos[gi] + GATE_POST_OFFSET * lateral)
        posts.append(gate_pos[gi] - GATE_POST_OFFSET * lateral)
    return np.array(posts, dtype=np.float64)


def _biased(gate_pos, gate_quat, use_bias):
    gp = gate_pos.copy()
    if use_bias:
        for gi, b in GATE_LATERAL_BIAS.items():
            if 0 <= gi < len(gp) and b:
                gp[gi] = gp[gi] + b * Rotation.from_quat(gate_quat[gi]).as_matrix()[:, 1]
    return gp


def _build_plan(start_pos, start_vel, gate_pos, gate_quat, obstacles, target_gate, settings):
    det = np.concatenate([obstacles, _post_obstacles(gate_pos, gate_quat)], axis=0)
    wp = build_waypoints(start_pos, start_vel, gate_pos, gate_quat, det, target_gate, settings)
    wp, knot_times, curve = repair_obstacles(wp, start_vel, gate_pos, det, settings)
    wp, knot_times, curve = enforce_geofence(wp, start_vel, gate_pos, det, settings)
    return wp, float(knot_times[-1]), curve


def _crossing(curve, t_total, center, forward):
    """Planned lateral miss (m) and entry angle (deg) where the spline crosses a gate."""
    ts = np.linspace(0.0, t_total, 2000)
    pts = np.asarray(curve(ts), dtype=np.float64)
    j = int(np.argmin(np.linalg.norm(pts - center, axis=1)))
    rel = pts[j] - center
    f = forward / (np.linalg.norm(forward) + 1e-12)
    lateral_miss = float(np.linalg.norm(rel - np.dot(rel, f) * f))
    tangent = np.asarray(curve.derivative(1)(ts[j]), dtype=np.float64)
    tn = float(np.linalg.norm(tangent))
    entry_deg = float(np.degrees(np.arccos(np.clip(np.dot(tangent / tn, f), -1.0, 1.0)))) if tn > 1e-9 else 0.0
    return lateral_miss, entry_deg


def evaluate(config: str = "real_track.toml", bias: bool = True, **overrides):
    """Report planned-path bound margins, swing width, and traversal time for each replan."""
    gate_pos, gate_quat, obstacles, drone, low, high = _load_track(config)
    n = len(gate_pos)
    settings = PlannerSettings(**{**V8, **overrides})
    if settings.geofence_margin > 0 and settings.arena_low is None:
        settings = dataclasses.replace(
            settings, arena_low=tuple(low.tolist()), arena_high=tuple(high.tolist())
        )
    gp_plan = _biased(gate_pos, gate_quat, bias)
    forwards = [Rotation.from_quat(gate_quat[i]).as_matrix()[:, 0] for i in range(n)]

    print("=" * 80)
    extra = f"  overrides={overrides}" if overrides else ""
    print(f"config={config}  bias={bias}{extra}")
    print(f"  bounds: x[{low[0]:.2f},{high[0]:.2f}] y[{low[1]:.2f},{high[1]:.2f}] "
          f"z[{low[2]:.2f},{high[2]:.2f}]  geofence_margin={settings.geofence_margin}")

    # Lateral (x,y) margin is what the "leaves the zone" abort is about; the z-floor margin at
    # t=0 is just the drone sitting on the ground at start, so report lateral separately.
    global_lat, global_y, global_x, full_t = np.inf, 0.0, 0.0, None
    worst_miss, worst_entry = 0.0, 0.0
    for g in range(n):
        if g == 0:
            start, vel = drone.copy(), np.zeros(3)
        else:
            start = gate_pos[g - 1] + settings.d_post * forwards[g - 1]
            vel = forwards[g - 1] * settings.v_cruise
        wp, t_total, curve = _build_plan(start, vel, gp_plan, gate_quat, obstacles, g, settings)
        ts = np.linspace(0.0, t_total, 600)
        pts = np.asarray(curve(ts), dtype=np.float64)
        y_margin = float(min(np.min(pts[:, 1] - low[1]), np.min(high[1] - pts[:, 1])))
        x_margin = float(min(np.min(pts[:, 0] - low[0]), np.min(high[0] - pts[:, 0])))
        lat_margin = min(x_margin, y_margin)
        ymax, xmax = float(np.max(np.abs(pts[:, 1]))), float(np.max(np.abs(pts[:, 0])))
        # Planned crossing accuracy for the gate this replan targets (vs the REAL, unbiased centre).
        miss, entry = _crossing(curve, t_total, gate_pos[g], forwards[g])
        worst_miss, worst_entry = max(worst_miss, miss), max(worst_entry, entry)
        if g == 0:
            full_t = t_total
        global_lat = min(global_lat, lat_margin)
        global_y, global_x = max(global_y, ymax), max(global_x, xmax)
        flag = "  <-- OUT OF BOUNDS" if lat_margin < 0 else ("  <-- TIGHT" if lat_margin < 0.10 else "")
        print(f"  replan g={g}: t={t_total:5.2f}s  |y|max={ymax:.3f} (mgn {y_margin:+.3f})  "
              f"|x|max={xmax:.3f} (mgn {x_margin:+.3f})  gate-miss={miss:.3f} entry={entry:4.1f}deg{flag}")

    print("-" * 80)
    print(f"  WORST lateral margin over all replans: {global_lat:+.3f} m  "
          f"[|y|max={global_y:.3f}/lim {high[1]:.2f}, |x|max={global_x:.3f}/lim {high[0]:.2f}]")
    print(f"  WORST planned gate-miss={worst_miss:.3f} m (opening half-width 0.20)  "
          f"max entry angle={worst_entry:.1f} deg")
    print(f"  full-route planned time (g=0): {full_t:.2f}s")
    return dict(lat_margin=global_lat, ymax=global_y, xmax=global_x, full_t=full_t,
                gate_miss=worst_miss, entry_deg=worst_entry)


if __name__ == "__main__":
    fire.Fire(evaluate)
