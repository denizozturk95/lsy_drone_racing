"""v10.6 planner: v10.4's gate-funnel chain + GUARDED waypoint smoothing.

Per plan build, on the actual revealed geometry, with the shipped v10.4 plan as a checked
fallback:

1. BUILD BOTH. The v10.4 waypoint chain unchanged, then a smoothed copy: free waypoints
   (clearance/apex/run-in/repair points) pulled toward their neighbours' midpoint and pushed back
   out of obstacle keep-outs. Never moved: each gate's approach/gate/exit triplet, the chain ends,
   the replan-continuity arc, and the whole leg of every reversal transition (the U-turn swing is
   necessary geometry -- smoothing it tightened the fold and crashed the reversal in flight).
2. PARITY CAPS, not hope. The smoothed plan carries speed caps DERIVED FROM THE UNSMOOTHED
   profile: per gate the base max inside [s_gate - reveal, s_gate]; per obstacle passage,
   arc-interval caps run-matched against the base plan's passage of the same obstacle. So by
   construction the smoothed plan is never faster than v10.4 exactly where speed kills.
3. GEOMETRY GUARD + STICKY MODE. The smoothed route is adopted only if every sampled gate-plane
   crossing inside the frame threads the opening centre, clearance to every real obstacle holds,
   and the capped profile beats the base by min_gain_s. The decision is made ONCE per episode and
   replans inherit it (per-rebuild re-deciding teleported the reference ~0.5 m and piled crashes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.controller_core_v8.timing import repair_obstacles
from lsy_drone_racing.control.controller_core_v8.trajectory import ReferencePlan
from lsy_drone_racing.control.controller_core_v9_1.speed_profile import SpeedProfile
from lsy_drone_racing.control.controller_core_v10_4.trajectory import (
    ReferenceManager as _ReferenceManager,
)
from lsy_drone_racing.control.controller_core_v10_4.trajectory import build_waypoints

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline

    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation
    from lsy_drone_racing.control.controller_core_v10_6.settings import SmoothPlannerSettings

_FRAME_HALF_M = 0.36  # gate frame outer half-extent (0.72 m square)


@dataclass(frozen=True)
class SmoothedPlan(ReferencePlan):
    """A reference plan plus the parity speed caps that make the smoothed route shippable.

    ``gate_window_caps`` has one entry per remaining gate (np.inf = no cap); ``obstacle_caps``
    one (s_lo, s_hi, v_max) arc-interval row per obstacle passage of this plan's own curve.
    Both are np.inf/empty when ``smoothed`` is False (the plan is byte-identical to v10.4's).
    """

    gate_window_caps: NDArray[np.float64]
    obstacle_caps: NDArray[np.float64]
    smoothed: bool


def smooth_waypoints(
    waypoints: NDArray[np.float64],
    gates_pos: NDArray[np.float64],
    gates_quat: NDArray[np.float64],
    obstacles_pos: NDArray[np.float64],
    clearance: float,
    pull: float,
    iterations: int,
) -> NDArray[np.float64]:
    """Pull free waypoints toward their neighbours' midpoint, then back out of obstacles.

    Protected (never moved): the chain ends, the first three points (start state and the
    replan-continuity / liftoff arc), each remaining gate's approach/gate/exit triplet, and the
    whole leg of every REVERSAL transition (same cos < -0.3 branch the planner takes) -- the
    U-turn swing points are geometrically necessary; pulling them inward tightens the fold until
    the turn stops being trackable.
    """
    out = waypoints.copy()
    protected = {0, 1, 2, len(out) - 1}
    gate_idx = [int(np.argmin(np.linalg.norm(out - gate, axis=1))) for gate in gates_pos]
    for idx in gate_idx:
        protected.update({idx - 1, idx, idx + 1})
    for i in range(len(gates_pos) - 1):
        forward = Rotation.from_quat(gates_quat[i]).as_matrix()[:2, 0]
        to_next = (gates_pos[i + 1] - gates_pos[i])[:2]
        denom = float(np.linalg.norm(forward) * np.linalg.norm(to_next))
        if denom > 1e-9 and float(np.dot(forward, to_next)) / denom < -0.3:
            protected.update(range(gate_idx[i] + 1, gate_idx[i + 1]))
    for _ in range(iterations):
        for i in range(1, len(out) - 1):
            if i in protected:
                continue
            out[i] += pull * (0.5 * (out[i - 1] + out[i + 1]) - out[i])
            for obstacle in obstacles_pos:
                delta = out[i, :2] - obstacle[:2]
                norm = float(np.linalg.norm(delta))
                if norm < clearance:
                    direction = delta / norm if norm > 1e-6 else np.array([1.0, 0.0])
                    out[i, :2] = obstacle[:2] + direction * clearance
    return out


def _profile(
    curve: CubicSpline, t_total: float, v_cap: float, a_lat_max: float, v_min: float, n: int = 400
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Sample the spline densely: (arc grid, points, curvature-limited speed)."""
    taus = np.linspace(0.0, float(t_total), n)
    pts = np.asarray(curve(taus), dtype=np.float64)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    v = SpeedProfile(curve, t_total, v_cap, a_lat_max, v_min, n).at_arc(s)
    return s, pts, v


def _predicted_time(s: NDArray[np.float64], v: NDArray[np.float64]) -> float:
    """Ride time of the profile, sum(ds / v) with midpoint speeds."""
    return float(np.sum(np.diff(s) / np.maximum(0.5 * (v[1:] + v[:-1]), 1e-3)))


def _gate_arc_positions(
    s: NDArray[np.float64], pts: NDArray[np.float64], gates_pos: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Arc length of the dense sample nearest to each gate centre."""
    return np.array([s[int(np.argmin(np.linalg.norm(pts - g, axis=1)))] for g in gates_pos])


def _window_caps(
    s: NDArray[np.float64],
    pts: NDArray[np.float64],
    v: NDArray[np.float64],
    gates_pos: NDArray[np.float64],
    window_pre: float,
) -> NDArray[np.float64]:
    """Per gate: the profile's max speed inside the reveal window [s_gate - pre, s_gate]."""
    caps = np.empty(len(gates_pos))
    for i, s_g in enumerate(_gate_arc_positions(s, pts, gates_pos)):
        mask = (s >= s_g - window_pre) & (s <= s_g)
        caps[i] = float(np.max(v[mask])) if np.any(mask) else np.inf
    return caps


def _passage_runs(
    pts: NDArray[np.float64], obstacle: NDArray[np.float64], radius: float
) -> list[NDArray[np.intp]]:
    """Contiguous sample-index runs where the path is within ``radius`` of the obstacle (XY)."""
    near = np.linalg.norm(pts[:, :2] - obstacle[:2], axis=1) < radius
    edges = np.nonzero(np.diff(near.astype(int)))[0] + 1
    return [run for run in np.split(np.arange(len(near)), edges) if near[run[0]]]


def _obstacle_caps(
    base: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    cand: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    obstacles_pos: NDArray[np.float64],
    radius: float,
) -> NDArray[np.float64]:
    """Arc-interval caps (s_lo, s_hi, v_max) on the candidate path's obstacle passages.

    ``base``/``cand`` are (arc grid, points, profile speed) triples. Each contiguous candidate
    passage within ``radius`` of an obstacle is capped at the base profile's max over the
    ORDER-MATCHED base passage of that obstacle -- per-passage parity. When the passage counts
    differ, every candidate passage gets the slowest base passage's cap; when the base path never
    enters the tube, its speed at the closest approach.
    """
    s_b, pts_b, v_b = base
    s_c, pts_c, v_c = cand
    rows: list[tuple[float, float, float]] = []
    for obstacle in obstacles_pos:
        cand_runs = _passage_runs(pts_c, obstacle, radius)
        if not cand_runs:
            continue
        base_runs = _passage_runs(pts_b, obstacle, radius)
        if base_runs:
            base_caps = [float(np.max(v_b[run])) for run in base_runs]
        else:
            d = np.linalg.norm(pts_b[:, :2] - obstacle[:2], axis=1)
            base_caps = [float(v_b[int(np.argmin(d))])]
        matched = len(base_caps) == len(cand_runs)
        pad = 0.06  # a couple of grid steps, so resampling in the path view can't shrink the tube
        for k, run in enumerate(cand_runs):
            cap = base_caps[k] if matched else min(base_caps)
            rows.append((float(s_c[run[0]]) - pad, float(s_c[run[-1]]) + pad, cap))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 3)


def _crossings_centred(
    pts: NDArray[np.float64],
    gates_pos: NDArray[np.float64],
    gates_quat: NDArray[np.float64],
    tol: float,
) -> bool:
    """True if every plane crossing inside a gate's frame square threads the opening centre."""
    for gate, quat in zip(gates_pos, gates_quat):
        rot = Rotation.from_quat(quat).as_matrix()
        rel = pts - gate
        f = rel @ rot[:, 0]
        for i in np.nonzero(np.signbit(f[:-1]) != np.signbit(f[1:]))[0]:
            w = f[i] / (f[i] - f[i + 1])
            p = rel[i] + w * (rel[i + 1] - rel[i])
            lat, vert = abs(float(p @ rot[:, 1])), abs(float(p @ rot[:, 2]))
            inside_frame = lat < _FRAME_HALF_M and vert < _FRAME_HALF_M
            if inside_frame and (lat > tol or vert > tol):
                return False
    return True


def _make_plan(
    frame: DroneObservation,
    curve: CubicSpline,
    knots: NDArray[np.float64],
    waypoints: NDArray[np.float64],
    gate_caps: NDArray[np.float64],
    obs_caps: NDArray[np.float64],
    smoothed: bool,
) -> SmoothedPlan:
    """Assemble a SmoothedPlan, snapshotting the frame the way v10.4's build does."""
    return SmoothedPlan(
        curve=curve,
        t_total=float(knots[-1]),
        waypoints=waypoints,
        gate_pos_snapshot=frame.gate_pos.copy(),
        obstacle_pos_snapshot=frame.obstacles_pos.copy(),
        built_target_gate=frame.target_gate,
        gate_window_caps=gate_caps,
        obstacle_caps=obs_caps,
        smoothed=smoothed,
    )


class ReferenceManager(_ReferenceManager):
    """v10.4's ReferenceManager building smoothed plans behind the parity-cap guard."""

    def __init__(
        self,
        settings: SmoothPlannerSettings,
        replan_gate_delta_m: float,
        replan_obstacle_delta_m: float,
        v_cap: float,
        a_lat_max: float,
        v_min: float,
    ):
        """Store the speed-profile budget the guard scores candidate plans with."""
        super().__init__(settings, replan_gate_delta_m, replan_obstacle_delta_m)
        self._v_cap = float(v_cap)
        self._a_lat_max = float(a_lat_max)
        self._v_min = float(v_min)
        self._mode: str | None = None  # episode-sticky: "smooth" or "base", set by build 1
        self.decisions: list[str] = []  # per build: "accept" or the guard that rejected

    def reset(self) -> None:
        """Forget the cached plan and the episode's sticky smoothing mode."""
        super().reset()
        self._mode = None

    def build(
        self,
        start_pos: NDArray[np.float64],
        start_vel: NDArray[np.float64],
        frame: DroneObservation,
    ) -> SmoothedPlan:
        """Build the v10.4 plan and a smoothed candidate; ship the candidate only if it guards."""
        settings: SmoothPlannerSettings = self._settings
        planning_obstacles = self._planning_obstacles(frame)
        first = max(frame.target_gate, 0)
        remaining_pos = np.asarray(frame.gate_pos, dtype=np.float64)[first:]
        remaining_quat = np.asarray(frame.gate_quat, dtype=np.float64)[first:]

        raw = build_waypoints(
            start_pos,
            start_vel,
            frame.gate_pos,
            frame.gate_quat,
            planning_obstacles,
            frame.target_gate,
            settings,
        )
        base_wps, base_knots, base_curve = repair_obstacles(
            raw, start_vel, frame.gate_pos, planning_obstacles, settings
        )

        def base_plan(reason: str) -> SmoothedPlan:
            if self._mode is None:
                self._mode = "base"
            self.decisions.append(reason)
            return _make_plan(
                frame,
                base_curve,
                base_knots,
                base_wps,
                np.full(len(remaining_pos), np.inf),
                np.empty((0, 3)),
                False,
            )

        # The smooth-or-not decision is STICKY for the episode: a mid-flight flip between the
        # smoothed and base routes teleports the reference under the drone by up to ~0.5 m. Build 1
        # decides; replans keep the mode and only the safety guards below may demote a rebuild.
        if self._mode == "base":
            return base_plan("base-locked")
        first_build = self._mode is None

        smooth_raw = smooth_waypoints(
            raw,
            remaining_pos,
            remaining_quat,
            planning_obstacles,
            settings.r_obs + 0.06,
            settings.smooth_pull,
            settings.smooth_iters,
        )
        if np.allclose(smooth_raw, raw):  # nothing was free to move (e.g. sharp slalom)
            return base_plan("noop")
        cand_wps, cand_knots, cand_curve = repair_obstacles(
            smooth_raw, start_vel, frame.gate_pos, planning_obstacles, settings
        )

        budget = (self._v_cap, self._a_lat_max, self._v_min)
        s_b, pts_b, v_b = _profile(base_curve, float(base_knots[-1]), *budget)
        s_c, pts_c, v_c = _profile(cand_curve, float(cand_knots[-1]), *budget)
        gate_caps = _window_caps(s_b, pts_b, v_b, remaining_pos, settings.reveal_window_m)
        obs_caps = _obstacle_caps(
            (s_b, pts_b, v_b),
            (s_c, pts_c, v_c),
            np.asarray(frame.obstacles_pos),
            settings.obs_cap_radius,
        )

        # Guard 1/2: crossings centred and real-obstacle clearance held on the candidate.
        if not _crossings_centred(pts_c, remaining_pos, remaining_quat, settings.cross_tol_m):
            return base_plan("crossings")
        if (
            len(frame.obstacles_pos)
            and float(
                np.min(
                    np.linalg.norm(pts_c[:, None, :2] - frame.obstacles_pos[None, :, :2], axis=2)
                )
            )
            < settings.r_obs
        ):
            return base_plan("clearance")

        # Guard 3 (build 1 only -- replans inherit the mode): the candidate must still win AFTER
        # the parity caps are priced in.
        if first_build:
            v_capped = v_c.copy()
            for s_g, cap in zip(_gate_arc_positions(s_c, pts_c, remaining_pos), gate_caps):
                mask = (s_c >= s_g - settings.reveal_window_m) & (s_c <= s_g)
                v_capped[mask] = np.minimum(v_capped[mask], cap)
            for s_lo, s_hi, cap in obs_caps:
                mask = (s_c >= s_lo) & (s_c <= s_hi)
                v_capped[mask] = np.minimum(v_capped[mask], cap)
            if _predicted_time(s_c, v_capped) > _predicted_time(s_b, v_b) - settings.min_gain_s:
                return base_plan("gain")

        self._mode = "smooth"
        self.decisions.append("accept")
        return _make_plan(frame, cand_curve, cand_knots, cand_wps, gate_caps, obs_caps, True)
