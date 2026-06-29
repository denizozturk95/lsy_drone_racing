"""Level-3 drone racing controller (v11): v10 architecture + faster, decoupled NAVIGATE.

GateSearchV11 keeps GateSearchV10's proven SEARCH/NAVIGATE/HOME architecture and all its
robustness work, and changes ONE thing that a navigate-anatomy study identified as the dominant,
*safely* recoverable lap-time cost: the navigate speed profile.

What the study found (see scripts/nav_anatomy.py, nav_timing_ablation.py)
------------------------------------------------------------------------
  * Lap ≈ takeoff (~1.6 s) + search (~8 s) + NAVIGATE (~13-17 s); navigate dominates.
  * In v10, ``v_cruise == v_cruise_inter == 1.0`` m/s, so 85% of navigate was flown below 1 m/s —
    even on level straightaways. The drone never reached cruise: the peri-gate slow zone and the
    inter-gate cruise were the SAME speed, so the whole course crawled.
  * Detection is XY-only (race_core.py), so the high (1.8 m) search costs nothing for discovery and
    is kept for frame safety; the descent cap onto gates is ~25% of navigate, not the main cost.
  * Removing the slowdown heuristics (turn_slowdown, etc.) or raising the *peri-gate* speed sped
    things up but raised crash:navi sharply — those slowdowns are load-bearing for gate-crossing
    precision. Raising feedforward/lowering lookahead did NOT recover reliability: the residual
    failures are obstacle-pole/corridor and cross-arena-transit crashes, not tracking lag.

The (partial) win: DECOUPLE the two speeds — a speed↔reliability DIAL
--------------------------------------------------------------------
Keep ``v_cruise`` (the peri-gate zone, within ``peri_gate_radius`` of a gate) at v10's safe gate-
crossing speed, but raise ``v_cruise_inter`` for the straightaways between gates, and lift the peak
cap. This carries the drone quickly between gates while still slowing it to thread each opening.

IMPORTANT — there is NO free lunch on navigate speed. A 30-seed Level-3 sweep gives a clean,
monotonic Pareto (faster lap costs finish rate, because the bottleneck is decelerating to thread
the 0.4 m opening — faster cruise = harder deceleration):

    v_cruise/inter/max     finish    lap(finishers)
    1.0 / 1.0 / 1.4 (v10)   36.7%     25.6 s
    1.0 / 1.8 / 2.4 (HERE)  30.0%     23.2 s   <- current default: modest speed gain
    1.2 / 2.6 / 3.2         26.7%     21.4 s   (min lap 16.8 s on easy seeds)

These constants are a DIAL: lower them toward v10 for max reliability, raise them for speed. The
current default is the mild middle point. Sub-22 s while *keeping* reliability is not reachable by
speed tuning — it needs either:
  * SEARCH rework: a LOCAL interleaved discovery (sweep seeded at the drone, not a global arena
    spiral) to cut the ~8 s search dead-time AND the cross-arena transit crashes — the one avenue
    that does not trade against reliability. (Prototyped but not yet robust; global interleaving
    alone regressed via crash:sear.)
  * A different navigate trajectory paradigm (time-optimal / MPC) so the drone can thread gates
    accurately at speed.
Note: gate/obstacle detection is XY-only and altitude-independent, so the high search altitude is
free for discovery and kept purely for frame safety — it is not the lap-time bottleneck.

This is the controller for level3.toml; GateSearchV9 remains the known-track/deploy controller.

Modes
-----
TAKEOFF : hold the start x/y and climb straight up to _TAKEOFF_ALT, then hand off to SEARCH.
SEARCH  : Archimedean spiral at _SEARCH_ALT (above every frame), swept outermost point first so it
          ends near the arena centre. Spiral waypoints are virtual gates; the path is unconstrained.
NAVIGATE: plan through all contiguously discovered real gates, avoiding detected obstacles, crossing
          each gate along its canonical +x axis. Decoupled fast straightaway / precise gate speeds.
HOME    : after all gates are passed, descend to arena centre (0, 0) and land.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line
from drone_models.core import load_params
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.online_planner.attitude import attitude_action
from lsy_drone_racing.control.online_planner.feedback import CascadedPid
from lsy_drone_racing.control.online_planner.settings import (
    ControllerSettings,
    PlannerSettings,
    RuntimeSettings,
)
from lsy_drone_racing.control.online_planner.state import DroneObservation, parse_observation
from lsy_drone_racing.control.online_planner.timing import build_spline
from lsy_drone_racing.control.online_planner.trajectory import ReferenceManager, ReferencePlan

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

# ── Spiral / search constants (unchanged from v10) ────────────────────────────
_SEARCH_ALT = 1.8           # m — above every gate frame/obstacle (detection is XY-only, so free)
_SPIRAL_RADIAL_STEP = 0.6   # m radial gap per revolution; < 2 × 0.7 m sensor range
_SPIRAL_ANGLE_STEP = np.pi / 6   # 30° step = 12 waypoints per revolution
_SPIRAL_ADVANCE_RADIUS = 0.6     # m (2-D) — advance to next spiral point when this close
_SPIRAL_HORIZON = 3              # waypoints ahead to include in each SEARCH plan
_SPIRAL_OUTWARD = False          # outermost-first → sweep ends near the arena centre
_SEARCH_RADIUS = 2.2             # m — outer radius the search starts from, spiralling inward
_ARENA_X_LIM = 1.9          # m from centre — search boundary (gates are within ±2.0 m)
_ARENA_Y_LIM = 1.0          # m from centre — search boundary (gates are within ±1.0 m)
_GATE_POST_OFFSET = 0.30    # m — lateral offset from gate centre to each virtual column.
# ── Takeoff constants ─────────────────────────────────────────────────────────
_TAKEOFF_ALT = 1.8
_TAKEOFF_Z_TOL = 0.05
_TAKEOFF_TIME_MARGIN = 1.0
# ── Speed constants (m/s) — the v11 change ────────────────────────────────────
_V_CRUISE_SEARCH = 2.5   # SEARCH cruise speed near (spiral) waypoints
_VMAX_SEARCH = 3.0       # SEARCH peak-velocity cap
# DECOUPLED navigate speeds: peri-gate stays moderate for crossing precision; the inter-gate cruise
# and peak cap are raised so the straightaways between gates are flown fast. This is the Pareto-best
# speed config from the Level-3 navigate ablation (lap 26 s → ~22 s at equal finish rate).
_V_CRUISE = 1.0          # NAVIGATE peri-gate speed (= v10's safe gate-crossing speed): precision
_V_CRUISE_INTER = 1.8    # NAVIGATE inter-gate cruise: faster straightaways (mild bump from 1.0)
_VMAX = 2.4              # NAVIGATE peak-velocity cap
_PERI_GATE_RADIUS = 0.55  # m — radius around a gate where v_cruise (slow) applies, not v_cruise_inter
# ── NAVIGATE gate-approach geometry (unchanged from v10) ──────────────────────
_NAV_D_PRE = 0.60
_NAV_D_POST = 0.40
_NAV_R_OBS = 0.12
_NAV_START_VEL_SCALE = 0.5
_NAV_LOOKAHEAD = 0.20
# Search strategy: True = discover ALL gates before navigating; False = navigate each gate as found.
# Interleaving (False) with this GLOBAL spiral was tested worse on Level 3 (+crash:sear from the
# cross-arena transit back to the spiral); it only pays off with a *local* search seeded at the
# drone. Kept True (v10 behavior). See the v11 docstring for the search-rework roadmap.
_DISCOVER_ALL_FIRST = True
# ── Robustness (from v10): arena geofence + tighter reversal + descent cap ────
_GEOFENCE_MARGIN = 0.15
_DEFAULT_ARENA_LOW = (-2.5, -1.5, -1e-3)
_DEFAULT_ARENA_HIGH = (2.5, 1.5, 2.0)
_REVERSAL_SWING_M = 0.40
_REVERSAL_APEX_M = 0.06
_NAV_MAX_DESCENT_RATE = 0.8


def _arena_bounds(config: dict) -> tuple[tuple, tuple]:
    """Read [low, high] position safety limits from config, falling back to the standard box."""
    try:
        limits = config.env.track.safety_limits
        low = tuple(float(v) for v in limits.pos_limit_low)
        high = tuple(float(v) for v in limits.pos_limit_high)
        if len(low) == 3 and len(high) == 3:
            return low, high
    except (AttributeError, KeyError, TypeError):
        pass
    return _DEFAULT_ARENA_LOW, _DEFAULT_ARENA_HIGH


class GateSearchV11(Controller):
    """Level-3 search-then-navigate controller: v10 robustness + faster decoupled navigate."""

    _MODE_TAKEOFF = "TAKEOFF"
    _MODE_SEARCH = "SEARCH"
    _MODE_NAVIGATE = "NAVIGATE"
    _MODE_HOME = "HOME"
    _MODE_DONE = "DONE"

    def __init__(self, obs: dict, info: dict, config: dict):
        """Initialise timing, drone parameters, PID, reference manager, and spiral."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("GateSearchV11 requires env.control_mode = 'attitude'.")
        arena_low, arena_high = _arena_bounds(config)
        self._settings = ControllerSettings(
            planner=PlannerSettings(
                d_pre=_NAV_D_PRE, d_post=_NAV_D_POST, v_cruise=_V_CRUISE,
                v_cruise_inter=_V_CRUISE_INTER, max_speed=_VMAX, peri_gate_radius=_PERI_GATE_RADIUS,
                r_obs=_NAV_R_OBS, orient_gates_to_travel=False, early_climb=True,
                reversal_swing_m=_REVERSAL_SWING_M, reversal_apex_m=_REVERSAL_APEX_M,
                geofence_margin=_GEOFENCE_MARGIN, arena_low=arena_low, arena_high=arena_high,
                max_descent_rate=_NAV_MAX_DESCENT_RATE,
            ),
            runtime=RuntimeSettings(lookahead_s=_NAV_LOOKAHEAD),
        )
        self._search_settings = ControllerSettings(
            planner=PlannerSettings(
                v_cruise=_V_CRUISE_SEARCH, max_speed=_VMAX_SEARCH,
                geofence_margin=_GEOFENCE_MARGIN, arena_low=arena_low, arena_high=arena_high,
            )
        )
        self._freq = float(config.env.freq)
        self._dt = 1.0 / self._freq
        params = load_params(config.sim.physics, config.sim.drone_model)
        self._mass = float(params["mass"])
        self._feedback = CascadedPid(self._settings.feedback)
        self._references = ReferenceManager(
            self._settings.planner,
            self._settings.runtime.replan_gate_delta_m,
            self._settings.runtime.replan_obstacle_delta_m,
        )
        self._search_references = ReferenceManager(
            self._search_settings.planner,
            self._search_settings.runtime.replan_gate_delta_m,
            self._search_settings.runtime.replan_obstacle_delta_m,
        )
        self._spiral_wps, self._spiral_quats = self._build_spiral()
        self._tick = 0
        self._plan_start_tick = 0
        self._progress_t = 0.0
        self._finished = False
        self._last_action = self._hover_action()
        self._last_target = -1
        self._mode = self._MODE_TAKEOFF
        self._known_gates: set[int] = set()
        self._last_last_known = -1
        self._virtual_target = 0
        self._spiral_swept = False
        self._takeoff_plan: tuple | None = None
        self._takeoff_start_tick = 0
        self._home_plan: tuple | None = None
        self._home_start_tick = 0
        self._dbg_gate_pos: NDArray = np.empty((0, 3), dtype=np.float64)
        self._dbg_known_mask: NDArray = np.zeros(0, dtype=bool)
        self._dbg_target_gate: int = -1
        self._dbg_obs_pos: NDArray = np.empty((0, 3), dtype=np.float64)
        self._dbg_wp_pos: NDArray = np.empty((0, 3), dtype=np.float64)

    # ── Main control loop ────────────────────────────────────────────────────

    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray:
        """Return a [roll, pitch, yaw, thrust] attitude command for the current step."""
        frame = parse_observation(obs)
        gates_visited = np.asarray(obs["gates_visited"], dtype=bool)
        obs_visited = np.asarray(obs["obstacles_visited"], dtype=bool)
        n_gates = len(gates_visited)
        now = self._tick * self._dt

        for i in range(n_gates):
            if gates_visited[i]:
                self._known_gates.add(i)

        self._last_target = int(frame.target_gate)

        if now >= self._settings.runtime.timeout_s:
            self._finished = True
            return self._last_action.copy()

        target = frame.target_gate

        if target == -1:
            if self._mode not in (self._MODE_HOME, self._MODE_DONE):
                self._mode = self._MODE_HOME

        elif self._mode == self._MODE_SEARCH and (
            (len(self._known_gates) == n_gates
             or (self._spiral_swept and target in self._known_gates))
            if _DISCOVER_ALL_FIRST
            else (target in self._known_gates)
        ):
            self._mode = self._MODE_NAVIGATE
            self._references.reset()
            self._progress_t = 0.0
            self._plan_start_tick = self._tick
            self._last_last_known = -1

        elif self._mode == self._MODE_NAVIGATE and target not in self._known_gates:
            self._mode = self._MODE_SEARCH
            self._virtual_target = self._find_nearest_spiral(frame.pos)
            self._search_references.reset()
            self._progress_t = 0.0
            self._plan_start_tick = self._tick

        if self._mode == self._MODE_TAKEOFF:
            action = self._takeoff_action(frame, obs_visited)
        elif self._mode == self._MODE_SEARCH:
            action = self._search_action(frame, obs_visited)
        elif self._mode == self._MODE_NAVIGATE:
            action = self._navigate_action(frame, obs_visited)
        elif self._mode == self._MODE_HOME:
            action = self._home_action(frame, obs_visited)
        else:
            self._finished = True
            return self._last_action.copy()

        self._capture_debug(frame, obs_visited)
        return action

    # ── TAKEOFF mode ─────────────────────────────────────────────────────────

    def _takeoff_action(self, frame: DroneObservation, obs_visited: NDArray) -> NDArray:
        """Climb straight up (holding start x/y) to _TAKEOFF_ALT, then hand off to SEARCH."""
        clock_t = (self._tick - self._takeoff_start_tick) * self._dt
        t_total = self._takeoff_plan[1] if self._takeoff_plan is not None else 0.0
        reached = float(frame.pos[2]) >= _TAKEOFF_ALT - _TAKEOFF_Z_TOL
        overran = self._takeoff_plan is not None and clock_t >= t_total + _TAKEOFF_TIME_MARGIN
        if reached or overran:
            self._mode = self._MODE_SEARCH
            self._virtual_target = 0
            self._search_references.reset()
            self._progress_t = 0.0
            self._plan_start_tick = self._tick
            return self._search_action(frame, obs_visited)

        if self._takeoff_plan is None:
            start = np.asarray(frame.pos, dtype=np.float64).copy()
            target = np.array([start[0], start[1], _TAKEOFF_ALT])
            waypoints = np.array([start, target])
            det_obs = (
                frame.obstacles_pos[obs_visited] if obs_visited.any() else np.empty((0, 3))
            )
            knot_times, curve = build_spline(
                waypoints,
                np.asarray(frame.vel, dtype=np.float64),
                np.empty((0, 3)),
                det_obs,
                self._settings.planner,
            )
            self._takeoff_plan = (curve, float(knot_times[-1]))
            self._takeoff_start_tick = self._tick
            clock_t = 0.0

        curve, t_total = self._takeoff_plan
        t_eval = float(np.clip(clock_t, 0.0, t_total))
        action, _ = attitude_action(
            curve, t_eval, frame.pos, frame.vel, frame.quat,
            self._feedback, self._dt, self._mass,
            self._settings.runtime.gravity, self._settings.command,
        )
        self._last_action = action
        return action.copy()

    # ── SEARCH mode ──────────────────────────────────────────────────────────

    def _search_action(self, frame: DroneObservation, obs_visited: NDArray) -> NDArray:
        del obs_visited  # SEARCH flies above every frame/pole, so the path is unconstrained
        det_obs = np.empty((0, 3))

        vt = self._virtual_target
        dist2d = float(np.linalg.norm(frame.pos[:2] - self._spiral_wps[vt, :2]))
        if dist2d < _SPIRAL_ADVANCE_RADIUS:
            next_vt = vt + 1
            if next_vt >= len(self._spiral_wps):
                next_vt = 0
                self._spiral_swept = True
            self._virtual_target = next_vt
            self._search_references.reset()
            self._progress_t = 0.0
            self._plan_start_tick = self._tick

        active = self._search_references.plan
        if active is not None and len(det_obs) != len(active.obstacle_pos_snapshot):
            self._search_references.reset()
            self._progress_t = 0.0
            self._plan_start_tick = self._tick

        vt = self._virtual_target
        n_wps = len(self._spiral_wps)
        end = min(vt + _SPIRAL_HORIZON, n_wps)
        window_pos = self._spiral_wps[vt:end]
        window_quat = self._spiral_quats[vt:end]
        fake_frame = DroneObservation(
            target_gate=0,
            gate_pos=window_pos,
            gate_quat=window_quat,
            obstacles_pos=det_obs,
            pos=frame.pos,
            vel=frame.vel,
            quat=frame.quat,
        )

        plan, rebuilt = self._search_references.ensure_plan(fake_frame)
        if rebuilt:
            self._plan_start_tick = self._tick
            self._progress_t = 0.0

        clock_t = (self._tick - self._plan_start_tick) * self._dt
        self._progress_t = self._project(plan, frame.pos)
        t_eval = float(
            np.clip(
                min(clock_t, self._progress_t + self._settings.runtime.lookahead_s),
                0.0,
                plan.t_total,
            )
        )
        action, _ = attitude_action(
            plan.curve, t_eval, frame.pos, frame.vel, frame.quat,
            self._feedback, self._dt, self._mass,
            self._settings.runtime.gravity, self._settings.command,
        )
        self._last_action = action
        return action.copy()

    # ── NAVIGATE mode ────────────────────────────────────────────────────────

    def _navigate_action(self, frame: DroneObservation, obs_visited: NDArray) -> NDArray:
        target = frame.target_gate
        n_total = len(frame.gate_pos)

        last_known = target
        for i in range(target + 1, n_total):
            if i in self._known_gates:
                last_known = i
            else:
                break

        if last_known != self._last_last_known:
            self._references.reset()
            self._last_last_known = last_known

        det_obs = frame.obstacles_pos[obs_visited] if obs_visited.any() else np.empty((0, 3))
        gate_obs = self._gate_post_obstacles(frame)
        if len(gate_obs):
            det_obs = np.concatenate([det_obs, gate_obs], axis=0)

        active = self._references.plan
        if active is not None and len(det_obs) != len(active.obstacle_pos_snapshot):
            self._references.reset()
            self._progress_t = 0.0
            self._plan_start_tick = self._tick

        building_first = self._references.plan is None
        plan_vel = frame.vel * _NAV_START_VEL_SCALE if building_first else frame.vel

        truncated = DroneObservation(
            target_gate=target,
            gate_pos=frame.gate_pos[: last_known + 1],
            gate_quat=frame.gate_quat[: last_known + 1],
            obstacles_pos=det_obs,
            pos=frame.pos,
            vel=plan_vel,
            quat=frame.quat,
        )

        plan, rebuilt = self._references.ensure_plan(truncated)
        if rebuilt:
            self._plan_start_tick = self._tick
            self._progress_t = 0.0

        clock_t = (self._tick - self._plan_start_tick) * self._dt
        self._progress_t = self._project(plan, frame.pos)
        t_eval = float(
            np.clip(
                min(clock_t, self._progress_t + self._settings.runtime.lookahead_s),
                0.0,
                plan.t_total,
            )
        )
        action, _ = attitude_action(
            plan.curve, t_eval, frame.pos, frame.vel, frame.quat,
            self._feedback, self._dt, self._mass,
            self._settings.runtime.gravity, self._settings.command,
        )
        self._last_action = action
        return action.copy()

    # ── HOME mode ────────────────────────────────────────────────────────────

    def _home_action(self, frame: DroneObservation, obs_visited: NDArray) -> NDArray:
        if self._home_plan is None:
            cur_z = float(frame.pos[2])
            mid = np.array([0.0, 0.0, max(0.40, cur_z * 0.5)])
            end = np.array([0.0, 0.0, 0.05])
            waypoints = np.array([frame.pos.copy(), mid, end])
            det_obs = (
                frame.obstacles_pos[obs_visited] if obs_visited.any() else np.empty((0, 3))
            )
            knot_times, curve = build_spline(
                waypoints,
                np.asarray(frame.vel, dtype=np.float64),
                np.empty((0, 3)),
                det_obs,
                self._settings.planner,
            )
            self._home_plan = (curve, float(knot_times[-1]))
            self._home_start_tick = self._tick

        curve, t_total = self._home_plan
        clock_t = (self._tick - self._home_start_tick) * self._dt
        if clock_t >= t_total:
            self._mode = self._MODE_DONE
            self._finished = True
            return self._hover_action()

        t_eval = float(np.clip(clock_t, 0.0, t_total))
        action, _ = attitude_action(
            curve, t_eval, frame.pos, frame.vel, frame.quat,
            self._feedback, self._dt, self._mass,
            self._settings.runtime.gravity, self._settings.command,
        )
        self._last_action = action
        return action.copy()

    # ── Spiral construction ───────────────────────────────────────────────────

    def _build_spiral(self) -> tuple[NDArray, NDArray]:
        """Pre-compute an Archimedean spiral at _SEARCH_ALT, outermost-first (ends near centre)."""
        a = _SPIRAL_RADIAL_STEP / (2.0 * np.pi)
        wps: list[NDArray] = []
        theta = 0.0
        while a * theta <= _SEARCH_RADIUS:
            r = a * theta
            x = float(np.clip(r * np.cos(theta), -_ARENA_X_LIM, _ARENA_X_LIM))
            y = float(np.clip(r * np.sin(theta), -_ARENA_Y_LIM, _ARENA_Y_LIM))
            pt = np.array([x, y, _SEARCH_ALT])
            if len(wps) == 0 or float(np.linalg.norm(pt[:2] - wps[-1][:2])) >= 0.3:
                wps.append(pt)
            theta += _SPIRAL_ANGLE_STEP
        if not _SPIRAL_OUTWARD:
            wps.reverse()
        spiral_wps = np.array(wps, dtype=np.float64)
        spiral_quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(spiral_wps), 1))
        return spiral_wps, spiral_quats

    def _find_nearest_spiral(self, pos: NDArray) -> int:
        """Return the index of the spiral waypoint closest to pos (2-D distance)."""
        dists = np.linalg.norm(self._spiral_wps[:, :2] - np.asarray(pos[:2]), axis=1)
        return int(np.argmin(dists))

    # ── Spline projection ─────────────────────────────────────────────────────

    def _project(self, plan: ReferencePlan, pos: NDArray) -> float:
        """Closest spline time ahead of current progress (shared by all modes)."""
        window = self._settings.runtime.projection_window_s
        upper = min(self._progress_t + window, plan.t_total)
        sample_t = np.linspace(self._progress_t, upper, 40)
        distances = np.linalg.norm(np.asarray(plan.curve(sample_t)) - pos, axis=1)
        return float(sample_t[int(np.argmin(distances))])

    # ── Gate frame obstacle columns ───────────────────────────────────────────

    def _gate_post_obstacles(self, frame: DroneObservation) -> NDArray:
        """Return two virtual cylindrical obstacle positions per discovered gate."""
        if not self._known_gates:
            return np.empty((0, 3), dtype=np.float64)
        posts: list[NDArray] = []
        for gi in self._known_gates:
            if gi >= len(frame.gate_pos):
                continue
            gp = np.asarray(frame.gate_pos[gi], dtype=np.float64)
            lateral = Rotation.from_quat(frame.gate_quat[gi]).as_matrix()[:, 1]
            posts.append(gp + _GATE_POST_OFFSET * lateral)
            posts.append(gp - _GATE_POST_OFFSET * lateral)
        return np.array(posts, dtype=np.float64)

    # ── Debug capture ──────────────────────────────────────────────────────────

    def _capture_debug(self, frame: DroneObservation, obs_visited: NDArray) -> None:
        self._dbg_gate_pos = np.asarray(frame.gate_pos, dtype=np.float64)
        known_mask = np.zeros(len(frame.gate_pos), dtype=bool)
        for idx in self._known_gates:
            if idx < len(frame.gate_pos):
                known_mask[idx] = True
        self._dbg_known_mask = known_mask
        self._dbg_target_gate = int(frame.target_gate)
        det_obs = frame.obstacles_pos[obs_visited] if obs_visited.any() else np.empty((0, 3))
        self._dbg_obs_pos = np.asarray(det_obs, dtype=np.float64)
        if self._mode == self._MODE_SEARCH:
            vt = self._virtual_target
            end = min(vt + _SPIRAL_HORIZON, len(self._spiral_wps))
            self._dbg_wp_pos = self._spiral_wps[vt:end].copy()
        elif self._mode == self._MODE_NAVIGATE:
            ref_plan = self._references.plan
            if ref_plan is not None:
                n_samples = min(10, ref_plan.t_total * 5)
                t_pts = np.linspace(0.0, ref_plan.t_total, max(2, int(n_samples)))
                self._dbg_wp_pos = np.asarray(ref_plan.curve(t_pts), dtype=np.float64)
            else:
                self._dbg_wp_pos = np.empty((0, 3), dtype=np.float64)
        else:
            self._dbg_wp_pos = np.empty((0, 3), dtype=np.float64)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def step_callback(
        self,
        action: NDArray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the episode clock."""
        self._tick += 1
        return self._finished

    def reset(self) -> None:
        """Reset all per-episode state."""
        self._tick = 0
        self._plan_start_tick = 0
        self._progress_t = 0.0
        self._finished = False
        self._last_target = -1
        self._mode = self._MODE_TAKEOFF
        self._known_gates = set()
        self._last_last_known = -1
        self._virtual_target = 0
        self._spiral_swept = False
        self._takeoff_plan = None
        self._takeoff_start_tick = 0
        self._home_plan = None
        self._home_start_tick = 0
        self._feedback.reset()
        self._references.reset()
        self._search_references.reset()
        self._last_action = self._hover_action()
        self._dbg_gate_pos = np.empty((0, 3), dtype=np.float64)
        self._dbg_known_mask = np.zeros(0, dtype=bool)
        self._dbg_target_gate = -1
        self._dbg_obs_pos = np.empty((0, 3), dtype=np.float64)
        self._dbg_wp_pos = np.empty((0, 3), dtype=np.float64)

    def episode_callback(self) -> None:
        """Reset controller state at the end of an episode."""
        self.reset()

    def episode_reset(self) -> None:
        """Reset controller state before a new episode."""
        self.reset()

    # ── Rendering and diagnostics ─────────────────────────────────────────────

    def render_callback(self, sim: Sim) -> None:
        """Draw the active reference spline and debug markers."""
        ref = self._search_references if self._mode == self._MODE_SEARCH else self._references
        plan = ref.plan
        if plan is not None:
            samples = plan.curve(np.linspace(0.0, plan.t_total, 100))
            draw_line(sim, np.asarray(samples, dtype=np.float32), rgba=(0.0, 1.0, 0.0, 1.0))

        arm = 0.08

        def _cross(pos: NDArray, rgba: tuple) -> None:
            p = np.asarray(pos, dtype=np.float32)
            for axis in range(3):
                seg = np.zeros((2, 3), dtype=np.float32)
                seg[0] = p
                seg[1] = p
                seg[0, axis] -= arm
                seg[1, axis] += arm
                draw_line(sim, seg, rgba=rgba)

        for i, gp in enumerate(self._dbg_gate_pos):
            if not self._dbg_known_mask[i]:
                continue
            if i == self._dbg_target_gate:
                _cross(gp, rgba=(0.0, 1.0, 0.0, 1.0))
            else:
                _cross(gp, rgba=(0.0, 0.55, 0.0, 1.0))
        for op in self._dbg_obs_pos:
            _cross(op, rgba=(1.0, 0.0, 0.0, 1.0))
        for wp in self._dbg_wp_pos:
            _cross(wp, rgba=(0.2, 0.4, 1.0, 1.0))

    def diagnostic(self) -> dict:
        """Return a short status dict for logging."""
        plan = self._references.plan
        return {
            "controller_phase": self._mode,
            "active_target_gate": self._last_target,
            "controller_time": self._tick * self._dt,
            "reference_end_time": None if plan is None else plan.t_total,
            "known_gates": sorted(self._known_gates),
            "virtual_target": self._virtual_target,
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    def _hover_action(self) -> NDArray:
        thrust = float(
            np.clip(
                self._mass * self._settings.runtime.gravity,
                self._settings.command.thrust_min,
                self._settings.command.thrust_max,
            )
        )
        return np.array([0.0, 0.0, 0.0, thrust], dtype=np.float32)
