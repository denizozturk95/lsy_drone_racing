"""Level-3 drone racing controller (v12): time-OPTIMAL navigate (TOPP) for fast laps.

GateSearchV12 keeps v10/v11's SEARCH/HOME architecture and robustness, and replaces the NAVIGATE
trajectory paradigm. v10/v11 time-parameterized the gate path with a fixed cruise speed plus
heuristic slowdowns (peri-gate / turn / t_min_seg); a navigate-anatomy study found this forced 85%
of navigate below 1 m/s even on straightaways, and that raising the cruise cap just crashed the
drone (the heuristic path was not dynamically feasible to track at speed).

v12 navigate = Time-Optimal Path Parameterization (TOPP)
--------------------------------------------------------
The gate *geometry* (run-in / gate / run-out / clearance waypoints) is unchanged, but its time
parameterization is now computed to be as fast as the drone's dynamics allow (see
online_planner.timing.build_spline_topp):

  * curvature limit:  v(s) <= sqrt(a_lat / kappa(s)) — straightaways run at ``max_speed``; the path
    slows ONLY through curves and gate turns, by exactly as much as the lateral-accel budget needs.
  * precision caps:   v <= topp_v_gate near a gate opening, v <= topp_v_obs near an obstacle, plus
    the descent-rate cap onto low gates.
  * feasibility:      a forward/backward pass bounds tangential accel by ``topp_a_tang``, so the
    whole trajectory respects the drone's real thrust/tilt authority and is therefore TRACKABLE at
    speed. Feedforward is raised (the reference is now feasible, so anticipating it helps instead of
    fighting the tracker).

Limits come from the cf21B_500: thrust-to-weight ~1.88 (collective thrust_max 0.8 N, mass 0.043 kg)
gives ~13-15 m/s² of usable lateral accel; topp_a_lat/topp_a_tang are set well inside that.

SEARCH, TAKEOFF and HOME keep the classic time-allocation (their planners have use_topp=False):
TOPP is only for the gate course, where carrying speed between gates is the whole point.

RESULT (30 Level-3 seeds): a genuine Pareto improvement over v10
---------------------------------------------------------------
    controller                  finish    lap(mean / min)
    v10 baseline                36.7%      25.6 / 19.9 s
    v12 (this, all fixes)       36.7%      21.9 / 18.0 s   <- SAME reliability, ~14% faster

How it got there. Plain TOPP started at only 20% finish (faster laps but more crashes). A crash
locator (scripts/crash_locate.py) showed the dominant failure was the path clipping a NON-target
gate's frame while transiting past it, plus a secondary at-gate cross-track tail. Three stacked
fixes closed the gap (20% -> 30% -> 35% -> 36.7%):
  1. SHORT planning horizon (_NAV_HORIZON_GATES) + gate WALLS (_gate_keepouts): plan/cross only the
     next 1-2 gates and wall off every OTHER known gate, so the spline routes AROUND it instead of
     grazing its frame. Biggest single gain.
  2. A MODERATE, v12-ONLY tighter tracker (higher position/velocity gains + ff 0.75) to pull in the
     ~0.14 m gate-zone cross-track tail. The shared FeedbackSettings defaults (v9/deploy) are NOT
     touched.
  3. Lower straightaway/gate speed (vmax 2.2, v_gate 0.9, gentler braking) to remove the high-speed
     gate-clip cluster.

The result is the first controller that is BOTH as reliable as v10 AND meaningfully faster (min lap
18 s). <10 s remains infeasible (it is below the known-track lap floor). This is the controller for
level3.toml; GateSearchV9 remains the known-track/deploy controller.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line
from drone_models.core import load_params
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.online_planner.attitude import attitude_action
from lsy_drone_racing.control.online_planner.feedback import CascadedPid
from lsy_drone_racing.control.online_planner.settings import (
    CommandSettings,
    ControllerSettings,
    FeedbackProfile,
    FeedbackSettings,
    PlannerSettings,
    RuntimeSettings,
)
from lsy_drone_racing.control.online_planner.state import DroneObservation, parse_observation
from lsy_drone_racing.control.online_planner.timing import build_spline
from lsy_drone_racing.control.online_planner.trajectory import ReferenceManager, ReferencePlan

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

# ── Spiral / search constants (unchanged from v10/v11) ────────────────────────
_SEARCH_ALT = 1.8
_SPIRAL_RADIAL_STEP = 0.6
_SPIRAL_ANGLE_STEP = np.pi / 6
_SPIRAL_ADVANCE_RADIUS = 0.6
_SPIRAL_HORIZON = 3
_SPIRAL_OUTWARD = False
_SEARCH_RADIUS = 2.2
_ARENA_X_LIM = 1.9
_ARENA_Y_LIM = 1.0
_GATE_POST_OFFSET = 0.30
# ── Takeoff constants ─────────────────────────────────────────────────────────
_TAKEOFF_ALT = 1.8
_TAKEOFF_Z_TOL = 0.05
_TAKEOFF_TIME_MARGIN = 1.0
# ── Search speeds ─────────────────────────────────────────────────────────────
_V_CRUISE_SEARCH = 2.5
_VMAX_SEARCH = 3.0
# ── TOPP navigate parameters (the v12 change) ─────────────────────────────────
# max_speed is the straightaway ceiling; the curvature/accel limits decide where it is actually
# reached. v_gate is the precise gate-crossing speed. a_lat/a_tang are kept well inside the drone's
# ~13-15 m/s² envelope so the time-optimal profile stays trackable.
# Best-validated v12 envelope: fast straights (TOPP), v10-equivalent gate-crossing speed. Gives the
# fastest finishing laps (~19-22 s vs v10's 24-29 s) at a reliability cost — the residual at-gate
# clips are an inner-loop tracking-precision limit (raising/lowering the approach speed both hurt),
# not a trajectory-shape problem (the path is gate-exact and a_lat-bounded by construction).
_NAV_VMAX = 2.2           # straightaway ceiling (lowered from 2.6: high-speed gate clips were a cluster)
_TOPP_A_LAT = 4.5         # cornering accel bound (well inside the ~13-15 m/s² envelope)
_TOPP_A_TANG = 3.0        # accel/brake bound (gentler → settles earlier into the gate)
_TOPP_V_GATE = 0.9        # gate-crossing speed (slightly under v10's 1.0 for threading margin)
_TOPP_V_OBS = 1.0
_TOPP_V_STOP = 0.4
_NAV_FEEDFORWARD = 0.75    # anticipate the feasible reference (reduces cross-track lag at the gate)
_NAV_LOOKAHEAD_OVERRIDE = 0.20
# MODERATE tighter tracker for v12 ONLY (shared FeedbackSettings defaults that v9/deploy use are
# untouched). Pulls in the ~0.14 m gate-zone cross-track tail that — once the gate-wall avoidance
# removed the off-path clips — became the dominant remaining at-gate crash. Kept modest (esp. the
# z-gain) to avoid the aggressive-descent ground crashes a stiffer tune caused.
_FB_KP = (0.85, 0.85, 1.80)   # position gains (default 0.60/0.60/1.65)
_FB_KI = (0.05, 0.05, 0.05)
_FB_KD = (0.45, 0.45, 0.55)   # velocity-loop gains (default 0.35/0.35/0.50)
_FB_OUTER_I_LIMIT = (1.5, 1.5, 0.4)
# ── NAVIGATE gate-approach geometry (unchanged from v10/v11) ──────────────────
_NAV_D_PRE = 0.60
_NAV_D_POST = 0.40
_NAV_R_OBS = 0.12
_PERI_GATE_RADIUS = 0.75   # radius of the v_gate (slow, precise) zone
# Plan only this many gates ahead of the target. A short horizon keeps each navigate spline short so
# it cannot bulge sideways into a gate further down the order; the gates beyond the horizon are
# instead BLOCKED (routed around) by _gate_keepouts. The dominant Level-3 crash was the path
# clipping a NON-target gate's frame mid-transit (crash_locate.py); this is the fix.
_NAV_HORIZON_GATES = 1
# Off-path (non-chain) known gates are walled off with this many keep-out posts spanning the frame
# width, so the path detours around them instead of grazing a side/top bar.
_GATE_BLOCK_OFFSETS = (-0.36, -0.18, 0.0, 0.18, 0.36)
_NAV_START_VEL_SCALE = 0.5
_NAV_LOOKAHEAD = _NAV_LOOKAHEAD_OVERRIDE
_DISCOVER_ALL_FIRST = True
# ── Robustness (from v10): geofence + tighter reversal + descent cap ──────────
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


class GateSearchV12(Controller):
    """Level-3 search-then-navigate controller with a time-optimal (TOPP) navigate phase."""

    _MODE_TAKEOFF = "TAKEOFF"
    _MODE_SEARCH = "SEARCH"
    _MODE_NAVIGATE = "NAVIGATE"
    _MODE_HOME = "HOME"
    _MODE_DONE = "DONE"

    def __init__(self, obs: dict, info: dict, config: dict):
        """Initialise timing, drone parameters, PID, reference manager, and spiral."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("GateSearchV12 requires env.control_mode = 'attitude'.")
        arena_low, arena_high = _arena_bounds(config)
        nav_planner = PlannerSettings(
            d_pre=_NAV_D_PRE, d_post=_NAV_D_POST, max_speed=_NAV_VMAX,
            peri_gate_radius=_PERI_GATE_RADIUS, r_obs=_NAV_R_OBS,
            orient_gates_to_travel=False, early_climb=True,
            reversal_swing_m=_REVERSAL_SWING_M, reversal_apex_m=_REVERSAL_APEX_M,
            geofence_margin=_GEOFENCE_MARGIN, arena_low=arena_low, arena_high=arena_high,
            max_descent_rate=_NAV_MAX_DESCENT_RATE,
            use_topp=True, topp_a_lat=_TOPP_A_LAT, topp_a_tang=_TOPP_A_TANG,
            topp_v_gate=_TOPP_V_GATE, topp_v_obs=_TOPP_V_OBS, topp_v_stop=_TOPP_V_STOP,
        )
        self._settings = ControllerSettings(
            planner=nav_planner,
            feedback=FeedbackSettings(
                profile=FeedbackProfile(
                    np.array(_FB_KP, dtype=np.float64), np.array(_FB_KI, dtype=np.float64),
                    np.array(_FB_KD, dtype=np.float64), np.array(_FB_OUTER_I_LIMIT, dtype=np.float64),
                )
            ),
            command=CommandSettings(feedforward_scale=_NAV_FEEDFORWARD),
            runtime=RuntimeSettings(lookahead_s=_NAV_LOOKAHEAD),
        )
        # Vertical climbs/descents (TAKEOFF, HOME) use the classic builder, not TOPP.
        self._vertical_planner = dataclasses.replace(nav_planner, use_topp=False)
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
                self._vertical_planner,
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
        # Short planning horizon: cross only the next _NAV_HORIZON_GATES gates this plan; the rest
        # are walls to route around (set below). Keeps the spline from bulging into a far gate.
        last_known = min(last_known, target + _NAV_HORIZON_GATES)

        if last_known != self._last_last_known:
            self._references.reset()
            self._last_last_known = last_known

        det_obs = frame.obstacles_pos[obs_visited] if obs_visited.any() else np.empty((0, 3))
        chain = set(range(target, last_known + 1))  # gates this plan crosses (funnel, not block)
        gate_obs = self._gate_keepouts(frame, chain)
        if len(gate_obs):
            det_obs = np.concatenate([det_obs, gate_obs], axis=0) if len(det_obs) else gate_obs

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
                self._vertical_planner,
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

    def _gate_keepouts(self, frame: DroneObservation, chain: set[int]) -> NDArray:
        """Virtual keep-out posts for every known gate.

        Gates in ``chain`` (the ones this plan actually crosses) get the two funnel posts at the
        opening edges so the spline is centred through the opening. Every OTHER known gate is walled
        off across its full frame width, so the path routes AROUND it rather than grazing a frame bar
        — the dominant Level-3 crash was clipping a non-target gate while transiting past it.
        """
        if not self._known_gates:
            return np.empty((0, 3), dtype=np.float64)
        posts: list[NDArray] = []
        for gi in self._known_gates:
            if gi >= len(frame.gate_pos):
                continue
            gp = np.asarray(frame.gate_pos[gi], dtype=np.float64)
            lateral = Rotation.from_quat(frame.gate_quat[gi]).as_matrix()[:, 1]
            if gi in chain:
                posts.append(gp + _GATE_POST_OFFSET * lateral)
                posts.append(gp - _GATE_POST_OFFSET * lateral)
            else:
                for off in _GATE_BLOCK_OFFSETS:
                    posts.append(gp + off * lateral)
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
                n_samples = min(20, ref_plan.t_total * 5)
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
