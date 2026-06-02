"""search-then-navigate drone racing controller with shared reference manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line
from drone_models.core import load_params
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.online_planner.attitude import attitude_action
from lsy_drone_racing.control.online_planner.feedback import CascadedPid
from lsy_drone_racing.control.online_planner.settings import ControllerSettings, PlannerSettings
from lsy_drone_racing.control.online_planner.state import DroneObservation, parse_observation
from lsy_drone_racing.control.online_planner.timing import build_spline
from lsy_drone_racing.control.online_planner.trajectory import ReferenceManager, ReferencePlan

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

_SEARCH_ALT = 1.8
_SPIRAL_RADIAL_STEP = 0.6
_SPIRAL_ANGLE_STEP = np.pi / 6
_SPIRAL_ADVANCE_RADIUS = 0.6
_SPIRAL_HORIZON = 3
_ARENA_X_LIM = 2.2
_ARENA_Y_LIM = 1.3
_GATE_POST_OFFSET = 0.36
_TAKEOFF_ALT = 1.8
_TAKEOFF_Z_TOL = 0.05
_TAKEOFF_TIME_MARGIN = 1.0


class GateSearchV1(Controller):
    """search-then-navigate controller for fully random gate positions."""

    _MODE_TAKEOFF = "TAKEOFF"
    _MODE_SEARCH = "SEARCH"
    _MODE_NAVIGATE = "NAVIGATE"
    _MODE_HOME = "HOME"
    _MODE_DONE = "DONE"

    def __init__(self, obs: dict, info: dict, config: dict):
        """initialise timing parameters pid reference manager and spiral."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("this controller requires attitude control mode")
        self._settings = ControllerSettings(planner=PlannerSettings(d_pre=0.60, d_post=0.40))
        self._search_settings = ControllerSettings(
            planner=PlannerSettings(v_cruise=0.55, max_speed=0.70)
        )
        self._freq = float(config.env.freq)
        self._dt = 1.0 / self._freq
        params = load_params("so_rpy_rotor", config.sim.drone_model)
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
        self._takeoff_plan: tuple | None = None
        self._takeoff_start_tick = 0
        self._home_plan: tuple | None = None
        self._home_start_tick = 0
        self._dbg_gate_pos: NDArray = np.empty((0, 3), dtype=np.float64)
        self._dbg_known_mask: NDArray = np.zeros(0, dtype=bool)
        self._dbg_target_gate: int = -1
        self._dbg_obs_pos: NDArray = np.empty((0, 3), dtype=np.float64)
        self._dbg_wp_pos: NDArray = np.empty((0, 3), dtype=np.float64)

    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray:
        """return an attitude command for the current step."""
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

        elif self._mode == self._MODE_SEARCH and target in self._known_gates:
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
            n_wps = len(self._spiral_wps)
            end = min(vt + _SPIRAL_HORIZON, n_wps)
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

        return action

    def _takeoff_action(self, frame: DroneObservation, obs_visited: NDArray) -> NDArray:
        """climb straight up to the takeoff altitude then hand off to search."""
        clock_t = (self._tick - self._takeoff_start_tick) * self._dt
        t_total = self._takeoff_plan[1] if self._takeoff_plan is not None else 0.0
        reached = float(frame.pos[2]) >= _TAKEOFF_ALT - _TAKEOFF_Z_TOL
        overran = self._takeoff_plan is not None and clock_t >= t_total + _TAKEOFF_TIME_MARGIN
        if reached or overran:
            self._mode = self._MODE_SEARCH
            self._virtual_target = self._find_nearest_spiral(frame.pos)
            self._search_references.reset()
            self._progress_t = 0.0
            self._plan_start_tick = self._tick
            return self._search_action(frame, obs_visited)

        if self._takeoff_plan is None:
            start = np.asarray(frame.pos, dtype=np.float64).copy()
            target = np.array([start[0], start[1], _TAKEOFF_ALT])
            waypoints = np.array([start, target])
            det_obs = frame.obstacles_pos[obs_visited] if obs_visited.any() else np.empty((0, 3))
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
            curve,
            t_eval,
            frame.pos,
            frame.vel,
            frame.quat,
            self._feedback,
            self._dt,
            self._mass,
            self._settings.runtime.gravity,
            self._settings.command,
        )
        self._last_action = action
        return action.copy()

    def _search_action(self, frame: DroneObservation, obs_visited: NDArray) -> NDArray:
        del obs_visited
        det_obs = np.empty((0, 3))

        vt = self._virtual_target
        dist2d = float(np.linalg.norm(frame.pos[:2] - self._spiral_wps[vt, :2]))
        if dist2d < _SPIRAL_ADVANCE_RADIUS:
            next_vt = vt + 1
            if next_vt >= len(self._spiral_wps):
                next_vt = 0
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
            plan.curve,
            t_eval,
            frame.pos,
            frame.vel,
            frame.quat,
            self._feedback,
            self._dt,
            self._mass,
            self._settings.runtime.gravity,
            self._settings.command,
        )
        self._last_action = action
        return action.copy()

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

        truncated = DroneObservation(
            target_gate=target,
            gate_pos=frame.gate_pos[: last_known + 1],
            gate_quat=frame.gate_quat[: last_known + 1],
            obstacles_pos=det_obs,
            pos=frame.pos,
            vel=frame.vel,
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
            plan.curve,
            t_eval,
            frame.pos,
            frame.vel,
            frame.quat,
            self._feedback,
            self._dt,
            self._mass,
            self._settings.runtime.gravity,
            self._settings.command,
        )
        self._last_action = action
        return action.copy()

    def _home_action(self, frame: DroneObservation, obs_visited: NDArray) -> NDArray:
        if self._home_plan is None:
            cur_z = float(frame.pos[2])
            mid = np.array([0.0, 0.0, max(0.40, cur_z * 0.5)])
            end = np.array([0.0, 0.0, 0.05])
            waypoints = np.array([frame.pos.copy(), mid, end])
            det_obs = frame.obstacles_pos[obs_visited] if obs_visited.any() else np.empty((0, 3))
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
            curve,
            t_eval,
            frame.pos,
            frame.vel,
            frame.quat,
            self._feedback,
            self._dt,
            self._mass,
            self._settings.runtime.gravity,
            self._settings.command,
        )
        self._last_action = action
        return action.copy()

    def _build_spiral(self) -> tuple[NDArray, NDArray]:
        """pre-compute an archimedean spiral covering the arena at search altitude."""
        a = _SPIRAL_RADIAL_STEP / (2.0 * np.pi)
        arena_diag = float(np.sqrt(_ARENA_X_LIM**2 + _ARENA_Y_LIM**2))
        wps: list[NDArray] = []
        theta = 0.0
        while a * theta <= arena_diag + _SPIRAL_RADIAL_STEP:
            r = a * theta
            x = float(np.clip(r * np.cos(theta), -_ARENA_X_LIM, _ARENA_X_LIM))
            y = float(np.clip(r * np.sin(theta), -_ARENA_Y_LIM, _ARENA_Y_LIM))
            pt = np.array([x, y, _SEARCH_ALT])
            if len(wps) == 0 or float(np.linalg.norm(pt[:2] - wps[-1][:2])) >= 0.3:
                wps.append(pt)
            theta += _SPIRAL_ANGLE_STEP
        spiral_wps = np.array(wps, dtype=np.float64)
        spiral_quats = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (len(spiral_wps), 1))
        return spiral_wps, spiral_quats

    def _find_nearest_spiral(self, pos: NDArray) -> int:
        """return the index of the spiral waypoint closest to pos."""
        dists = np.linalg.norm(self._spiral_wps[:, :2] - np.asarray(pos[:2]), axis=1)
        return int(np.argmin(dists))

    def _project(self, plan: ReferencePlan, pos: NDArray) -> float:
        """closest spline time ahead of current progress shared by all modes."""
        window = self._settings.runtime.projection_window_s
        upper = min(self._progress_t + window, plan.t_total)
        sample_t = np.linspace(self._progress_t, upper, 40)
        distances = np.linalg.norm(np.asarray(plan.curve(sample_t)) - pos, axis=1)
        return float(sample_t[int(np.argmin(distances))])

    def _gate_post_obstacles(self, frame: DroneObservation) -> NDArray:
        """return two virtual obstacle columns per discovered gate."""
        if not self._known_gates:
            return np.empty((0, 3), dtype=np.float64)
        posts: list[NDArray] = []
        for gi in self._known_gates:
            gp = np.asarray(frame.gate_pos[gi], dtype=np.float64)
            lateral = Rotation.from_quat(frame.gate_quat[gi]).as_matrix()[:, 1]
            posts.append(gp + _GATE_POST_OFFSET * lateral)
            posts.append(gp - _GATE_POST_OFFSET * lateral)
        return np.array(posts, dtype=np.float64)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """advance the episode clock."""
        self._tick += 1
        return self._finished

    def reset(self) -> None:
        """reset all per-episode state."""
        self._tick = 0
        self._plan_start_tick = 0
        self._progress_t = 0.0
        self._finished = False
        self._last_target = -1
        self._mode = self._MODE_TAKEOFF
        self._known_gates = set()
        self._last_last_known = -1
        self._virtual_target = 0
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
        self.reset()

    def episode_reset(self) -> None:
        self.reset()

    def render_callback(self, sim: Sim) -> None:
        """draw the active reference spline and debug markers."""
        ref = self._search_references if self._mode == self._MODE_SEARCH else self._references
        plan = ref.plan
        if plan is not None:
            samples = plan.curve(np.linspace(0.0, plan.t_total, 100))
            draw_line(sim, np.asarray(samples, dtype=np.float32), rgba=(0.0, 1.0, 0.0, 1.0))

        arm = 0.08

        def _cross(pos: NDArray, rgba: tuple) -> None:
            """draw a three axis cross at pos using three draw_line calls."""
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
        """return a short status dict for logging."""
        plan = self._references.plan
        return {
            "controller_phase": self._mode,
            "active_target_gate": self._last_target,
            "controller_time": self._tick * self._dt,
            "reference_end_time": None if plan is None else plan.t_total,
            "known_gates": sorted(self._known_gates),
            "virtual_target": self._virtual_target,
        }

    def _hover_action(self) -> NDArray:
        thrust = float(
            np.clip(
                self._mass * self._settings.runtime.gravity,
                self._settings.command.thrust_min,
                self._settings.command.thrust_max,
            )
        )
        return np.array([0.0, 0.0, 0.0, thrust], dtype=np.float32)
