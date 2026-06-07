"""Known-track controller (v6): one offline minimum-snap trajectory through all gates.

Where v3–v5 use the *reactive* clamped-cubic planner (greedy, rebuilt as poses refine), v6
exploits the fact that on a known/measured track the whole layout is available at ``t = 0``:
it solves ONE globally-smooth minimum-snap trajectory through every gate up front
(:mod:`lsy_drone_racing.control.online_planner.minsnap`).

Design
------
Routing reuses the reactive planner's proven waypoint chain (``build_waypoints`` +
``repair_obstacles``): it already handles gate approach/exit along the normal, short->tall
climbs, the post-gate reversal U-turns, and obstacle push-off. v6 only changes the *time
parameterisation* — a minimum-snap polynomial (``online_planner.minsnap``) through those
waypoints instead of a clamped cubic. Minimising snap (4th derivative) yields low jerk/accel, so
the path stays trackable at higher speed without drifting into frames; segment times are
stretched until peak velocity AND acceleration are feasible.

The trajectory is built once from the start pose and **rebuilt only when a remaining gate's
observed pose drifts** beyond a threshold (sim track-perturbation; on the exact measured real
track it never rebuilds). Tracking, HOME and the known-track assumptions match GateSearchV3.

Use only on a known track (``randomize = false`` / a ``save_track_as_config.py`` output). For
blind Level 3 use :class:`GateSearchV2`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line
from drone_models.core import load_params

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.online_planner.attitude import attitude_action
from lsy_drone_racing.control.online_planner.feedback import CascadedPid
from lsy_drone_racing.control.online_planner.minsnap import MinSnapTrajectory, build_minsnap
from lsy_drone_racing.control.online_planner.settings import ControllerSettings, PlannerSettings
from lsy_drone_racing.control.online_planner.state import DroneObservation, parse_observation
from lsy_drone_racing.control.online_planner.timing import build_spline, repair_obstacles
from lsy_drone_racing.control.online_planner.trajectory import build_waypoints

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

# ── Trajectory parameters ──────────────────────────────────────────────────────
# Routing reuses the reactive planner's proven waypoint chain (build_waypoints + repair_obstacles
# from online_planner): it already handles gate approach/exit along the normal, short->tall
# climbs, the post-gate reversal U-turns, and obstacle push-off. v6 only swaps the time
# parameterisation: a minimum-snap polynomial through those waypoints instead of a clamped cubic,
# which is smoother (lower jerk/accel) and therefore trackable at higher speed.
_PLANNER = PlannerSettings(  # geometry/routing only (d_pre/d_post/r_obs/normal-crossing/climb)
    d_pre=0.60, d_post=0.40, r_obs=0.20, orient_gates_to_travel=False, early_climb=True
)
_V_NOMINAL = 1.6     # m/s — uniform seed speed for per-segment time allocation
_V_MAX = 2.2         # m/s — peak velocity cap on the fitted min-snap (re-capped after fitting)
_A_MAX = 7.0         # m/s^2 — peak acceleration cap. Segments are stretched until BOTH the
#                      velocity and acceleration peaks fit (stretching time by f scales v by 1/f
#                      and a by 1/f^2); without the accel cap the drone cannot track the turn
#                      into a gate at speed and clips the frame.
_T_MIN_SEG = 0.40    # s — minimum segment duration
_LOOKAHEAD = 0.15    # s — reference look-ahead for tracking
_REBUILD_DELTA = 0.08  # m — rebuild when a remaining gate/obstacle pose moves more than this
_PROJ_WINDOW = 0.6   # s — forward window for spline progress projection


class GateSearchV6(Controller):
    """Offline minimum-snap navigate controller for known tracks."""

    _MODE_NAVIGATE = "NAVIGATE"
    _MODE_HOME = "HOME"
    _MODE_DONE = "DONE"

    def __init__(self, obs: dict, info: dict, config: dict):
        """Initialise timing, drone parameters, PID; the trajectory is built on first step."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("GateSearchV6 requires env.control_mode = 'attitude'.")
        self._settings = ControllerSettings()
        self._freq = float(config.env.freq)
        self._dt = 1.0 / self._freq
        params = load_params(config.sim.physics, config.sim.drone_model)
        self._mass = float(params["mass"])
        self._feedback = CascadedPid(self._settings.feedback)
        # Per-episode state
        self._tick = 0
        self._plan_start_tick = 0
        self._progress_t = 0.0
        self._finished = False
        self._last_action = self._hover_action()
        self._last_target = -1
        self._mode = self._MODE_NAVIGATE
        self._traj: MinSnapTrajectory | None = None
        self._built_target = 0
        self._gate_snapshot: NDArray = np.empty((0, 3))
        self._obs_snapshot: NDArray = np.empty((0, 3))
        self._home_plan: tuple | None = None
        self._home_start_tick = 0
        # Debug visualisation
        self._dbg_gate_pos: NDArray = np.empty((0, 3), dtype=np.float64)
        self._dbg_target_gate: int = -1
        self._dbg_obs_pos: NDArray = np.empty((0, 3), dtype=np.float64)

    # ── Main control loop ────────────────────────────────────────────────────

    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray:
        """Return a [roll, pitch, yaw, thrust] attitude command for the current step."""
        frame = parse_observation(obs)
        now = self._tick * self._dt
        self._last_target = int(frame.target_gate)

        if now >= self._settings.runtime.timeout_s:
            self._finished = True
            return self._last_action.copy()

        if frame.target_gate == -1 and self._mode not in (self._MODE_HOME, self._MODE_DONE):
            self._mode = self._MODE_HOME

        if self._mode == self._MODE_NAVIGATE:
            action = self._navigate_action(frame)
        elif self._mode == self._MODE_HOME:
            action = self._home_action(frame)
        else:
            self._finished = True
            return self._last_action.copy()

        self._dbg_gate_pos = np.asarray(frame.gate_pos, dtype=np.float64)
        self._dbg_target_gate = int(frame.target_gate)
        self._dbg_obs_pos = np.asarray(frame.obstacles_pos, dtype=np.float64)
        return action

    # ── NAVIGATE mode ────────────────────────────────────────────────────────

    def _navigate_action(self, frame: DroneObservation) -> NDArray:
        if self._needs_rebuild(frame):
            self._traj = self._build_trajectory(frame)
            self._built_target = frame.target_gate
            self._gate_snapshot = frame.gate_pos.copy()
            self._obs_snapshot = frame.obstacles_pos.copy()
            self._plan_start_tick = self._tick
            self._progress_t = 0.0

        traj = self._traj
        clock_t = (self._tick - self._plan_start_tick) * self._dt
        self._progress_t = self._project(traj, frame.pos)
        t_eval = float(np.clip(min(clock_t, self._progress_t + _LOOKAHEAD), 0.0, traj.t_total))
        action, _ = attitude_action(
            traj, t_eval, frame.pos, frame.vel, frame.quat,
            self._feedback, self._dt, self._mass,
            self._settings.runtime.gravity, self._settings.command,
        )
        self._last_action = action
        return action.copy()

    def _needs_rebuild(self, frame: DroneObservation) -> bool:
        """Rebuild on first call or when a remaining gate/obstacle pose drifts past threshold."""
        if self._traj is None:
            return True
        target = frame.target_gate
        rem = frame.gate_pos[target:]
        snap = self._gate_snapshot[target:] if target < len(self._gate_snapshot) else rem
        if len(rem) != len(snap):
            return True
        if len(rem) and float(np.max(np.linalg.norm(rem - snap, axis=1))) > _REBUILD_DELTA:
            return True
        if len(frame.obstacles_pos) == len(self._obs_snapshot) and len(self._obs_snapshot):
            if float(np.max(np.linalg.norm(frame.obstacles_pos - self._obs_snapshot, axis=1))) > (
                _REBUILD_DELTA
            ):
                return True
        return False

    # ── Trajectory construction ────────────────────────────────────────────────

    def _build_trajectory(self, frame: DroneObservation) -> MinSnapTrajectory:
        """Build the reactive planner's routed waypoint chain, then fit min-snap through it.

        ``build_waypoints`` + ``repair_obstacles`` give an obstacle-aware, reversal-aware,
        climb-aware waypoint chain (the same routing v5 uses). We then time-parameterise it as a
        minimum-snap polynomial (peak velocity + acceleration capped) instead of a clamped cubic.
        """
        gates_pos = np.asarray(frame.gate_pos, dtype=np.float64)
        obstacles = np.asarray(frame.obstacles_pos, dtype=np.float64)
        start_vel = np.asarray(frame.vel, dtype=np.float64)
        waypoints = build_waypoints(
            frame.pos, start_vel, gates_pos, frame.gate_quat, obstacles,
            frame.target_gate, _PLANNER,
        )
        waypoints, _, _ = repair_obstacles(waypoints, start_vel, gates_pos, obstacles, _PLANNER)
        # Pin only the endpoints: start = current velocity, final stop = rest; interior free.
        vels: list[NDArray | None] = [start_vel] + [None] * (len(waypoints) - 2) + [np.zeros(3)]
        return self._build_capped(waypoints, vels)

    def _build_capped(
        self, waypoints: NDArray, vels: list[NDArray | None]
    ) -> MinSnapTrajectory:
        """Allocate times by distance, then stretch until peak velocity AND accel are feasible.

        Stretching all segment times by ``f`` scales velocity by ``1/f`` and acceleration by
        ``1/f^2``; we take ``f = max(peak_v/V_MAX, sqrt(peak_a/A_MAX))`` so both peaks fit.
        """
        seg_len = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        times = np.maximum(seg_len / _V_NOMINAL, _T_MIN_SEG)
        traj = build_minsnap(waypoints, times, vels)
        for _ in range(8):
            samples = np.linspace(0.0, traj.t_total, max(100, 16 * len(waypoints)))
            peak_v = float(np.max(np.linalg.norm(traj.derivative(1)(samples), axis=1)))
            peak_a = float(np.max(np.linalg.norm(traj.derivative(2)(samples), axis=1)))
            factor = max(peak_v / _V_MAX, np.sqrt(peak_a / _A_MAX))
            if factor <= 1.03:
                break
            times = times * factor
            traj = build_minsnap(waypoints, times, vels)
        return traj

    def _project(self, traj: MinSnapTrajectory, pos: NDArray) -> float:
        """Closest trajectory time ahead of current progress."""
        upper = min(self._progress_t + _PROJ_WINDOW, traj.t_total)
        sample_t = np.linspace(self._progress_t, upper, 40)
        distances = np.linalg.norm(np.asarray(traj(sample_t)) - pos, axis=1)
        return float(sample_t[int(np.argmin(distances))])

    # ── HOME mode ────────────────────────────────────────────────────────────

    def _home_action(self, frame: DroneObservation) -> NDArray:
        if self._home_plan is None:
            cur_z = float(frame.pos[2])
            mid = np.array([0.0, 0.0, max(0.40, cur_z * 0.5)])
            end = np.array([0.0, 0.0, 0.05])
            waypoints = np.array([frame.pos.copy(), mid, end])
            knot_times, curve = build_spline(
                waypoints,
                np.asarray(frame.vel, dtype=np.float64),
                np.empty((0, 3)),
                np.empty((0, 3)),
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
        self._mode = self._MODE_NAVIGATE
        self._traj = None
        self._built_target = 0
        self._gate_snapshot = np.empty((0, 3))
        self._obs_snapshot = np.empty((0, 3))
        self._home_plan = None
        self._home_start_tick = 0
        self._feedback.reset()
        self._last_action = self._hover_action()
        self._dbg_gate_pos = np.empty((0, 3), dtype=np.float64)
        self._dbg_target_gate = -1
        self._dbg_obs_pos = np.empty((0, 3), dtype=np.float64)

    def episode_callback(self) -> None:
        """Reset controller state at the end of an episode."""
        self.reset()

    def episode_reset(self) -> None:
        """Reset controller state before a new episode."""
        self.reset()

    # ── Rendering and diagnostics ─────────────────────────────────────────────

    def render_callback(self, sim: Sim) -> None:
        """Draw the active min-snap trajectory and gate/obstacle markers."""
        if self._traj is not None:
            samples = self._traj(np.linspace(0.0, self._traj.t_total, 120))
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
            rgba = (0.0, 1.0, 0.0, 1.0) if i == self._dbg_target_gate else (0.0, 0.55, 0.0, 1.0)
            _cross(gp, rgba=rgba)
        for op in self._dbg_obs_pos:
            _cross(op, rgba=(1.0, 0.0, 0.0, 1.0))

    def diagnostic(self) -> dict:
        """Return a short status dict for logging."""
        return {
            "controller_phase": self._mode,
            "active_target_gate": self._last_target,
            "controller_time": self._tick * self._dt,
            "reference_end_time": None if self._traj is None else self._traj.t_total,
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
