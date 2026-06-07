"""Known-track navigate controller (v4): no search, with early climb to tall gates.

GateSearchV4 extends GateSearchV3 (the no-search sibling of
:class:`~lsy_drone_racing.control.gate_search_v2`) with one change: it sets
``PlannerSettings.early_climb=True`` so an upward gate-to-gate transition reaches the full
next-gate height *early* and flies level into the run-in, instead of finishing the climb late
and clipping the bottom frame bar of a tall gate (the dominant remaining failure on v3).

Like GateSearchV3 it assumes the gate (and obstacle) positions are *known from the first step*
and plans a single global spline through all of them straight away — no TAKEOFF climb, no
Archimedean spiral SEARCH phase.

When is that true?
------------------
``obs["gates_pos"]`` reports a gate's true pose only once it has been *visited* (drone within
``sensor_range``); otherwise it reports the **nominal** pose (``envs/race_core.py:obs``). On a
KNOWN track the nominal pose already equals the real pose, so the mask is a no-op and the real
layout is available at ``t = 0``. This holds for:

* ``level0/1/2`` configs (``randomize = false``; the config holds the real layout), and
* a ``real_track.toml`` captured by ``scripts/save_track_as_config.py`` — it measures the
  physical track via Vicon, writes the real gate/obstacle poses into the config and sets
  ``randomize = false``. After that, both sim and ``real_race_env`` surface the true poses
  from the start (``nominal == real``).

It does NOT hold for blind Level 3 (``randomize = true``), where every gate's nominal pose is
the degenerate ``[0, 0, h]`` origin stack. Use :class:`GateSearchV2` (spiral search) there.

Design
------
The shared ``online_planner`` already lifts off from the ground on its own: when the start
height is below ``liftoff_z_threshold`` it inserts a climbing waypoint toward the first gate
(``online_planner/trajectory.py:build_waypoints``). So this controller is just:

NAVIGATE : from ``t = 0``, plan one clamped-cubic spline through every remaining gate using the
           observed poses, avoiding the observed obstacles. Replans automatically when the
           target gate advances or any observed pose shifts.
HOME     : after the last gate (``target_gate == -1``), descend to arena centre and land.

``orient_gates_to_travel`` is False (same as GateSearchV2's NAVIGATE): gates are crossed along
their canonical +x axis, the direction the environment counts a pass in
(``envs/utils.py:gate_passed`` requires gate-local -x -> +x), which every track layout respects.
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
from lsy_drone_racing.control.online_planner.trajectory import ReferenceManager

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline

# ── NAVIGATE gate-approach geometry (carried over from GateSearchV2's NAVIGATE) ────────
# d_pre/d_post set the straight run-in / run-out length aligned with the gate normal; a
# longer run-in reduces corner-cutting at the gate plane (but too long hurts — measured: a
# 0.60 m run-in clearly beat 0.80/1.00 m, which forced late climbs and slower runs).
# r_obs is the obstacle keep-out radius. GateSearchV2 tightened this to 0.12 m for Level-3's
# *thin* (0.03 m) poles placed right at the gate; on a known track the obstacles are full
# poles with room to spare, so a wider 0.20 m keep-out is correct. Measured on level2 (12
# seeds): 0.12 -> 7/12 finishes, 0.20 -> 10/12; 0.24 over-corrects and clips gate frames.
_NAV_D_PRE = 0.60
_NAV_D_POST = 0.40
_NAV_R_OBS = 0.20
# Speeds (m/s). Peri-gate cruise is kept modest for crossing precision; tune up for speed.
_V_CRUISE = 1.0          # cruise speed near gates
_V_CRUISE_INTER = 1.0    # cruise speed between gates
_VMAX = 1.4              # peak-velocity cap
# Reference look-ahead (s). Smaller => track the spline point nearer current progress,
# reducing corner-cutting (and the body tilt that clips a rotor on the gate frame).
_NAV_LOOKAHEAD = 0.20
# Virtual columns ±_GATE_POST_OFFSET along each gate's lateral axis funnel the spline through
# the opening (gate frame outer half-width is 0.36 m; tighter centres the run-in => fewer clips).
_GATE_POST_OFFSET = 0.30


class GateSearchV4(Controller):
    """No-search navigate controller + early climb for known tracks (level0/1/2 / saved track)."""

    _MODE_NAVIGATE = "NAVIGATE"
    _MODE_HOME = "HOME"
    _MODE_DONE = "DONE"

    def __init__(self, obs: dict, info: dict, config: dict):
        """Initialise timing, drone parameters, PID, and the reference manager."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("GateSearchV4 requires env.control_mode = 'attitude'.")
        # orient_gates_to_travel=False: cross every gate along its canonical +x axis, the
        # direction the env requires for a counted pass (gate-local -x -> +x).
        # early_climb=True: reach tall-gate height early so the run-in is level (v4 fix).
        self._settings = ControllerSettings(
            planner=PlannerSettings(
                d_pre=_NAV_D_PRE, d_post=_NAV_D_POST, v_cruise=_V_CRUISE,
                v_cruise_inter=_V_CRUISE_INTER, max_speed=_VMAX,
                r_obs=_NAV_R_OBS, orient_gates_to_travel=False, early_climb=True,
            ),
            runtime=RuntimeSettings(lookahead_s=_NAV_LOOKAHEAD),
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
        # Per-episode state — also fully re-initialised in reset()
        self._tick = 0
        self._plan_start_tick = 0
        self._progress_t = 0.0
        self._finished = False
        self._last_action = self._hover_action()
        self._last_target = -1
        self._mode = self._MODE_NAVIGATE
        self._home_plan: tuple | None = None
        self._home_start_tick = 0
        # Debug visualisation state — written by compute_control, read by render_callback
        self._dbg_gate_pos: NDArray = np.empty((0, 3), dtype=np.float64)
        self._dbg_target_gate: int = -1
        self._dbg_obs_pos: NDArray = np.empty((0, 3), dtype=np.float64)
        self._dbg_wp_pos: NDArray = np.empty((0, 3), dtype=np.float64)

    # ── Main control loop ────────────────────────────────────────────────────

    def compute_control(self, obs: dict, info: dict | None = None) -> NDArray:
        """Return a [roll, pitch, yaw, thrust] attitude command for the current step."""
        frame = parse_observation(obs)
        now = self._tick * self._dt
        self._last_target = int(frame.target_gate)

        if now >= self._settings.runtime.timeout_s:
            self._finished = True
            return self._last_action.copy()

        # ── Mode transitions ────────────────────────────────────────────────
        if frame.target_gate == -1 and self._mode not in (self._MODE_HOME, self._MODE_DONE):
            self._mode = self._MODE_HOME

        # ── Dispatch ────────────────────────────────────────────────────────
        if self._mode == self._MODE_NAVIGATE:
            action = self._navigate_action(frame)
        elif self._mode == self._MODE_HOME:
            action = self._home_action(frame)
        else:
            self._finished = True
            return self._last_action.copy()

        self._capture_debug(frame)
        return action

    # ── NAVIGATE mode ────────────────────────────────────────────────────────

    def _navigate_action(self, frame: DroneObservation) -> NDArray:
        # Trust the observation: every gate and obstacle pose is taken as known. Feed all
        # observed obstacles (plus per-gate funnel columns) to the planner so the path avoids
        # them from the start, not only once they enter sensor range.
        det_obs = frame.obstacles_pos
        gate_obs = self._gate_post_obstacles(frame)
        if len(gate_obs):
            det_obs = np.concatenate([det_obs, gate_obs], axis=0)

        # Guard against an obstacle-count change between plans (would break _needs_plan's
        # element-wise pose diff). The count is constant here, so this is defensive only.
        active = self._references.plan
        if active is not None and len(det_obs) != len(active.obstacle_pos_snapshot):
            self._references.reset()
            self._progress_t = 0.0
            self._plan_start_tick = self._tick

        nav_frame = DroneObservation(
            target_gate=frame.target_gate,
            gate_pos=frame.gate_pos,
            gate_quat=frame.gate_quat,
            obstacles_pos=det_obs,
            pos=frame.pos,
            vel=frame.vel,
            quat=frame.quat,
        )

        plan, rebuilt = self._references.ensure_plan(nav_frame)
        if rebuilt:
            self._plan_start_tick = self._tick
            self._progress_t = 0.0

        clock_t = (self._tick - self._plan_start_tick) * self._dt
        self._progress_t = self._project(plan.curve, plan.t_total, frame.pos)
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

    def _home_action(self, frame: DroneObservation) -> NDArray:
        if self._home_plan is None:
            cur_z = float(frame.pos[2])
            # Descend to arena centre in two segments
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

    # ── Gate frame obstacle columns ───────────────────────────────────────────

    def _gate_post_obstacles(self, frame: DroneObservation) -> NDArray:
        """Two virtual cylindrical obstacles per gate at the outer frame edges.

        Placed at ±_GATE_POST_OFFSET along each gate's lateral axis so the planner routes
        through the opening rather than clipping a frame bar. The planner's 2-D avoidance
        uses XY only; Z is set to the gate centre height for visual clarity.
        """
        posts: list[NDArray] = []
        for gi in range(len(frame.gate_pos)):
            gp = np.asarray(frame.gate_pos[gi], dtype=np.float64)
            lateral = Rotation.from_quat(frame.gate_quat[gi]).as_matrix()[:, 1]
            posts.append(gp + _GATE_POST_OFFSET * lateral)
            posts.append(gp - _GATE_POST_OFFSET * lateral)
        return np.array(posts, dtype=np.float64) if posts else np.empty((0, 3), dtype=np.float64)

    # ── Spline projection ─────────────────────────────────────────────────────

    def _project(self, curve: CubicSpline, t_total: float, pos: NDArray) -> float:
        """Closest spline time ahead of current progress."""
        window = self._settings.runtime.projection_window_s
        upper = min(self._progress_t + window, t_total)
        sample_t = np.linspace(self._progress_t, upper, 40)
        distances = np.linalg.norm(np.asarray(curve(sample_t)) - pos, axis=1)
        return float(sample_t[int(np.argmin(distances))])

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
        self._home_plan = None
        self._home_start_tick = 0
        self._feedback.reset()
        self._references.reset()
        self._last_action = self._hover_action()
        self._dbg_gate_pos = np.empty((0, 3), dtype=np.float64)
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

    def _capture_debug(self, frame: DroneObservation) -> None:
        """Snapshot marker state for render_callback (drawn with draw_line)."""
        self._dbg_gate_pos = np.asarray(frame.gate_pos, dtype=np.float64)
        self._dbg_target_gate = int(frame.target_gate)
        self._dbg_obs_pos = np.asarray(frame.obstacles_pos, dtype=np.float64)
        plan = self._references.plan
        if self._mode == self._MODE_NAVIGATE and plan is not None:
            n_samples = max(2, int(min(20, plan.t_total * 5)))
            t_pts = np.linspace(0.0, plan.t_total, n_samples)
            self._dbg_wp_pos = np.asarray(plan.curve(t_pts), dtype=np.float64)
        else:
            self._dbg_wp_pos = np.empty((0, 3), dtype=np.float64)

    def render_callback(self, sim: Sim) -> None:
        """Draw the active reference spline and debug markers."""
        plan = self._references.plan
        if plan is not None:
            samples = plan.curve(np.linspace(0.0, plan.t_total, 100))
            draw_line(sim, np.asarray(samples, dtype=np.float32), rgba=(0.0, 1.0, 0.0, 1.0))

        arm = 0.08  # half-length of each cross arm in metres

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
            if i == self._dbg_target_gate:
                _cross(gp, rgba=(0.0, 1.0, 0.0, 1.0))   # bright green — target gate
            else:
                _cross(gp, rgba=(0.0, 0.55, 0.0, 1.0))  # dim green — other gates
        for op in self._dbg_obs_pos:
            _cross(op, rgba=(1.0, 0.0, 0.0, 1.0))       # red — obstacles
        for wp in self._dbg_wp_pos:
            _cross(wp, rgba=(0.2, 0.4, 1.0, 1.0))       # blue — planned spline samples

    def diagnostic(self) -> dict:
        """Return a short status dict for logging."""
        plan = self._references.plan
        return {
            "controller_phase": self._mode,
            "active_target_gate": self._last_target,
            "controller_time": self._tick * self._dt,
            "reference_end_time": None if plan is None else plan.t_total,
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
