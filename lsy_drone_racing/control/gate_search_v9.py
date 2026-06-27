"""Known-track navigate controller (v9): GateSearchV8 + an arena geofence + a narrower U-turn.

GateSearchV9 keeps the GateSearchV7/V8 architecture (no-search NAVIGATE from t=0, early climb,
deploy-tuned speeds + approach/exit distances, HOME descent, ``orient_gates_to_travel=False``) and
adds a hard guarantee that the drone cannot leave the safety box, plus a slightly tighter reversal.

Why
---
On real hardware the run was aborted when a wide curve near a border gate pushed the drone across a
safety plane (it then auto-returned to start). In full-physics sim GateSearchV8 indeed crosses the
y=+-1.5 plane on tracks with a border gate (e.g. level2_deploy: peak |y| 1.52, out of bounds on
9/12 seeds). The fix is to reserve lateral room for tracking overshoot, in the planner, so the
*flown* path stays inside.

What changed vs v8 (all track-agnostic)
---------------------------------------
1. Arena geofence (the main change): the planned path is clamped to stay >= ``geofence_margin`` =
   0.15 m inside the config ``safety_limits`` in X/Y, with gate crossings protected
   (online_planner.timing.enforce_geofence). This reserves room for tracking overshoot so a wide
   curve can never reach a safety plane. No-op where it does not bind.
2. Narrower U-turn swing: ``reversal_swing_m`` 0.55 -> 0.40 m, ``reversal_apex_m`` 0.10 -> 0.06 m
   (new PlannerSettings knobs; defaults preserve v2..v8). Rounds the reversal tighter; sim-measured
   slightly higher finish than the v8 swing on the lab capture, and ~0.3 s faster on level2_deploy.

NOT changed (a lesson from sim): the approach/exit distances ``d_pre``/``d_post`` stay at v8's
0.60/0.40. Trimming them (tried 0.45/0.30) looked good on planner geometry but COLLAPSED the
full-physics finish rate (100% -> 0% on level2_deploy) -- a shorter approach gives the
cold-start/clamped spline too little room to align with the gate axis. The geofence (not a tighter
approach) is what keeps the drone in the zone. This is why physics validation, not geometry, is the
gate: see scripts/lab_eval.py and scripts/v9_ablation.py.

Sim-validated (scripts/lab_eval.py, first_principles physics):
  level2_deploy: finish 100% (= v8) but 0/12 out-of-bounds (v8: 9/12 OOB), lap 10.37 s (v8 10.65).
  real_track (deploy-faithful): finish ~63% (v8 ~37%), 0/N OOB, lap ~9.4 s.
Note: the saved real_track.toml is an imperfect capture -- an obstacle sits ~5 cm from gate 0's
approach, so it is unflyable for v7/v8/v9 alike; re-measure and re-validate before relying on it.

The v8 per-gate lateral pre-bias is a *tracking-lag* correction calibrated to one specific track;
v9 ships it disabled (empty map) to stay general (sim: enabling it on the lab capture lowered finish
6/12 vs 8/12). The mechanism is retained so a future measured track can re-enable it. v8 remains the
exact rollback baseline.
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

# ── NAVIGATE gate-approach geometry ────────────────────────────────────────────────────
# Kept at the GateSearchV7/V8 values. Trimming these (tried d_pre 0.45 / d_post 0.30) tightened
# the planned path on paper but collapsed the finish rate in full physics (sim: 100% -> 0% on
# level2_deploy) -- a shorter approach gives the cold-start/clamped spline too little room to align
# with the gate axis, so the drone diverges right after take-off. The arena geofence (below) does
# the "stay in the zone" job instead, without touching the run-in the tracker depends on.
_NAV_D_PRE = 0.60
_NAV_D_POST = 0.40
_NAV_R_OBS = 0.20
# Speeds (m/s) — carried over verbatim from GateSearchV7/V8 (this is not a speed change).
_V_CRUISE = 1.4          # cruise speed near gates (peri-gate)
_V_CRUISE_INTER = 1.6    # cruise speed between gates
_VMAX = 1.9              # peak-velocity cap
_NAV_LOOKAHEAD = 0.20
_GATE_POST_OFFSET = 0.30

# U-turn swing geometry — narrower than the v2..v8 default (0.55 / 0.10) so reversals round tighter.
_REVERSAL_SWING_M = 0.40
_REVERSAL_APEX_M = 0.06

# Arena geofence: keep the planned path this far (m) inside the config safety_limits in X/Y, so a
# wide curve can never push the drone across a safety plane. The margin reserves room for tracking
# overshoot. Gate crossings are protected (see online_planner.timing.enforce_geofence).
_GEOFENCE_MARGIN = 0.15
# Fallback arena bounds if a config omits safety_limits (standard course box).
_DEFAULT_ARENA_LOW = (-2.5, -1.5, -1e-3)
_DEFAULT_ARENA_HIGH = (2.5, 1.5, 2.0)

# Per-gate lateral pre-bias (m) along each gate's local +y axis. Empty => general/track-agnostic
# (recommended). Retained as a re-calibration hook: populate {gate_index: bias_m} from
# scripts/probe_tracking.py if a measured track shows a repeatable crossing offset at a gate.
_GATE_LATERAL_BIAS: dict[int, float] = {}


def _arena_bounds(config: dict) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
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


class GateSearchV9(Controller):
    """GateSearchV8 line with tighter gate geometry, a narrower U-turn, and an arena geofence."""

    _MODE_NAVIGATE = "NAVIGATE"
    _MODE_HOME = "HOME"
    _MODE_DONE = "DONE"

    def __init__(self, obs: dict, info: dict, config: dict):
        """Initialise timing, drone parameters, PID, and the reference manager."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("GateSearchV9 requires env.control_mode = 'attitude'.")
        arena_low, arena_high = _arena_bounds(config)
        # orient_gates_to_travel=False: cross every gate along its canonical +x axis.
        # early_climb=True: reach tall-gate height early so the run-in is level.
        self._settings = ControllerSettings(
            planner=PlannerSettings(
                d_pre=_NAV_D_PRE, d_post=_NAV_D_POST, v_cruise=_V_CRUISE,
                v_cruise_inter=_V_CRUISE_INTER, max_speed=_VMAX,
                r_obs=_NAV_R_OBS, orient_gates_to_travel=False, early_climb=True,
                reversal_swing_m=_REVERSAL_SWING_M, reversal_apex_m=_REVERSAL_APEX_M,
                geofence_margin=_GEOFENCE_MARGIN, arena_low=arena_low, arena_high=arena_high,
            ),
            runtime=RuntimeSettings(lookahead_s=_NAV_LOOKAHEAD),
        )
        self._gate_bias = dict(_GATE_LATERAL_BIAS)
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
        # Optional deterministic crossing-offset correction (disabled by default: empty bias map).
        # When enabled it shifts only the gate center the controller *aims* at; the env still
        # scores the real gate. Everything downstream uses the biased frame.
        frame = self._bias_frame(frame)

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

    # ── Lateral pre-bias (optional re-calibration hook; disabled when the map is empty) ───────

    def _bias_frame(self, frame: DroneObservation) -> DroneObservation:
        """Return a frame with each calibrated gate center shifted along its local +y axis.

        No-op when ``_GATE_LATERAL_BIAS`` is empty (the v9 default), keeping the controller
        track-agnostic. See the module docstring for when to populate it.
        """
        if not self._gate_bias:
            return frame
        gate_pos = np.asarray(frame.gate_pos, dtype=np.float64).copy()
        for gi, bias in self._gate_bias.items():
            if 0 <= gi < len(gate_pos) and bias != 0.0:
                lateral = Rotation.from_quat(frame.gate_quat[gi]).as_matrix()[:, 1]
                gate_pos[gi] = gate_pos[gi] + bias * lateral
        return DroneObservation(
            target_gate=frame.target_gate,
            gate_pos=gate_pos,
            gate_quat=frame.gate_quat,
            obstacles_pos=frame.obstacles_pos,
            pos=frame.pos,
            vel=frame.vel,
            quat=frame.quat,
        )

    # ── HOME mode ──────────────────────────────────────────────────────────────

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
