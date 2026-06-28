"""Level-3 drone racing controller (v10): SEARCH / NAVIGATE / HOME, hardened for robustness.

GateSearchV10 is the Level-3 (fully random tracks) controller: it discovers gates by sweeping the
arena, then navigates them. It is GateSearchV2's search/navigate architecture plus the robustness
work proven on the deploy track in GateSearchV9:

  * Arena geofence (online_planner.timing.enforce_geofence) on BOTH the search and navigate plans,
    so the spline can never push the drone toward the +-3 m sim wall (where it is disabled) -- v2
    routinely flew out to |y|~2.1 m. Gate/spiral crossings are protected; bounds come from the
    config safety_limits.
  * Narrower reversal swing (reversal_swing_m/apex_m) for tighter, in-bounds U-turns.

It uses ONE planner option from v2: ``orient_gates_to_travel=False`` for NAVIGATE, so gates are
crossed along their canonical +x axis -- the direction the env counts as a pass
(``envs/utils.py:gate_passed``, -x -> +x); the Level-3 randomizer orients every gate's +x along
the intended traversal order. SEARCH feeds Archimedean-spiral waypoints as virtual gates so the
planner sweeps the arena; only sensor-confirmed obstacles ever reach the planner (Level-3 nominal
positions are the [0,0,h] origin stack and must never be planned through).

This is the controller for level3.toml; GateSearchV9 remains the known-track/deploy controller.

Modes
-----
TAKEOFF : initial mode.  Hold the start x/y and climb straight up to a fixed
          altitude (_TAKEOFF_ALT), then hand off to SEARCH.
SEARCH  : Archimedean spiral at _SEARCH_ALT (above the obstacle poles), swept outermost
          point first so it ends near the arena centre.  Spiral waypoints are treated as
          virtual gates; the search path is left unconstrained.
NAVIGATE: plan through all contiguously discovered real gates, avoiding only detected
          obstacles, crossing each gate along its canonical +x axis.
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

# ── Spiral / search constants ─────────────────────────────────────────────────
# Arena safety limits from level3.toml: x ∈ [-2.5, 2.5], y ∈ [-1.5, 1.5].
# We keep 0.3 m inside the hard limits on each axis.
_SEARCH_ALT = 1.8           # m — between gate heights 0.7 m and 1.2 m
_SPIRAL_RADIAL_STEP = 0.6   # m radial gap per revolution; < 2 × 0.7 m sensor range
_SPIRAL_ANGLE_STEP = np.pi / 6   # 30° step = 12 waypoints per revolution
_SPIRAL_ADVANCE_RADIUS = 0.6     # m (2-D) — advance to next spiral point when this close
_SPIRAL_HORIZON = 3              # waypoints ahead to include in each SEARCH plan
# False (legacy, much better): outermost-first — fly out to _SEARCH_RADIUS then spiral inward,
# ending the sweep near the arena centre, a good hub from which any gate-0 location is
# reachable. True (spiral outward from centre) ends the sweep at the edge and tested far worse
# (7.5% vs 40% on 40 seeds), so it is disabled.
_SPIRAL_OUTWARD = False
# Gates are only ever placed within border_margin (0.5 m) of the hard limits, i.e. inside
# ±2.0 m (x) × ±1.0 m (y) (see envs/randomize.py:build_random_track_fn). The search box is
# pulled in to those bounds: full coverage is still guaranteed (sensor range is 0.7 m) while
# keeping the fast spiral well clear of the ±2.5/±1.5 m hard boundary, where the uncapped
# outbound dash used to overshoot and disable the drone.
_SEARCH_RADIUS = 2.2             # m — outer radius the search starts from, spiralling inward
_ARENA_X_LIM = 1.9          # m from centre — search boundary (gates are within ±2.0 m)
_ARENA_Y_LIM = 1.0          # m from centre — search boundary (gates are within ±1.0 m)
_GATE_SKIP_RADIUS = 1.85    # m — skip spiral waypoints within this XY distance of a detected gate
# Virtual columns are placed ±_GATE_POST_OFFSET along the gate lateral axis to funnel the
# spline through the opening (the gate frame outer half-width is 0.36 m).
_GATE_POST_OFFSET = 0.30    # m — lateral offset from gate centre to each virtual column.
# Tighter than the 0.36 m outer frame edge: the virtual columns squeeze the approach/exit
# waypoints closer to the opening centre (a straighter, better-centred run-in => fewer frame
# clips), while staying clear of the gate-centre waypoint's obstacle-repair trigger (~0.24 m
# at r_obs=0.12). This was the change that pushed the Level-3 finish rate to 50%.
# ── Takeoff constants ─────────────────────────────────────────────────────────
_TAKEOFF_ALT = 1.8          # m — straight-up climb target before SEARCH begins (tunable)
_TAKEOFF_Z_TOL = 0.05       # m — switch to SEARCH when within this of _TAKEOFF_ALT
_TAKEOFF_TIME_MARGIN = 1.0  # s — fallback handoff after the takeoff spline overruns by this
# ── Speed constants (m/s) ─────────────────────────────────────────────────────
_V_CRUISE_SEARCH = 2.5   # SEARCH cruise speed near (spiral) waypoints
_VMAX_SEARCH = 3.0       # SEARCH peak-velocity cap
_V_CRUISE = 1.0          # NAVIGATE cruise speed near gates (peri-gate; keep low for precision)
_V_CRUISE_INTER = 1.0    # NAVIGATE cruise speed BETWEEN gates (raise for faster traversal)
_VMAX = 1.4              # NAVIGATE peak-velocity cap
# ── NAVIGATE gate-approach geometry ───────────────────────────────────────────
# d_pre/d_post set the length of the straight run-in / run-out aligned with the gate
# normal; a longer run-in reduces corner-cutting at the gate plane.  r_obs is the
# obstacle keep-out radius: Level-3 obstacles are thin poles (0.015 m) placed in the
# approach corridor *close to* the gate, so a large keep-out shoves the path toward the
# far frame bar.  A tighter keep-out keeps the crossing near the opening centre while
# still clearing the (thin) pole + drone radius (~0.07 m needed).
_NAV_D_PRE = 0.60
_NAV_D_POST = 0.40
_NAV_R_OBS = 0.12
# Scale applied to the drone's velocity when building the FIRST navigate plan after a
# search handoff. The drone carries cross-arena spiral momentum that, used as the spline
# start boundary condition, forces a wrong-way U-turn loop toward the first gate. Damping
# it (0.0 = plan from rest) makes the spline head straight at the gate; the real velocity
# is still used for tracking, so the drone simply decelerates onto the new plan. 0.5 gave
# the best average gates passed (it heads more directly at gate 0 without a hard transient).
# Kept at v2's 0.5: a Level-3 ablation showed 0.0 (plan from rest) lowered finish (it over-damps
# the run-in), and 0.5 was best (7/14 vs 6/14).
_NAV_START_VEL_SCALE = 0.5
# Reference look-ahead time (s). Smaller => the drone tracks the spline point closer to its
# current progress instead of aiming ahead, which reduces corner-cutting (and the body tilt
# that makes a rotor clip the gate frame) on the curved gate approach. Default v6 value 0.20.
_NAV_LOOKAHEAD = 0.20
# Search strategy: True = discover ALL gates before navigating; False = navigate each gate as found
_DISCOVER_ALL_FIRST = True
# ── Robustness (v10): arena geofence + tighter reversal ───────────────────────
# Keep the planned path this far (m) inside the config safety_limits in X/Y on BOTH search and
# navigate, so the spline never drives the drone toward the ±3 m sim wall (where it is disabled).
# Gates are within ±2.0/±1.0 m (well inside the fence), so this never blocks reaching a gate.
_GEOFENCE_MARGIN = 0.15
_DEFAULT_ARENA_LOW = (-2.5, -1.5, -1e-3)
_DEFAULT_ARENA_HIGH = (2.5, 1.5, 2.0)
# Narrower U-turn swing than the v2..v8 default 0.55/0.10 (carried over from GateSearchV9).
_REVERSAL_SWING_M = 0.40
_REVERSAL_APEX_M = 0.06
# Cap (m/s) on the planned descent rate during NAVIGATE. The search sweeps at 1.8 m (above the
# obstacle poles); without this the drone, dropping to a 0.7/1.2 m gate, crosses the gate plane
# too high and clips the top frame bar (the dominant Level-3 crash). Stretches only descending
# segments so the drop stays trackable.
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


class GateSearchV10(Controller):
    """Level-3 search-then-navigate controller, geofenced + tuned for robustness."""

    _MODE_TAKEOFF = "TAKEOFF"
    _MODE_SEARCH = "SEARCH"
    _MODE_NAVIGATE = "NAVIGATE"
    _MODE_HOME = "HOME"
    _MODE_DONE = "DONE"

    def __init__(self, obs: dict, info: dict, config: dict):
        """Initialise timing, drone parameters, PID, reference manager, and spiral."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("GateSearchV10 requires env.control_mode = 'attitude'.")
        arena_low, arena_high = _arena_bounds(config)
        # Navigate settings: wider gate approach funnel reduces spline curvature at
        # the gate plane, keeping the trajectory closer to the opening center.
        # orient_gates_to_travel=False makes the planner cross every gate along its
        # canonical +x axis — the direction the env requires for a pass (Level-3 gates
        # are oriented +x along the intended traversal order). With the travel-flip the
        # drone reaches the opening but crosses backwards and the pass is never counted.
        # geofence + tighter reversal (v10): keep the path inside the arena, round U-turns tighter.
        self._settings = ControllerSettings(
            planner=PlannerSettings(
                d_pre=_NAV_D_PRE, d_post=_NAV_D_POST, v_cruise=_V_CRUISE,
                v_cruise_inter=_V_CRUISE_INTER, max_speed=_VMAX,
                r_obs=_NAV_R_OBS, orient_gates_to_travel=False, early_climb=True,
                reversal_swing_m=_REVERSAL_SWING_M, reversal_apex_m=_REVERSAL_APEX_M,
                geofence_margin=_GEOFENCE_MARGIN, arena_low=arena_low, arena_high=arena_high,
                max_descent_rate=_NAV_MAX_DESCENT_RATE,
            ),
            runtime=RuntimeSettings(lookahead_s=_NAV_LOOKAHEAD),
        )
        # Search settings: slower cruise reduces committed distance between replans,
        # giving the drone more time to detect gates and obstacles before passing them.
        # geofence here too: clamp the fast spiral sweep so it never reaches the sim wall.
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
        # Pre-compute spiral once; reused across episodes
        self._spiral_wps, self._spiral_quats = self._build_spiral()
        # Per-episode state — also fully re-initialised in reset()
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
        # Debug visualisation state — written by compute_control, read by render_callback
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

        # Track all gates whose true positions are now known
        for i in range(n_gates):
            if gates_visited[i]:
                self._known_gates.add(i)

        self._last_target = int(frame.target_gate)

        if now >= self._settings.runtime.timeout_s:
            self._finished = True
            return self._last_action.copy()

        target = frame.target_gate

        # ── Mode transitions ────────────────────────────────────────────────
        if target == -1:
            if self._mode not in (self._MODE_HOME, self._MODE_DONE):
                self._mode = self._MODE_HOME

        elif self._mode == self._MODE_SEARCH and (
            (len(self._known_gates) == n_gates  # primary: discover ALL gates first
             or (self._spiral_swept and target in self._known_gates))  # fallback after a full sweep
            if _DISCOVER_ALL_FIRST
            else (target in self._known_gates)  # interleaved: navigate each gate as discovered
        ):
            # Discover-all-first: stay in SEARCH until every gate is known, then navigate
            # through all of them in order.  Fallback: once the spiral has swept the whole
            # arena without finding them all, revert to the original "navigate the target
            # once it is known" behavior so the drone still makes progress before timeout.
            self._mode = self._MODE_NAVIGATE
            self._references.reset()
            self._progress_t = 0.0
            self._plan_start_tick = self._tick
            self._last_last_known = -1

        elif self._mode == self._MODE_NAVIGATE and target not in self._known_gates:
            # Just passed all currently known gates; need to search for the next
            self._mode = self._MODE_SEARCH
            self._virtual_target = self._find_nearest_spiral(frame.pos)
            self._search_references.reset()
            self._progress_t = 0.0
            self._plan_start_tick = self._tick

        # ── Dispatch ────────────────────────────────────────────────────────
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

        # ── DEBUG VISUALISATION ──────────────────────────────────────────────
        # Capture marker state here; render_callback draws it using draw_line.
        self._dbg_gate_pos = np.asarray(frame.gate_pos, dtype=np.float64)
        known_mask = np.zeros(len(frame.gate_pos), dtype=bool)
        for idx in self._known_gates:
            if idx < len(frame.gate_pos):
                known_mask[idx] = True
        self._dbg_known_mask = known_mask
        self._dbg_target_gate = int(frame.target_gate)
        det_obs = frame.obstacles_pos[obs_visited] if obs_visited.any() else np.empty((0, 3))
        self._dbg_obs_pos = np.asarray(det_obs, dtype=np.float64)
        # Planned waypoints: spiral window in SEARCH, gate positions in NAVIGATE
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
        # ── END DEBUG VISUALISATION ──────────────────────────────────────────

        return action

    # ── TAKEOFF mode ─────────────────────────────────────────────────────────

    def _takeoff_action(self, frame: DroneObservation, obs_visited: NDArray) -> NDArray:
        """Climb straight up (holding start x/y) to _TAKEOFF_ALT, then hand off to SEARCH.

        Reuses the same build_spline + attitude_action machinery as HOME mode: a
        2-waypoint vertical spline whose endpoints share the start x/y so the clamped
        cubic only moves in z.  The shared PID is intentionally NOT reset at handoff so
        the thrust integrator stays continuous into SEARCH.
        """
        clock_t = (self._tick - self._takeoff_start_tick) * self._dt
        t_total = self._takeoff_plan[1] if self._takeoff_plan is not None else 0.0
        reached = float(frame.pos[2]) >= _TAKEOFF_ALT - _TAKEOFF_Z_TOL
        overran = self._takeoff_plan is not None and clock_t >= t_total + _TAKEOFF_TIME_MARGIN
        if reached or overran:
            self._mode = self._MODE_SEARCH
            # Start at the outermost spiral waypoint (index 0) so the drone first flies
            # out to _SEARCH_RADIUS, then spirals back inward to the centre.
            self._virtual_target = 0
            self._search_references.reset()
            self._progress_t = 0.0
            self._plan_start_tick = self._tick
            return self._search_action(frame, obs_visited)

        if self._takeoff_plan is None:
            start = np.asarray(frame.pos, dtype=np.float64).copy()
            target = np.array([start[0], start[1], _TAKEOFF_ALT])  # hold x/y, climb in z
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
        del obs_visited  # unused in SEARCH: the search path is left unconstrained (see below)
        # SEARCH flies the sweep above the track, so the search path is left fully
        # unconstrained: no obstacles (real OR virtual gate-frame) are passed to the
        # planner.  NAVIGATE keeps the full avoidance logic (see _navigate_action).
        # Disabled SEARCH-only avoidance (kept for easy revert):
        # det_obs = frame.obstacles_pos[obs_visited] if obs_visited.any() else np.empty((0, 3))
        # gate_obs = self._gate_post_obstacles(frame)
        # if len(gate_obs):
        #     det_obs = np.concatenate([det_obs, gate_obs], axis=0)
        det_obs = np.empty((0, 3))

        # Advance the virtual target when the drone is close enough in 2-D
        vt = self._virtual_target
        dist2d = float(np.linalg.norm(frame.pos[:2] - self._spiral_wps[vt, :2]))
        if dist2d < _SPIRAL_ADVANCE_RADIUS:
            next_vt = vt + 1
            if next_vt >= len(self._spiral_wps):
                next_vt = 0  # spiral exhausted — restart from centre
                self._spiral_swept = True  # arena fully swept ≥ once → enable navigate fallback
            self._virtual_target = next_vt
            self._search_references.reset()
            self._progress_t = 0.0
            self._plan_start_tick = self._tick

        # SEARCH-only gate-avoidance heuristic — DISABLED.  This block skipped spiral
        # waypoints within _GATE_SKIP_RADIUS of a detected gate (because such waypoints
        # become protected gate-centre waypoints the planner can't push off).  With the
        # search path now unconstrained the spiral no longer routes around gates, so the
        # skip is unnecessary.  Kept commented for easy revert.  NAVIGATE is unaffected.
        # if self._known_gates:
        #     gate_xys = frame.gate_pos[sorted(self._known_gates), :2]
        #     n_wps = len(self._spiral_wps)
        #     advances = 0
        #     while advances < _SPIRAL_HORIZON:
        #         vt = self._virtual_target
        #         if vt >= n_wps - 1:
        #             break
        #         dists = np.linalg.norm(gate_xys - self._spiral_wps[vt, :2], axis=1)
        #         if dists.min() < _GATE_SKIP_RADIUS:
        #             self._virtual_target = vt + 1
        #             self._search_references.reset()
        #             self._progress_t = 0.0
        #             self._plan_start_tick = self._tick
        #             advances += 1
        #         else:
        #             break

        # If the detected-obstacle count changed the stored snapshot has a different
        # shape and trajectory._needs_plan would raise a numpy shape error — rebuild.
        active = self._search_references.plan
        if active is not None and len(det_obs) != len(active.obstacle_pos_snapshot):
            self._search_references.reset()
            self._progress_t = 0.0
            self._plan_start_tick = self._tick

        # Inject a short window of spiral waypoints as virtual gates.
        # Only plan to the next _SPIRAL_HORIZON points so each plan is rebuilt
        # frequently with up-to-date obstacle data (short-horizon fix).
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

        # Chain through all contiguously discovered gates from the current target
        last_known = target
        for i in range(target + 1, n_total):
            if i in self._known_gates:
                last_known = i
            else:
                break  # gate ordering is fixed; stop at first unknown

        # Expanding the plan horizon triggers a rebuild
        if last_known != self._last_last_known:
            self._references.reset()
            self._last_last_known = last_known

        # Only sensor-confirmed obstacles — same rule as search mode
        det_obs = frame.obstacles_pos[obs_visited] if obs_visited.any() else np.empty((0, 3))
        gate_obs = self._gate_post_obstacles(frame)
        if len(gate_obs):
            det_obs = np.concatenate([det_obs, gate_obs], axis=0)

        # Guard against shape mismatch in _needs_plan when obstacle count changes
        active = self._references.plan
        if active is not None and len(det_obs) != len(active.obstacle_pos_snapshot):
            self._references.reset()
            self._progress_t = 0.0
            self._plan_start_tick = self._tick

        # Damp the start velocity for the first plan after the search handoff so the spline
        # heads straight to the gate instead of looping to honour cross-arena search
        # momentum. Tracking still uses the real velocity (see attitude_action below).
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
            # Descend to arena centre in two segments
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
        """Pre-compute an Archimedean spiral at _SEARCH_ALT.

        Waypoints are generated from the centre outward up to _SEARCH_RADIUS. With
        _SPIRAL_OUTWARD the drone follows them centre-first (it sweeps outward from where it
        took off); otherwise they are reversed so index 0 is the outermost point (legacy: fly
        out to the radius first, then spiral back inward).
        """
        a = _SPIRAL_RADIAL_STEP / (2.0 * np.pi)
        wps: list[NDArray] = []
        theta = 0.0
        while a * theta <= _SEARCH_RADIUS:
            r = a * theta
            x = float(np.clip(r * np.cos(theta), -_ARENA_X_LIM, _ARENA_X_LIM))
            y = float(np.clip(r * np.sin(theta), -_ARENA_Y_LIM, _ARENA_Y_LIM))
            pt = np.array([x, y, _SEARCH_ALT])
            # Skip points that are too close to the previous one (can arise from clamping)
            if len(wps) == 0 or float(np.linalg.norm(pt[:2] - wps[-1][:2])) >= 0.3:
                wps.append(pt)
            theta += _SPIRAL_ANGLE_STEP
        if not _SPIRAL_OUTWARD:
            wps.reverse()  # outermost first → fly out to _SEARCH_RADIUS, then spiral inward
        spiral_wps = np.array(wps, dtype=np.float64)
        # Identity quaternion [0,0,0,1]: gate faces +x; _oriented_forward() in
        # trajectory.py will flip to match the actual travel direction at plan time.
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
        """Return two virtual cylindrical obstacle positions per discovered gate.

        Columns are placed at the outer frame edges (±_GATE_POST_OFFSET along the
        gate lateral axis) so the planner routes through the opening rather than
        clipping the frame bars.  The planner's 2-D avoidance uses XY only, so Z
        is set to the gate centre height for visual clarity only.
        """
        if not self._known_gates:
            return np.empty((0, 3), dtype=np.float64)
        posts: list[NDArray] = []
        for gi in self._known_gates:
            gp = np.asarray(frame.gate_pos[gi], dtype=np.float64)
            lateral = Rotation.from_quat(frame.gate_quat[gi]).as_matrix()[:, 1]
            posts.append(gp + _GATE_POST_OFFSET * lateral)
            posts.append(gp - _GATE_POST_OFFSET * lateral)
        return np.array(posts, dtype=np.float64)

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

        # ── DEBUG VISUALISATION ──────────────────────────────────────────────
        arm = 0.08  # half-length of each cross arm in metres

        def _cross(pos: NDArray, rgba: tuple) -> None:
            """Draw a 3-axis cross at pos using three two-point draw_line calls."""
            p = np.asarray(pos, dtype=np.float32)
            for axis in range(3):
                seg = np.zeros((2, 3), dtype=np.float32)
                seg[0] = p
                seg[1] = p
                seg[0, axis] -= arm
                seg[1, axis] += arm
                draw_line(sim, seg, rgba=rgba)

        # Green crosses at detected gate centres; bright green for current target
        for i, gp in enumerate(self._dbg_gate_pos):
            if not self._dbg_known_mask[i]:
                continue
            if i == self._dbg_target_gate:
                _cross(gp, rgba=(0.0, 1.0, 0.0, 1.0))   # bright green — target gate
            else:
                _cross(gp, rgba=(0.0, 0.55, 0.0, 1.0))  # dim green — other known gates

        # Red crosses at detected (sensor-confirmed) obstacle positions
        for op in self._dbg_obs_pos:
            _cross(op, rgba=(1.0, 0.0, 0.0, 1.0))

        # Blue crosses at current planned waypoints
        for wp in self._dbg_wp_pos:
            _cross(wp, rgba=(0.2, 0.4, 1.0, 1.0))
        # ── END DEBUG VISUALISATION ──────────────────────────────────────────

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
