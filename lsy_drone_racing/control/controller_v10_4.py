"""Gate-aware time-optimal MPCC with a fast launch (controller_v10_4).

v10.4 is v10.3 with the START PHASE rebuilt. Profiling showed ~2.6 s of the ~8.1 s level2 lap
is spent before gate 0 (PID climb to 0.5 m: 0.54 s; ramp-throttled run to the gate: ~2.1 s)
against a curvature-limited floor of ~1.45 s. Four coordinated changes close part of that gap
without touching gate-crossing speeds (the proven +/-0.15 m reveal-precision ceiling).
Measured, paired-seed 3 x 20 runs (seeds 42/7/123, same tracks for both controllers):
v10.3 54/60 at 8.20 s -> v10.4 52/60 at 8.01 s (finish within paired noise, launch crashes
2/60), and on the sharp-slalom edge track 9/10 vs v10.3's 3/10 at the same seed. The full
speed/robustness frontier and every condemned variant are ledgered in controller_core_v10_4/cockpit.py.

1. MINI-TAKEOFF (controller_core_v10_4.takeoff): the XY-held PID climb stays -- it is what keeps the
   start inside the env's +/-0.02 m floor-touch carve-out while the rotors spin up from zero
   (a from-ground MPCC launch is unsafe by construction: the env hard-disables on any floor
   touch outside the carve-out, and every episode begins with ~40 ms of thrust-free free-fall).
   But it climbs on a FIXED 0.55 s spline to 0.42 m (v8: speed-derived with a 0.6 s floor, to
   0.5 m), handing off ~0.15 s earlier with upward momentum along the plan's initial tangent.

2. LAUNCH RAMP (controller_core_v10_4.cockpit): the progress-rate ramp is retuned (0.08/2.0 -> 0.25/2.4).
   Hotter ramps (0.30/1.6) were probed exhaustively -- with gate-0 caps, hand-off altitudes
   0.22/0.42, and obstacle keep-outs 0.20/0.26 -- and ALL pile failures into the gate-0 reveal
   corridor at 1.5-1.9 s: that corridor tolerates ~2.4 m/s under +/-0.15 m randomisation
   however it is approached, so the ramp is the binding launch knob, just warmer than v10.3's.

3. HONEST COLD START (controller_core_v10_4.mpcc): on the first solve of a plan the progress rate is
   clamped to the drone's measured speed (was: fabricated at the full ramp cap, even from
   rest) and a hover-stationary warm start at the measured pose is seeded into the RTI (was:
   linearised at the all-zeros iterate, where the tilt/thrust constraints have zero lateral
   gradient). This also de-risks v10.3's residual hand-off crash class (1-2/20).

4. GATE-0 APPROACH CAP (controller_core_v10_4.arc_path): the hot ramp re-exposes the launch window --
   gate 0's +/-0.15 m reveal correction AND the obstacle-0 reveal swerve right before it, both
   revealed only at the 0.7 m sensor range. Gate 0's approach is therefore always capped at
   V_GATE_REACT (2.5 m/s) while it is the target; the backward pass slows the obstacle-0
   passage as a side effect. Measured on the hostile seed: launch crashes 5 -> 2 at zero time
   cost. (The same machinery supports REACTIVE caps on any gate whose revealed pose moved
   more than REACT_DELTA_M -- measured a net loss mid-race and shipped disabled; and a
   clearance-geometry trim dial in controller_core_v10_4.trajectory, also shipped neutral. See the
   cockpit ledger for the numbers.)

Everything else -- the OCP, the gate-aware contouring weight, the replan-continuity rebase, the
speed budget (3.2 / 8.5), and the render overlays -- is v10.3, hence v10.4 (not v11).
Implemented as a thin subclass of ControllerV103. REQUIRES the acados environment -- run under
``pixi run``; the compiled solver is shared with v10.3 (identical OCP).
"""

from __future__ import annotations

import numpy as np

from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.controller_v10_3 import ControllerV103
from lsy_drone_racing.control.controller_core_v10_4.arc_path import GateArcPath
from lsy_drone_racing.control.controller_core_v10_4.mpcc import MPCC
from lsy_drone_racing.control.controller_core_v10_4.settings import ControllerSettings
from lsy_drone_racing.control.controller_core_v10_4.takeoff import TakeoffPhase
from lsy_drone_racing.control.controller_core_v10_4.trajectory import ReferenceManager


class ControllerV104(ControllerV103):
    """v10.3's racing MPCC behind a mini-takeoff, a launch ramp, and an honest cold start."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v10.3, then swap in the v10.4 launch components."""
        super().__init__(obs, info, config)
        self._settings = ControllerSettings()
        mpcc = self._settings.mpcc
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCC(mpcc, a_max)  # honest cold start; solver shared with v10.3
        # Refresh the cached knobs the inherited flow reads (only the ramp values differ).
        self._v_theta_max = mpcc.v_theta_max
        self._ramp_s, self._ramp_start = mpcc.ramp_s, mpcc.ramp_start
        self._a_lat_max, self._v_min = mpcc.a_lat_max, mpcc.v_min
        self._w_base, self._w_gate, self._gate_sigma = (
            mpcc.w_contour_base, mpcc.w_contour_gate, mpcc.gate_sigma,
        )
        self._v_gate = mpcc.v_gate
        self._gate_v_pre, self._gate_v_post = mpcc.gate_v_pre, mpcc.gate_v_post
        # Rebuild the takeoff with the v10.4 mini-climb (the inherited one was built with the
        # v8 defaults captured before this settings swap).
        self._takeoff = TakeoffPhase(self._settings, self._settings.takeoff)
        # Plan through the trimmed clearance geometry (v10.4-owned planner copy).
        self._references = ReferenceManager(
            self._settings.planner,
            self._settings.runtime.replan_gate_delta_m,
            self._settings.runtime.replan_obstacle_delta_m,
        )
        # Reactive per-gate cap knobs (read when (re)building the GateArcPath).
        self._v_gate_react = mpcc.v_gate_react
        self._react_delta = mpcc.react_delta_m
        self._react_v_pre, self._react_v_post = mpcc.react_v_pre, mpcc.react_v_post
        self._gate_nominal: np.ndarray | None = None  # gate poses before any reveal
        self._path = None
        self._s = 0.0

    def reset(self) -> None:
        """Reset v10.3 state plus the nominal gate-pose snapshot."""
        super().reset()
        self._gate_nominal = None

    def _track_action(self, frame):
        """v10.3's flow, flagging moved (revealed) gates with a reactive approach speed cap."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if self._gate_nominal is None:  # first NAVIGATE tick: nothing is revealed yet
            self._gate_nominal = np.asarray(frame.gate_pos, dtype=np.float64).copy()
        if rebuilt or self._path is None:
            first = max(frame.target_gate, 0)
            gates_ahead = plan.gate_pos_snapshot[first:]
            deltas = np.linalg.norm(
                plan.gate_pos_snapshot[first:] - self._gate_nominal[first:], axis=1
            )
            caps = np.where(deltas > self._react_delta, self._v_gate_react, np.inf)
            if first == 0:  # gate 0 is always capped: launch-window protection (see cockpit)
                caps[0] = self._v_gate_react
            path = GateArcPath(
                plan.curve, plan.t_total, self._v_theta_max, self._a_lat_max, self._v_min,
                gates_ahead, self._w_base, self._w_gate, self._gate_sigma,
                caps, self._react_v_pre, self._react_v_post,
            )
            self._s = path.project(frame.pos, 0.0)
            if self._path is None:  # episode start / first plan: cold start (honest, see mpcc)
                self._mpcc.set_path(path)
            else:  # mid-flight replan: keep the warm start and the solver memory
                self._mpcc.rebase(path, self._s)
            self._path = path
        self._s = self._path.project(frame.pos, self._s)
        elapsed = (self._tick - self._nav_start_tick) * self._dt
        ramp = min(1.0, self._ramp_start + (1.0 - self._ramp_start) * elapsed / self._ramp_s)
        accel = self._mpcc.solve(frame.pos, frame.vel, self._s, self._v_theta_max * ramp)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)
