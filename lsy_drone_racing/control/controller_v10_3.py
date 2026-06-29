"""Gate-aware time-optimal MPCC that stays warm across replans (controller_v10_3).

v10.3 is v10.1 with REPLAN CONTINUITY. v10.1 fully resets the MPCC on every replan -- warm start
discarded, acados memory wiped, fallback command zeroed -- and replans fire right after every
gate pass (target-gate advance) and at essentially every gate reveal (pose delta > 0.05 m at the
0.7 m sensor range under +/-0.15 m gate randomisation), i.e. ~0.2 s before each gate at speed.
The post-replan command therefore came from one cold SQP-RTI iteration on a fabricated
linearisation (thbar marching at the full progress cap, vth0 hard-set to the cap), with hover
thrust as the failure fallback -- the solver was at its dumbest exactly where precision binds.
v10.3 rebases instead (controller_core_v10_3.mpcc.MPCC.rebase): the previous solution's world-frame
states/controls are kept as the warm start (the world did not change, only the reference) and
the progress row is re-anchored onto the new path, so the solver stays sharp through the reveal
correction and the gate exit.

Measured on level2 paired 20-run track sequences (scripts/compare_v10_3.py, same seeds for both
controllers): seed 42: v10.1 14/20 at avg 8.30 s -> v10.3 19/20 at avg 8.10 s. Faster AND far
more reliable at the identical speed budget; the gain is concentrated where the analysis said it
would be (gate-reveal corrections and gate exits).

v10.3 also carries an optional GATE-WINDOW SPEED CAP (controller_core_v10_3.arc_path.GateArcPath): speed
capped to V_GATE inside a Gaussian window at each gate's arc-position with a feasibility-
repairing backward/forward pass, to buy reveal-correction margin (~v^2-scaled) and fund a higher
straight budget. On level2 it is OFF -- measured worse (see controller_core_v10_3.cockpit for the numbers:
the gates are too closely spaced, the cap costs more than the raised budget recovers). It is the
knob for long-straight tracks or real-flight margin.

The OCP, its states/parameters, the gate-aware contouring weight, the speed budget, the planner,
and the takeoff are all v10.1 -- only the warm-start handling (and optionally the path's speed
profile) changes, hence v10.3 (not v11). Implemented as a thin subclass of ControllerV101.
REQUIRES the acados environment -- run under ``pixi run`` (sets ACADOS_SOURCE_DIR + the tera
renderer); the C solver is code-generated and compiled once per process into c_generated_code/.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line

from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.controller_v10_1 import ControllerV101
from lsy_drone_racing.control.controller_core_v10_3.arc_path import GateArcPath
from lsy_drone_racing.control.controller_core_v10_3.mpcc import MPCC
from lsy_drone_racing.control.controller_core_v10_3.settings import ControllerSettings

if TYPE_CHECKING:
    from crazyflow import Sim

    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation


class ControllerV103(ControllerV101):
    """v10.1's gate-aware time-optimal MPCC with replan continuity and a gate speed cap."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v10.1, then swap in the v10.3 settings and the rebase-capable MPCC."""
        super().__init__(obs, info, config)
        self._settings = ControllerSettings()
        mpcc = self._settings.mpcc
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCC(mpcc, a_max)
        # Refresh the cached limits v10's flow reads, from the v10.3 settings.
        self._v_theta_max = mpcc.v_theta_max
        self._ramp_s, self._ramp_start = mpcc.ramp_s, mpcc.ramp_start
        self._a_lat_max, self._v_min = mpcc.a_lat_max, mpcc.v_min
        self._w_base, self._w_gate, self._gate_sigma = (
            mpcc.w_contour_base, mpcc.w_contour_gate, mpcc.gate_sigma,
        )
        # Gate-window speed cap knobs (read when (re)building the GateArcPath).
        self._v_gate = mpcc.v_gate
        self._gate_v_pre, self._gate_v_post = mpcc.gate_v_pre, mpcc.gate_v_post
        self._path: GateArcPath | None = None
        self._s = 0.0

    def _track_action(self, frame: DroneObservation) -> np.ndarray:
        """Fly the plan; mid-flight replans rebase the solver instead of resetting it."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if rebuilt or self._path is None:
            gates_ahead = plan.gate_pos_snapshot[max(frame.target_gate, 0):]
            path = GateArcPath(
                plan.curve, plan.t_total, self._v_theta_max, self._a_lat_max, self._v_min,
                gates_ahead, self._w_base, self._w_gate, self._gate_sigma,
                self._v_gate, self._gate_v_pre, self._gate_v_post,
            )
            self._s = path.project(frame.pos, 0.0)
            if self._path is None:  # episode start / first plan: cold start as in v10.1
                self._mpcc.set_path(path)
            else:  # mid-flight replan: keep the warm start and the solver memory
                self._mpcc.rebase(path, self._s)
            self._path = path
        self._s = self._path.project(frame.pos, self._s)  # advance the foot-point anchor
        elapsed = (self._tick - self._nav_start_tick) * self._dt
        ramp = min(1.0, self._ramp_start + (1.0 - self._ramp_start) * elapsed / self._ramp_s)
        accel = self._mpcc.solve(frame.pos, frame.vel, self._s, self._v_theta_max * ramp)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)

    def render_callback(self, sim: Sim) -> None:
        """Draw the active plan (green, as v9/v10.1) plus the MPCC's predicted horizon (cyan).

        The cyan line is the solver's own intended trajectory over the next ~0.9 s -- the thing
        to watch when judging aggression: it shows how hard the optimiser plans to cut toward a
        gate after a reveal replan, and (thanks to the v10.3 rebase) it stays continuous across
        replans instead of restarting from a cold guess.
        """
        super().render_callback(sim)
        pred = self._mpcc.predicted_positions()
        if pred is not None and self._mode == self._MODE_NAVIGATE:
            draw_line(sim, pred.astype(np.float32), rgba=(0.0, 0.9, 1.0, 1.0))
