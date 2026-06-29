"""Gate-aware time-optimal MPCC: fast launch + dynamics-aware anchor (controller_v10_5).

v10.5 is the MERGE of the two sibling branches of the v10.x tree (see controller_core_v10_5.__init__ for the
lineage). It takes v10.4's everything -- the mini-takeoff, the hot launch ramp, the honest cold
start, the replan-continuity rebase, and the reactive per-gate caps (8.01 s level2 flagship) --
and re-applies v10.2's DYNAMICS-AWARE PROGRESS ANCHOR on top of it.

The problem v10.4 left open: it still anchors the MPCC's progress state with the GLOBAL geometric
projection (``self._path.project``), the exact mechanism v10.2 proved can teleport the anchor
~1-2 m across a path fold on a sharp slalom and skip the gate. v10.4's slalom robustness came from
LAUNCH fixes; the fold-teleport failure mode is structurally still present mid-race.

The fix is one substitution in v10.4's ``_track_action``. Instead of anchoring progress at the
global nearest path point, anchor it to the SOLVER'S OWN predicted progress one step ahead
(``predicted_progress()``), which is dynamics-feasible -- it advances at the bounded rate vth, so
it cannot teleport -- and let a geometric search correct it only within +/- PROJ_BAND_M
(``project_near``). The far fold leg lies outside the band and can never be selected; the gate-apex
motion lies inside it, so tracking is unchanged everywhere except the fold, where the skip is gone.

The merge composes cleanly with v10.4's two cold-start paths:

1. Cold start (episode start / first plan): after ``set_path``, ``_x_sol`` is None ->
   ``predicted_progress()`` returns None -> geometric fallback for exactly one step, identical to
   v10.2's handling. v10.4's honest cold start then seeds an anchored solution, so from step 2 the
   predicted-progress anchor is live.
2. Rebase (mid-flight replan -- the case neither parent had together): ``rebase`` keeps ``_x_sol``
   and re-anchors its progress row onto the new path, so ``predicted_progress()`` is already in
   NEW-PATH arc coordinates. The anchor works across replans with no special case -- strictly
   better than v10.2, which lost the prediction (and the fold protection) for one step at every
   replan.

The OCP, the horizon (18), and every launch/racing knob are v10.4's, so the compiled acados solver
is shared with v10.4 (codegen namespace ``controller_v10_4``). Implemented as a thin subclass of
ControllerV104; the only behavioural change is the anchor line plus always-on anchor telemetry
(per-step jump + band-edge rate; see ``anchor_telemetry``) used to validate the fold protection and
drive the PROJ_BAND_M sweep. REQUIRES the acados environment -- run under ``pixi run``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.controller_v10_4 import ControllerV104
from lsy_drone_racing.control.controller_core_v10_5.arc_path import GateArcPath
from lsy_drone_racing.control.controller_core_v10_5.mpcc import MPCC
from lsy_drone_racing.control.controller_core_v10_5.settings import ControllerSettings

if TYPE_CHECKING:
    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation


class ControllerV105(ControllerV104):
    """v10.4's launch-optimised racing MPCC with v10.2's dynamics-aware progress anchor."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v10.4, then swap in the v10.5 MPCC (predicted-progress) and the anchor band."""
        super().__init__(obs, info, config)
        self._settings = ControllerSettings()
        mpcc = self._settings.mpcc
        a_max = self._command.thrust_max / self._mass
        # v10.5 MPCC adds predicted_progress(); same cache key as v10.4 -> shares the solver.
        self._mpcc = MPCC(mpcc, a_max)
        # Refresh the cached knobs the inherited v10.4 flow reads (all values identical to v10.4;
        # re-read so a future v10.5-only retune of any of them is picked up here).
        self._v_theta_max = mpcc.v_theta_max
        self._ramp_s, self._ramp_start = mpcc.ramp_s, mpcc.ramp_start
        self._a_lat_max, self._v_min = mpcc.a_lat_max, mpcc.v_min
        self._w_base, self._w_gate, self._gate_sigma = (
            mpcc.w_contour_base, mpcc.w_contour_gate, mpcc.gate_sigma,
        )
        self._v_gate_react = mpcc.v_gate_react
        self._react_delta = mpcc.react_delta_m
        self._react_v_pre, self._react_v_post = mpcc.react_v_pre, mpcc.react_v_post
        # The v10.2 anchor band: half-width of the geometric correction around the prediction.
        self._proj_band = mpcc.proj_band_m
        self._gate_nominal: np.ndarray | None = None
        self._path: GateArcPath | None = None
        self._s = 0.0
        self._reset_anchor_telemetry()

    def _reset_anchor_telemetry(self) -> None:
        """Clear the per-episode anchor diagnostics (max jump, jump samples, band-edge rate)."""
        self._anchor_prev_s: float | None = None
        self._anchor_jumps: list[float] = []  # |s_t - s_{t-1}| in a plan (rebuild steps excluded)
        self._band_edge_hits = 0   # project_near results pinned at the +/- band edge
        self._band_calls = 0       # project_near calls (i.e. steps with a live prediction)

    def reset(self) -> None:
        """Reset v10.4 state (incl. the nominal gate snapshot) plus the anchor telemetry."""
        super().reset()
        self._reset_anchor_telemetry()

    def _track_action(self, frame: DroneObservation) -> np.ndarray:
        """v10.4's flow verbatim, but the geometric anchor becomes the predicted-progress anchor."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if self._gate_nominal is None:  # first NAVIGATE tick: nothing is revealed yet
            self._gate_nominal = np.asarray(frame.gate_pos, dtype=np.float64).copy()
        new_plan = rebuilt or self._path is None
        if new_plan:
            first = max(frame.target_gate, 0)
            gates_ahead = plan.gate_pos_snapshot[first:]
            deltas = np.linalg.norm(
                plan.gate_pos_snapshot[first:] - self._gate_nominal[first:], axis=1
            )
            caps = np.where(deltas > self._react_delta, self._v_gate_react, np.inf)
            if first == 0:  # gate 0 is always capped: launch-window protection (see v10.4 cockpit)
                caps[0] = self._v_gate_react
            path = GateArcPath(
                plan.curve, plan.t_total, self._v_theta_max, self._a_lat_max, self._v_min,
                gates_ahead, self._w_base, self._w_gate, self._gate_sigma,
                caps, self._react_v_pre, self._react_v_post,
            )
            self._s = path.project(frame.pos, 0.0)
            if self._path is None:  # episode start / first plan: cold start (honest, see mpcc)
                self._mpcc.set_path(path)
            else:  # mid-flight replan: keep the warm start and re-anchor progress onto the new path
                self._mpcc.rebase(path, self._s)
            self._path = path
        # --- v10.5 anchor (replaces v10.4's `self._s = self._path.project(frame.pos, self._s)`) ---
        th_pred = self._mpcc.predicted_progress()
        if th_pred is None:  # first solve of a plan (honest cold start pending): geometric fallback
            self._s = self._path.project(frame.pos, self._s)
        else:  # dynamics-feasible anchor, geometric correction within +/- the band
            self._s = self._path.project_near(frame.pos, th_pred, self._proj_band)
            self._band_calls += 1
            if abs(self._s - th_pred) >= self._proj_band - 1e-9:  # search pinned at the band edge
                self._band_edge_hits += 1
        # Telemetry: per-step anchor jump within a plan (rebuild steps cross arc-coordinate frames,
        # so they are excluded -- they are not the within-fold teleport the metric watches for).
        if not new_plan and self._anchor_prev_s is not None:
            self._anchor_jumps.append(abs(self._s - self._anchor_prev_s))
        self._anchor_prev_s = self._s
        elapsed = (self._tick - self._nav_start_tick) * self._dt
        ramp = min(1.0, self._ramp_start + (1.0 - self._ramp_start) * elapsed / self._ramp_s)
        accel = self._mpcc.solve(frame.pos, frame.vel, self._s, self._v_theta_max * ramp)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)

    def anchor_telemetry(self) -> dict[str, float | int]:
        """Per-episode progress-anchor diagnostics (the v10.2-style fold-teleport signature).

        ``max_jump_m`` is the worst single-step anchor motion within a plan -- v10.2's success
        signature is ~0.7 m (vs ~2.0 m for a fold teleport). ``band_edge_rate`` is the fraction of
        live-prediction steps where the geometric correction was pinned at the +/- band edge; a
        high rate in runs that finish is the signal to widen PROJ_BAND_M (see the cockpit).
        """
        jumps = np.asarray(self._anchor_jumps, dtype=np.float64)
        edge_rate = (self._band_edge_hits / self._band_calls) if self._band_calls else 0.0
        return {
            "n_steps": int(jumps.size),
            "max_jump_m": float(jumps.max()) if jumps.size else 0.0,
            "p99_jump_m": float(np.percentile(jumps, 99)) if jumps.size else 0.0,
            "n_jumps_gt_1m": int((jumps > 1.0).sum()),
            "band_calls": int(self._band_calls),
            "band_edge_rate": edge_rate,
        }
