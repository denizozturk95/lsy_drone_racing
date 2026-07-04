"""MPCC drone-racing controller for known tracks (v9).

Gate-aware planner (v8 path geometry) + model-predictive contouring controller.
TAKEOFF → NAVIGATE phases. REQUIRES the acados environment (``pixi run``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line
from drone_models.core import load_params

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.common_controller_v2.feedback import CascadedPid
from lsy_drone_racing.control.common_controller_v2.state import parse_observation
from lsy_drone_racing.control.controller_core_v8.takeoff import TakeoffPhase
from lsy_drone_racing.control.controller_core_v8.trajectory import ReferenceManager
from lsy_drone_racing.control.controller_core_v9.mpcc import MPCC, sample_path
from lsy_drone_racing.control.controller_core_v9.settings import ControllerSettings

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation
    from lsy_drone_racing.control.controller_core_v8.trajectory import ReferencePlan


class ControllerV9(Controller):
    """Flies v8's gate-aware path with an MPCC that goes as fast as the limits allow."""

    _MODE_TAKEOFF = "TAKEOFF"
    _MODE_NAVIGATE = "NAVIGATE"

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the planner, takeoff phase, and the MPCC."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("controller_v9 requires env.control_mode = 'attitude'.")
        self._settings = ControllerSettings()
        self._freq = float(config.env.freq)
        self._dt = 1.0 / self._freq
        params = load_params(config.sim.physics, config.sim.drone_model)
        self._mass = float(params["mass"])
        self._gravity = self._settings.runtime.gravity
        self._command = self._settings.command
        self._feedback = CascadedPid(self._settings.feedback)
        self._references = ReferenceManager(
            self._settings.planner,
            self._settings.runtime.replan_gate_delta_m,
            self._settings.runtime.replan_obstacle_delta_m,
        )
        self._takeoff = TakeoffPhase(self._settings, self._settings.takeoff)
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCC(self._settings.mpcc, a_max)
        self._arc = np.arange(self._settings.mpcc.horizon + 1) * self._mpcc.ds
        self._ramp_s = self._settings.mpcc.ramp_s
        self._ramp_start = self._settings.mpcc.ramp_start
        self._tick = 0
        self._nav_start_tick = 0
        self._progress_t = 0.0
        self._finished = False
        self._mode = self._MODE_TAKEOFF
        self._last_target = -1
        self._last_action = self._hover_action()

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return a [roll, pitch, yaw, thrust] attitude action for the current step."""
        frame = parse_observation(obs)
        self._last_target = frame.target_gate
        if self._tick / self._freq >= self._settings.runtime.timeout_s or frame.target_gate == -1:
            self._finished = True
            return self._last_action.copy()

        if self._mode == self._MODE_TAKEOFF:
            if not self._takeoff.is_complete(frame, self._tick, self._dt):
                action = self._takeoff.action(
                    frame, self._feedback, self._tick, self._dt, self._mass, self._gravity
                )
                self._last_action = action
                return action.copy()
            self._mode = self._MODE_NAVIGATE
            self._references.reset()
            self._mpcc.reset()
            self._progress_t = 0.0
            self._nav_start_tick = self._tick

        action = self._track_action(frame)
        self._last_action = action
        return action.copy()

    def _track_action(self, frame: DroneObservation) -> NDArray[np.floating]:
        """(Re)plan the global path and let the MPCC fly it within the actuator limits."""
        plan, rebuilt = self._references.ensure_plan(frame)
        if rebuilt:
            self._progress_t = 0.0
        self._progress_t = self._project(plan, frame.pos)
        elapsed = (self._tick - self._nav_start_tick) * self._dt
        ramp = min(1.0, self._ramp_start + (1.0 - self._ramp_start) * elapsed / self._ramp_s)
        ref_p, ref_t = sample_path(plan.curve, self._progress_t, plan.t_total, self._arc * ramp)
        accel = self._mpcc.solve(frame.pos, frame.vel, ref_p, ref_t)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)

    def _project(self, plan: ReferencePlan, pos: NDArray[np.floating]) -> float:
        """Find the spline time of the closest point ahead of the current progress."""
        upper = min(self._progress_t + self._settings.runtime.projection_window_s, plan.t_total)
        sample_t = np.linspace(self._progress_t, upper, 40)
        distances = np.linalg.norm(np.asarray(plan.curve(sample_t)) - pos, axis=1)
        return float(sample_t[int(np.argmin(distances))])

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the controller clock after an environment step."""
        self._tick += 1
        return self._finished

    def reset(self) -> None:
        """Reset per-episode controller state."""
        self._tick = 0
        self._progress_t = 0.0
        self._finished = False
        self._mode = self._MODE_TAKEOFF
        self._last_target = -1
        self._feedback.reset()
        self._references.reset()
        self._takeoff.reset()
        self._mpcc.reset()
        self._last_action = self._hover_action()

    def episode_callback(self) -> None:
        """Reset state after an episode completes."""
        self.reset()

    def episode_reset(self) -> None:
        """Reset state before the next episode starts."""
        self.reset()

    def render_callback(self, sim: Sim) -> None:
        """Draw the active path (green) during navigation."""
        plan = self._references.plan
        if plan is not None and self._mode == self._MODE_NAVIGATE:
            samples = plan.curve(np.linspace(0.0, plan.t_total, 100))
            draw_line(sim, np.asarray(samples, dtype=np.float32), rgba=(0.0, 1.0, 0.0, 1.0))

    def diagnostic(self) -> dict[str, float | int | str | None]:
        """Return a short status summary for debugging and logging."""
        plan = self._references.plan
        return {
            "controller_phase": "FINISHED" if self._finished else self._mode,
            "active_target_gate": self._last_target,
            "controller_time": min(self._tick / self._freq, self._settings.runtime.timeout_s),
            "reference_end_time": None if plan is None else plan.t_total,
        }

    def _hover_action(self) -> NDArray[np.float32]:
        """Build a level hover action that holds altitude at startup and on finish."""
        thrust = float(
            np.clip(self._mass * self._gravity, self._command.thrust_min, self._command.thrust_max)
        )
        return np.array([0.0, 0.0, 0.0, thrust], dtype=np.float32)
