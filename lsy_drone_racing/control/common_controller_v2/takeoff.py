"""Dedicated vertical takeoff phase for the controller_v6 controller.

v6 otherwise plans a single global cubic spline from the drone's current state through
all gates. At lift-off that couples the takeoff dynamics to the cruise speed: the
C²-continuous spline ramps vertical acceleration to whatever the downstream segments
demand, so raising the cruise speed makes the drone whip off the ground — a large
vertical-thrust swing at low altitude (while also tilting laterally toward the first
gate) destabilises the climb.

This phase decouples the two. It flies a clean rest-to-altitude *vertical* climb
(holding the start x/y), tracked with the same :func:`attitude_action` machinery as the
main loop, then reports completion so the controller can hand off to gate tracking from
a stable near-hover. The climb duration is derived from a target climb speed (not the
cruise speed), so lift-off aggressiveness is independent of how fast the race is flown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.common_controller_v2.attitude import attitude_action

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.common_controller_v2.feedback import CascadedPid
    from lsy_drone_racing.control.common_controller_v2.settings import ControllerSettings
    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation

# Floor on the climb duration so a tiny start-to-target height can't produce a near-zero
# time (and therefore an enormous endpoint acceleration). 0.6 s keeps the worst-case
# short climb gentle.
_MIN_CLIMB_TIME = 0.6


class TakeoffPhase:
    """A self-contained vertical climb from the ground to a target altitude.

    The climb is a clamped cubic from the current position straight up to
    ``[x, y, alt]`` with a zero end-velocity boundary condition (settles to a hover at
    the top). It builds its plan lazily on the first :meth:`action` call so it captures
    the true post-reset start pose, and exposes :meth:`is_complete` for the controller's
    mode handoff.
    """

    def __init__(
        self,
        settings: ControllerSettings,
        alt: float,
        climb_speed: float,
        z_tol: float,
        time_margin: float,
    ):
        """Store the climb geometry/timing; the plan is built lazily (see :meth:`action`)."""
        self._settings = settings
        self._alt = float(alt)
        self._climb_speed = max(float(climb_speed), 1e-3)
        self._z_tol = float(z_tol)
        self._time_margin = float(time_margin)
        self._curve: CubicSpline | None = None
        self._t_total = 0.0
        self._start_tick = 0

    def reset(self) -> None:
        """Forget the cached climb plan so the next episode rebuilds from its start pose."""
        self._curve = None
        self._t_total = 0.0
        self._start_tick = 0

    def is_complete(self, frame: DroneObservation, tick: int, dt: float) -> bool:
        """Return True once the drone has reached the target altitude (or the climb overran).

        Safe to call before the plan exists: it then reports completion purely from the
        measured altitude, so an already-airborne start skips takeoff entirely.
        """
        reached = float(frame.pos[2]) >= self._alt - self._z_tol
        if self._curve is None:
            return reached
        overran = (tick - self._start_tick) * dt >= self._t_total + self._time_margin
        return reached or overran

    def action(
        self,
        frame: DroneObservation,
        feedback: CascadedPid,
        tick: int,
        dt: float,
        mass: float,
        gravity: float,
    ) -> NDArray[np.float32]:
        """Return the [roll, pitch, yaw, thrust] action tracking the vertical climb."""
        if self._curve is None:
            self._build(frame, tick)
        t_eval = float(np.clip((tick - self._start_tick) * dt, 0.0, self._t_total))
        action, _ = attitude_action(
            self._curve,
            t_eval,
            frame.pos,
            frame.vel,
            frame.quat,
            feedback,
            dt,
            mass,
            gravity,
            self._settings.command,
        )
        return action

    def _build(self, frame: DroneObservation, tick: int) -> None:
        """Build the rest-to-altitude vertical climb spline from the current pose."""
        start = np.asarray(frame.pos, dtype=np.float64).copy()
        start_vel = np.asarray(frame.vel, dtype=np.float64).copy()
        # Near the floor and nearly stationary, a spurious downward start velocity would
        # clamp the spline to a downward slope and drive it below z=0. Force the vertical
        # boundary condition non-negative so the climb opens upward.
        if float(np.linalg.norm(start_vel)) < 0.3:
            start_vel[2] = max(float(start_vel[2]), 0.0)
        target = np.array([start[0], start[1], self._alt])  # hold x/y, climb in z
        height = max(self._alt - float(start[2]), 0.0)
        # Duration from the target climb speed (NOT the cruise speed) → peak vertical
        # accel ≈ 6 × height / t_climb², kept gentle and cruise-speed-independent.
        t_climb = max(height / self._climb_speed, _MIN_CLIMB_TIME)
        knot_times = np.array([0.0, t_climb])
        bc = ((1, start_vel), (1, np.zeros(3)))  # settle to a hover at the top
        self._curve = CubicSpline(knot_times, np.vstack([start, target]), bc_type=bc)
        self._t_total = float(t_climb)
        self._start_tick = tick
