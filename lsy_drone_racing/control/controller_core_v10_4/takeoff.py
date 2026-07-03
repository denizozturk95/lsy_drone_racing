"""Mini-takeoff for controller_v10_4: the v8 XY-held climb, but short, low, and fixed-duration.

The PID-tracked vertical climb is load-bearing in this environment: the env hard-disables the
drone on ANY floor touch once it drifts more than +/-0.02 m XY from its start (the takeoff-pad
carve-out), and the rotors spin up from zero so every episode begins with ~40 ms of thrust-free
free-fall onto the floor clamp. Holding the start XY through that window is what makes episode
starts survivable, so the phase stays.

What it does NOT need is v8's 0.5 m of altitude at a 0.6 s minimum spline time (profiled
hand-off at ~0.54 s). The racing MPCC only needs enough height that a few centimetres of
tracking sag cannot reach the floor; the plan's initial tangent is a ~65 deg climb anyway, so
handing off at 0.22 m with upward momentum points the drone along the plan. This subclass
replaces the duration rule with a fixed TAKEOFF_CLIMB_TIME (0.40 s: peak vertical accel
6*h/T^2 ~ 7.9 m/s^2, worst-case thrust ~0.77 N < the 0.8 N cap) -- the altitude-based
is_complete() hands off as soon as the measured climb reaches alt - z_tol (~0.26 s).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.controller_core_v8.takeoff import TakeoffPhase as _TakeoffPhase

if TYPE_CHECKING:
    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation
    from lsy_drone_racing.control.controller_core_v10_4.settings import LaunchTakeoffSettings
    from lsy_drone_racing.control.controller_core_v8.settings import ControllerSettings


class TakeoffPhase(_TakeoffPhase):
    """v8's XY-held climb with a fixed (short) spline duration instead of the 0.6 s floor."""

    def __init__(self, settings: ControllerSettings, takeoff: LaunchTakeoffSettings):
        """Store the v8 fields plus the fixed climb duration."""
        super().__init__(settings, takeoff)
        self._climb_time = float(takeoff.climb_time)

    def _build(self, frame: DroneObservation, tick: int) -> None:
        """Build the climb spline exactly as v8 does, but over the fixed duration."""
        start = np.asarray(frame.pos, dtype=np.float64).copy()
        start_vel = np.asarray(frame.vel, dtype=np.float64).copy()
        if float(np.linalg.norm(start_vel)) < 0.3:  # keep the spline opening upward (as v8)
            start_vel[2] = max(float(start_vel[2]), 0.0)
        target = np.array([start[0], start[1], self._alt])  # hold x/y, climb in z
        knot_times = np.array([0.0, self._climb_time])
        bc = ((1, start_vel), (1, np.zeros(3)))  # settle toward hover at the top
        self._curve = CubicSpline(knot_times, np.vstack([start, target]), bc_type=bc)
        self._t_total = self._climb_time
        self._start_tick = tick
