"""controller_v17 — Level-3 hybrid: blind-spiral discovery + v10.5's MPCC tunnel navigate.

The MPCC tunnel navigate (controller_v10_5) threads tight gate+obstacle corridors far better than
the spline-waypoint navigate of the gate_search line: with perfect detection it finishes level3 at
65% / ~8 s vs gate_search's 55% / ~16 s. Its only gap on level3 is that, with sensor_range 0.7 m,
it commits to a reference through nominal positions and flies it fast into an unrevealed obstacle.

v17 inserts a SEARCH mode between v10.5's TAKEOFF and NAVIGATE: the drone climbs above every pole
and flies a blind Archimedean spiral (level3 nominal gate positions are placeholders at the origin,
so the search must COVER the arena), revealing the true gate and obstacle positions. Once all gates
are discovered it hands off to v10.5's NAVIGATE -- which now plans its tunnel over TRUE positions.

SEARCH tracks one smooth spiral spline with a PD acceleration command through v10.5's
_vector_to_attitude (NOT the takeoff CascadedPid, which cannot hold altitude in horizontal flight).
The reference is CLUTCHED (steps forward only while the drone keeps up) so it can neither run away
nor jump between nearby spiral loops, and the horizontal acceleration is CAPPED so the thrust vector
never tilts far enough to drop the drone into the ground / pole tops. NAVIGATE is v10.5 verbatim.
REQUIRES the acados environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.common_controller_v2.state import parse_observation
from lsy_drone_racing.control.controller_v10_5 import ControllerV105

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation

_SEARCH_ALT = 1.8       # m — above every gate frame / obstacle (detection is XY-only, so free)
_SEARCH_SPEED = 1.1     # m/s — spiral cruise (gentle reveal pass, not a race)
_CATCH_RADIUS = 0.7     # m — advance the reference only while the drone is within this of it
_KP, _KD = 6.0, 4.0     # PD gains: position error + reference-velocity feed-forward -> accel
_SPIRAL_RADIAL_STEP = 0.6
_SPIRAL_ANGLE_STEP = np.pi / 6
_SEARCH_RADIUS = 2.2
_ARENA_X_LIM = 1.9
_ARENA_Y_LIM = 1.0


class ControllerV17(ControllerV105):
    """v10.5 with a blind-spiral gate-discovery SEARCH pass before its MPCC navigate."""

    _MODE_SEARCH = "SEARCH"

    def __init__(self, obs: dict, info: dict, config: dict):
        """Build v10.5, then add the discovery-search state."""
        super().__init__(obs, info, config)
        self._known_gates: set[int] = set()
        self._search_curve: CubicSpline | None = None
        self._search_t_total = 0.0
        self._search_t = 0.0

    def reset(self) -> None:
        """Reset v10.5 state plus the search state."""
        super().reset()
        self._known_gates = set()
        self._search_curve = None
        self._search_t_total = 0.0
        self._search_t = 0.0

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """TAKEOFF -> SEARCH (reveal gates) -> NAVIGATE (v10.5 MPCC over true positions)."""
        frame = parse_observation(obs)
        self._last_target = frame.target_gate
        if self._tick / self._freq >= self._settings.runtime.timeout_s or frame.target_gate == -1:
            self._finished = True
            return self._last_action.copy()

        gates_visited = np.asarray(obs["gates_visited"], dtype=bool)
        n_gates = len(gates_visited)
        self._known_gates.update(int(i) for i in np.nonzero(gates_visited)[0])

        # ── TAKEOFF: v10.5/v8 vertical climb ──────────────────────────────────
        if self._mode == self._MODE_TAKEOFF:
            if not self._takeoff.is_complete(frame, self._tick, self._dt):
                action = self._takeoff.action(
                    frame, self._feedback, self._tick, self._dt, self._mass, self._gravity
                )
                self._last_action = action
                return action.copy()
            self._mode = self._MODE_SEARCH
            self._build_search_curve(frame)

        # ── SEARCH: reveal the gates (and their neighbouring obstacles) ───────
        if self._mode == self._MODE_SEARCH:
            done = (
                len(self._known_gates) >= n_gates
                or self._search_t >= self._search_t_total - 1e-3
            )
            if not done:
                action = self._search_action(frame)
                self._last_action = action
                return action.copy()
            # hand off to v10.5 NAVIGATE with true positions; cold-start its MPCC/anchor
            self._mode = self._MODE_NAVIGATE
            self._references.reset()
            self._mpcc.reset()
            self._progress_t = 0.0
            self._nav_start_tick = self._tick
            self._path = None
            self._gate_nominal = None

        # ── NAVIGATE: v10.5 MPCC tunnel, verbatim ────────────────────────────
        action = self._track_action(frame)
        self._last_action = action
        return action.copy()

    def _build_search_curve(self, frame: DroneObservation) -> None:
        """One smooth spline: climb straight to search alt (above every pole), then spiral out."""
        start = np.asarray(frame.pos, dtype=np.float64)
        climb = np.array([start[0], start[1], _SEARCH_ALT])
        a = _SPIRAL_RADIAL_STEP / (2.0 * np.pi)
        spiral: list[np.ndarray] = []
        theta = 0.0
        while a * theta <= _SEARCH_RADIUS:
            r = a * theta
            x = float(np.clip(r * np.cos(theta), -_ARENA_X_LIM, _ARENA_X_LIM))
            y = float(np.clip(r * np.sin(theta), -_ARENA_Y_LIM, _ARENA_Y_LIM))
            pt = np.array([x, y, _SEARCH_ALT])
            if not spiral or float(np.linalg.norm(pt[:2] - spiral[-1][:2])) >= 0.3:
                spiral.append(pt)
            theta += _SPIRAL_ANGLE_STEP
        pts = np.vstack([start, climb] + spiral)
        seg = np.maximum(np.linalg.norm(np.diff(pts, axis=0), axis=1), 1e-3)
        knots = np.concatenate([[0.0], np.cumsum(seg)]) / _SEARCH_SPEED
        bc = ((1, np.asarray(frame.vel, dtype=np.float64)), (1, np.zeros(3)))
        self._search_curve = CubicSpline(knots, pts, bc_type=bc)
        self._search_t_total = float(knots[-1])
        self._search_t = 0.0

    def _search_action(self, frame: DroneObservation) -> NDArray[np.floating]:
        """Clutched PD tracking via the navigate-style accel -> attitude path (NOT the PID).

        v10.5's attitude conversion correctly trades thrust against tilt, so the drone holds
        altitude in horizontal flight; the takeoff CascadedPid does not, which made it fall.
        """
        pos = np.asarray(frame.pos, dtype=np.float64)
        vel = np.asarray(frame.vel, dtype=np.float64)
        ref = np.asarray(self._search_curve(self._search_t), dtype=np.float64)
        if float(np.linalg.norm(pos - ref)) < _CATCH_RADIUS:
            self._search_t = min(self._search_t + self._dt, self._search_t_total)
        ref_vel = np.asarray(self._search_curve(self._search_t, 1), dtype=np.float64)
        accel = _KP * (ref - pos) + _KD * (ref_vel - vel)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)
