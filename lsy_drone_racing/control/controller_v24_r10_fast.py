"""controller_v24_r10_fast — v24_r10 with a faster, tighter search spiral (PARAMETERS ONLY).

Same algorithm as v24_r10 / v17: blind Archimedean-spiral discovery, then v10.5 MPCC navigate.
The ONLY change is the spiral's numeric budget — arm spacing (tighter), traversal speed (faster),
and the PD tracking gains that let the drone keep up. Both methods below are byte-for-byte v17's,
with the constants swapped; no control-flow change. Spiral knobs read from env so they can be swept:
SPIRAL_SPEED, SPIRAL_STEP, SPIRAL_RADIUS, SPIRAL_KP, SPIRAL_KD, SPIRAL_CATCH.
"""

from __future__ import annotations

import os

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import controller_v17 as _v17
from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.controller_v24_r10 import ControllerV24R10

_SPEED = float(os.environ.get("SPIRAL_SPEED", "2.4"))  # was 1.1
_STEP = float(os.environ.get("SPIRAL_STEP", "1.2"))  # was 0.6 (arm spacing; sensor swath ~1.4 m)
_RADIUS = float(os.environ.get("SPIRAL_RADIUS", "2.3"))  # was 2.2 (>=~2.2 to reach gate 2)
_KP = float(os.environ.get("SPIRAL_KP", "9.0"))  # was 6.0
_KD = float(os.environ.get("SPIRAL_KD", "5.0"))  # was 4.0
_CATCH = float(os.environ.get("SPIRAL_CATCH", "0.8"))  # was 0.7 (clutch radius)


class ControllerV24R10Fast(ControllerV24R10):
    """v24_r10 with the search spiral re-parameterized for speed on the near-nominal final track."""

    def _build_search_curve(self, frame) -> None:
        """v17._build_search_curve with tuned spiral constants (control flow unchanged)."""
        start = np.asarray(frame.pos, dtype=np.float64)
        climb = np.array([start[0], start[1], _v17._SEARCH_ALT])
        a = _STEP / (2.0 * np.pi)
        spiral: list[np.ndarray] = []
        theta = 0.0
        while a * theta <= _RADIUS:
            r = a * theta
            x = float(np.clip(r * np.cos(theta), -_v17._ARENA_X_LIM, _v17._ARENA_X_LIM))
            y = float(np.clip(r * np.sin(theta), -_v17._ARENA_Y_LIM, _v17._ARENA_Y_LIM))
            pt = np.array([x, y, _v17._SEARCH_ALT])
            if not spiral or float(np.linalg.norm(pt[:2] - spiral[-1][:2])) >= 0.3:
                spiral.append(pt)
            theta += _v17._SPIRAL_ANGLE_STEP
        pts = np.vstack([start, climb] + spiral)
        seg = np.maximum(np.linalg.norm(np.diff(pts, axis=0), axis=1), 1e-3)
        knots = np.concatenate([[0.0], np.cumsum(seg)]) / _SPEED
        bc = ((1, np.asarray(frame.vel, dtype=np.float64)), (1, np.zeros(3)))
        self._search_curve = CubicSpline(knots, pts, bc_type=bc)
        self._search_t_total = float(knots[-1])
        self._search_t = 0.0

    def _search_action(self, frame):
        """v17._search_action with tuned PD gains / clutch radius (control flow unchanged)."""
        pos = np.asarray(frame.pos, dtype=np.float64)
        vel = np.asarray(frame.vel, dtype=np.float64)
        ref = np.asarray(self._search_curve(self._search_t), dtype=np.float64)
        if float(np.linalg.norm(pos - ref)) < _CATCH:
            self._search_t = min(self._search_t + self._dt, self._search_t_total)
        ref_vel = np.asarray(self._search_curve(self._search_t, 1), dtype=np.float64)
        accel = _KP * (ref - pos) + _KD * (ref_vel - vel)
        thrust_vector = self._mass * (accel + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)
