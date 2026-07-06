"""controller_v24_r10_fast — v24_r10 with a faster search spiral (PARAMETERS ONLY).

Same algorithm as v24_r10 / v17: blind Archimedean-spiral discovery, then v10.5 MPCC navigate.
The ONLY change is the spiral's numeric budget; both methods below are byte-for-byte v17's with the
constants swapped, no control-flow change.

Tuned on final.toml (thread-pinned, deterministic). Stock v24_r10 = 85% @ 25.6 s.
BAKED: SPEED 2.0 + gentle gains KP 5.5 / KD 3.0 -> 90% @ 16.4 s (N=50). ~9 s faster AND more
reliable than stock. Two levers, found empirically:
  * SPEED shrinks the ~17 s search tax (2.0 -> ~11 s). Alone it drops SR (73% @ 2.0) via a
    takeoff->search entry-transient crash (drone over-tilts before it's climbed, hits ground ~t4s).
  * The entry crash is cured by LOWERING the PD gains, esp. KD: at SPEED 2.0, KD 4->3 lifts SR
    70%->90%. Higher gains crash it; KD 3.5/4 regress; SEARCH_ALT below 1.8 regresses.
"Tighter" (wider arm spacing) is NOT usable — STEP above ~0.6 opens coverage gaps that miss gates;
RADIUS must stay >=~2.2 to reach gate 2. Knobs env-overridable: SPIRAL_SPEED/STEP/RADIUS/KP/KD/CATCH/ALT.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import controller_v17 as _v17
from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.controller_core_v10_5.settings import MPCCSettings as _V105MPCC
from lsy_drone_racing.control.controller_core_v24.mpcc import MPCC as _MPCCv24
from lsy_drone_racing.control.controller_v24_r10 import ControllerV24R10

_SPEED = float(os.environ.get("SPIRAL_SPEED", "2.0"))  # was 1.1; faster spiral traversal
_STEP = float(os.environ.get("SPIRAL_STEP", "0.6"))  # arm spacing; >0.6 opens coverage gaps
_RADIUS = float(os.environ.get("SPIRAL_RADIUS", "2.3"))  # was 2.2 (>=~2.2 to reach gate 2)
_KP = float(os.environ.get("SPIRAL_KP", "5.5"))  # gentle; higher crashes the entry transient
_KD = float(os.environ.get("SPIRAL_KD", "3.0"))  # gentle vel gain — the key to surviving entry @ SPEED 2.0
_CATCH = float(os.environ.get("SPIRAL_CATCH", "0.7"))  # clutch radius; tighter hurt
_ALT = float(os.environ.get("SPIRAL_ALT", str(_v17._SEARCH_ALT)))  # search alt; floor ~1.6 (poles 1.52m)

# --- navigate-phase speed budget (v10_5_max's max-attack values; env-tunable) ---
_NAV_VTHETA = float(os.environ.get("NAV_VTHETA", "4.5"))  # v_theta_max (progress speed), was ~3.0
_NAV_ALAT = float(os.environ.get("NAV_ALAT", "10.5"))  # a_lat_max (cornering), was ~6.x
_NAV_ATHETA = float(os.environ.get("NAV_ATHETA", "12.0"))  # a_theta_max (progress accel)
_NAV_MU = float(os.environ.get("NAV_MU", "4.0"))  # progress-reward weight


@dataclass(frozen=True)
class _FastNavMPCC(_V105MPCC):
    """v10.5 MPCC settings with v10_5_max's hotter speed budget (parameters only)."""

    mu: float = _NAV_MU
    v_max: float = 4.5
    v_theta_max: float = _NAV_VTHETA
    a_theta_max: float = _NAV_ATHETA
    a_lat_max: float = _NAV_ALAT
    tilt_ratio: float = 1.0
    ramp_start: float = 0.40
    ramp_s: float = 1.6
    v_gate_react: float = 999.0
    proj_band_m: float = 0.8


class ControllerV24R10Fast(ControllerV24R10):
    """v24_r10 with a faster search spiral AND a hotter navigate budget (near-nominal final track)."""

    def __init__(self, obs: dict, info: dict, config: dict):
        """Build v24_r10, then rebuild the navigate MPCC on the hot budget (mirrors v10_5_max)."""
        super().__init__(obs, info, config)  # v24_r10: stock-budget MPCCv24, _r_avoid=0.10
        mpcc = _FastNavMPCC()
        a_max = self._command.thrust_max / self._mass
        self._mpcc = _MPCCv24(mpcc, a_max)  # fresh acados codegen keyed on the new budget
        # refresh every cached knob v10.5's _track_action reads (mirrors v10_5_max.__init__)
        self._v_theta_max = mpcc.v_theta_max
        self._ramp_s, self._ramp_start = mpcc.ramp_s, mpcc.ramp_start
        self._a_lat_max, self._v_min = mpcc.a_lat_max, mpcc.v_min
        self._w_base, self._w_gate, self._gate_sigma = (
            mpcc.w_contour_base, mpcc.w_contour_gate, mpcc.gate_sigma,
        )
        self._v_gate_react = mpcc.v_gate_react
        self._react_delta = mpcc.react_delta_m
        self._react_v_pre, self._react_v_post = mpcc.react_v_pre, mpcc.react_v_post
        self._proj_band = mpcc.proj_band_m

    def _build_search_curve(self, frame) -> None:
        """v17._build_search_curve with tuned spiral constants (control flow unchanged)."""
        start = np.asarray(frame.pos, dtype=np.float64)
        climb = np.array([start[0], start[1], _ALT])
        a = _STEP / (2.0 * np.pi)
        spiral: list[np.ndarray] = []
        theta = 0.0
        while a * theta <= _RADIUS:
            r = a * theta
            x = float(np.clip(r * np.cos(theta), -_v17._ARENA_X_LIM, _v17._ARENA_X_LIM))
            y = float(np.clip(r * np.sin(theta), -_v17._ARENA_Y_LIM, _v17._ARENA_Y_LIM))
            pt = np.array([x, y, _ALT])
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
