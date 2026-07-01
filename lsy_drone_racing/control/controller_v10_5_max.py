"""controller_v10_5_max: v10.5 with a MAX-ATTACK speed budget (the "go way faster" build).

v10.5 flies a good line but paces well below its own curvature profile (the "solver-pace gap":
v10.x flies 0.5-0.9 m/s under its profile everywhere because pacing is a cost-side balance). This
build keeps v10.5's architecture UNCHANGED -- planner, dynamics-aware anchor, gate contour spikes,
mini-takeoff -- and only reshapes the speed cockpit:

  MU          1.5 -> 4.0   progress reward: drive vth hard at the profile (close the pace gap)
  V_THETA_MAX 3.2 -> 4.5   straight-line top speed
  V_MAX       3.2 -> 4.5   (kept consistent with V_THETA_MAX)
  A_THETA_MAX 8.0 -> 12.0  let vth ramp to the higher cap out of corners
  A_LAT_MAX   8.5 -> 10.5  corner speed. cf21B_500 a_max = 0.8 N / 0.04338 kg = 18.4 m/s^2
                           (T/W ~ 1.9); level-flight lateral ceiling at tilt 1.0 is ~9.8, so 10.5
                           borrows a little vertical through corners -- aggressive but realizable.
  TILT_RATIO  0.85 -> 1.0  45 deg cornering authority
  RAMP        0.25/2.4 -> 0.40/1.6  hot launch
  V_GATE_REACT 2.5 -> off  DROP the gate-0 launch cap and keep reactive caps off (max attack)
  PROJ_BAND   0.6 -> 0.8   the anchor's legitimate per-step advance scales with speed; widen the
                           correction band so it never clamps (still < 1.0, above which the fold
                           teleport returns -- v10.2 ledger).

HONEST CAVEAT -- the target (final.toml): this is a Level-3 layout with [env.track] randomize=false
BUT [env.randomizations] still perturbs gate pos +/-0.15 m, gate yaw, obstacles +/-0.15 m and mass
EVERY reset, revealed only within 0.7 m. So the reveal ceiling BINDS here exactly as it does on
randomized level2, where the v10.4 ledger measured that raising any gate's window speed -- and
dropping the launch cap in particular -- trades finish-rate for lap time ("EIGHT mechanisms all
landed ON the frontier"). Expect this build to be much faster on the finishes it makes and to shed
finishes vs v10.5 (gate-0 corridor + gate crossings at speed under the +/-0.15 reveal). Measure on
final.toml and dial MU / V_THETA_MAX / A_LAT_MAX / the dropped cap back toward v10.5 to taste.

REQUIRES the acados environment (a fresh solver is code-generated: the cache key changes with MU,
V_THETA_MAX, A_THETA_MAX, TILT_RATIO). Run under ``pixi run``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.controller_core_v10_5.mpcc import MPCC
from lsy_drone_racing.control.controller_core_v10_5.settings import (
    ControllerSettings as _V105ControllerSettings,
)
from lsy_drone_racing.control.controller_core_v10_5.settings import MPCCSettings as _V105MPCCSettings
from lsy_drone_racing.control.controller_v10_5 import ControllerV105

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class _MaxAttackMPCCSettings(_V105MPCCSettings):
    """v10.5's MPCC settings with the max-attack speed budget (see module docstring)."""

    mu: float = 4.0
    v_max: float = 4.5
    v_theta_max: float = 4.5
    a_theta_max: float = 12.0
    a_lat_max: float = 10.5
    tilt_ratio: float = 1.0
    ramp_start: float = 0.40
    ramp_s: float = 1.6
    v_gate_react: float = 999.0  # drop the gate-0 launch cap (min(profile, 999) == profile)
    proj_band_m: float = 0.8


@dataclass(frozen=True)
class _MaxAttackControllerSettings(_V105ControllerSettings):
    """v10.5's full settings stack with the max-attack MPCC budget swapped in."""

    mpcc: _MaxAttackMPCCSettings = field(default_factory=_MaxAttackMPCCSettings)


class ControllerV105Max(ControllerV105):
    """v10.5 with a max-attack speed budget: same line, driven far harder along it."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v10.5, then swap in the max-attack settings and rebuild the (re-keyed) solver."""
        super().__init__(obs, info, config)
        self._settings = _MaxAttackControllerSettings()
        mpcc = self._settings.mpcc
        a_max = self._command.thrust_max / self._mass
        # New cache key (mu/v_theta_max/a_theta_max/tilt_ratio changed) -> fresh acados codegen.
        self._mpcc = MPCC(mpcc, a_max)
        # Refresh every cached knob the inherited v10.5 _track_action reads (mirrors v10.5.__init__).
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
