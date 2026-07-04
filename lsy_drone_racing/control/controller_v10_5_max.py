"""v10.5 with a max-attack speed budget (v10_5_max).

Same architecture, reshaped budget: MU 4.0, V_THETA_MAX 4.5, A_LAT_MAX 10.5, TILT_RATIO 1.0,
hotter ramp (0.40/1.6), dropped gate-0 cap. Faster on finishes, sheds some vs v10.5 under
+/-0.15 m reveal. REQUIRES the acados environment (``pixi run``).
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
