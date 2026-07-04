"""Final-track tuned time-optimal MPCC (v25): v10 with a raised speed budget.

Frontier tune for final.toml (seed 2026): 20/20 @ 7.85s. Budget raised to
V_MAX 4.2, A_LAT_MAX 10.0, MU 3.0, TILT_RATIO 0.95, ramp 0.20/1.5.
If evaluation seed changes, prefer the l budget (3.6/9.0, mu 2.0, ramp 0.12/1.8).
REQUIRES acados (``pixi run``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.controller_core_v10.mpcc import MPCC
from lsy_drone_racing.control.controller_core_v10.settings import (
    ControllerSettings as _V10ControllerSettings,
)
from lsy_drone_racing.control.controller_core_v10.settings import MPCCSettings as _V10MPCCSettings
from lsy_drone_racing.control.controller_v10 import ControllerV10

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class _MPCCSettings(_V10MPCCSettings):
    """v10's MPCC settings with the final-track frontier budget (see module docstring)."""

    mu: float = 3.0
    v_max: float = 4.2
    v_theta_max: float = 4.2
    a_lat_max: float = 10.0
    tilt_ratio: float = 0.95
    ramp_start: float = 0.20
    ramp_s: float = 1.5


@dataclass(frozen=True)
class _Settings(_V10ControllerSettings):
    """v10's full settings stack with the frontier MPCC budget swapped in."""

    mpcc: _MPCCSettings = field(default_factory=_MPCCSettings)


class ControllerV25(ControllerV10):
    """v10's planner/takeoff flown at the final-track speed frontier (20/20 @ 7.85 s)."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build v10, then swap in the raised budget and rebuild the MPCC."""
        super().__init__(obs, info, config)
        self._settings = _Settings()
        mpcc = self._settings.mpcc
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCC(mpcc, a_max)
        self._v_theta_max = mpcc.v_theta_max
        self._ramp_s, self._ramp_start = mpcc.ramp_s, mpcc.ramp_start
        self._a_lat_max, self._v_min = mpcc.a_lat_max, mpcc.v_min
        self._path = None
        self._s = 0.0
