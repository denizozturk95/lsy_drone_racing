"""Configuration for the controller_v9.1 controller.

Inherits v9's planner/takeoff/command/runtime settings, but builds its own MPCC settings
(so v9.1 can be tuned without touching v9) and adds the progress-governor settings. The
MPCCSettings dataclass shape is reused from v9; only the values come from the v9.1 cockpit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsy_drone_racing.control.controller_core_v9.settings import ControllerSettings as _V9ControllerSettings
from lsy_drone_racing.control.controller_core_v9.settings import MPCCSettings
from lsy_drone_racing.control.controller_core_v9_1 import cockpit as cp


def _mpcc_settings() -> MPCCSettings:
    """Build the v9.1-owned MPCC settings from the v9.1 cockpit."""
    return MPCCSettings(
        horizon=cp.HORIZON,
        step_dt=cp.STEP_DT,
        v_ref=cp.V_REF,
        v_max=cp.V_MAX,
        ramp_s=cp.RAMP_S,
        ramp_start=cp.RAMP_START,
        tilt_ratio=cp.TILT_RATIO,
        a_z_min=cp.A_Z_MIN,
        w_contour=cp.W_CONTOUR,
        w_lag=cp.W_LAG,
        w_accel=cp.W_ACCEL,
        max_iter=cp.MAX_ITER,
        gravity=cp.GRAVITY,
    )


@dataclass(frozen=True)
class ProgressSettings:
    """Progress-governor knobs (from the cockpit)."""

    min_progress_rate: float = cp.MIN_PROGRESS_RATE
    max_lead_t: float = cp.MAX_LEAD_T
    v_stall: float = cp.V_STALL
    t_stall: float = cp.T_STALL
    stall_boost: float = cp.STALL_BOOST


@dataclass(frozen=True)
class ControllerSettings(_V9ControllerSettings):
    """v9's planner/takeoff/command/runtime plus v9.1's own MPCC and the progress governor."""

    mpcc: MPCCSettings = field(default_factory=_mpcc_settings)
    progress: ProgressSettings = field(default_factory=ProgressSettings)
