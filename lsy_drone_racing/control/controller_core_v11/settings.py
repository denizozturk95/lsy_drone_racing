"""Configuration for the controller_v11 controller.

Extends v10.5's settings with the v11 tunnel/reveal knobs. The OCP CHANGES in v11 (two extra
tunnel constraints, five extra per-stage parameters), so MPCCSettings carries the new fields
and controller_core_v11.mpcc builds its own solver under the ``controller_v11`` codegen namespace -- the
compiled v10.4/v10.5 solver is NOT shared (different dimensions; see the v10.4 ledger on why
reusing a namespace across dimensions segfaults).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsy_drone_racing.control.controller_core_v10_5.settings import (
    ControllerSettings as _V105ControllerSettings,
)
from lsy_drone_racing.control.controller_core_v10_5.settings import MPCCSettings as _V105MPCCSettings
from lsy_drone_racing.control.controller_core_v11 import cockpit as cp


@dataclass(frozen=True)
class MPCCSettings(_V105MPCCSettings):
    """v10.5's MPCC settings, de-spiked cost, plus the tunnel and reveal-cap knobs."""

    w_contour_base: float = cp.W_CONTOUR_BASE
    w_contour_gate: float = cp.W_CONTOUR_GATE
    gate_sigma: float = cp.GATE_SIGMA
    tunnel_w_max: float = cp.TUNNEL_W_MAX
    tunnel_w_gate: float = cp.TUNNEL_W_GATE
    tunnel_h_max: float = cp.TUNNEL_H_MAX
    tunnel_h_gate: float = cp.TUNNEL_H_GATE
    tunnel_taper_m: float = cp.TUNNEL_TAPER_M
    tunnel_w_min: float = cp.TUNNEL_W_MIN
    tunnel_obs_margin: float = cp.TUNNEL_OBS_MARGIN
    tunnel_post_margin: float = cp.TUNNEL_POST_MARGIN
    tunnel_z_floor: float = cp.TUNNEL_Z_FLOOR
    tunnel_z_ceil: float = cp.TUNNEL_Z_CEIL
    tunnel_curv_frac: float = cp.TUNNEL_CURV_FRAC
    tunnel_slack_l1: float = cp.TUNNEL_SLACK_L1
    tunnel_slack_l2: float = cp.TUNNEL_SLACK_L2
    v_gate_reveal: float = cp.V_GATE_REVEAL
    reveal_cap_all: bool = cp.REVEAL_CAP_ALL
    reveal_pre_m: float = cp.REVEAL_PRE_M
    reveal_post_m: float = cp.REVEAL_POST_M
    launch_cap_pre_m: float = cp.LAUNCH_CAP_PRE_M


@dataclass(frozen=True)
class ControllerSettings(_V105ControllerSettings):
    """v10.5's settings with the v11 tunnel MPCC."""

    mpcc: MPCCSettings = field(default_factory=MPCCSettings)
