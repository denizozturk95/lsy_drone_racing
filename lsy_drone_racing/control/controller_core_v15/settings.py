"""Configuration for the controller_v15 controller.

Extends v11's settings with the arena keep-in barrier knobs. The OCP cost expression changes
in v11.2 (the arena hinge is added to ``cost_expr_ext_cost``), so the v11 solver is NOT reused:
controller_core_v15.mpcc builds its own solver under the ``controller_v11_2`` codegen namespace (never share a
namespace across differing generated code; see the v10.4 dlopen segfault note). The OCP
DIMENSIONS are unchanged from v11 (the arena bounds are static constants baked into the cost,
not per-stage parameters), so the path view and parameter packing are reused verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lsy_drone_racing.control.controller_core_v11.settings import (
    ControllerSettings as _V11ControllerSettings,
)
from lsy_drone_racing.control.controller_core_v11.settings import MPCCSettings as _V11MPCCSettings
from lsy_drone_racing.control.controller_core_v15 import cockpit as cp


@dataclass(frozen=True)
class MPCCSettings(_V11MPCCSettings):
    """v11's tunnel MPCC settings plus the arena keep-in barrier knobs."""

    arena_x_min: float = cp.ARENA_X_MIN
    arena_x_max: float = cp.ARENA_X_MAX
    arena_y_min: float = cp.ARENA_Y_MIN
    arena_y_max: float = cp.ARENA_Y_MAX
    arena_inset: float = cp.ARENA_INSET
    arena_ramp_m: float = cp.ARENA_RAMP_M
    arena_clip_beta: float = cp.ARENA_CLIP_BETA
    w_arena: float = cp.W_ARENA
    v_arena: float = cp.V_ARENA


@dataclass(frozen=True)
class ControllerSettings(_V11ControllerSettings):
    """v11's settings with the arena-aware MPCC."""

    mpcc: MPCCSettings = field(default_factory=MPCCSettings)
