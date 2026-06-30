"""controller_v24_r16 — v24 with a wider 0.16 m keep-out (ensemble variant).

Same solver as v24 (radius is a runtime parameter). r=0.16 gives off-route poles more margin (safer
on route-obstacle tracks) at the cost of over-avoidance near gates — ANTI-CORRELATED with v24@0.13
and @0.10. Pooled in the best-of-N ensemble, it wins yet-different seeds.
"""

from __future__ import annotations

from lsy_drone_racing.control.controller_v24 import ControllerV24


class ControllerV24R16(ControllerV24):
    """v24 with a 0.16 m obstacle keep-out (anti-correlated ensemble member)."""

    def __init__(self, obs: dict, info: dict, config: dict):
        """Build v24, then widen the keep-out radius."""
        super().__init__(obs, info, config)
        self._r_avoid = 0.16
