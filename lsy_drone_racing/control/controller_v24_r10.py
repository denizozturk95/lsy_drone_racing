"""controller_v24_r10 — v24 with a tighter 0.10 m keep-out (ensemble variant).

Same solver as v24 (radius is a runtime parameter). r=0.10 recovers the centred-doorway seeds v24
at 0.13 collapses on, at the cost of margin on off-route poles elsewhere — i.e. it is ANTI-CORRELATED
with v24@0.13. Pooled in the best-of-N ensemble (scripts/ensemble_select.py), it wins different seeds.
"""

from __future__ import annotations

from lsy_drone_racing.control.controller_v24 import ControllerV24


class ControllerV24R10(ControllerV24):
    """v24 with a 0.10 m obstacle keep-out (anti-correlated ensemble member)."""

    def __init__(self, obs: dict, info: dict, config: dict):
        """Build v24, then tighten the keep-out radius."""
        super().__init__(obs, info, config)
        self._r_avoid = 0.10
