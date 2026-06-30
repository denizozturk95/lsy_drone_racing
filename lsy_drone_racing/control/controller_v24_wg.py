"""controller_v24_wg — v24 with a higher gate contour weight (tighter gate-centerline tracking).

Floor diagnosis: navigate failures are marginal GATE CLIPS — the plan threads the gate centre but
the MPCC carries ~0.12 m contour error there, which (with the 0.07 m drone half-width) nearly fills
the +-0.20 m opening, so adverse gate randomisation clips the frame. Raising w_gate makes the MPCC
hug the gate line harder -> smaller contour error at the gate -> bigger clip margin. Runtime path
parameter, same solver. Aimed at the dominant floor mechanism; also an ensemble pool member.
"""

from __future__ import annotations

from lsy_drone_racing.control.controller_v24 import ControllerV24


class ControllerV24WG(ControllerV24):
    """v24 with the gate contouring weight raised from 20 to 45 (tighter gate tracking)."""

    def __init__(self, obs: dict, info: dict, config: dict):
        """Build v24, then raise the gate contour weight."""
        super().__init__(obs, info, config)
        self._w_gate = 45.0
