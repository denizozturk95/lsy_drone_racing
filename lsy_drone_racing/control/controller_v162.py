"""controller_v162: v16 with a deadlock-proof MPCC solve (the g3->g4 hover fix).

v16 flew great until the g2->g3 U-turn fold, where a single failed SQP-RTI step froze the MPCC's
predicted progress (the reference anchor reads it) and the drone hovered/wobbled in place for
seconds before drifting free -- the "slow recovery lap" the v11 cockpit flagged for this line. v162
is v16 UNCHANGED except that the tunnel MPCC is swapped for controller_core_v162.RobustMPCC, whose
solve() recovers a failed warm step with an in-tick cold re-solve instead of freezing. Successful
solves, planner, tunnel, caps, and every knob are v16's. REQUIRES the acados environment -- run
under ``pixi run``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.controller_core_v162.mpcc import RobustMPCC
from lsy_drone_racing.control.controller_v16 import ControllerV16

if TYPE_CHECKING:
    pass


class ControllerV162(ControllerV16):
    """v16 with the solver-failure-robust MPCC (identical flight, no hover deadlock)."""

    def __init__(self, obs: dict[str, np.ndarray], info: dict, config: dict):
        """Build v16, then swap its tunnel MPCC for the deadlock-proof RobustMPCC (same solver)."""
        super().__init__(obs, info, config)
        a_max = self._command.thrust_max / self._mass
        self._mpcc = RobustMPCC(self._settings.mpcc, a_max)
