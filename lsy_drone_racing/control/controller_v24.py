"""controller_v24 — v17 search + obstacle-constrained MPCC navigate (avoidance inside the OCP).

v17 crashes mostly because its v10.5 MPCC only TRACKS the planner's reference, and the planner
inflates obstacle clearance to r_obs=0.20 m (~2x the real ~0.085 m drone-to-pole collision), which
closes physically-passable doorways. v24 keeps v17's blind-spiral search and flow verbatim but
swaps the navigate MPCC for the v24 core, which carries soft per-obstacle keep-out constraints in
the OCP at the real radius. The optimiser then threads off-centre doorways the planner declared
blocked, while real margin (and the gate contour weight) is preserved. REQUIRES the acados env.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.controller_core_v24.mpcc import MPCC as MPCCv24
from lsy_drone_racing.control.controller_v17 import ControllerV17

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.common_controller_v2.state import DroneObservation

_R_AVOID = 0.13     # m, drone-centre -> pole-centre keep-out. Single global value: per-obstacle
                    # radii (gate-proximity-tiered) were tried and relocated failures, not removed —
                    # the doorway win/lose boundary is dynamic, not geometric. 0.13 is the best net.


class ControllerV24(ControllerV17):
    """v17 with the obstacle-constrained MPCC core in the navigate phase."""

    def __init__(self, obs: dict, info: dict, config: dict):
        """Build v17, then replace the navigate MPCC with the v24 obstacle-constrained solver."""
        super().__init__(obs, info, config)
        a_max = self._command.thrust_max / self._mass
        self._mpcc = MPCCv24(self._settings.mpcc, a_max)
        self._revealed_obs_xy = np.zeros((0, 2), dtype=np.float64)
        self._r_avoid = _R_AVOID  # per-instance keep-out radius (variants override; runtime param)

    def reset(self) -> None:
        """Reset v17 state plus the v24 MPCC and the revealed-obstacle buffer."""
        super().reset()
        self._mpcc.reset()
        self._revealed_obs_xy = np.zeros((0, 2), dtype=np.float64)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Stash the revealed obstacle XY (true positions), then run v17's flow."""
        op = np.asarray(obs["obstacles_pos"], dtype=np.float64).reshape(-1, 3)
        vis = np.asarray(obs["obstacles_visited"], dtype=bool)
        self._revealed_obs_xy = op[vis, :2] if vis.any() else op[:0, :2]
        return super().compute_control(obs, info)

    def _track_action(self, frame: DroneObservation) -> np.ndarray:
        """Tag each revealed pole with a gate-proximity keep-out radius, feed the OCP, then navigate."""
        self._mpcc.set_obstacles(self._obstacle_triples())
        return super()._track_action(frame)

    def _obstacle_triples(self) -> NDArray[np.float64]:
        """(m, 3) [x, y, r] per revealed pole, all at the single global keep-out radius."""
        poles = self._revealed_obs_xy
        if len(poles) == 0:
            return poles.reshape(0, 3)
        triples = np.empty((len(poles), 3), dtype=np.float64)
        triples[:, :2] = poles
        triples[:, 2] = self._r_avoid
        return triples
