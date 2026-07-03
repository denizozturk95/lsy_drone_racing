"""Gate-aware time-optimal MPCC core for controller_v10_3: v10.1's solver + REPLAN CONTINUITY.

The OCP is byte-for-byte v10.1's -- v10.3 changes only what happens when the reference path is
swapped mid-flight. v10.1's ``set_path`` performs a full reset: the warm start is discarded, the
acados solver memory is wiped, and the fallback command is zeroed. Replans fire whenever the
target gate advances (right after EVERY gate pass at ~3 m/s) and whenever an observed gate moves
more than the 0.05 m replan threshold -- which at level2's 0.7 m sensor range under +/-0.15 m
gate randomisation means at essentially every gate reveal, ~0.2 s before the gate. So the next
command after each replan came from ONE cold SQP-RTI iteration on a fabricated linearisation
(thbar marching at the full progress cap, vth0 hard-set to the cap), with hover thrust as the
failure fallback -- at exactly the moments where gate precision binds.

``rebase`` swaps the path WITHOUT resetting: the previous solution's world-frame states and
controls are kept (the world did not change, only the reference, so they remain dynamically
feasible and an excellent warm start), and the progress row is re-anchored by forward-projecting
the predicted positions onto the new path. The progress rates carry over: old and new paths
nearly coincide around the drone (a replan moves a gate by <= ~0.2 m). Episode starts and the
first plan still cold-start through ``set_path`` exactly as in v10.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.controller_core_v10_1.mpcc import MPCC as _MPCC

if TYPE_CHECKING:
    from lsy_drone_racing.control.controller_core_v10_3.arc_path import GateArcPath

__all__ = ["MPCC"]


class MPCC(_MPCC):
    """v10.1's MPCC plus warm-start-preserving path swaps for mid-flight replans."""

    def rebase(self, path: GateArcPath, s0: float) -> None:
        """Swap the reference path without discarding the warm start or the solver memory.

        ``s0`` is the drone's progress on the NEW path. The predicted progress row is rebuilt by
        projecting each predicted position onto the new path forward from ``s0`` (the forward
        search makes the row monotone by construction); states, rates, and controls are kept.
        Falls back to a full ``set_path`` reset when there is no previous solution to preserve.
        """
        if self._x_sol is None:
            self.set_path(path)
            return
        self._path = path
        th = np.empty(self._x_sol.shape[1])
        th[0] = float(s0)
        for k in range(1, th.shape[0]):
            th[k] = path.project(self._x_sol[0:3, k], th[k - 1])
        self._x_sol[6] = th

    def predicted_positions(self) -> np.ndarray | None:
        """The solver's predicted positions over the horizon (for overlays), or None."""
        if self._x_sol is None:
            return None
        return self._x_sol[0:3].T.copy()
