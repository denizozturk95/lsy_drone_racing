"""Gate-aware arc-length view for v10.5: v10.4's reactive caps + v10.2's bounded projection.

v10.5 is the merge of the two sibling v10.x branches. The path view therefore needs BOTH:

- v10.4's per-gate reactive approach-speed cap (the launch-window / moved-gate protection that
  funds the fast start) -- inherited verbatim by subclassing controller_core_v10_4.arc_path.GateArcPath,
  including its constructor signature (gate_caps, react_v_pre, react_v_post);
- v10.2's BOUNDED-CORRECTION projection ``project_near`` (the dynamics-aware anchor that kills
  the sharp-slalom fold teleport) -- the only thing added here, copied verbatim from v10.2.

The two are orthogonal: ``project_near`` reads the dense arc tables (``self._s``/``self._pts``,
built by the v10/v10.1 base and untouched by the v10.4 cap, which only reshapes ``self._vcurv``),
so the reactive speed profile and the bounded anchor compose with no interaction. See
controller_v10_5 for the full lineage and why the merge is byte-compatible with v10.4's solver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.controller_core_v10_4.arc_path import GateArcPath as _GateArcPath

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GateArcPath(_GateArcPath):
    """v10.4's reactive-cap GateArcPath plus v10.2's band-restricted nearest-point search."""

    def project_near(
        self, pos: NDArray[np.float64], center: float, band: float
    ) -> float:
        """Arc length of the nearest path point within ``+/- band`` of ``center`` (v10.2).

        ``center`` is the solver's predicted progress for this step. Restricting the search to a
        tight band around it keeps the progress anchor glued to the drone (the band is wider than
        the legitimate per-step fold advance) while making it impossible to select a far leg of a
        self-folding path (which lies outside the band) -- so the anchor can no longer teleport
        across a fold and skip a gate.
        """
        lo = int(np.searchsorted(self._s, center - band))
        hi = max(int(np.searchsorted(self._s, center + band)), lo + 1)
        nearest = int(np.argmin(np.linalg.norm(self._pts[lo:hi] - pos, axis=1)))
        return float(self._s[lo + nearest])
