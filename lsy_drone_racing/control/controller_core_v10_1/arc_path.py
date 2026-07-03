"""Gate-aware arc-length view of a plan spline for the v10.1 time-optimal MPCC.

Extends v10's ArcPath (dense arc-length tables of path point, unit tangent, curvature speed)
with the arc-positions of the gates and a per-arc contouring weight that spikes around each
gate. The MPCC reads this weight at every predicted progress and passes it as a per-stage
parameter, so the drone is forced to hug the path exactly where a gate must be threaded while
staying free on the straights -- which is what lets v10.1 carry a higher speed budget than v10
without overshooting gates (gate precision, not actuator authority, is the level2 lap-time cap).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.controller_core_v10.arc_path import ArcPath

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline


class GateArcPath(ArcPath):
    """v10 ArcPath plus gate arc-positions and a gate-spiked contouring-weight profile."""

    def __init__(
        self,
        curve: CubicSpline,
        t_total: float,
        v_cap: float,
        a_lat_max: float,
        v_min: float,
        gate_pos: NDArray[np.float64],
        w_base: float,
        w_gate: float,
        gate_sigma: float,
        n: int = 600,
    ):
        """Build the v10 arc tables, then project each gate centre onto the path arc length."""
        super().__init__(curve, t_total, v_cap, a_lat_max, v_min, n)
        self._w_base = float(w_base)
        self._w_gate = float(w_gate)
        self._inv2s2 = 1.0 / (2.0 * float(gate_sigma) ** 2)
        # Arc length of each gate = arc of the nearest dense sample to the gate centre.
        gates = np.asarray(gate_pos, dtype=np.float64).reshape(-1, 3)
        self._gate_arcs = np.array(
            [self._s[int(np.argmin(np.linalg.norm(self._pts - g, axis=1)))] for g in gates],
            dtype=np.float64,
        )

    def w_contour(self, arc: NDArray[np.float64]) -> NDArray[np.float64]:
        """Per-arc contouring weight: a baseline plus a Gaussian bump at each gate's arc length."""
        a = np.asarray(arc, dtype=np.float64)
        w = np.full(a.shape, self._w_base, dtype=np.float64)
        for s_g in self._gate_arcs:
            w += self._w_gate * np.exp(-((a - s_g) ** 2) * self._inv2s2)
        return w
