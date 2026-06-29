"""Gate-aware arc-length view with a GATE-WINDOW SPEED CAP for the v10.3 time-optimal MPCC.

Subclasses v10.1's GateArcPath (dense arc tables, gate arc-positions, gate-spiked contouring
weight -- all reused unchanged) and additionally caps the curvature-limited speed profile inside
a Gaussian window around each gate's arc-position. The curvature profile only brakes where the
PATH bends, so a straight-approach gate is otherwise crossed at full V_MAX -- but the true gate
pose is only revealed at the 0.7 m sensor range and can be ~0.2 m off nominal, and the lateral
acceleration needed to absorb that correction scales with v^2. Capping the crossing speed to
V_GATE makes the correction cheap and deterministic at every gate, which is what lets v10.3 carry
a higher straight budget than v10.1 without losing gates.

After the cap, a backward pass limits deceleration into each window and a forward pass limits
acceleration out of it (same longitudinal budget the SpeedProfile uses), so the profile the MPCC
sees stays dynamically feasible -- the braking starts upstream of the window automatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.controller_core_v10_1.arc_path import GateArcPath as _GateArcPath

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline


class GateArcPath(_GateArcPath):
    """v10.1's GateArcPath plus a feasibility-repaired speed cap at each gate window."""

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
        v_gate: float,
        gate_v_pre: float,
        gate_v_post: float,
        n: int = 600,
    ):
        """Build the v10.1 tables, then cap the speed profile inside each gate window.

        The window is asymmetric: ``[s_gate - gate_v_pre, s_gate + gate_v_post]``. The reveal
        correction must be absorbed on the APPROACH (between the 0.7 m sensor reveal and the
        gate plane), so that side carries the cap; the exit can take speed back immediately.
        The backward pass below shapes the deceleration into the window, so ``gate_v_pre``
        does not need to cover the braking distance.
        """
        super().__init__(
            curve, t_total, v_cap, a_lat_max, v_min, gate_pos, w_base, w_gate, gate_sigma, n
        )
        v_gate, v_cap = float(v_gate), float(v_cap)
        if v_gate >= v_cap:  # cap disabled -> exactly v10.1's profile
            return
        v = self._vcurv.copy()
        for s_g in self._gate_arcs:
            window = (self._s >= s_g - float(gate_v_pre)) & (self._s <= s_g + float(gate_v_post))
            v[window] = np.minimum(v[window], v_gate)
        ds = np.diff(self._s)
        for i in range(len(v) - 2, -1, -1):  # backward: brake in time for each gate window
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_lat_max * ds[i]))
        for i in range(1, len(v)):  # forward: accelerate out within the longitudinal limit
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_lat_max * ds[i - 1]))
        self._vcurv = v
