"""Gate-aware arc-length view for v10.4: a REACTIVE per-gate approach speed cap.

The binding reliability constraint of the whole v10.x line is the gate-reveal correction: the
true gate pose appears only at the 0.7 m sensor range and can be up to ~0.2 m off nominal, and
absorbing that lateral correction scales ~v^2 with the arrival speed. A PERMANENT gate speed
cap was condemned twice (it taxes every gate on every lap, costing more than it saves); but the
correction is only expensive when the gate actually MOVED -- and the controller knows each
gate's revealed delta the moment it replans.

So the cap is reactive: the controller passes a per-gate cap array, set to V_GATE_REACT for
exactly those gates whose revealed pose differs from nominal by more than REACT_DELTA_M (and
np.inf for the rest), and this path view caps the speed profile inside an approach-heavy window
([s_gate - pre, s_gate + post]) around the flagged gates only. Lucky draws fly at full pace;
unlucky ones brake for the one gate that needs it. The backward/forward passes keep the braking
into / acceleration out of each window dynamically feasible (braking forms upstream
automatically -- at reveal range the drone only needs ~2-3 m/s^2 to shed the difference).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.controller_core_v10_3.arc_path import GateArcPath as _GateArcPath

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline


class GateArcPath(_GateArcPath):
    """v10.3's GateArcPath with per-gate (reactive) approach speed caps."""

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
        gate_caps: NDArray[np.float64],
        react_v_pre: float,
        react_v_post: float,
        n: int = 600,
    ):
        """Build the v10.3 tables (global cap disabled), then cap the flagged gates' windows.

        ``gate_caps`` has one entry per row of ``gate_pos``: the approach-window speed cap for
        that gate, or np.inf to leave it at the curvature profile.
        """
        super().__init__(
            curve, t_total, v_cap, a_lat_max, v_min, gate_pos, w_base, w_gate, gate_sigma,
            v_gate=1e9, gate_v_pre=0.0, gate_v_post=0.0, n=n,  # permanent cap OFF
        )
        caps = np.asarray(gate_caps, dtype=np.float64).reshape(-1)
        active = [(s_g, c) for s_g, c in zip(self._gate_arcs, caps) if c < float(v_cap)]
        if not active:
            return
        v = self._vcurv.copy()
        for s_g, cap in active:
            window = (self._s >= s_g - float(react_v_pre)) & (self._s <= s_g + float(react_v_post))
            v[window] = np.minimum(v[window], cap)
        ds = np.diff(self._s)
        for i in range(len(v) - 2, -1, -1):  # backward: brake into each window in time
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_lat_max * ds[i]))
        for i in range(1, len(v)):  # forward: accelerate out within the longitudinal limit
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_lat_max * ds[i - 1]))
        self._vcurv = v
