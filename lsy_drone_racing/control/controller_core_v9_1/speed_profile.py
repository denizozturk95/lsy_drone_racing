"""Curvature-aware speed profile for the controller v9-family reference (wired into v9.2).

v9/v9.1 recede the MPCC reference at a CONSTANT rate (V_REF), which is why the cockpit caps
V_REF low: a constant rate that is safe through the tight gate turns is slow on the straights.
This module shapes the recede rate to the path instead -- slow into corners, fast on
straights -- so the straights can run near V_MAX while the turns self-brake, without changing
the MPCC itself (which stays the safety net). It lives here but is consumed by controller_v9.2.

It is a textbook time-optimal path parameterisation under a friction-circle limit: a lateral
cap from the path curvature, v_curv = sqrt(a_lat_max / kappa), then a forward+backward pass
bounding longitudinal acceleration. Curvature uses scipy's analytic spline derivatives, so
nothing numerical is hand-rolled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline

_EPS = 1e-9


class SpeedProfile:
    """A curvature-limited speed v(s) along one plan spline, sampled on an arc-length grid."""

    def __init__(
        self,
        curve: CubicSpline,
        t_total: float,
        v_max: float,
        a_lat_max: float,
        v_min: float,
        n: int = 300,
    ):
        """Precompute v(s) from the path curvature plus a forward/backward accel-limit pass."""
        taus = np.linspace(0.0, float(t_total), n)
        d1 = np.asarray(curve.derivative(1)(taus), dtype=np.float64)
        d2 = np.asarray(curve.derivative(2)(taus), dtype=np.float64)
        speed = np.linalg.norm(d1, axis=1)
        s = np.concatenate([[0.0], np.cumsum((speed[:-1] + speed[1:]) * 0.5 * np.diff(taus))])
        kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / (speed**3 + _EPS)
        v = np.clip(np.sqrt(a_lat_max / (kappa + _EPS)), v_min, v_max)  # friction-circle cap
        ds = np.diff(s)
        for i in range(n - 2, -1, -1):  # backward: brake in time for the upcoming corner
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_lat_max * ds[i]))
        for i in range(1, n):  # forward: accelerate no faster than the longitudinal limit
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_lat_max * ds[i - 1]))
        self._taus, self._s, self._v = taus, s, v

    def arc_at_time(self, t: float) -> float:
        """Arc length at spline time t (maps the controller's progress onto the profile)."""
        return float(np.interp(t, self._taus, self._s))

    def at_arc(self, s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Curvature-limited speed at the given arc lengths (clamped past the path end).

        Used by v10 to cap its progress rate by curvature; np.interp flat-extends, so arc
        values past the path end take the final speed.
        """
        return np.interp(np.asarray(s, dtype=np.float64), self._s, self._v)

    def arc_offsets(self, s0: float, n: int, dt: float) -> NDArray[np.float64]:
        """Cumulative arc distances ahead of s0 after n steps of dt at the profiled speed."""
        offsets = np.zeros(n + 1, dtype=np.float64)
        
        arc = float(s0)
        for k in range(1, n + 1):
            arc += float(np.interp(arc, self._s, self._v)) * dt
            offsets[k] = arc - s0
        return offsets
