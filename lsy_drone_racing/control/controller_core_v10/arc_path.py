"""Arc-length view of a plan spline for the v10 time-optimal MPCC (acados).

The acados MPCC linearises the reference once per control step (SQP-RTI), so it does not need
the spline inside the solver -- it only needs, at each predicted progress theta_k, the path
point, the unit tangent, and the curvature-limited speed. This helper precomputes dense
arc-length tables from the planner's spline and evaluates those three quantities at arbitrary
arc lengths (numpy), plus a forward projection to anchor progress to the drone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.controller_core_v9_1.speed_profile import SpeedProfile

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline

_EPS = 1e-9
_PROJ_WINDOW_M = 2.0  # forward window (m) for the projection search


class ArcPath:
    """Dense arc-length tables (point, unit tangent, curvature speed) for one plan spline."""

    def __init__(
        self,
        curve: CubicSpline,
        t_total: float,
        v_cap: float,
        a_lat_max: float,
        v_min: float,
        n: int = 600,
    ):
        """Sample the plan densely and precompute arc length, unit tangent, and v_curv(s)."""
        taus = np.linspace(0.0, float(t_total), n)
        pts = np.asarray(curve(taus), dtype=np.float64)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        tan = np.gradient(pts, s, axis=0)  # dp/ds ~ unit tangent (arc-length parameterised)
        tan /= np.linalg.norm(tan, axis=1, keepdims=True) + _EPS
        self._s, self._pts, self._tan = s, pts, tan
        self._vcurv = SpeedProfile(curve, t_total, v_cap, a_lat_max, v_min).at_arc(s)
        self.total = float(s[-1])

    def project(self, pos: NDArray[np.float64], s_min: float) -> float:
        """Arc length of the nearest path point within a forward window of s_min."""
        lo = int(np.searchsorted(self._s, s_min))
        hi = max(int(np.searchsorted(self._s, s_min + _PROJ_WINDOW_M)), lo + 1)
        nearest = int(np.argmin(np.linalg.norm(self._pts[lo:hi] - pos, axis=1)))
        return float(self._s[lo + nearest])

    def eval(
        self, arc: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return (point, unit tangent, curvature-speed) at each arc length (clamped to path)."""
        a = np.clip(np.asarray(arc, dtype=np.float64), 0.0, self.total)
        point = np.column_stack([np.interp(a, self._s, self._pts[:, i]) for i in range(3)])
        tan = np.column_stack([np.interp(a, self._s, self._tan[:, i]) for i in range(3)])
        tan /= np.linalg.norm(tan, axis=1, keepdims=True) + _EPS
        vcurv = np.interp(a, self._s, self._vcurv)
        return point, tan, vcurv
