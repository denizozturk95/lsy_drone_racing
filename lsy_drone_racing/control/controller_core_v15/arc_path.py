"""Arena-aware path view for v11.2: v11's tunnel/reveal tables over a keep-in-clipped SPLINE.

The measured root cause of the out-of-bounds risk is the reference SPLINE itself: it routes to
y ~ 1.4-1.6 / x ~ 1.55 on the gate-1 exit arc (probe: scripts/_arena_probe.py), at and past the
smaller real-arena edge, and the drone faithfully tracks it. So v11.2 keeps the reference inside
the real arena AT THE SOURCE: the plan spline is resampled, each sample softly saturated into the
keep-in box (sim safety box shrunk by ``arena_inset``), and a fresh CubicSpline refit through the
clipped samples. The whole v11 path view (arc length, tangent, curvature speed, gate arcs, tunnel
tables, projection anchor) is then built from that clipped curve by the parent -- so the contour
reference, the progress anchor, and the tunnel centre all agree (clipping only the OCP's
reference, as an earlier version did, made the lag term and the geometric anchor fight at the
gate-1 exit and cost reliability).

The clip is SMOOTH (a soft, two-sided saturation, not a hard ``np.clip``): it leaves the interior
untouched, asymptotically approaches the wall from inside, and introduces no sharp corner that
would spike the curvature speed cap or pinch the tunnel. Gates sit well inside the box, so the
gate-passing geometry is preserved; only the between-gate wall-ward bulge is flattened.

An optional border SPEED CAP (``v_arena``, default disabled) additionally paces the profile where
the reference enters the ramp band -- the worst residual excursions are inertial, so slowing the
border zone tightens containment, at a measured reliability cost (see cockpit). The MPC position
barrier (controller_core_v15.mpcc) supplies the soft lateral margin on top of the clipped spline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.controller_core_v11.arc_path import TunnelArcPath, _limit_longitudinal

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.controller_core_v15.settings import MPCCSettings


def _soft_saturate(
    v: NDArray[np.float64], lo: float, hi: float, beta: float
) -> NDArray[np.float64]:
    """Smoothly saturate ``v`` into ``[lo, hi]``: ~identity inside, asymptotic at the walls.

    ``hi - softplus(hi - v)`` caps from above (-> hi as v -> inf, -> v for v << hi); composing the
    mirror image caps from below. ``beta`` (m) sets the transition width -- small beta -> closer to
    a hard clip but still C-infinity, so the refit spline has no curvature spike at the wall.
    """
    v = hi - beta * np.logaddexp(0.0, (hi - v) / beta)
    return lo + beta * np.logaddexp(0.0, (v - lo) / beta)


class ArenaTunnelArcPath(TunnelArcPath):
    """v11's tunnel path view built over a spline soft-clipped into the real-arena keep-in box."""

    def __init__(
        self,
        curve: CubicSpline,
        t_total: float,
        settings: MPCCSettings,
        gate_pos: NDArray[np.float64],
        obstacles_pos: NDArray[np.float64],
        posts_pos: NDArray[np.float64],
        gate_is_target_zero: bool,
        n: int = 600,
    ):
        """Soft-clip the plan spline into the keep-in box, then build v11's tables over it."""
        s = settings
        lo = (s.arena_x_min + s.arena_inset, s.arena_y_min + s.arena_inset)
        hi = (s.arena_x_max - s.arena_inset, s.arena_y_max - s.arena_inset)
        beta = float(s.arena_clip_beta)
        taus = np.linspace(0.0, float(t_total), max(n, 2))
        pts = np.asarray(curve(taus), dtype=np.float64).copy()
        pts[:, 0] = _soft_saturate(pts[:, 0], lo[0], hi[0], beta)
        pts[:, 1] = _soft_saturate(pts[:, 1], lo[1], hi[1], beta)
        clipped = CubicSpline(taus, pts)
        super().__init__(
            clipped, t_total, settings, gate_pos, obstacles_pos, posts_pos, gate_is_target_zero, n
        )

        # Optional border speed cap (default off via a large v_arena): pace the profile down where
        # the clipped reference still enters the ramp band, so the drone can turn inside the wall.
        arena_lo = np.array([lo[0], lo[1]], dtype=np.float64)
        arena_hi = np.array([hi[0], hi[1]], dtype=np.float64)
        ramp = float(s.arena_ramp_m)
        pxy = self._pts[:, :2]
        in_band = np.any((pxy > arena_hi - ramp) | (pxy < arena_lo + ramp), axis=1)
        if np.isfinite(s.v_arena) and np.any(in_band):
            v = self._vcurv.copy()
            v[in_band] = np.minimum(v[in_band], float(s.v_arena))
            self._vcurv = _limit_longitudinal(v, self._s, float(self._a_lat_max_v11))
