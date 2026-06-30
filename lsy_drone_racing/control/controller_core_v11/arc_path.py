"""Arc-length path view for v11: v10.5's anchor + TUNNEL TABLES + all-gate reveal caps.

Extends v10.5's GateArcPath with the two things the v11 OCP consumes per stage:

1. TUNNEL TABLES (``tunnel(arc)``): a lateral unit basis n(s) (horizontal, perpendicular to
   the path tangent) and half-extents W(s), H(s) of the prismatic tunnel -- W_MAX on the open
   path, clipped so the tunnel edge respects every obstacle's keep-out, tapered down to the
   gate opening within TUNNEL_TAPER_M of each gate's arc position, and H additionally bounded
   by the floor/ceiling margins. The OCP constrains the predicted positions to |e.n| <= W,
   |e.b| <= H (b = tangent x n), softly.

2. REVEAL CAPS ON EVERY GATE: the curvature profile is capped at v_reveal inside every gate's
   [s_g - pre, s_g + post] window (the information constraint the tunnel does not remove),
   plus v10.4's wide launch window on gate 0 while it is the target. Both reuse the v10.4 cap
   machinery semantics with the usual backward/forward feasibility passes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.controller_core_v10_5.arc_path import GateArcPath as _GateArcPath

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline

    from lsy_drone_racing.control.controller_core_v11.settings import MPCCSettings

_EPS = 1e-9


def _limit_longitudinal(
    v: NDArray[np.float64], s: NDArray[np.float64], a_max: float
) -> NDArray[np.float64]:
    """Backward (brake in time) and forward (accelerate out) passes over a speed profile."""
    ds = np.diff(s)
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * a_max * ds[i]))
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * a_max * ds[i - 1]))
    return v


class TunnelArcPath(_GateArcPath):
    """v10.5's GateArcPath plus tunnel half-extent tables and all-gate reveal caps."""

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
        """Build the v10.5 tables, cap every gate's reveal window, and lay the tunnel.

        ``posts_pos`` are the frame-post columns of ALL gates (passed ones included -- the
        frames are physical regardless of target); they clip the tunnel like obstacles but
        with the smaller post margin. ``gate_is_target_zero`` widens gate 0's cap window to
        the launch corridor (v10.4's launch protection) -- True only while gate 0 is the
        current target.
        """
        s = settings
        # Reveal caps via the inherited v10.4 cap machinery: every remaining gate when
        # reveal_cap_all is set, otherwise none here (gate 0's launch cap is applied below).
        caps = np.full(len(gate_pos), float(s.v_gate_reveal) if s.reveal_cap_all else np.inf)
        super().__init__(
            curve,
            t_total,
            s.v_theta_max,
            s.a_lat_max,
            s.v_min,
            gate_pos,
            s.w_contour_base,
            s.w_contour_gate,
            s.gate_sigma,
            caps,
            s.reveal_pre_m,
            s.reveal_post_m,
            n,
        )
        if gate_is_target_zero and len(self._gate_arcs):
            v = self._vcurv.copy()
            s_g0 = self._gate_arcs[0]
            window = (self._s >= s_g0 - s.launch_cap_pre_m) & (self._s <= s_g0 + s.reveal_post_m)
            v[window] = np.minimum(v[window], s.v_gate_reveal)
            self._vcurv = _limit_longitudinal(v, self._s, s.a_lat_max)

        # --- Tunnel tables over the dense samples ---
        tangent = self._tan
        lat = np.column_stack([tangent[:, 1], -tangent[:, 0], np.zeros(len(tangent))])
        norms = np.linalg.norm(lat, axis=1)
        # Near-vertical tangents (launch climb) have no horizontal normal: carry the last
        # valid one forward (world-x before any valid sample exists).
        lat[norms < 0.2] = np.nan
        lat[0] = lat[0] if norms[0] >= 0.2 else np.array([1.0, 0.0, 0.0])
        for i in range(1, len(lat)):
            if np.isnan(lat[i, 0]):
                lat[i] = lat[i - 1]
        lat /= np.linalg.norm(lat, axis=1, keepdims=True) + _EPS

        # Curvature clamp: the OCP's linearised reference is only trustworthy within
        # ~1/kappa of the path. kappa is recovered from the (uncapped-by-windows) curvature
        # profile: kappa = a_lat / v_curv^2.
        kappa = s.a_lat_max / np.maximum(self._vcurv, 1e-3) ** 2
        w = np.minimum(
            np.full(len(self._s), float(s.tunnel_w_max)),
            np.maximum(float(s.tunnel_curv_frac) / np.maximum(kappa, 1e-6), s.tunnel_w_min),
        )
        clip_sets = (
            (np.asarray(obstacles_pos, dtype=np.float64).reshape(-1, 3), s.tunnel_obs_margin),
            (np.asarray(posts_pos, dtype=np.float64).reshape(-1, 3), s.tunnel_post_margin),
        )
        for points, margin in clip_sets:
            for point in points:
                d = np.linalg.norm(self._pts[:, :2] - point[:2], axis=1)
                w = np.minimum(w, np.maximum(d - margin, s.tunnel_w_min))
        h = np.minimum(
            np.full(len(self._s), float(s.tunnel_h_max)),
            np.maximum(
                np.minimum(self._pts[:, 2] - s.tunnel_z_floor, s.tunnel_z_ceil - self._pts[:, 2]),
                s.tunnel_w_min,
            ),
        )
        for s_g in self._gate_arcs:  # narrow to the opening into each gate
            ramp = np.minimum(np.abs(self._s - s_g) / float(s.tunnel_taper_m), 1.0)
            w = np.minimum(w, s.tunnel_w_gate + (s.tunnel_w_max - s.tunnel_w_gate) * ramp)
            h = np.minimum(h, s.tunnel_h_gate + (s.tunnel_h_max - s.tunnel_h_gate) * ramp)
        self._lat, self._w, self._h = lat, w, h
        self._a_lat_max_v11 = float(s.a_lat_max)  # for subclasses re-running the passes

    def tunnel(
        self, arc: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return (lateral unit basis, half-width, half-height) at each arc length."""
        a = np.clip(np.asarray(arc, dtype=np.float64), 0.0, self.total)
        lat = np.column_stack([np.interp(a, self._s, self._lat[:, i]) for i in range(3)])
        lat /= np.linalg.norm(lat, axis=1, keepdims=True) + _EPS
        return lat, np.interp(a, self._s, self._w), np.interp(a, self._s, self._h)
