"""Offline minimum-snap polynomial trajectory through a known set of gates.

The whole track is known at ``t = 0`` on a known/measured track, so instead of the reactive
clamped-cubic planner we solve ONE globally-smooth trajectory up front. Per axis we fit
piecewise 7th-order polynomials (8 coefficients) that minimise the integral of snap (the 4th
derivative) subject to:

* fixed position at every waypoint (start, each gate centre, a stop point past the last gate),
* the drone starts and ends at rest (velocity = accel = jerk = 0),
* **velocity at each gate is pinned along the gate normal** (``v_gate * gate_x_axis``) so the
  drone crosses straight through the opening — this is what removes the reactive planner's
  post-gate reversal loops and lateral frame clips,
* velocity/accel/jerk continuity at every interior knot.

Each segment is solved in normalised time ``s = (t - t_i) / T_i ∈ [0, 1]`` (physical derivative
``p^(d)_t = T_i^{-d} p^(d)_s``) for numerical conditioning, then assembled into a curve object
that mimics the ``scipy`` ``CubicSpline`` interface (``curve(t)`` and ``curve.derivative(n)``)
expected by ``online_planner.attitude.attitude_action``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray

_ORDER = 7  # 7th-order polynomials (8 coefficients); snap is the 4th derivative
_N = _ORDER + 1


def _basis_row(d: int, s: float) -> NDArray[np.float64]:
    """Row r such that ``r @ a`` is the d-th derivative of ``sum_k a_k s^k`` at ``s``."""
    row = np.zeros(_N)
    for k in range(d, _N):
        coef = 1.0
        for m in range(d):
            coef *= k - m
        row[k] = coef * (s ** (k - d))
    return row


def _snap_cost_unit() -> NDArray[np.float64]:
    """8x8 snap cost ``∫_0^1 (p'''' )^2 ds`` for a unit-time segment in normalised s."""
    q = np.zeros((_N, _N))
    d = 4
    for i in range(d, _N):
        for j in range(d, _N):
            ci = cj = 1.0
            for m in range(d):
                ci *= i - m
                cj *= j - m
            power = i + j - 2 * d + 1
            q[i, j] = ci * cj / power
    return q


def _solve_axis(
    positions: NDArray[np.float64],
    times: NDArray[np.float64],
    vel: list[float | None],
) -> NDArray[np.float64]:
    """Solve the per-axis min-snap QP; return ``(n_seg, 8)`` normalised-s coefficients.

    Args:
        positions: ``(n_seg + 1,)`` waypoint positions for this axis.
        times: ``(n_seg,)`` segment durations.
        vel: length ``n_seg + 1`` physical velocity at each knot (``None`` = free/continuous;
            endpoints are always pinned to rest here).
    """
    n = len(times)
    q_unit = _snap_cost_unit()
    dim = _N * n
    q_full = np.zeros((dim, dim))
    for i in range(n):
        q_full[_N * i : _N * i + _N, _N * i : _N * i + _N] = q_unit * times[i] ** -7

    rows: list[NDArray[np.float64]] = []
    rhs: list[float] = []

    def seg_row(i: int, d: int, s: float) -> NDArray[np.float64]:
        row = np.zeros(dim)
        row[_N * i : _N * i + _N] = _basis_row(d, s)
        return row

    def add(row: NDArray[np.float64], value: float) -> None:
        rows.append(row)
        rhs.append(value)

    # Position at both ends of every segment.
    for i in range(n):
        add(seg_row(i, 0, 0.0), positions[i])
        add(seg_row(i, 0, 1.0), positions[i + 1])
    # Endpoint velocity (vel[0]/vel[n], 0 = rest); accel and jerk = 0 at both ends.
    # Physical v maps to normalised p'_s via p'_s = T * v.
    v_start = 0.0 if vel[0] is None else vel[0]
    v_end = 0.0 if vel[n] is None else vel[n]
    add(seg_row(0, 1, 0.0), times[0] * v_start)
    add(seg_row(0, 2, 0.0), 0.0)
    add(seg_row(0, 3, 0.0), 0.0)
    add(seg_row(n - 1, 1, 1.0), times[n - 1] * v_end)
    add(seg_row(n - 1, 2, 1.0), 0.0)
    add(seg_row(n - 1, 3, 1.0), 0.0)
    # Interior knots: knot k = waypoint i+1 between segments i and i+1.
    for i in range(n - 1):
        k = i + 1
        t_i, t_j = times[i], times[i + 1]
        if vel[k] is not None:
            # Pin physical velocity on both sides: p'_s(1) = T_i * v, p'_s(0) = T_j * v.
            add(seg_row(i, 1, 1.0), t_i * vel[k])
            add(seg_row(i + 1, 1, 0.0), t_j * vel[k])
        else:
            # Velocity continuity: T_i^{-1} p'_s,i(1) = T_j^{-1} p'_s,j(0).
            add(seg_row(i, 1, 1.0) / t_i - seg_row(i + 1, 1, 0.0) / t_j, 0.0)
        # Accel and jerk continuity (physical).
        add(seg_row(i, 2, 1.0) / t_i**2 - seg_row(i + 1, 2, 0.0) / t_j**2, 0.0)
        add(seg_row(i, 3, 1.0) / t_i**3 - seg_row(i + 1, 3, 0.0) / t_j**3, 0.0)

    a_mat = np.asarray(rows)
    b_vec = np.asarray(rhs)
    n_con = len(b_vec)
    kkt = np.zeros((dim + n_con, dim + n_con))
    kkt[:dim, :dim] = 2.0 * q_full
    kkt[:dim, dim:] = a_mat.T
    kkt[dim:, :dim] = a_mat
    sol = np.linalg.solve(kkt, np.concatenate([np.zeros(dim), b_vec]))
    return sol[:dim].reshape(n, _N)


class MinSnapTrajectory:
    """Piecewise-polynomial trajectory with a CubicSpline-like interface."""

    def __init__(self, coeffs: NDArray[np.float64], knots: NDArray[np.float64], deriv: int = 0):
        """Store ``(n_seg, 8, 3)`` normalised-s coefficients, knot times, derivative order."""
        self.coeffs = coeffs
        self.knots = np.asarray(knots, dtype=np.float64)
        self.deriv = deriv
        self.seg_times = np.diff(self.knots)

    @property
    def t_total(self) -> float:
        """Total trajectory duration."""
        return float(self.knots[-1])

    def __call__(self, t: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the (deriv-th derivative of the) trajectory at scalar or array ``t``."""
        scalar = np.ndim(t) == 0
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
        n_seg = len(self.seg_times)
        seg = np.clip(np.searchsorted(self.knots, t_arr, side="right") - 1, 0, n_seg - 1)
        out = np.zeros((len(t_arr), 3))
        for idx in range(len(t_arr)):
            i = int(seg[idx])
            t_i = self.seg_times[i]
            s = float(np.clip((t_arr[idx] - self.knots[i]) / t_i, 0.0, 1.0))
            row = _basis_row(self.deriv, s) * (t_i ** -self.deriv)
            out[idx] = row @ self.coeffs[i]
        return out[0] if scalar else out

    def derivative(self, n: int) -> MinSnapTrajectory:
        """Return the n-th derivative as another MinSnapTrajectory."""
        return MinSnapTrajectory(self.coeffs, self.knots, self.deriv + n)


def build_minsnap(
    waypoints: NDArray[np.float64],
    times: NDArray[np.float64],
    vel: list[NDArray[np.float64] | None],
) -> MinSnapTrajectory:
    """Build a 3-axis min-snap trajectory.

    Args:
        waypoints: ``(n_seg + 1, 3)`` positions (start, gates..., stop).
        times: ``(n_seg,)`` segment durations.
        vel: length ``n_seg + 1`` of per-knot velocity vectors (``None`` = free); endpoints
            are pinned to rest internally regardless.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    n_seg = len(waypoints) - 1
    coeffs = np.zeros((n_seg, _N, 3))
    for axis in range(3):
        vel_axis = [None if v is None else float(v[axis]) for v in vel]
        coeffs[:, :, axis] = _solve_axis(waypoints[:, axis], times, vel_axis)
    knots = np.concatenate([[0.0], np.cumsum(times)])
    return MinSnapTrajectory(coeffs, knots)


def gate_normal(gate_quat: NDArray[np.float64]) -> NDArray[np.float64]:
    """Gate crossing direction = gate-local +x axis (env counts a pass from -x to +x)."""
    return Rotation.from_quat(np.asarray(gate_quat, dtype=np.float64)).as_matrix()[:, 0]
