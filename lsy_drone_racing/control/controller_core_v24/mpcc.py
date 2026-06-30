"""Obstacle-constrained MPCC core for controller_v24.

The v10.x MPCC has NO obstacle awareness: it tracks the reference curve, and all avoidance lives
in the reference planner, which inflates clearance to r_obs=0.20 m -- ~2x the real drone-to-pole
collision distance (drone box half-extent 0.07 + pole capsule radius 0.015 ~ 0.085 m). That
conservatism CLOSES doorways (a pole near a gate opening) the drone could physically thread, which
is the dominant level-3 crash class.

v24 adds the avoidance INSIDE the OCP: the K obstacle XY positions become solver parameters and
con_h gains one soft constraint per obstacle, ||drone_xy - obs_k|| >= _R_AVOID. _R_AVOID sits
between the real collision radius and the planner's 0.20, so the optimiser threads off-centre
doorways the planner declared blocked while keeping real margin. The constraints are SOFT (slacked)
so a genuinely-blocked (centred-pole) doorway degrades to minimum penetration instead of failing
the QP. Everything else -- horizon, cost, warm/cold start, rebase, progress anchor -- is v10.5's,
replicated verbatim with the parameter vector extended by the obstacle block.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from lsy_drone_racing.control.controller_core_v10_1.mpcc import _NX
from lsy_drone_racing.control.controller_core_v10_5.mpcc import MPCC as _MPCC

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.controller_core_v10_1.settings import MPCCSettings

__all__ = ["MPCC"]

_K = 4                # max obstacles fed to the OCP (level-3 has 4; pass them all)
_NP24 = 9 + 3 * _K    # path params p[0:9] + K obstacle triples (x, y, r) p[9:21]
_FAR = 50.0           # m, dummy position for unused/unrevealed obstacle slots (constraint inert)
_Z_OBS_L = 1.0e2      # linear slack penalty on each obstacle constraint (= speed-cap; higher -> NaN)
_Z_OBS_Q = 1.0e1      # quadratic slack penalty on each obstacle constraint (= speed-cap)
_SOLVER_CACHE: dict[tuple, AcadosOcpSolver] = {}  # v24-owned (own codegen namespace)


def _build_v24_solver(s: MPCCSettings, a_max: float) -> AcadosOcpSolver:
    """v10.5's OCP byte-for-byte, with K soft obstacle keep-out constraints added to con_h."""
    n, dt, g = int(s.horizon), float(s.step_dt), float(s.gravity)
    x, u, p = ca.SX.sym("x", _NX), ca.SX.sym("u", 4), ca.SX.sym("p", _NP24)
    pos, vel, th, vth = x[0:3], x[3:6], x[6], x[7]
    acc, dvth = u[0:3], u[3]
    pref, tan, thbar, vcurv, w_contour = p[0:3], p[3:6], p[6], p[7], p[8]

    model = AcadosModel()
    model.name = "controller_v24_gtopt"
    model.x, model.u, model.p = x, u, p
    model.f_expl_expr = ca.vertcat(vel, acc, vth, dvth)
    model.xdot = ca.SX.sym("xdot", _NX)
    model.f_impl_expr = model.xdot - model.f_expl_expr

    err = pos - (pref + tan * (th - thbar))
    lag = ca.dot(err, tan)
    perp = err - lag * tan
    contour = w_contour * ca.sumsqr(perp) + s.w_lag * lag**2
    model.cost_expr_ext_cost = contour - s.mu * vth + s.w_accel * ca.sumsqr(acc) + s.r_dv * dvth**2
    model.cost_expr_ext_cost_e = contour - s.mu * vth

    thrust = acc + ca.vertcat(0.0, 0.0, g)
    cons = [
        a_max**2 - ca.sumsqr(thrust),
        thrust[2] - s.a_z_min,
        (s.tilt_ratio * thrust[2]) ** 2 - ca.sumsqr(thrust[0:2]),
        vcurv**2 - ca.sumsqr(vel),
        vcurv - vth,
    ]
    for k in range(_K):  # soft keep-out per obstacle: ||xy - obs_k||^2 - r_k^2 >= 0 (r_k per-stage)
        ok = p[9 + 3 * k : 9 + 3 * k + 2]
        rk = p[9 + 3 * k + 2]
        cons.append(ca.sumsqr(pos[0:2] - ok) - rk**2)
    con_h = ca.vertcat(*cons)
    model.con_h_expr = model.con_h_expr_0 = con_h
    nh = 5 + _K

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = n
    ocp.solver_options.tf = n * dt
    ocp.cost.cost_type = ocp.cost.cost_type_e = "EXTERNAL"
    ocp.parameter_values = np.zeros(_NP24)

    inf = 1e9
    ocp.constraints.lh = ocp.constraints.lh_0 = np.zeros(nh)
    ocp.constraints.uh = ocp.constraints.uh_0 = np.full(nh, inf)
    # soften the two speed caps (3,4) AND every obstacle constraint (5..5+K-1)
    soft = np.array([3, 4, *range(5, 5 + _K)])
    ocp.constraints.idxsh = ocp.constraints.idxsh_0 = soft
    zl = np.array([1e2, 1e2, *([_Z_OBS_L] * _K)])
    Zl = np.array([1e1, 1e1, *([_Z_OBS_Q] * _K)])
    ocp.cost.zl = ocp.cost.zu = zl
    ocp.cost.Zl = ocp.cost.Zu = Zl
    ocp.cost.zl_0, ocp.cost.zu_0 = zl, zl
    ocp.cost.Zl_0, ocp.cost.Zu_0 = Zl, Zl

    ocp.constraints.idxbu = np.array([3])
    ocp.constraints.lbu = np.array([-s.a_theta_max])
    ocp.constraints.ubu = np.array([s.a_theta_max])
    ocp.constraints.x0 = np.zeros(_NX)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    # The keep-out constraint ||xy-obs||^2 >= r^2 is non-convex; near penetration its exact Hessian
    # makes the Lagrangian indefinite -> CONVEXIFY can emit NaN (HPIPM status 3). LM stabilises it.
    ocp.solver_options.levenberg_marquardt = 1.0e-2
    ocp.solver_options.qp_solver_iter_max = int(s.max_iter)
    return AcadosOcpSolver(ocp, json_file="c_generated_code/controller_v24.json", verbose=False)


class MPCC(_MPCC):
    """v10.5's MPCC with K soft obstacle keep-out constraints inside the OCP."""

    def __init__(self, settings: MPCCSettings, a_max: float):
        """Build (or reuse) the v24 obstacle-constrained solver; else identical to v10.4 setup."""
        self.n = int(settings.horizon)
        self._dt = float(settings.step_dt)
        self._a_theta = float(settings.a_theta_max)
        key = (round(a_max, 4), *(round(float(getattr(settings, f)), 4) for f in (
            "horizon", "step_dt", "v_theta_max", "a_theta_max", "tilt_ratio", "a_z_min",
            "mu", "w_lag", "w_accel", "r_dv", "gravity", "max_iter")))
        if key not in _SOLVER_CACHE:
            _SOLVER_CACHE[key] = _build_v24_solver(settings, a_max)
        self._solver = _SOLVER_CACHE[key]
        self._path = None
        self._obs_flat = np.tile([_FAR, _FAR, 0.0], _K)  # all-far/inert until set_obstacles
        self.reset()

    def set_obstacles(self, obs_xyr: NDArray[np.float64]) -> None:
        """Load up to _K obstacle triples (x, y, r); unused slots stay far with r=0 (inert)."""
        flat = np.tile([_FAR, _FAR, 0.0], _K)
        pts = np.asarray(obs_xyr, dtype=np.float64).reshape(-1, 3)[:_K]  # ponytail: first K; ok (4)
        flat[: 3 * len(pts)] = pts.reshape(-1)
        self._obs_flat = flat

    def solve(
        self, pos: NDArray[np.float64], vel: NDArray[np.float64], th0: float, vth_cap: float
    ) -> NDArray[np.float64]:
        """v10.5 solve (cold = honest measured-state seed; warm = shifted), p extended by obstacles."""
        n, dt, sv, ob = self.n, self._dt, self._solver, self._obs_flat
        vth_cap = max(float(vth_cap), 1e-3)
        cold = self._x_sol is None
        if cold:
            vth0 = min(vth_cap, float(np.linalg.norm(vel)) + self._a_theta * dt)
            thbar = th0 + np.arange(n + 1) * vth0 * dt
        else:
            thbar = np.concatenate([self._x_sol[6, 1:], self._x_sol[6, -1:]])
            thbar = np.maximum.accumulate(np.maximum(thbar + (th0 - thbar[0]), th0))
        pref, tan, vcurv = self._path.eval(thbar)
        vcurv = np.minimum(vcurv, vth_cap)
        wc = self._path.w_contour(thbar)
        for k in range(n + 1):
            sv.set(k, "p", np.concatenate([pref[k], tan[k], [thbar[k], vcurv[k], wc[k]], ob]))
        vth0 = min(vth0, float(vcurv[0])) if cold else float(self._x_sol[7, 1])
        x0 = np.concatenate([pos, vel, [th0, vth0]])
        sv.set(0, "lbx", x0)
        sv.set(0, "ubx", x0)
        if cold:  # hover-stationary seed at the measured pose (v10.4 honest cold start)
            for k in range(n + 1):
                sv.set(k, "x", np.concatenate([pos, vel, [thbar[k], vth0]]))
            for k in range(n):
                sv.set(k, "u", np.zeros(4))
        else:  # warm start from the shifted previous solution (v10.1)
            x_warm = np.column_stack([self._x_sol[:, 1:], self._x_sol[:, -1:]])
            u_warm = np.column_stack([self._u_sol[:, 1:], self._u_sol[:, -1:]])
            x_warm[6] += th0 - x_warm[6, 0]
            for k in range(n + 1):
                sv.set(k, "x", x_warm[:, k])
            for k in range(n):
                sv.set(k, "u", u_warm[:, k])
        if sv.solve() == 0:
            self._x_sol = np.column_stack([sv.get(k, "x") for k in range(n + 1)])
            self._u_sol = np.column_stack([sv.get(k, "u") for k in range(n)])
            self._last_acc = self._u_sol[0:3, 0].copy()
        return self._last_acc.copy()
