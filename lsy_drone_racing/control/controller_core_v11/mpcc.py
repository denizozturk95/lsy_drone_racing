"""Tunnel-constrained time-optimal MPCC core for controller_v11 (acados SQP-RTI).

The OCP is v10.1's time-optimal contouring formulation with TWO additions (the MPCC++ move,
Krinner et al. RSS 2024 -- see the cockpit for why):

- five extra per-stage parameters: the tunnel's lateral unit basis n (3) and half-extents
  W, H -- supplied by controller_core_v11.arc_path.TunnelArcPath at the predicted progress;
- two extra nonlinear constraints: W^2 - (e.n)^2 >= 0 and H^2 - (e.b)^2 >= 0, where
  e = pos - (pref + tan*(th - thbar)) is the linearised path error and b = tan x n.
  SOFT (slacked) like the speed caps, so a reveal-jump of the tunnel degrades gracefully
  instead of making the QP infeasible.

The contouring weight stays a per-stage parameter but arrives as a low constant (de-spiked);
-MU*vth then sets the pace against the tunnel walls. The solve/rebase/cold-start flow is the
v10.1 + v10.3 + v10.4 + v10.5 machinery copied with the parameter packing extended (the house
pattern; the parameter count changes, so every sv.set(k, "p", ...) site must change with it).
Own codegen namespace ``controller_v11`` (different dimensions than v10.x -- never share; see the
v10.4 dlopen segfault note). REQUIRES the acados environment -- run under ``pixi run``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.controller_core_v11.arc_path import TunnelArcPath
    from lsy_drone_racing.control.controller_core_v11.settings import MPCCSettings

__all__ = ["MPCC"]

# states [pos3, vel3, th, vth]; controls [acc3, dvth];
# params [pref3, tan3, thbar, vcurv, w_contour, n_lat3, w_half, h_half]
_NX, _NU, _NP = 8, 4, 14
_SOLVER_CACHE: dict[tuple, AcadosOcpSolver] = {}


def _build_v11_solver(s: MPCCSettings, a_max: float) -> AcadosOcpSolver:
    """Code-generate the v10.1 OCP plus the soft tunnel constraints (controller_v11 namespace)."""
    n, dt, g = int(s.horizon), float(s.step_dt), float(s.gravity)
    x, u, p = ca.SX.sym("x", _NX), ca.SX.sym("u", _NU), ca.SX.sym("p", _NP)
    pos, vel, th, vth = x[0:3], x[3:6], x[6], x[7]
    acc, dvth = u[0:3], u[3]
    pref, tan, thbar, vcurv, w_contour = p[0:3], p[3:6], p[6], p[7], p[8]
    n_lat, w_half, h_half = p[9:12], p[12], p[13]

    model = AcadosModel()
    model.name = "controller_v11_tunnel"
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
    b_vert = ca.cross(tan, n_lat)  # vertical-ish tunnel basis (unit: tan and n_lat are unit)
    con_h = ca.vertcat(
        a_max**2 - ca.sumsqr(thrust),
        thrust[2] - s.a_z_min,
        (s.tilt_ratio * thrust[2]) ** 2 - ca.sumsqr(thrust[0:2]),
        vcurv**2 - ca.sumsqr(vel),  # speed <= curvature/reveal cap (soft)
        vcurv - vth,  # progress rate <= cap (soft)
        w_half**2 - ca.dot(err, n_lat) ** 2,  # lateral tunnel wall (soft)
        h_half**2 - ca.dot(err, b_vert) ** 2,  # vertical tunnel wall (soft)
    )
    model.con_h_expr = model.con_h_expr_0 = con_h

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = n
    ocp.solver_options.tf = n * dt
    ocp.cost.cost_type = ocp.cost.cost_type_e = "EXTERNAL"
    ocp.parameter_values = np.zeros(_NP)

    inf = 1e9
    ocp.constraints.lh = ocp.constraints.lh_0 = np.zeros(7)
    ocp.constraints.uh = ocp.constraints.uh_0 = np.full(7, inf)
    # Soften the speed caps (3, 4) and the tunnel walls (5, 6); tunnel violations are priced
    # higher -- they are frame/obstacle proximity, not pacing.
    ocp.constraints.idxsh = ocp.constraints.idxsh_0 = np.array([3, 4, 5, 6])
    z_lin = np.array([1e2, 1e2, s.tunnel_slack_l1, s.tunnel_slack_l1])
    z_quad = np.array([1e1, 1e1, s.tunnel_slack_l2, s.tunnel_slack_l2])
    ocp.cost.zl = ocp.cost.zu = z_lin
    ocp.cost.Zl = ocp.cost.Zu = z_quad
    ocp.cost.zl_0, ocp.cost.zu_0 = z_lin, z_lin
    ocp.cost.Zl_0, ocp.cost.Zu_0 = z_quad, z_quad
    ocp.constraints.idxbu = np.array([3])
    ocp.constraints.lbu = np.array([-s.a_theta_max])
    ocp.constraints.ubu = np.array([s.a_theta_max])
    ocp.constraints.x0 = np.zeros(_NX)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.qp_solver_iter_max = int(s.max_iter)
    return AcadosOcpSolver(ocp, json_file="c_generated_code/controller_v11.json", verbose=False)


class MPCC:
    """Tunnel-constrained time-optimal contouring controller (world-frame acceleration out)."""

    def __init__(self, settings: MPCCSettings, a_max: float):
        """Build (or reuse) the v11 solver; the cache key covers every OCP-shaping knob."""
        self.n = int(settings.horizon)
        self._dt = float(settings.step_dt)
        key = (
            round(a_max, 4),
            *(
                round(float(getattr(settings, f)), 4)
                for f in (
                    "horizon",
                    "step_dt",
                    "v_theta_max",
                    "a_theta_max",
                    "tilt_ratio",
                    "a_z_min",
                    "mu",
                    "w_lag",
                    "w_accel",
                    "r_dv",
                    "gravity",
                    "max_iter",
                    "tunnel_slack_l1",
                    "tunnel_slack_l2",
                )
            ),
        )
        if key not in _SOLVER_CACHE:
            _SOLVER_CACHE[key] = _build_v11_solver(settings, a_max)
        self._solver = _SOLVER_CACHE[key]
        self._path: TunnelArcPath | None = None
        self._a_theta = float(settings.a_theta_max)
        self.reset()

    def set_path(self, path: TunnelArcPath) -> None:
        """Load the tunnel path view used to linearise the reference each step."""
        self._path = path
        self.reset()

    def reset(self) -> None:
        """Forget the warm start between episodes / on a new path."""
        self._x_sol: NDArray[np.float64] | None = None
        self._u_sol: NDArray[np.float64] | None = None
        self._last_acc = np.zeros(3)
        self._solver.reset()

    def rebase(self, path: TunnelArcPath, s0: float) -> None:
        """v10.3's warm path swap: keep states/controls, re-anchor progress on the new path."""
        if self._x_sol is None:
            self.set_path(path)
            return
        self._path = path
        th = np.empty(self._x_sol.shape[1])
        th[0] = float(s0)
        for k in range(1, th.shape[0]):
            th[k] = path.project(self._x_sol[0:3, k], th[k - 1])
        self._x_sol[6] = th

    def predicted_progress(self) -> float | None:
        """v10.5's dynamics-aware anchor source: progress one step ahead, or None."""
        if self._x_sol is None:
            return None
        return float(self._x_sol[6, 1])

    def predicted_positions(self) -> np.ndarray | None:
        """The solver's predicted positions over the horizon (for overlays), or None."""
        if self._x_sol is None:
            return None
        return self._x_sol[0:3].T.copy()

    def _set_stage_params(self, thbar: NDArray[np.float64], vth_cap: float) -> NDArray[np.float64]:
        """Pack and load the per-stage parameters; return the (capped) vcurv per stage."""
        pref, tan, vcurv = self._path.eval(thbar)
        vcurv = np.minimum(vcurv, vth_cap)
        wc = self._path.w_contour(thbar)
        n_lat, w_half, h_half = self._path.tunnel(thbar)
        for k in range(self.n + 1):
            self._solver.set(
                k,
                "p",
                np.concatenate(
                    [pref[k], tan[k], [thbar[k], vcurv[k], wc[k]], n_lat[k], [w_half[k], h_half[k]]]
                ),
            )
        return vcurv

    def solve(
        self, pos: NDArray[np.float64], vel: NDArray[np.float64], th0: float, vth_cap: float
    ) -> NDArray[np.float64]:
        """One RTI iteration: v10.1's warm flow / v10.4's honest cold start, tunnel params in."""
        n, dt, sv = self.n, self._dt, self._solver
        vth_cap = max(float(vth_cap), 1e-3)
        if self._x_sol is not None:  # warm: shifted previous solution (v10.1 flow)
            thbar = np.concatenate([self._x_sol[6, 1:], self._x_sol[6, -1:]])
            thbar += th0 - thbar[0]
            thbar = np.maximum.accumulate(np.maximum(thbar, th0))
            vcurv = self._set_stage_params(thbar, vth_cap)
            vth0 = float(self._x_sol[7, 1])
            x0 = np.concatenate([pos, vel, [th0, vth0]])
            sv.set(0, "lbx", x0)
            sv.set(0, "ubx", x0)
            x_warm = np.column_stack([self._x_sol[:, 1:], self._x_sol[:, -1:]])
            u_warm = np.column_stack([self._u_sol[:, 1:], self._u_sol[:, -1:]])
            x_warm[6] += th0 - x_warm[6, 0]
            for k in range(n + 1):
                sv.set(k, "x", x_warm[:, k])
            for k in range(n):
                sv.set(k, "u", u_warm[:, k])
        else:  # cold: honest start at the measured state (v10.4 flow)
            vth0 = min(vth_cap, float(np.linalg.norm(vel)) + self._a_theta * dt)
            thbar = th0 + np.arange(n + 1) * vth0 * dt
            vcurv = self._set_stage_params(thbar, vth_cap)
            vth0 = min(vth0, float(vcurv[0]))
            x0 = np.concatenate([pos, vel, [th0, vth0]])
            sv.set(0, "lbx", x0)
            sv.set(0, "ubx", x0)
            for k in range(n + 1):
                sv.set(k, "x", np.concatenate([pos, vel, [thbar[k], vth0]]))
            for k in range(n):
                sv.set(k, "u", np.zeros(4))
        if sv.solve() == 0:
            self._x_sol = np.column_stack([sv.get(k, "x") for k in range(n + 1)])
            self._u_sol = np.column_stack([sv.get(k, "u") for k in range(n)])
            self._last_acc = self._u_sol[0:3, 0].copy()
        return self._last_acc.copy()
