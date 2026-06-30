"""Gate-aware time-optimal MPCC core for controller_v10_4: honest cold starts.

The OCP and the replan-continuity rebase are byte-for-byte v10.3's -- v10.4 changes only the
COLD start (the first solve of a plan, i.e. the takeoff hand-off and episode starts). v10.3
inherits two fabrications from v10.1 there:

1. vth0 is hard-set into x0 as min(ramp cap, v_curv[0]) -- the path's curvature profile does
   not encode "starts at rest" (its forward pass begins at the cap), so from a standstill the
   solver is TOLD it is already moving at the full ramped cap and its first command lunges.
2. No warm start is loaded at all: SQP-RTI linearises the whole horizon at the all-zeros
   iterate (position at the world origin, metres from the drone), and the tilt/thrust-circle
   constraints have zero lateral gradient at u = 0, so the first QP step is effectively
   unconstrained sideways. This transient is the v10.3 residual hand-off crash class (1-2/20).

The fix: clamp vth0 to the drone's measured speed plus one step of progress acceleration, march
the linearisation points at that honest rate, and seed a hover-stationary warm start at the
measured pose (x_k = [pos, vel, thbar_k, vth0], u_k = 0). Warm solves are untouched -- they
already shift the previous solution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from lsy_drone_racing.control.controller_core_v10_1.mpcc import _NP, _NX
from lsy_drone_racing.control.controller_core_v10_3.mpcc import MPCC as _MPCC

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.controller_core_v10_1.settings import MPCCSettings

__all__ = ["MPCC"]

_SOLVER_CACHE: dict[tuple, AcadosOcpSolver] = {}  # v10.4-owned (own codegen namespace)


def _build_v10_4_solver(s: MPCCSettings, a_max: float) -> AcadosOcpSolver:
    """Code-generate the v10.1 OCP, byte-for-byte, under a v10.4-owned artifact name.

    v10.4 changes the horizon, which changes the solver DIMENSIONS. Reusing v10.1's builder
    verbatim would regenerate into the same model name / json / dylib -- and dlopen returns the
    already-loaded other-dimension image for the new solver object, which segfaults when both
    live in one process (the v10.4 __init__ chain builds the v10.3 solver first). Same OCP,
    own namespace.
    """
    n, dt, g = int(s.horizon), float(s.step_dt), float(s.gravity)
    x, u, p = ca.SX.sym("x", _NX), ca.SX.sym("u", 4), ca.SX.sym("p", _NP)
    pos, vel, th, vth = x[0:3], x[3:6], x[6], x[7]
    acc, dvth = u[0:3], u[3]
    pref, tan, thbar, vcurv, w_contour = p[0:3], p[3:6], p[6], p[7], p[8]

    model = AcadosModel()
    model.name = "controller_v10_4_gtopt"
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
    con_h = ca.vertcat(
        a_max**2 - ca.sumsqr(thrust),
        thrust[2] - s.a_z_min,
        (s.tilt_ratio * thrust[2]) ** 2 - ca.sumsqr(thrust[0:2]),
        vcurv**2 - ca.sumsqr(vel),
        vcurv - vth,
    )
    model.con_h_expr = model.con_h_expr_0 = con_h

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = n
    ocp.solver_options.tf = n * dt
    ocp.cost.cost_type = ocp.cost.cost_type_e = "EXTERNAL"
    ocp.parameter_values = np.zeros(_NP)

    inf = 1e9
    ocp.constraints.lh = ocp.constraints.lh_0 = np.zeros(5)
    ocp.constraints.uh = ocp.constraints.uh_0 = np.full(5, inf)
    ocp.constraints.idxsh = ocp.constraints.idxsh_0 = np.array([3, 4])
    ns = 2
    ocp.cost.zl = ocp.cost.zu = 1e2 * np.ones(ns)
    ocp.cost.Zl = ocp.cost.Zu = 1e1 * np.ones(ns)
    ocp.cost.zl_0, ocp.cost.zu_0 = ocp.cost.zl, ocp.cost.zu
    ocp.cost.Zl_0, ocp.cost.Zu_0 = ocp.cost.Zl, ocp.cost.Zu
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
    return AcadosOcpSolver(ocp, json_file="c_generated_code/controller_v10_4.json", verbose=False)


class MPCC(_MPCC):
    """v10.3's MPCC with an honest (measured-speed, hover-seeded) cold start."""

    def __init__(self, settings: MPCCSettings, a_max: float):
        """Build (or reuse) the v10.4-named solver and stash the progress-acceleration bound."""
        self.n = int(settings.horizon)
        self._dt = float(settings.step_dt)
        key = (round(a_max, 4), *(round(float(getattr(settings, f)), 4) for f in (
            "horizon", "step_dt", "v_theta_max", "a_theta_max", "tilt_ratio", "a_z_min",
            "mu", "w_lag", "w_accel", "r_dv", "gravity", "max_iter")))
        if key not in _SOLVER_CACHE:
            _SOLVER_CACHE[key] = _build_v10_4_solver(settings, a_max)
        self._solver = _SOLVER_CACHE[key]
        self._path = None
        self._a_theta = float(settings.a_theta_max)
        self.reset()

    def solve(
        self, pos: NDArray[np.float64], vel: NDArray[np.float64], th0: float, vth_cap: float
    ) -> NDArray[np.float64]:
        """One RTI iteration; cold starts are linearised at the measured state, not a fiction."""
        if self._x_sol is not None:  # warm path: v10.3 behaviour, untouched
            return super().solve(pos, vel, th0, vth_cap)
        n, dt, sv = self.n, self._dt, self._solver
        vth_cap = max(float(vth_cap), 1e-3)
        # Honest progress rate: the drone's actual speed plus one step of progress accel.
        vth0 = min(vth_cap, float(np.linalg.norm(vel)) + self._a_theta * dt)
        thbar = th0 + np.arange(n + 1) * vth0 * dt
        pref, tan, vcurv = self._path.eval(thbar)
        vcurv = np.minimum(vcurv, vth_cap)
        wc = self._path.w_contour(thbar)
        for k in range(n + 1):
            sv.set(k, "p", np.concatenate([pref[k], tan[k], [thbar[k], vcurv[k], wc[k]]]))
        vth0 = min(vth0, float(vcurv[0]))
        x0 = np.concatenate([pos, vel, [th0, vth0]])
        sv.set(0, "lbx", x0)
        sv.set(0, "ubx", x0)
        # Hover-stationary warm seed at the measured pose: the RTI linearises where the drone
        # IS (zero-acceleration hover thrust), not at the all-zeros iterate at the origin.
        for k in range(n + 1):
            sv.set(k, "x", np.concatenate([pos, vel, [thbar[k], vth0]]))
        for k in range(n):
            sv.set(k, "u", np.zeros(4))
        if sv.solve() == 0:
            self._x_sol = np.column_stack([sv.get(k, "x") for k in range(n + 1)])
            self._u_sol = np.column_stack([sv.get(k, "u") for k in range(n)])
            self._last_acc = self._u_sol[0:3, 0].copy()
        return self._last_acc.copy()
