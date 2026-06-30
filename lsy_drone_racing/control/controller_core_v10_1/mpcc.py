"""Gate-aware time-optimal MPCC core for controller_v10_1 (acados SQP-RTI, real-time).

This is v10's time-optimal MPCC -- progress ``th`` and rate ``vth`` as states, cost ``-mu*vth``
so traversal speed is an output, a friction-circle curvature cap braking corners, solved in
real time by acados SQP-RTI with the reference linearised out of the solver -- with ONE change:
the contouring weight is no longer a single constant. It is a PER-STAGE PARAMETER that spikes
where the path crosses a gate and relaxes elsewhere.

Why: on this track the lap time is capped by gate-passing precision under +/-0.15 m gate
randomisation, not by actuator authority. With a uniform contouring weight, raising the speed
budget makes the drone trade contour error for progress everywhere -- including at the gate --
so it clips/overshoots and the finish rate collapses (that is the v9.x/v10 ceiling). Spiking the
contouring weight in a window around each gate forces the drone to hug the line exactly where a
gate must be threaded, while leaving it free on the straights, so the speed budget (V_MAX,
A_LAT_MAX, V_THETA_MAX) can be raised without losing gates. Acceleration stays the control (a
zero-order-hold command, matching the plant), so the proven v10 solver structure is unchanged
apart from the extra per-stage parameter.

The per-stage weight at each predicted progress is supplied by the path view (see
controller_core_v10_1.arc_path: a v10 ArcPath that also knows the gate arc-positions and returns the gate
weight w_contour(theta)). Requires the acados environment (run under ``pixi run``); the C solver
is code-generated and compiled once per process and cached.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.controller_core_v10_1.arc_path import GateArcPath
    from lsy_drone_racing.control.controller_core_v10_1.settings import MPCCSettings

__all__ = ["MPCC"]

# states [pos3, vel3, th, vth]; controls [acc3, dvth];
# params [pref3, tan3, thbar, vcurv, w_contour]  <- w_contour is now per-stage
_NX, _NU, _NP = 8, 4, 9
_SOLVER_CACHE: dict[tuple, AcadosOcpSolver] = {}  # build the C solver once per process


def _build_solver(s: MPCCSettings, a_max: float) -> AcadosOcpSolver:
    """Code-generate and compile the acados SQP-RTI OCP solver for the given settings."""
    n, dt, g = int(s.horizon), float(s.step_dt), float(s.gravity)
    x, u, p = ca.SX.sym("x", _NX), ca.SX.sym("u", _NU), ca.SX.sym("p", _NP)
    pos, vel, th, vth = x[0:3], x[3:6], x[6], x[7]
    acc, dvth = u[0:3], u[3]
    pref, tan, thbar, vcurv, w_contour = p[0:3], p[3:6], p[6], p[7], p[8]

    model = AcadosModel()
    model.name = "controller_v10_1_gtopt"
    model.x, model.u, model.p = x, u, p
    model.f_expl_expr = ca.vertcat(vel, acc, vth, dvth)
    model.xdot = ca.SX.sym("xdot", _NX)
    model.f_impl_expr = model.xdot - model.f_expl_expr

    # Reference linearised around theta_bar: p_ref(th) ~ pref + tan*(th - thbar).
    err = pos - (pref + tan * (th - thbar))
    lag = ca.dot(err, tan)
    perp = err - lag * tan
    contour = w_contour * ca.sumsqr(perp) + s.w_lag * lag**2  # w_contour is per-stage (gate-aware)
    model.cost_expr_ext_cost = contour - s.mu * vth + s.w_accel * ca.sumsqr(acc) + s.r_dv * dvth**2
    model.cost_expr_ext_cost_e = contour - s.mu * vth  # terminal: keep rewarding progress

    thrust = acc + ca.vertcat(0.0, 0.0, g)
    con_h = ca.vertcat(
        a_max**2 - ca.sumsqr(thrust),                                  # |thrust| <= a_max
        thrust[2] - s.a_z_min,                                         # keep lift
        (s.tilt_ratio * thrust[2]) ** 2 - ca.sumsqr(thrust[0:2]),      # tilt
        vcurv**2 - ca.sumsqr(vel),                                     # speed <= curvature cap (soft)
        vcurv - vth,                                                   # progress rate <= cap (soft)
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
    ocp.constraints.idxsh = ocp.constraints.idxsh_0 = np.array([3, 4])  # soften the speed caps
    ns = 2
    ocp.cost.zl = ocp.cost.zu = 1e2 * np.ones(ns)
    ocp.cost.Zl = ocp.cost.Zu = 1e1 * np.ones(ns)
    ocp.cost.zl_0, ocp.cost.zu_0 = ocp.cost.zl, ocp.cost.zu
    ocp.cost.Zl_0, ocp.cost.Zu_0 = ocp.cost.Zl, ocp.cost.Zu
    ocp.constraints.idxbu = np.array([3])  # |dvth| <= a_theta_max
    ocp.constraints.lbu = np.array([-s.a_theta_max])
    ocp.constraints.ubu = np.array([s.a_theta_max])
    ocp.constraints.x0 = np.zeros(_NX)  # placeholder; set each step

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.qp_solver_iter_max = int(s.max_iter)
    return AcadosOcpSolver(ocp, json_file="c_generated_code/controller_v10_1.json", verbose=False)


class MPCC:
    """Real-time gate-aware time-optimal contouring controller (world-frame acceleration out)."""

    def __init__(self, settings: MPCCSettings, a_max: float):
        """Build (or reuse) the acados SQP-RTI solver for these settings."""
        self.n = int(settings.horizon)
        self._dt = float(settings.step_dt)
        key = (round(a_max, 4), *(round(float(getattr(settings, f)), 4) for f in (
            "horizon", "step_dt", "v_theta_max", "a_theta_max", "tilt_ratio", "a_z_min",
            "mu", "w_lag", "w_accel", "r_dv", "gravity", "max_iter")))
        if key not in _SOLVER_CACHE:
            _SOLVER_CACHE[key] = _build_solver(settings, a_max)
        self._solver = _SOLVER_CACHE[key]
        self._path: GateArcPath | None = None
        self.reset()

    def set_path(self, path: GateArcPath) -> None:
        """Load the gate-aware arc-length path used to linearise the reference each step."""
        self._path = path
        self.reset()

    def reset(self) -> None:
        """Forget the warm start between episodes / on a new path."""
        self._x_sol: NDArray[np.float64] | None = None
        self._u_sol: NDArray[np.float64] | None = None
        self._last_acc = np.zeros(3)
        self._solver.reset()  # clear the (cached, shared) solver's internal memory

    def solve(
        self, pos: NDArray[np.float64], vel: NDArray[np.float64], th0: float, vth_cap: float
    ) -> NDArray[np.float64]:
        """Run one real-time iteration and return the first acceleration command."""
        n, dt, sv = self.n, self._dt, self._solver
        vth_cap = max(float(vth_cap), 1e-3)
        # Linearisation points: predicted progress per stage (shifted last solution, re-anchored).
        if self._x_sol is not None:
            thbar = np.concatenate([self._x_sol[6, 1:], self._x_sol[6, -1:]])
            thbar += th0 - thbar[0]
        else:
            thbar = th0 + np.arange(n + 1) * vth_cap * dt
        thbar = np.maximum.accumulate(np.maximum(thbar, th0))
        pref, tan, vcurv = self._path.eval(thbar)
        vcurv = np.minimum(vcurv, vth_cap)
        wc = self._path.w_contour(thbar)  # per-stage gate-aware contouring weight
        for k in range(n + 1):
            sv.set(k, "p", np.concatenate([pref[k], tan[k], [thbar[k], vcurv[k], wc[k]]]))
        vth0 = float(self._x_sol[7, 1]) if self._x_sol is not None else min(vth_cap, float(vcurv[0]))
        x0 = np.concatenate([pos, vel, [th0, vth0]])
        sv.set(0, "lbx", x0)
        sv.set(0, "ubx", x0)
        if self._x_sol is not None:  # warm start from the shifted previous solution
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
