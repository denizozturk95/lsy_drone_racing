from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import TYPE_CHECKING

import casadi as ca
import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control._planner import Plan, PlannerConfig, build_plan
from lsy_drone_racing.control._racing_line import RacingLineConfig, build_racing_line_plan

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Unique suffix for acados generated artifacts so parallel benchmark workers
# (each with a distinct ``LSY_FAST_WORKER_ID``) do not clobber each other's
# generated C code / JSON.
_WORKER_SUFFIX = (
    f"_w{os.environ['LSY_FAST_WORKER_ID']}" if os.environ.get("LSY_FAST_WORKER_ID") else ""
)
_MODEL_NAME = f"gate_aware_fast_v3_s55_t757{_WORKER_SUFFIX}"

_DIAG_PATH = Path("/tmp/lsy_diagnostics.csv")


def _build_ocp(
    Tf: float,
    N: int,
    parameters: dict,
    n_obstacles: int,
    r_safe: float,
    w_obs: float,
    n_wings: int = 4,
    r_wing: float = 0.13,
    w_wing: float = 80000.0,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Build an acados OCP with parametric soft obstacle + current-gate wing constraints.

    Parameters (``model.p``) layout:
      [0 : 2*n_obstacles]      — obstacles xy (2D), constrained in xy
      [2*n_obstacles : 2*n_obstacles + 3*n_wings] — wing centers xyz (3D)
    """
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    n_params = 2 * n_obstacles + 3 * n_wings
    obs_p = ca.MX.sym("obs_p", n_params)
    model = AcadosModel()
    model.name = _MODEL_NAME
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U
    model.p = obs_p

    ocp = AcadosOcp()
    ocp.model = model

    nx = X.rows()
    nu = U.rows()
    ny = nx + nu
    ny_e = nx

    ocp.solver_options.N_horizon = N

    # Tuned for cf21B_500 (m≈43g, T/W≈1.9). xy weights 75 (vs base 50) for
    # tighter lateral tracking through the 0.4 m gate openings; vel weights
    # 12 (vs 10) for smoother approaches.
    Q = np.diag([75.0, 75.0, 400.0, 1.0, 1.0, 1.0, 12.0, 12.0, 12.0, 5.0, 5.0, 5.0])
    Rcost = np.diag([1.0, 1.0, 1.0, 50.0])
    Q_e = Q.copy()

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q, Rcost)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx = Vx
    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu
    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e
    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # Attitude state bounds on roll/pitch kept generous (±1.20 rad ≈ 69°) so
    # that the MPC's QP stays feasible even when the drone physically tilts
    # past the "preferred" range during aggressive gate transitions. The cost
    # function (Q diag=1.0 on rpy) still discourages large tilts, but we no
    # longer make the OCP infeasible when x0's measured rpy exceeds the bound.
    # Command (input) bounds remain tighter at ±0.80 rad since they directly
    # drive the attitude controller; the drone's actual roll can overshoot
    # commanded via body dynamics under large angular rates.
    # Yaw bound widened to ±1.5 rad: at ±0.5 the OCP went infeasible during
    # fast gate-2/3 transitions, causing stale-control ground strikes.
    ocp.constraints.lbx = np.array([-1.20, -1.20, -1.5])
    ocp.constraints.ubx = np.array([1.20, 1.20, 1.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])
    ocp.constraints.lbu = np.array([-0.80, -0.80, -1.0, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.80, 0.80, 1.0, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.x0 = np.zeros((nx,))

    # Soft constraints: squared 2D distance to each obstacle >= r_safe**2, plus
    # squared 3D distance to each wing of the current target gate >= r_wing**2.
    p_drone_xy = X[0:2]
    p_drone_xyz = X[0:3]
    dist_sq_terms = []
    for j in range(n_obstacles):
        ox = obs_p[2 * j]
        oy = obs_p[2 * j + 1]
        dx = p_drone_xy[0] - ox
        dy = p_drone_xy[1] - oy
        dist_sq_terms.append(dx * dx + dy * dy)
    wing_offset = 2 * n_obstacles
    for k in range(n_wings):
        wx = obs_p[wing_offset + 3 * k]
        wy = obs_p[wing_offset + 3 * k + 1]
        wz = obs_p[wing_offset + 3 * k + 2]
        dx = p_drone_xyz[0] - wx
        dy = p_drone_xyz[1] - wy
        dz = p_drone_xyz[2] - wz
        dist_sq_terms.append(dx * dx + dy * dy + dz * dz)
    n_soft = n_obstacles + n_wings
    model.con_h_expr = ca.vertcat(*dist_sq_terms)
    ocp.constraints.lh = np.concatenate(
        [np.full(n_obstacles, r_safe**2), np.full(n_wings, r_wing**2)]
    )
    ocp.constraints.uh = np.full(n_soft, 1e6)
    ocp.constraints.idxsh = np.arange(n_soft)
    ocp.cost.zl = np.zeros(n_soft)
    ocp.cost.zu = np.zeros(n_soft)
    ocp.cost.Zl = np.concatenate([np.full(n_obstacles, w_obs), np.full(n_wings, w_wing)])
    ocp.cost.Zu = np.zeros(n_soft)

    ocp.parameter_values = np.zeros(n_params)

    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 40
    ocp.solver_options.nlp_solver_max_iter = 10
    ocp.solver_options.tf = Tf

    solver = AcadosOcpSolver(ocp, json_file=f"c_generated_code/{_MODEL_NAME}.json", verbose=False)
    return solver, ocp


class GateAwareFastV3S55T757(Controller):
    """Attitude-mode MPC with parametric obstacle soft constraints."""

    _run_counter = 0

    N = 35
    PLAN_PAD = 200
    N_OBSTACLES = 4
    N_WINGS = 4
    R_SAFE = 0.20
    R_WING = 0.13
    W_OBS = 150000.0
    W_WING = 80000.0
    WING_OFFSET = 0.28  # distance along gate y/z axis to wing midpoint
    USE_RACING_LINE = False
    PLANNER = PlannerConfig(
        d_pre=0.28, d_post=0.18, v_cruise=2.60, v_cruise_inter=4.00, t_min_seg=0.24, r_obs=0.22
    )
    RACING_LINE = RacingLineConfig(
        v_cruise=1.8, t_min_seg=0.15, max_accel=9.0, max_vel=4.0, r_obs=0.22
    )

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict) -> None:
        """Build MPC + initial plan."""
        super().__init__(obs, info, config)
        self._dt = 1.0 / config.env.freq
        self._T_HORIZON = self.N * self._dt

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = _build_ocp(
            self._T_HORIZON,
            self.N,
            self.drone_params,
            self.N_OBSTACLES,
            self.R_SAFE,
            self.W_OBS,
            n_wings=self.N_WINGS,
            r_wing=self.R_WING,
            w_wing=self.W_WING,
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._finished = False
        self._plan_spline_ticks = 0
        self._pos_samples = np.zeros((0, 3))
        self._vel_samples = np.zeros((0, 3))
        self._replan(obs, start_vel=np.zeros(3), target_gate=0)

        self._prev_gates_visited = np.asarray(obs["gates_visited"]).copy()
        self._prev_obstacles_visited = np.asarray(obs["obstacles_visited"]).copy()

        GateAwareFastV3S55T757._run_counter += 1
        self._run_idx = GateAwareFastV3S55T757._run_counter
        self._terminal_state: dict | None = None
        self._last_live_state: dict | None = None
        self._flight_tick = 0  # never reset by _replan
        if self._run_idx == 1 and _DIAG_PATH.exists():
            _DIAG_PATH.unlink()

    # ---- planning ---------------------------------------------------------

    def _replan(
        self,
        obs: dict[str, NDArray[np.floating]],
        start_vel: NDArray[np.floating],
        target_gate: int,
    ) -> None:
        if self.USE_RACING_LINE:
            plan: Plan = build_racing_line_plan(
                start_pos=np.asarray(obs["pos"], dtype=np.float64),
                start_vel=np.asarray(start_vel, dtype=np.float64),
                gates_pos=obs["gates_pos"],
                gates_quat=obs["gates_quat"],
                obstacles_pos=obs["obstacles_pos"],
                target_gate=target_gate,
                cfg=self.RACING_LINE,
            )
        else:
            plan = build_plan(
                start_pos=np.asarray(obs["pos"], dtype=np.float64),
                start_vel=np.asarray(start_vel, dtype=np.float64),
                gates_pos=obs["gates_pos"],
                gates_quat=obs["gates_quat"],
                obstacles_pos=obs["obstacles_pos"],
                target_gate=target_gate,
                cfg=self.PLANNER,
            )
        n_samples = max(int(np.ceil(plan.t_total / self._dt)), 2)
        ts = np.arange(n_samples) * self._dt
        ts = np.clip(ts, 0.0, plan.t_total)
        pos = plan.pos_spline(ts)
        vel = plan.vel_spline(ts)
        pad_pos = np.tile(pos[-1], (self.PLAN_PAD, 1))
        pad_vel = np.zeros((self.PLAN_PAD, 3))
        self._pos_samples = np.vstack([pos, pad_pos])
        self._vel_samples = np.vstack([vel, pad_vel])
        self._plan_spline_ticks = n_samples
        self._tick = 0

    def _current_gate_wings(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Return (N_WINGS, 3) world-frame midpoints of the current target gate's frame wings.

        Frame wings are left/right along the gate y-axis and top/bottom along world z.
        Parked far away when no live gate exists (target_gate == -1, or out of range).
        """
        tgt = int(obs["target_gate"])
        if tgt < 0 or tgt >= len(obs["gates_pos"]):
            return np.full((self.N_WINGS, 3), 100.0)
        gp = np.asarray(obs["gates_pos"][tgt], dtype=np.float64)
        y_axis = R.from_quat(obs["gates_quat"][tgt]).as_matrix()[:, 1]
        left = gp + self.WING_OFFSET * y_axis
        right = gp - self.WING_OFFSET * y_axis
        top = gp + self.WING_OFFSET * np.array([0.0, 0.0, 1.0])
        bottom = gp - self.WING_OFFSET * np.array([0.0, 0.0, 1.0])
        return np.stack([left, right, top, bottom])

    # ---- controller API ---------------------------------------------------

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Return the MPC's first-step attitude command for the current observation."""
        if self._tick >= self._plan_spline_ticks + self.PLAN_PAD - self.N - 1:
            self._finished = True
        i = min(self._tick, self._plan_spline_ticks + self.PLAN_PAD - self.N - 2)

        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], rpy, obs["vel"], drpy))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Build parameter vector: [obstacles_xy (flat), wings_xyz (flat)].
        obs_xy = np.asarray(obs["obstacles_pos"])[: self.N_OBSTACLES, :2].flatten()
        if obs_xy.size < 2 * self.N_OBSTACLES:
            obs_xy = np.concatenate([obs_xy, np.full(2 * self.N_OBSTACLES - obs_xy.size, 1e6)])
        wings_xyz = self._current_gate_wings(obs).flatten()
        param_vec = np.concatenate([obs_xy, wings_xyz])
        for j in range(self.N + 1):
            self._acados_ocp_solver.set(j, "p", param_vec)

        hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        yref = np.zeros((self.N, self._ny))
        yref[:, 0:3] = self._pos_samples[i : i + self.N]
        yref[:, 6:9] = self._vel_samples[i : i + self.N]
        yref[:, 15] = hover_thrust
        for j in range(self.N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        yref_e = np.zeros((self._ny_e,))
        yref_e[0:3] = self._pos_samples[i + self.N]
        yref_e[6:9] = self._vel_samples[i + self.N]
        self._acados_ocp_solver.set(self.N, "y_ref", yref_e)

        self._acados_ocp_solver.solve()
        u0 = self._acados_ocp_solver.get(0, "u")
        return np.asarray(u0, dtype=np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance tick counters, snapshot terminal state, and trigger replans on sense events."""
        self._tick += 1
        self._flight_tick += 1
        target_gate = int(obs["target_gate"])

        pos_now = np.asarray(obs["pos"]).copy()
        disabled_warped = bool(np.all(np.isclose(pos_now, -1.0, atol=1e-3)))
        if not disabled_warped:
            self._last_live_state = {
                "tick": self._flight_tick,
                "pos": pos_now,
                "vel": np.asarray(obs["vel"]).copy(),
                "target_gate": target_gate,
                "gates_pos": np.asarray(obs["gates_pos"]).copy(),
                "gates_quat": np.asarray(obs["gates_quat"]).copy(),
                "gates_visited": np.asarray(obs["gates_visited"]).copy(),
                "obstacles_pos": np.asarray(obs["obstacles_pos"]).copy(),
            }

        if terminated or truncated:
            ls = self._last_live_state or {}
            self._terminal_state = {
                **ls,
                "final_target_gate": target_gate,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }

        if target_gate == -1:
            self._finished = True
            return self._finished
        gates_visited = np.asarray(obs["gates_visited"])
        obstacles_visited = np.asarray(obs["obstacles_visited"])
        new_gate = bool((gates_visited & ~self._prev_gates_visited).any())
        new_obstacle = bool((obstacles_visited & ~self._prev_obstacles_visited).any())
        if new_gate or new_obstacle:
            self._replan(obs, start_vel=np.asarray(obs["vel"]), target_gate=target_gate)
        self._prev_gates_visited = gates_visited.copy()
        self._prev_obstacles_visited = obstacles_visited.copy()
        return self._finished

    def episode_callback(self) -> None:
        """Write a diagnostic row at episode end and reset per-episode state."""
        if self._terminal_state is not None:
            self._write_diagnostic(self._terminal_state)
        self._tick = 0
        self._flight_tick = 0
        self._terminal_state = None
        self._last_live_state = None

    def _write_diagnostic(self, ts: dict) -> None:
        pos = ts.get("pos", np.zeros(3))
        vel = ts.get("vel", np.zeros(3))
        tgt = ts.get("final_target_gate", ts.get("target_gate", -2))
        flight_time = ts.get("tick", 0) * self._dt
        success = tgt == -1
        outcome = "success" if success else ("timeout" if ts.get("truncated") else "crash")

        gate_pos = ts["gates_pos"]
        obs_pos = ts["obstacles_pos"]
        gate_dists = np.linalg.norm(gate_pos - pos[None, :], axis=1)
        obs_dists_2d = np.linalg.norm(obs_pos[:, :2] - pos[None, :2], axis=1)
        nearest_obs_idx = int(np.argmin(obs_dists_2d))
        nearest_obs_dist = float(obs_dists_2d[nearest_obs_idx])
        nearest_gate_idx = int(np.argmin(gate_dists))
        nearest_gate_dist = float(gate_dists[nearest_gate_idx])

        if 0 <= tgt < len(gate_pos):
            rot = R.from_quat(ts["gates_quat"][tgt]).as_matrix()
            rel = pos - gate_pos[tgt]
            local_x = float(rot[:, 0] @ rel)
            local_y = float(rot[:, 1] @ rel)
            local_z = float(rel[2])
        else:
            local_x = local_y = local_z = 0.0

        header = not _DIAG_PATH.exists()
        with open(_DIAG_PATH, "a", newline="") as f:
            w = csv.writer(f)
            if header:
                w.writerow(
                    [
                        "run",
                        "outcome",
                        "t_flight",
                        "target_gate",
                        "pos_x",
                        "pos_y",
                        "pos_z",
                        "vx",
                        "vy",
                        "vz",
                        "speed",
                        "near_obs_idx",
                        "near_obs_xy_dist",
                        "near_gate_idx",
                        "near_gate_3d_dist",
                        "tgt_local_x",
                        "tgt_local_y",
                        "tgt_local_z",
                        "gate0_dist",
                        "gate1_dist",
                        "gate2_dist",
                        "gate3_dist",
                        "obs0_xy",
                        "obs1_xy",
                        "obs2_xy",
                        "obs3_xy",
                    ]
                )
            w.writerow(
                [
                    self._run_idx,
                    outcome,
                    f"{flight_time:.3f}",
                    tgt,
                    f"{pos[0]:.3f}",
                    f"{pos[1]:.3f}",
                    f"{pos[2]:.3f}",
                    f"{vel[0]:.3f}",
                    f"{vel[1]:.3f}",
                    f"{vel[2]:.3f}",
                    f"{float(np.linalg.norm(vel)):.3f}",
                    nearest_obs_idx,
                    f"{nearest_obs_dist:.3f}",
                    nearest_gate_idx,
                    f"{nearest_gate_dist:.3f}",
                    f"{local_x:.3f}",
                    f"{local_y:.3f}",
                    f"{local_z:.3f}",
                    *[f"{d:.3f}" for d in gate_dists],
                    *[f"{d:.3f}" for d in obs_dists_2d],
                ]
            )
        self._tick = 0
