"""Feedforward trajectory tracker (v27): fly the racing line by the clock.

20/20 @ 6.89s official (final.toml seed 2026). Feedforward + PD tracker on an
offline-optimized racing line; no solver in the loop.
"""

from __future__ import annotations

import numpy as np
from drone_models.core import load_params
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.common_controller_v2.attitude import _vector_to_attitude
from lsy_drone_racing.control.common_controller_v2.feedback import CascadedPid
from lsy_drone_racing.control.common_controller_v2.state import parse_observation
from lsy_drone_racing.control.controller_core_v8.settings import ControllerSettings as _V8Settings
from lsy_drone_racing.control.controller_core_v10_4.settings import (
    ControllerSettings as _V104Settings,
)
from lsy_drone_racing.control.controller_core_v10_4.takeoff import TakeoffPhase as MiniTakeoff

D_PRE = np.array([0.621, 0.700, 0.318, 0.397])
D_POST = np.array([0.12, 0.130, 0.298, 0.0])
LEG_CP = np.array([
    [0.236, -0.521, 0.021],
    [-0.473, 0.345, -0.270],
    [-0.434, -0.583, 0.225],
    [0.607, -0.632, -0.063],
])
D_KNOT_PRE = 0.12
D_KNOT_POST = 0.08
RUNOUT = 0.35
MIN_PT_GAP = 0.06
MIN_MID_LEG = 0.60

V_CEIL = 3.8
V_GATE = np.array([1.8, 2.4, 2.4, 2.4])
R_GATE = 0.40
A_LAT = 8.5
A_H = 6.5
A_BRAKE = 4.5
V0_MIN = 0.25

KP = np.array([8.0, 8.0, 10.0])
KD = np.array([6.0, 6.0, 7.0])
KP_GATE = np.array([24.0, 24.0, 26.0])
KD_GATE = np.array([10.0, 10.0, 11.0])
R_BOOST = 1.2
A_CMD_MAX = 14.0
DT = 0.02


def _gate_normal(quat_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quat_xyzw).apply(np.array([1.0, 0.0, 0.0]))


def _waypoints(start: np.ndarray, gates: np.ndarray, quats: np.ndarray) -> np.ndarray:
    cross = np.array([_gate_normal(quats[i]) for i in range(4)])
    pts = [start.astype(np.float64)]
    prev_exit = start
    for i in range(4):
        entry = gates[i] - cross[i] * D_PRE[i]
        if np.linalg.norm(entry - prev_exit) > MIN_MID_LEG:
            pts.append(0.5 * (prev_exit + entry) + LEG_CP[i])
        pts += [entry, gates[i] - cross[i] * D_KNOT_PRE, gates[i],
                gates[i] + cross[i] * D_KNOT_POST]
        if i < 3:
            exit_p = gates[i] + cross[i] * D_POST[i]
            pts.append(exit_p)
            prev_exit = exit_p
        else:
            pts.append(gates[i] + cross[i] * RUNOUT)
    clean = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - clean[-1]) > MIN_PT_GAP:
            clean.append(p)
    return np.array(clean)


def build_reference(start: np.ndarray, gates: np.ndarray, quats: np.ndarray, v0: float = 0.0,
                    vel_dir: np.ndarray | None = None):
    """Compile the course into 50 Hz (pos, vel, acc) arrays + per-gate crossing indices."""
    way = _waypoints(start, gates, quats)
    chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(way, axis=0), axis=1))])
    if vel_dir is not None:  # hand-off with momentum: open the spline along current motion
        spl = CubicSpline(chord, way, axis=0, bc_type=((1, vel_dir), (2, np.zeros(3))))
    else:
        spl = CubicSpline(chord, way, axis=0)
    u = np.linspace(0.0, chord[-1], 900)
    P = spl(u)
    d1, d2 = spl(u, 1), spl(u, 2)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    speed_u = np.linalg.norm(d1, axis=1)
    tan = d1 / np.maximum(speed_u, 1e-9)[:, None]
    kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(speed_u**3, 1e-9)

    v_cap = np.minimum(np.sqrt(A_LAT / np.maximum(kappa, 1e-6)), V_CEIL)
    for i in range(4):
        near = np.linalg.norm(P - gates[i], axis=1) < R_GATE
        v_cap[near] = np.minimum(v_cap[near], V_GATE[i])

    v = v_cap.copy()
    v[0] = max(v0, V0_MIN)
    for i in range(len(s) - 1):
        a_lat = v[i] ** 2 * kappa[i]
        a_long = np.sqrt(max(A_H**2 - a_lat**2, 0.09))
        v[i + 1] = min(v[i + 1], np.sqrt(v[i] ** 2 + 2 * a_long * seg[i]))
    for i in range(len(s) - 2, -1, -1):
        a_lat = v[i + 1] ** 2 * kappa[i + 1]
        a_long = np.sqrt(max(A_BRAKE**2 - min(a_lat**2, A_BRAKE**2 * 0.96), 0.09))
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_long * seg[i]))

    t = np.concatenate([[0.0], np.cumsum(seg / np.maximum(0.5 * (v[1:] + v[:-1]), 1e-6))])
    t_grid = np.arange(0.0, t[-1], DT)
    s_t = np.interp(t_grid, t, s)
    pos = np.column_stack([np.interp(s_t, s, P[:, i]) for i in range(3)])
    tan_t = np.column_stack([np.interp(s_t, s, tan[:, i]) for i in range(3)])
    tan_t /= np.linalg.norm(tan_t, axis=1, keepdims=True) + 1e-9
    vel = tan_t * np.interp(s_t, s, v)[:, None]
    acc = np.gradient(vel, DT, axis=0)
    k_gate = [int(np.argmin(np.linalg.norm(pos - gates[i], axis=1))) for i in range(4)]
    return pos, vel, acc, k_gate


class ControllerV27(Controller):
    """Feedforward + PD tracker on the offline racing line (no solver in the loop)."""

    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("v27 requires attitude control mode")
        self._settings = _V8Settings()
        self._dt = 1.0 / float(config.env.freq)
        params = load_params(config.sim.physics, config.sim.drone_model)
        self._mass = float(params["mass"])
        self._gravity = self._settings.runtime.gravity
        self._command = self._settings.command
        self._feedback = CascadedPid(self._settings.feedback)
        v104 = _V104Settings()
        self._takeoff = MiniTakeoff(v104, v104.takeoff)
        self._nominal = np.asarray(obs["gates_pos"], dtype=np.float64).copy()
        self._nominal_quat = np.asarray(obs["gates_quat"], dtype=np.float64).copy()
        self._delta = np.zeros((4, 3))
        self._dyaw = np.zeros(4)
        self._obs_nominal = np.asarray(obs["obstacles_pos"], dtype=np.float64).copy()
        self._obs_latched = self._obs_nominal.copy()
        self._ref = None
        self._ref_i = 0
        self._tick = 0
        self._mode = "TAKEOFF"
        self._frozen_ticks = 0
        self._last_action = np.zeros(4)

    def compute_control(self, obs, info=None):
        frame = parse_observation(obs)
        if frame.target_gate == -1:
            return self._last_action.copy()
        if self._mode == "TAKEOFF":
            if not self._takeoff.is_complete(frame, self._tick, self._dt):
                action = self._takeoff.action(
                    frame, self._feedback, self._tick, self._dt, self._mass, self._gravity)
                self._last_action = action
                return action.copy()
            self._mode = "TRACK"
            v0 = float(np.linalg.norm(frame.vel))
            vel_dir = frame.vel / v0 if v0 > 0.3 else None
            self._ref = build_reference(frame.pos, self._nominal, self._nominal_quat,
                                        v0=v0, vel_dir=vel_dir)
            self._ref_i = 0
            pos_r = self._ref[0]
            self._r_safe = np.array([
                min(float(np.hypot(pos_r[:, 0] - ox, pos_r[:, 1] - oy).min()) - 0.02, 0.34)
                for ox, oy, _oz in self._obs_nominal])
        action = self._track(frame)
        self._last_action = action
        return action.copy()

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self._tick += 1
        return False

    def _warped(self, k: int) -> np.ndarray:
        """Reference position at index k with the full reveal warp applied.

        translate: piecewise-linear gate-delta field (exact delta at each crossing);
        rotate: latched yaw delta blended in around each crossing (aperture rotation);
        repel: keep the warped point out of latched obstacle cylinders.
        """
        pos_r, _, _, k_gate = self._ref
        kk = min(k, len(pos_r) - 1)
        p = pos_r[kk]
        knots_k = [0] + k_gate
        knots_d = [np.zeros(3)] + [self._delta[i] for i in range(4)]
        if kk >= knots_k[-1]:
            off = knots_d[-1].copy()
        else:
            j = max(int(np.searchsorted(knots_k, kk, side="right")) - 1, 0)
            k0, k1 = knots_k[j], knots_k[j + 1]
            w = 0.0 if k1 == k0 else (kk - k0) / (k1 - k0)
            off = knots_d[j] * (1 - w) + knots_d[j + 1] * w
        warped = p + off
        # obstacle repulsion against LATCHED poles (poles are full-height cylinders)
        for oi, (ox, oy, _oz) in enumerate(self._obs_latched):
            r_safe = self._r_safe[oi]
            d = np.hypot(warped[0] - ox, warped[1] - oy)
            if 1e-6 < d < r_safe:
                scale = r_safe / d
                warped = warped.copy()
                warped[0] = ox + (warped[0] - ox) * scale
                warped[1] = oy + (warped[1] - oy) * scale
        # gate-frame post repulsion (except crossing window)
        for g in range(4):
            if abs(kk - k_gate[g]) < 12:
                continue
            cg = self._nominal[g] + self._delta[g]
            n_nom = _gate_normal(self._nominal_quat[g])
            lat = np.array([-n_nom[1], n_nom[0], 0.0])
            for s_lat in (-0.36, 0.36):
                post = cg + lat * s_lat
                d = np.hypot(warped[0] - post[0], warped[1] - post[1])
                if 1e-6 < d < 0.22 and abs(warped[2] - post[2]) < 0.45:
                    scale = 0.22 / d
                    warped = warped.copy()
                    warped[0] = post[0] + (warped[0] - post[0]) * scale
                    warped[1] = post[1] + (warped[1] - post[1]) * scale
        return warped

    def _track(self, frame):
        for i in range(4):
            if np.linalg.norm(frame.gate_pos[i] - self._nominal[i]) > 1e-6:
                self._delta[i] = frame.gate_pos[i] - self._nominal[i]
                n_now = _gate_normal(frame.gate_quat[i])
                n_nom = _gate_normal(self._nominal_quat[i])
                self._dyaw[i] = np.arctan2(n_now[1], n_now[0]) - np.arctan2(n_nom[1], n_nom[0])
        for i in range(len(self._obs_latched)):
            if np.linalg.norm(frame.obstacles_pos[i] - self._obs_nominal[i]) > 1e-6:
                self._obs_latched[i] = frame.obstacles_pos[i]
        pos_r, vel_r, acc_r, k_gate = self._ref
        k = min(self._ref_i, len(pos_r) - 1)
        tgt0 = max(frame.target_gate, 0)
        near_gate = np.linalg.norm(
            frame.pos - (self._nominal[tgt0] + self._delta[tgt0])) < 1.2
        err_clock = np.linalg.norm(self._warped(k) - frame.pos)
        frozen = near_gate and err_clock >= 0.22
        half = near_gate and 0.14 <= err_clock < 0.22
        self._half_toggle = (getattr(self, "_half_toggle", 0) + 1) % 3 == 0
        if frozen and self._frozen_ticks <= 75:
            self._frozen_ticks += 1
        elif half and self._half_toggle:
            pass
        else:
            self._ref_i += 1
            if not frozen:
                self._frozen_ticks = 0
        p_cmd = self._warped(k)
        v_cmd = vel_r[k] + (self._warped(k + 1) - p_cmd) / DT - (pos_r[min(k + 1, len(pos_r) - 1)] - pos_r[k]) / DT
        tgt = max(frame.target_gate, 0)
        gate_now = self._nominal[tgt] + self._delta[tgt]
        boost = np.linalg.norm(frame.pos - gate_now) < R_BOOST
        kp, kd = (KP_GATE, KD_GATE) if boost else (KP, KD)
        k_ff = min(k + 4, len(pos_r) - 1)
        a_ff = np.zeros(3) if frozen else acc_r[k_ff]
        v_ref = np.zeros(3) if frozen else v_cmd
        acc = a_ff + kp * (p_cmd - frame.pos) + kd * (v_ref - frame.vel)
        n = np.linalg.norm(acc)
        if n > A_CMD_MAX:
            acc = acc * (A_CMD_MAX / n)
        thrust_vector = self._mass * (acc + np.array([0.0, 0.0, self._gravity]))
        return _vector_to_attitude(thrust_vector, frame.quat, self._command)
