"""Model-predictive contouring control (MPCC) core for controller_v9.

A compact point-mass MPCC. Over a short horizon it chases a reference that recedes along
the geometric path at v_ref and is kept on the path by a contouring (perpendicular)
penalty, while thrust, tilt, and velocity limits cap how fast it can follow. Speed
therefore comes out of the drone's actuator limits instead of hand-tuned cruise constants,
which is what lets it generalise to an arbitrary track. The optimiser plans world-frame
accelerations; the controller turns the first one into an attitude+thrust command.

The NLP is built once with parameters and re-solved (warm-started) every control step via
casadi + ipopt. No acados build required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.controller_core_v9.settings import MPCCSettings


def sample_path(
    curve: object, tau0: float, tau_end: float, arc_lengths: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return path points and unit tangents at given arc lengths ahead of tau0.

    The reference spline is arc-length re-sampled from the current projection forward, so
    the MPCC reference is geometric and independent of the spline's own time profile.
    """
    taus = np.linspace(tau0, tau_end, 200)
    points = np.asarray(curve(taus), dtype=np.float64)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    ref_p, ref_t = [], []
    for arc in arc_lengths:
        arc = min(float(arc), float(s[-1]))
        i = int(np.clip(np.searchsorted(s, arc), 1, len(taus) - 1))
        w = 0.0 if s[i] == s[i - 1] else (arc - s[i - 1]) / (s[i] - s[i - 1])
        ref_p.append(points[i - 1] * (1.0 - w) + points[i] * w)
        tangent = points[i] - points[i - 1]
        norm = float(np.linalg.norm(tangent))
        ref_t.append(tangent / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0]))
    return np.asarray(ref_p), np.asarray(ref_t)


class MPCC:
    """Receding-horizon contouring controller returning a world-frame acceleration."""

    def __init__(self, settings: MPCCSettings, a_max: float):
        """Build the parametric NLP once; it is re-solved each step with new parameters."""
        self.n = int(settings.horizon)
        self.ds = float(settings.v_ref) * float(settings.step_dt)
        dt, g = float(settings.step_dt), float(settings.gravity)

        opti = ca.Opti()
        pos = opti.variable(3, self.n + 1)
        vel = opti.variable(3, self.n + 1)
        acc = opti.variable(3, self.n)
        p0, v0 = opti.parameter(3), opti.parameter(3)
        ref_p, ref_t = opti.parameter(3, self.n + 1), opti.parameter(3, self.n + 1)
        gvec = ca.DM([0.0, 0.0, g])

        opti.subject_to(pos[:, 0] == p0)
        opti.subject_to(vel[:, 0] == v0)
        cost = 0
        for k in range(self.n):
            opti.subject_to(pos[:, k + 1] == pos[:, k] + vel[:, k] * dt + 0.5 * acc[:, k] * dt**2)
            opti.subject_to(vel[:, k + 1] == vel[:, k] + acc[:, k] * dt)
            thrust = acc[:, k] + gvec  # collective-thrust direction (per unit mass)
            opti.subject_to(ca.sumsqr(thrust) <= a_max**2)  # thrust magnitude limit
            opti.subject_to(thrust[2] >= settings.a_z_min)  # keep lift (no free-fall)
            opti.subject_to(ca.sumsqr(thrust[:2]) <= (settings.tilt_ratio * thrust[2]) ** 2)  # tilt
            opti.subject_to(ca.sumsqr(vel[:, k]) <= settings.v_max**2)
        for k in range(self.n + 1):
            err = pos[:, k] - ref_p[:, k]
            lag = ca.dot(err, ref_t[:, k])
            perp = err - lag * ref_t[:, k]
            cost += settings.w_contour * ca.sumsqr(perp) + settings.w_lag * lag**2
        cost += settings.w_accel * ca.sumsqr(acc)
        opti.minimize(cost)
        opti.solver(
            "ipopt",
            {"print_time": False},
            {"print_level": 0, "sb": "yes", "max_iter": int(settings.max_iter)},
        )
        self._opti, self._acc = opti, acc
        self._params = (p0, v0, ref_p, ref_t)
        self._last_acc = np.zeros((3, self.n))

    def solve(
        self,
        pos: NDArray[np.float64],
        vel: NDArray[np.float64],
        ref_p: NDArray[np.float64],
        ref_t: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the first optimal acceleration; fall back to the last plan on failure."""
        p0, v0, rp, rt = self._params
        self._opti.set_value(p0, pos)
        self._opti.set_value(v0, vel)
        self._opti.set_value(rp, ref_p.T)
        self._opti.set_value(rt, ref_t.T)
        self._opti.set_initial(self._acc, self._last_acc)
        try:
            sol = self._opti.solve()
            self._last_acc = np.asarray(sol.value(self._acc)).reshape(3, self.n)
        except RuntimeError:
            self._last_acc = np.roll(self._last_acc, -1, axis=1)  # shift previous plan
        return self._last_acc[:, 0].copy()

    def reset(self) -> None:
        """Forget the warm-start between episodes."""
        self._last_acc = np.zeros((3, self.n))
