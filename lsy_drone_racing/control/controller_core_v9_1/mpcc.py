"""MPCC solver for controller_v9.1 (isolated copy of v9's, with two robustness fixes).

Same point-mass contouring controller as controller_core_v9.mpcc, kept as a separate class so the
fixes below can't change v9's behaviour:

1. The velocity cap is applied to vel[1:] only, not the fixed initial velocity vel[0]. In
   v9 a hot turn entry (actual speed above v_max) made the NLP infeasible, so the solver
   threw and fell back to a stale plan. The initial velocity is a measurement, not a
   decision, so constraining it never helped and only caused failures.
2. On solve failure the fallback holds the last horizon command instead of wrapping the
   first (now consumed) command back to the end.

The unchanged arc-length helper sample_path is reused from v9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

from lsy_drone_racing.control.controller_core_v9.mpcc import sample_path  # unchanged helper, re-exported

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.controller_core_v9.settings import MPCCSettings

__all__ = ["MPCC", "sample_path"]


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
        # Cap only the decided velocities; vel[0] is the measured initial state (see module
        # docstring), so constraining it would only make a fast turn entry infeasible.
        for k in range(1, self.n + 1):
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
            # Shift the plan one step and hold the last command (don't wrap the consumed
            # first command back to the end).
            self._last_acc = np.roll(self._last_acc, -1, axis=1)
            self._last_acc[:, -1] = self._last_acc[:, -2]
        return self._last_acc[:, 0].copy()

    def reset(self) -> None:
        """Forget the warm-start between episodes."""
        self._last_acc = np.zeros((3, self.n))
