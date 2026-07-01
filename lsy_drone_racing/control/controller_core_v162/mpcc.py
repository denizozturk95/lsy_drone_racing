"""Solver-failure-robust MPCC for controller_v162 (v16's tunnel MPCC, deadlock-proofed).

THE BUG v162 FIXES (the ~10 s hover between gates 3 and 4):

v16 (== the "v11.1" line the v11 cockpit flags as "slow recovery laps") threads the g2->g3 U-turn
fold on the smoothed reference. There the tunnel pinches to its 2 cm floor (TUNNEL_W_MIN) beside a
passed gate's frame posts, the curvature clamp tightens it further, and v16's added obstacle/reveal
speed caps stack -- so a single SQP-RTI step occasionally returns a non-zero status.

v11's ``MPCC.solve`` treats that failure by returning the STALE ``self._last_acc`` and leaving
``self._x_sol`` UNTOUCHED. But the reference anchor in ``_track_action`` reads
``predicted_progress() == self._x_sol[6, 1]``: a frozen ``x_sol`` means a frozen anchor, the warm
start is rebuilt from the same states, and the IDENTICAL failing linearisation is replayed every
tick. The stale ``last_acc`` is a near-hover braking command (the drone was decelerating into the
fold), so the drone just wobbles in place until it physically drifts out of the bad basin -- the
observed multi-second pause.

THE FIX (bounded RTI recovery, no OCP/dimension change -- the compiled ``controller_v11`` solver is
reused verbatim): on a WARM-step failure, drop the warm start and re-solve ONCE from an honest cold
start at the measured state, this same tick. A cold solve does not depend on the frozen ``x_sol``,
so it breaks the deadlock and usually recovers with zero stall; if even the cold step fails, we
return hover for one tick with ``x_sol`` cleared, so the next tick cold-solves against the live
projection anchor instead of relocking. Successful solves are byte-identical to v16.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from lsy_drone_racing.control.controller_core_v11.mpcc import MPCC as _MPCC

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RobustMPCC(_MPCC):
    """v11's tunnel MPCC whose ``solve`` cannot freeze the progress anchor on a failed step."""

    def reset(self) -> None:
        """Forget the warm start (v11) and the consecutive-solve-failure counter."""
        super().reset()
        self._fail_streak = 0

    def solve(
        self, pos: NDArray[np.float64], vel: NDArray[np.float64], th0: float, vth_cap: float
    ) -> NDArray[np.float64]:
        """v11's RTI step, but a failed WARM solve falls back to a cold re-solve this tick."""
        return self._solve(pos, vel, th0, vth_cap, allow_cold_retry=True)

    def _solve(
        self,
        pos: NDArray[np.float64],
        vel: NDArray[np.float64],
        th0: float,
        vth_cap: float,
        allow_cold_retry: bool,
    ) -> NDArray[np.float64]:
        """v11.solve's body verbatim, with the failure tail replaced by the cold-restart recovery."""
        n, dt, sv = self.n, self._dt, self._solver
        vth_cap = max(float(vth_cap), 1e-3)
        warm = self._x_sol is not None
        if warm:  # warm: shifted previous solution (v10.1 flow)
            thbar = np.concatenate([self._x_sol[6, 1:], self._x_sol[6, -1:]])
            thbar += th0 - thbar[0]
            thbar = np.maximum.accumulate(np.maximum(thbar, th0))
            self._set_stage_params(thbar, vth_cap)
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
            self._fail_streak = 0
            return self._last_acc.copy()
        # --- solve failed: never freeze x_sol (the anchor reads x_sol[6, 1]) ---
        self._fail_streak += 1
        if warm and allow_cold_retry:
            # A failed WARM step would replay the same failing linearisation every tick and hover
            # the drone in place. Drop the warm start and re-solve honestly from the measured state
            # once, this same tick -- the cold step does not depend on the frozen x_sol.
            self.reset()
            return self._solve(pos, vel, th0, vth_cap, allow_cold_retry=False)
        # Cold step also failed (or we were cold already): x_sol is cleared, so predicted_progress()
        # returns None and _track_action falls back to projecting onto the live drone position next
        # tick -- no relock. Hold the (hover-level) last command for this single tick.
        return self._last_acc.copy()
