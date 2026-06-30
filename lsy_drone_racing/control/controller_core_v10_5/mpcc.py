"""Gate-aware time-optimal MPCC core for controller_v10_5.

v10.5 merges v10.4's launch/replan-continuity MPCC with v10.2's dynamics-aware progress anchor.
The solver itself is v10.4's, BYTE-FOR-BYTE: this subclass adds only the one read-only accessor
v10.2 introduced -- ``predicted_progress()`` -- which returns the solver's own progress one step
ahead (the th state at stage 1 of the last solution).

Because the OCP, the horizon (18), and every solver knob are v10.4's, the compiled acados solver
and its cache key are shared with v10.4 (codegen namespace ``controller_v10_4``); no ``_build_*`` is
added. v10.4's honest cold start (sets ``_x_sol`` on a successful first solve) and v10.3's
``rebase`` (rewrites row 6 onto the new path) both maintain the state this accessor reads, so the
predicted-progress anchor is live one step after every plan start AND immediately after every
mid-flight rebase -- strictly better coverage than v10.2, which lost the prediction for one step
at every replan. See controller_v10_5 for the full rationale.
"""

from __future__ import annotations

from lsy_drone_racing.control.controller_core_v10_4.mpcc import MPCC as _MPCC

__all__ = ["MPCC"]


class MPCC(_MPCC):
    """v10.4's MPCC (honest cold start + rebase) plus v10.2's predicted-progress accessor."""

    def predicted_progress(self) -> float | None:
        """Return the solver's progress one step ahead (th at stage 1), or None before any solve.

        This is the dynamics-feasible progress the OCP itself predicts for the next control step;
        because it evolves at the bounded progress rate vth, it cannot jump across a fold the way a
        raw geometric nearest-point search can. The controller uses it as the centre of a tight
        geometric correction so the progress anchor tracks the drone without ever teleporting.
        """
        if self._x_sol is None:
            return None
        return float(self._x_sol[6, 1])
