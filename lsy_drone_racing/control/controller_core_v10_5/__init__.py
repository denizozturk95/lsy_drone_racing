r"""Gate-aware time-optimal MPCC package for controller_v10_5.

v10.5 merges the two sibling branches of the v10.x tree. The tree is not linear -- v10.2 and
v10.3 are siblings off v10.1:

    v10.1 --> v10.2   dynamics-aware progress anchor (sharp-slalom gate-skip fix)
          \-> v10.3   replan continuity (rebase)  --> v10.4  fast launch + honest cold start
                                                              + reactive gate caps (8.01s)

v10.4 carries everything that makes the lap fast (the mini-takeoff, the hot launch ramp, the
honest cold start, the replan-continuity rebase, the reactive per-gate caps) but it still anchors
progress with the GLOBAL geometric projection -- the exact mechanism v10.2 proved can teleport the
progress anchor ~1-2 m across a path fold on a sharp slalom and skip the gate. v10.4's slalom
robustness came from launch fixes; the fold-teleport failure mode is structurally still present
mid-race.

v10.5 = v10.4's everything + v10.2's predicted-progress anchor. The two mechanisms touch disjoint
code: the anchor changes one line of ``_track_action`` plus two tiny accessors (``project_near``
on the path, ``predicted_progress`` on the MPCC, both copied verbatim from v10.2); all of v10.4's
launch/replan machinery is untouched. The OCP is byte-for-byte identical across the v10.x line, so
the compiled acados solver is shared with v10.4 (same horizon 18 -> same dimensions -> reuse the
``controller_v10_4`` codegen namespace; no new ``_build_*``).

The merge composes cleanly because v10.4's two cold-start paths both maintain the state the anchor
reads: after ``set_path`` (episode start / first plan) ``_x_sol`` is None, so ``predicted_progress``
returns None and the anchor falls back to the geometric projection for exactly one step until the
honest cold start seeds a solution; after ``rebase`` (mid-flight replan) ``_x_sol`` is kept and its
progress row re-anchored onto the new path, so the predicted-progress anchor is already in new-path
arc coordinates and works across replans with no special case -- strictly better than v10.2, which
lost the prediction (and the fold protection) for one step at every replan.

Implemented as a thin subclass of ControllerV104. REQUIRES the acados environment -- run under
``pixi run``.
"""
