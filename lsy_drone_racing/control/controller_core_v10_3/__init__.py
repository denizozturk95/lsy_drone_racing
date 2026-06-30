"""Gate-aware time-optimal MPCC package for controller_v10_3.

v10.3 is v10.1 made *seamless across replans and deliberate through gates*. It keeps everything
that makes v10.1 work -- v8's gate-aware planner, the vertical takeoff, the real-time acados
SQP-RTI time-optimal MPCC, the -mu*vth progress reward, the friction-circle curvature cap, and
the per-stage gate-aware contouring weight -- and changes two things about how that solver is fed:

1. REPLAN CONTINUITY (controller_core_v10_3.mpcc.MPCC.rebase). v10.1 fully resets the MPCC on every replan:
   warm start discarded, acados memory wiped, fallback command zeroed. Replans fire whenever the
   target gate advances (right after EVERY gate pass at ~3 m/s) and whenever an observed gate
   moves > 0.05 m -- which, at level2's 0.7 m sensor range under +/-0.15 m gate randomisation,
   means at essentially every gate reveal, ~0.2 s before the gate. So the solver was at its
   dumbest (one cold RTI iteration on a fabricated linearisation) exactly where precision binds.
   v10.3 instead REBASES: the previous solution's world-frame states/controls are kept (they stay
   dynamically feasible -- the world did not change, only the reference path), and the progress
   row is re-anchored by forward-projecting the predicted positions onto the new path.

2. GATE-WINDOW SPEED CAP (controller_core_v10_3.arc_path.GateArcPath). The curvature cap only brakes where
   the PATH bends, so a straight-approach gate is crossed at full V_MAX; the true gate pose
   appears 0.7 m out (~0.2 s), demanding up to ~0.21 m of lateral correction -- ~v^2-scaled, which
   is why raising V_MAX past 3.2 collapsed finish in the v10.1 sweeps. v10.3 caps speed to V_GATE
   inside a Gaussian window around each gate's arc-position (the same windows the v10.1 contouring
   weight uses) with a backward/forward pass keeping the braking feasible. Crossing gates at a
   deterministic V_GATE buys reveal-correction margin, which funds a higher straight budget
   (V_MAX = V_THETA_MAX raised above v10.1's 3.2).

The OCP, its states/parameters, and the solver structure are byte-for-byte v10.1's -- only the
path feed and the warm-start handling change, hence v10.3 rather than v11.
"""
