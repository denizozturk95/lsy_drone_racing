"""Arena-aware tunnel MPCC package for controller_v15 (real-arena keep-in).

v11.2 = v11 (tunnel-constrained time-optimal MPCC) with one addition: a SOFT ARENA KEEP-IN
BARRIER baked into the OCP cost. The motivation is deployment, not lap time -- the real flight
arena is slightly SMALLER than the sim safety box (``env.track.safety_limits`` in the configs:
x in [-2.5, 2.5], y in [-1.5, 1.5]); when the drone leaves the real bounds the run is stopped
and the lap is lost. v11's tunnel keeps the drone near the *path*, but the path itself (or a
wall-riding excursion / gate overshoot) can still swing toward an arena edge, and nothing in
v11 prices that.

The fix is a quadratic-hinge penalty on the predicted world (x, y) position that turns on
inside a ramp band before an inset keep-in wall and grows beyond it (controller_core_v15.mpcc). The wall
sits ``ARENA_INSET`` (default 0.10 m, the main knob) inside each sim safety wall, so the cost
punishes any predicted state that approaches the real arena edge. It is a soft cost (never a
hard constraint), so a gate sitting near an edge is still reachable -- the gate's contouring
spike out-prices the barrier locally; the barrier only dominates where there is no reason to be
near the wall.

Everything else is v11's: the tunnel/reveal path view (``controller_core_v11.arc_path.TunnelArcPath`` is
reused unchanged -- the barrier needs no per-stage parameters, the bounds are static), the
planner, launch, anchor, replan rebase, and telemetry. The OCP cost expression changes, so the
solver gets its OWN codegen namespace ``controller_v11_2`` (never share a namespace across differing
generated code; see the v10.4 dlopen segfault note). REQUIRES the acados environment (``pixi
run``).
"""
