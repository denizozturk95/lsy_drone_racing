"""Tunnel-constrained time-optimal MPCC package for controller_v11 (IMPROVEMENT_PLAN Phase 1).

v11 attacks the measured solver-pace gap (the v10.x MPCC flies 0.5-0.9 m/s below its own
curvature profile because gate precision is enforced through COST weights that pace the drone
everywhere -- RESEARCH_BRIEF.md section 3.2) with the MPCC++ move (Krinner et al., RSS 2024):

- gates and obstacles become a SOFT PRISMATIC TUNNEL CONSTRAINT around the reference path
  (controller_core_v11.mpcc + arc_path) -- wide on straights, clipped at obstacle keep-outs, narrowed to
  the opening at each gate;
- the contouring cost is de-spiked to a low constant, so -MU*vth sets the pace against the
  tunnel walls instead of against a weight schedule;
- the information constraint stays as REVEAL CAPS on every gate's 0.7 m approach window
  (Phase-0 bound: this costs only ~0.5 s of lap time; scripts/cpc_bound.py).

Everything else -- the planner (v10.4's trimmed gate-funnel chain), the launch (mini-takeoff,
ramp, honest cold start), the replan rebase, and the v10.2/v10.5 dynamics-aware anchor -- is
v10.5's, inherited or copied with the tunnel woven in. Own acados codegen namespace
``controller_v11`` (the OCP dimensions change). REQUIRES the acados environment (``pixi run``).
"""
