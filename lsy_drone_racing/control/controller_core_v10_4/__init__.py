"""Launch-optimised gate-aware time-optimal MPCC package for controller_v10_4.

v10.4 attacks the START PHASE, where a measured ~2.6 s of the ~8.1 s level2 lap is spent
(profiled: PID climb to 0.5 m completes at ~0.54 s, gate 0 is crossed at ~2.62 s, ramp-throttled
the whole way). The racing OCP, the gate-aware contouring weight, the replan-continuity rebase,
and the speed budget are all v10.3's -- only how the lap STARTS changes:

1. MINI-TAKEOFF (controller_core_v10_4.takeoff). The PID climb is kept -- it is what holds the start XY
   inside the env's +/-0.02 m floor-touch carve-out while the rotors spin up from zero (the env
   hard-disables on ANY floor contact once the drone drifts outside it, and every episode starts
   with ~40 ms of thrust-free free-fall; a from-ground MPCC launch is therefore unsafe by
   construction). But it now climbs only to 0.22 m on a fixed 0.40 s spline instead of 0.5 m at
   0.9 m/s: the hand-off happens at ~0.3 s with ~1 m/s of UPWARD momentum, roughly along the
   plan's steep initial tangent -- a smaller velocity mismatch than today's near-hover hand-off.

2. LAUNCH RAMP (controller_core_v10_4.cockpit). v10.3's progress-rate ramp (0.08, 2.0 s) was tuned to mask
   the 0.5 m hover hand-off transient. v10.4's hand-off is earlier, lower, and momentum-aligned,
   and the cold-start fixes below remove the solver-side fragility, so the ramp is retuned as a
   launch ramp. Gate-0 ARRIVAL speed stays curvature-capped (v_curv ~ 2.7 at the gate), so a
   hotter ramp shortens the run-up without raising the crossing speed -- the proven ceiling.

3. COLD-START SOLVER FIXES (controller_core_v10_4.mpcc). On the first solve of a plan v10.3 fabricates
   vth0 = min(cap, v_curv[0]) (= the full ramp cap, even from rest) and gives the RTI no warm
   start at all (linearisation at the all-zeros iterate). v10.4 clamps vth0 to the drone's
   actual speed (+ one step of progress acceleration) and seeds a hover-stationary warm start at
   the measured pose, so the first iterate is linearised where the drone actually is. This also
   de-risks the residual hand-off crash class (1-2/20 in v10.3).
"""
