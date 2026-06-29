"""Tunable constants (the "cockpit") for the controller_v15 controller.

v11.2 = v11 (tunnel MPCC) made to keep the drone inside the REAL flight arena, which is slightly
smaller than the sim safety box -- the deploy system aborts the run the instant the drone leaves
it. Three coordinated SOFT mechanisms, all reusing existing patterns:

1. SPLINE CLIP (controller_core_v15.arc_path): the plan spline is resampled, each sample softly saturated
   into the keep-in box (sim box shrunk by ARENA_INSET), and refit -- the whole v11 path view is
   built over the clipped curve, so reference, anchor, and tunnel agree.
2. ARENA SPEED CAP (controller_core_v15.arc_path, OPT-IN, default off): pace the profile down where the
   reference enters the ramp band, so the drone arrives slow enough to turn inside the wall.
3. MPC POSITION BARRIER (controller_core_v15.mpcc): a quadratic-hinge keep-in penalty on the predicted
   world (x, y), the soft lateral margin on top of (1):
     keep-in wall = sim_safety_wall -/+ ARENA_INSET ;  onset = wall -/+ ARENA_RAMP_M
     cost += W_ARENA * depth_into_band**2   (per horizontal axis/side, x and y only -- the
     vertical floor/ceiling is already the tunnel's H-extent, TUNNEL_Z_FLOOR/CEIL).

WHY ALL THREE: the measured root cause (scripts/_arena_probe.py) is that the reference SPLINE
itself routes to y ~ 1.42-1.62 / x ~ 1.55 on the gate-1 EXIT arc -- at/past a smaller real-arena
edge -- and the drone tracks it; the worst residual excursions are inertial (arrives too hot to
turn in). A position barrier alone could not win the tug-of-war against the contouring cost
pulling the drone onto the out-of-bounds spline (W_ARENA 200->1000 still let y reach 1.5-1.7), so
the spline is fixed at the source. An EARLIER eval-only clip (clip just the OCP reference, leave
the anchor on the unclipped path) made the lag term and the geometric anchor fight at the gate-1
exit and cost reliability -- clipping the curve so the whole view is consistent is what fixed
that.

MEASURED (scripts/_arena_probe.py = trajectory bounding box; scripts/compare_v10_5.py = paired
20-run finish/time; level2, seeds 42/7):
  CONTAINMENT (the point): reference spline y 1.62 -> <=1.40 (clipped into the box); flown drone
    y 1.56 -> ~1.43. With the opt-in speed cap (V_ARENA 2.5) the drone tightens to ~1.31.
  LEVEL2 SIM COST (overhead here -- see below): finish v11 50-55% -> v11.2 40-45% at the
    defaults (cap off); with the speed cap on, 25-40% (the cap is the reliability killer -- it
    paces the whole gate-1 exit and the re-accel into the gate-2 fold). Lap time +1-2 s, with
    occasional long recovery laps. v11 is itself a marginal/condemned ~50% base on level2; the
    keep-in unavoidably reshapes the gate-1 region because gate 1 (y 0.75, +0.15 randomised)
    sits near the north edge, and a marginal controller cannot fully absorb the reshape.

WHY THE SIM COST IS OVERHEAD, NOT THE TRADE: the sim arena is FULL size, so the keep-in only ever
takes here. The benefit -- not crossing the smaller real edge -- is NOT observable in this sim:
the env terminates on its own hardcoded pos_limit (~+/-3 m, race_core.py _disabled_drones), NOT
on the config's env.track.safety_limits, so shrinking that config did nothing to v11 (byte-equal
result). Validate the OOB benefit on hardware, or by setting the env's actual pos_limit.

TUNING for the real arena:
- ARENA_INSET (main knob, user-set 0.10): how much smaller the real arena is per side. The drone
  is held to ~keep-in wall + a few cm of inertial overshoot, so set inset >= the real gap PLUS a
  safety margin for that overshoot (e.g. gap 0.10 + margin 0.10 -> inset 0.20) for a hard keep-in.
- For tighter containment accept the reliability cost: turn on the speed cap (lower V_ARENA toward
  ~2.0-2.5), and/or widen ARENA_RAMP_M (but keep onset clear of the gates -- onset y ~ 1.10 at the
  defaults already sits only ~0.2 m above a +0.15 m-randomised gate 1).
- For less perturbation accept looser containment: raise ARENA_INSET's wall outward / lower
  W_ARENA. W_ARENA 2000 measured WORSE (stiff push destabilises the gate-1->2 transition).
"""

from __future__ import annotations

# Re-export every v11 knob unchanged (tunnel geometry, reveal caps, de-spiked contouring,
# the v10.x budget/launch/anchor/solver settings) -- and the whole ledger beneath it.
from lsy_drone_racing.control.controller_core_v11.cockpit import *  # noqa: F401,F403

# --- Arena keep-in soft barrier ---------------------------------------------------------------
# The sim safety box (env.track.safety_limits in the configs). The real arena is smaller; the
# barrier shrinks this box by ARENA_INSET per side to get the effective keep-in wall.
ARENA_X_MIN = -2.5
ARENA_X_MAX = 2.5
ARENA_Y_MIN = -1.5
ARENA_Y_MAX = 1.5

# MAIN KNOB: how much smaller the real arena is per side than the sim safety box. The soft
# keep-in wall sits ARENA_INSET inside each sim wall (x -> +/-2.4, y -> +/-1.4 at 0.10 m).
# Raise it if the real arena is more than 0.10 m tighter per side, or to fly more conservatively.
ARENA_INSET = 0.10

# Width (m) of the ramp band BEFORE the keep-in wall over which the penalty builds from zero,
# so the drone is nudged inward early instead of slammed at the wall. Wider = gentler/earlier
# pressure but eats more of the usable region; narrower = stiffer wall. The level2 racing region
# stays clear of the onset at 0.25 (onset x ~ +/-2.15, y ~ +/-1.15).
ARENA_RAMP_M = 0.30

# Smoothness (m) of the spline soft-clip into the keep-in box (controller_core_v15.arc_path): the reference
# is saturated into the box with this transition width. Small -> nearer a hard clip (tighter to
# the wall) but still smooth; too large -> pulls the interior in unnecessarily. 0.08 keeps the
# flattened bulge ~C-infinity so the refit spline has no curvature spike at the wall.
ARENA_CLIP_BETA = 0.08

# Penalty weight (m^-2): cost += W_ARENA * depth_into_band**2 per horizontal axis/side. The
# progress reward is -MU*vth ~ -4.8 at cruise; at 1000 the barrier reaches 1000 * 0.30**2 = 90 at
# the keep-in wall and grows quadratically beyond it -- a firm push-in, still under the tunnel
# slacks (1e3/1e2) so a genuine gate pass near an edge is never starved. W_ARENA 2000 measured
# WORSE on level2 (the stiffer push destabilises the gate-1 -> gate-2 transition); 1000 is the knee.
W_ARENA = 1000.0

# Border SPEED CAP (m/s), OPT-IN -- default OFF (99 > V_THETA_MAX 3.2, so it never binds). Lower
# it (toward ~2.0-2.5) to cap the curvature/reveal profile wherever the reference enters the ramp
# band, so the drone arrives slow enough to turn inside the wall and the residual inertial
# overshoot shrinks (measured drone y ~1.43 -> ~1.31 at 2.5). COSTS RELIABILITY: it paces the
# whole gate-1 exit and the re-accel into the gate-2 fold (measured 40-45% -> 25-40% finish on
# level2). Use only when containment must be tighter than the spline clip + barrier give alone.
V_ARENA = 99.0
