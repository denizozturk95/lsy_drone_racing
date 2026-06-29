"""Tunable constants (the "cockpit") for the controller_v11 controller.

VERDICT: CONDEMNED ON THIS CHALLENGE -- v10.5 remains the flagship. The tunnel formulation's
speed freedom is REAL (best finished laps 7.24-7.40 s on level2 vs v10.5's 7.36; 6.34 s on
the hairpin vs 6.48), but on 0.4 m openings (~3.3 drone-widths) with +/-0.15 m poses revealed
at 0.7 m (37% of the opening), every probed configuration converts the freed speed into
finish-rate loss and lands BELOW the v10.5 frontier point (19/20 @ 8.08 on the seed-42 draws):

  pure de-paced tunnel (several margin configs) ... 33-50% @ 7.3-7.9  (frame hits: aperture
                                                    corners, exit-climb ceiling rides,
                                                    U-turn outer-frame rotor clips)
  hybrid (v10.1 gate spikes restored) + all caps .. 14/20 @ 8.42     (most robust; the
                                                    spikes/caps/tapers reconstruct v10.5's
                                                    pacing, mid-leg gains get eaten)
  hybrid without all-gate reveal caps ............. 3/6              (gate-2/3 fold fails)
  + curvature-clamped tunnel width ................ 3/6              (folds persist)
  hairpin track (long straights, 10 runs paired) .. 4/10 @ 6.76 vs v10.5's 9/10 @ 6.96
  v11.1 (tunnel + v10.6 smoothed reference) ....... 5/6 @ 9.15      (slow recovery laps)

MECHANISM: MPCC++'s result assumed mm-accurate gate poses (the paper defers pose uncertainty
to future work); its gates are ~5 drone-widths. Here the reveal ceiling -- not the cost
formulation -- binds: wall-riding adds the tunnel offset to every reveal correction, and the
solver's linearised reference lies near path folds, so freedom anywhere near a gate or fold
is paid in crashes. This extends the v10.x nine-mechanism frontier with three more mechanism
classes (hard constraints, reference shortening under de-pacing, and their composition).
KEEP for: tracks with exact gate knowledge (level0/level1-style or scanned deployment) and
as the platform for any future reveal-aware tunnel-width schedule (the open research gap).

ORIGINAL DESIGN NOTE: v11 = v10.5 with the GATE-WEIGHT PACING REPLACED BY TUNNEL CONSTRAINTS
(IMPROVEMENT_PLAN.md Phase 1, after MPCC++ -- Krinner et al., RSS 2024, arXiv:2403.17551).
The measured problem it
attacks: v10.x's MPCC flies 0.5-0.9 m/s below its own curvature profile everywhere because the
gate-spiked contouring weight is a cost-side trade-off that paces the drone continuously (the
solver-pace gap, RESEARCH_BRIEF.md section 3.2; independently diagnosed as an MPCC formulation
pathology by the lab that invented quadrotor MPCC). The fix decouples speed from gate
precision:

1. TUNNEL CONSTRAINTS (controller_core_v11.mpcc + arc_path): predicted positions must stay inside a
   prismatic tunnel around the reference path -- wide on the straights, clipped near
   obstacles, narrowed to the gate opening at each gate plane. Soft constraints (slack
   penalties), so a reveal-jump of the tunnel cannot make the QP infeasible.
2. DE-SPIKED COST: the contouring weight drops to a low constant (W_CONTOUR_BASE 20+20 gate
   spikes -> 2.0 flat); -MU*vth then drives the pace against the tunnel walls instead of
   against a weight schedule.
3. REVEAL CAPS ON EVERY GATE (controller_core_v11.arc_path): the tunnel does NOT remove the information
   constraint -- the gate's true pose still appears only at 0.7 m and correcting +/-0.15 m at
   speed costs ~v^2 (Phase-0 bound: the reveal cap costs only ~0.5 s of lap time, see
   scripts/cpc_bound.py). Every gate's approach window [s_g - 0.7, s_g + 0.3] is capped at
   V_GATE_REVEAL. NOTE this is the "permanent gate cap" the v10.3 ledger condemned -- the
   condemnation does NOT transfer: it was measured against the gate-weight pacing, where the
   drone never reached the cap anyway (cap cost > gain). With the de-spiked cost the drone
   would otherwise cross gates at V_MAX, so the cap becomes the binding *information*
   constraint, exactly what the physics demands (lateral correction of 0.2 m within 0.7 m at
   2.5 m/s needs ~5 m/s^2 of the ~8 available; at 3.2 m/s it needs ~8.4 -- saturated).

Phase-0 context for every number here: time-optimal point-mass bound on level2 nominal is
2.85 s NAVIGATE at our V_MAX budget, 3.36 s with reveal caps, vs ~6.3 s predicted on the
current reference and ~7.6 s flown (v10.5, 54/60 @ 8.13 s incl. takeoff).
"""

from __future__ import annotations

# Re-export every v10.5 knob unchanged (budget 3.2/8.5, ramp, mini-takeoff, anchor band,
# solver settings...) -- and the whole v10.x ledger beneath it.
from lsy_drone_racing.control.controller_core_v10_5.cockpit import *  # noqa: F401,F403

# --- Hybrid contouring cost: free straights, stiff gates ---
# The pure MPCC++ form (constant low weight, tunnel only) was measured UNVIABLE here: our
# openings are ~3.3 drone-widths with a reveal error of 37% of the opening (MPCC++ had ~5
# widths and exact poses), and wall-riding adds the tunnel offset to every reveal correction
# -- 50% finish across three margin configs, failures rotating between aperture corners,
# exit-climb ceiling rides, and outer-frame clips. The hybrid keeps the LOW base weight (the
# de-pacing win: two-thirds of the lap is between tapers) and restores v10.1's PROVEN gate
# spike exactly where reveal corrections land, so the drone is centred and stiffly tracked
# through every opening. The v10.1 ledger's "do NOT drop W_BASE below 20" was measured
# without tunnel constraints; the tunnel now carries the off-gate safety.
W_CONTOUR_BASE = 2.0
W_CONTOUR_GATE = 20.0  # v10.1's gate spike (stacked on the low base)
GATE_SIGMA = 0.5

# --- Tunnel geometry (controller_core_v11.arc_path tables; per-stage params to the OCP) ---
TUNNEL_W_MAX = 0.40  # m, lateral half-width on open straights
TUNNEL_W_GATE = 0.10  # m, lateral half-width at a gate plane. The walls are RIDDEN under
# the progress reward plus ~0.03 m of slack lean (the constraint is
# in m^2 units, centimetre violations are cheap), so the aperture
# must price both in: measured at 0.13 the drone clipped frame bars
# at |offset| ~0.14-0.15 (contact starts at ~0.15 = 0.20 half-opening
# minus the body); at 0.11 corner hits at (0.12, 0.09) remained.
TUNNEL_H_MAX = 0.35  # m, vertical half-height on open straights
TUNNEL_H_GATE = 0.07  # m, vertical half-height at a gate plane (tighter than lateral: the
# clamped-cubic climbs make the TOP bar the measured hit point --
# exit-leg climbs ride the ceiling to z=+0.15 at 0.09+lean -- and the
# body is taller than it is slim with props)
TUNNEL_TAPER_M = 1.00  # m of arc over which the tunnel narrows into each gate. LONGER than
# the gate is wide on purpose: the tunnel lets the drone ride walls
# up to 0.4 m off-centre, but a gate reveal then adds that offset to
# the +/-0.15 m correction (a failure mode MPCC++ never had -- it
# assumed exact poses). Narrowing over the last metre re-centres the
# drone BEFORE the 0.7 m reveal, so the residual correction is the
# reveal delta alone. 0.5 m measured 2-3/6 with frame hits.
TUNNEL_W_MIN = 0.02  # m, floor under the clipping: where obstacles/frames pinch the tunnel
# to nothing, the drone must hug the planned line (which the repair
# loop keeps >= r_obs clear). A 0.10 floor was measured to LET THE
# TUNNEL REACH INTO KEEP-OUTS (tunnel edge ~0.12 m from an obstacle)
# -- the drone stayed inside the tunnel and still hit obstacles.
TUNNEL_OBS_MARGIN = 0.20  # m, obstacle keep-out the tunnel edge must respect (= R_OBS)
TUNNEL_POST_MARGIN = 0.18  # m, keep-out around EVERY gate's frame posts (+/-0.30 lateral of
# each gate centre, passed gates included -- the frames are physical
# whether or not a gate is the target). Covers the bar-to-outer-edge
# span (0.30->0.36) plus rotor reach: at 0.12 a U-turn pass clipped
# gate 2's outer frame at ~0.08 m. At the target gate the opening
# taper (0.10) stays the binding width (0.30-0.18=0.12 > 0.10).
TUNNEL_Z_FLOOR = 0.10  # m, world floor margin the tunnel must respect
TUNNEL_Z_CEIL = 1.90  # m, world ceiling margin
TUNNEL_CURV_FRAC = 0.5  # clamp W <= frac / kappa(s): the OCP linearises the reference as
# pref + tan*(th - thbar), which is only valid within ~1/kappa of the
# path -- at the g2->g3 U-turn fold a 0.4 m tunnel let the drone ride
# where the linearisation lies (measured: repeated gate-3 fails).
# Straights (kappa ~ 0) keep W_MAX; the hairpin (v_min 1.3 -> kappa
# ~ 5/m) tightens to ~0.10 m.

# --- Tunnel slack penalties (the constraints are SOFT: a reveal jump moves the tunnel under
# the predicted horizon and must degrade gracefully, never turn the QP infeasible). Pricier
# than the speed-cap slacks (1e2/1e1): a tunnel violation is a frame/obstacle hit. At 3e2/3e1
# the walls were measurably leaned through by ~0.02 m under the progress reward (the
# constraint is in m^2 units, so centimetre-scale violations are cheap) -- 1e3/1e2 ships. ---
TUNNEL_SLACK_L1 = 1e3
TUNNEL_SLACK_L2 = 1e2

# --- Reveal caps (the information constraint; see module docstring point 3) ---
V_GATE_REVEAL = 2.5  # m/s, approach cap (gate 0 always; all gates iff the flag below)
REVEAL_CAP_ALL = True  # cap EVERY gate's reveal window. Theory (Phase-0 bound) and the
# measured config ladder agree this is v11's most robust setting
# (14/20 vs 3/6 with it off -- without the caps the de-paced legs
# arrive hot at the gate-2/3 folds). Note v10.5 crossed some windows
# at up to 3.18 m/s and survived via spike-centred corrections, so
# the cap also taxes windows v10.5 exploited; see the verdict ledger.
REVEAL_PRE_M = 0.7  # m, window upstream of each gate plane (= sensor range)
REVEAL_POST_M = 0.3  # m, window past the plane (re-accelerate beyond the frame)
# Gate 0 keeps v10.4's WIDE launch window (the obstacle-0 reveal swerve sits in it; ledger:
# launch crashes 5 -> 2 at zero time cost). Applied only while gate 0 is the target.
LAUNCH_CAP_PRE_M = 1.4
