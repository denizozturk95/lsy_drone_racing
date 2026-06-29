"""Tunable constants (the "cockpit") for the controller_v10_5 controller.

v10.5 = v10.4 (fast launch: mini-takeoff, hot launch ramp, honest cold start, reactive gate caps)
+ v10.2's DYNAMICS-AWARE progress anchor. Every racing/launch knob -- speed budget, contouring
weights, ramp, takeoff, reactive caps, obstacle keep-out -- is inherited UNCHANGED from v10.4
(see controller_core_v10_4.cockpit and, beneath it, the whole v10.x ledger). The only new knob is
PROJ_BAND_M, the half-width of the geometric correction applied around the solver's predicted
progress.

Why the anchor is back: v10.4 still anchors progress with the GLOBAL geometric projection
(``self._path.project``), the exact mechanism v10.2 proved can teleport the anchor ~1-2 m across
a path fold on a sharp slalom and skip the gate. v10.4's 9/10 on the sharp-slalom edge track came
from LAUNCH fixes; the fold-teleport failure mode is structurally still present mid-race. v10.5
re-introduces v10.2's cure: anchor progress to the solver's own predicted progress
(dynamics-feasible, advances at the bounded rate vth, so it cannot teleport) and let a geometric
search correct it only within +/- PROJ_BAND_M.

Tuning PROJ_BAND_M (does NOT transfer blindly from v10.2): v10.2 measured 0.6 m at v10.1's
~8.3 s pace -- just above the legitimate per-step fold advance (~0.7 m/step) and below the fold
self-approach gap (~1 m). v10.4 (hence v10.5) flies faster, and the legitimate per-step advance
scales with speed, so start at 0.6, log the max single-step anchor motion on the slalom, and
sweep {0.6, 0.7, 0.8} only if the telemetry shows clamping (anchor pinned at the band edge in
runs that finish). Above ~1.0 the teleport returns -- hard upper bound from v10.2's ledger.

MEASURED LEDGER (paired-seed, scripts/compare_v10_5.py, SAME track draws as v10.4; ~±2/20 noise).
PROJ_BAND_M = 0.6 was NOT swept: across every track below the anchor band-edge rate was 0.0% and
the worst single-step anchor jump was 0.29-0.71 m (v10.2's clean signature ~0.7 m; a fold teleport
is ~2.0 m) with ZERO jumps > 1 m in any episode -- the band never clamped, so 0.6 m stands.
  level2 (seeds 42/7/123 × 20):  v10.4 52/60 @ 8.01 s -> v10.5 54/60 @ 8.13 s  (+2 finishes, time
                                 within paired noise; the predicted-progress anchor advances at the
                                 bounded rate vth so it lags the geometric pull-ahead by ~0.1 s.)
  level2_sharp_slalom (seed 42 × 20):  v10.4 15/20 -> v10.5 14/20 (within noise) -- but v10.5 keeps
                                 the STRUCTURAL no-teleport guarantee v10.4 cannot.
  extra tracks (seed 42 × 10 each: hairpin/boxloop/inout_dive/double/single, stress_synth_01-03):
                                 v10.4 69/80 vs v10.5 69/80 aggregate -- tied, every per-track delta
                                 within 10-run noise (inout_double +3 for v10.5; boxloop/dive/s02
                                 -1). No regression anywhere; anchor telemetry clean on all.
Net: v10.5 is a no-cost robustness merge -- it adds v10.2's fold protection to v10.4's fast launch
with finish >= v10.4 on the level2 gate and time within noise. It does NOT lower the lap-time floor
(the +/-0.15 m reveal ceiling is untouched); that is v10.51's (mass axis) and v11's job.
"""

from __future__ import annotations

# Re-export every v10.4 knob unchanged (budget 3.2/8.5, ramp 0.25/2.4, mini-takeoff, reactive
# gate caps, contouring, solver...) -- and, transitively, every v10.3/v10.1 knob beneath it.
from lsy_drone_racing.control.controller_core_v10_4.cockpit import *  # noqa: F401,F403

# --- Dynamics-aware progress anchor (the v10.2 mechanism, re-applied on the v10.4 base) ---
# Half-width (m) of the geometric correction applied around the solver's predicted progress. Just
# above the legitimate per-step fold advance and below the fold self-approach gap (~1 m), so the
# anchor tracks the drone through the fold but can never teleport across it. See docstring.
PROJ_BAND_M = 0.6
