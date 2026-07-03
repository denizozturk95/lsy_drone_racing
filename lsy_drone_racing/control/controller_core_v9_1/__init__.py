"""MPCC controller package for controller_v9.1.

v9.1 is v9 plus a fix for the mid-track stall: in v9 the MPCC reference was slaved to a
forward-windowed nearest-point projection of the drone, so at sharp turns and loops the
drone could overshoot, the projection would stop advancing, the receding reference would
freeze, and the controller settled into a permanent hover. v9.1 adds a progress governor
that guarantees the reference always creeps forward (bounded so it can't outrun a genuinely
blocked drone), which breaks that fixed point. It also hardens the MPCC solver. Everything
else (planner, takeoff, MPCC weights/limits) is inherited from v8/v9; only the governor
knobs live in controller_core_v9_1.cockpit.
"""
