"""Gate-aware time-optimal MPCC package for controller_v10_1.

v10.1 keeps everything that makes v10 work -- v8's gate-aware planner, the vertical takeoff, the
real-time acados SQP-RTI time-optimal MPCC, the -mu*vth progress reward, and the friction-circle
curvature cap -- and adds one mechanism: the contouring weight is no longer a single constant but
a per-stage value that spikes in a window around each gate's arc-position and relaxes on the
straights. On this track the lap time is capped by gate-passing precision under +/-0.15 m gate
randomisation, not by actuator authority, so v10's uniform contouring weight forces a compromise:
raising the speed budget trades contour error for progress everywhere (including at the gate) and
the drone overshoots. Spiking the weight exactly where a gate must be threaded lets v10.1 carry a
higher speed budget at equal-or-better finish (measured 19/20 @ 8.27s vs v10 16/20 @ 8.45s over
20 level2 runs). Acceleration stays the zero-order-hold control, so v10's proven solver structure
is unchanged apart from one extra per-stage parameter. Only the MPCC and how the path is built
change; the planner and takeoff are inherited from v8 via v9/v10.

(An earlier v10.1 tried a thrust-vector slew limit -- acceleration as a state, jerk the control --
and was rejected: the plant takes acceleration as a zero-order-hold command, so a jerk-as-state
model mismatched it and tracked worse than v10. See the package docstrings for details.)
"""
