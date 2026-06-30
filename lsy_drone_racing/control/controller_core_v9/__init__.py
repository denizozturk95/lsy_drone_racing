"""MPCC drone racing controller package for controller_v9.

v9 keeps the parts of controller v8 that work well (the global, gate-aware, obstacle/funnel-routed
spline, used here purely as path geometry, and the dedicated vertical takeoff) and replaces
v8's hand-tuned speed caps and cascaded-PID tracker with a model-predictive contouring
controller (controller_core_v9.mpcc). The MPCC flies as fast as the drone's thrust/tilt limits allow
along whatever path it's given, so speed generalises to arbitrary track geometry instead of
being a tuned constant. Only the MPCC knobs live in controller_core_v9.cockpit; the path and takeoff
tuning is inherited from controller_core_v8.
"""
