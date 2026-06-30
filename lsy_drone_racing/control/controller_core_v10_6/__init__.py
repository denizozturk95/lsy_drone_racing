"""v10.6 planner core: v10.4's gate-funnel chain + guarded waypoint smoothing.

Provides the smoothing ReferenceManager (trajectory.py) and its planner knobs (settings.py).
The OCP/solver are unchanged from v10.5, so v16 reuses the v11 compiled acados solver and only
swaps this planner in. REQUIRES the acados environment.
"""
