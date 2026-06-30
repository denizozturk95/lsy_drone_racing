"""Global-planner drone racing controller package for controller_v8.

v8 combines pieces of v6 and v7 for the known-track deployment case (the whole course is
scanned before flight): v6's global cubic-spline planner through all gates from t=0 and
its dedicated vertical takeoff phase, plus v7's gate-post funnels and canonical +x gate
crossing (orient_gates_to_travel=False). v7's SEARCH/discovery code is dropped on purpose,
since it only helps on genuinely unknown tracks and hurts when the layout is known.

Tuning flows one way: cockpit.py -> settings.py dataclass defaults -> the modules read
from settings. The shared v6 cockpit (controller_cockpit) is never imported here.
"""
