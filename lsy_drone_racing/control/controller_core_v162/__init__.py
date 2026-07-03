"""v162 core: v16's tunnel MPCC made robust to a failed SQP-RTI step.

Only the Python-side solve() recovery changes; the OCP, its dimensions, and the compiled
``controller_v11`` solver are v11/v16's, reused unchanged (no re-codegen). REQUIRES the acados
environment.
"""
