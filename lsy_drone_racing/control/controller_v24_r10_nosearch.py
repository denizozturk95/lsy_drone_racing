"""controller_v24_r10_nosearch — v24_r10 with the blind spiral search disabled.

v24_r10 flies TAKEOFF -> SEARCH (blind spiral to reveal gates) -> NAVIGATE. Now that
randomize=true hands nominal gate positions within ~0.2 m of true, the search is redundant:
NAVIGATE (v10.5 core) already reads gates from obs (nominal until sensed, true after), so it
can race the nominals directly and refine on reveal. This variant skips the spiral entirely and
keeps v24's obstacle-constrained navigate core. Tests search-on vs search-off head to head.
"""

from __future__ import annotations

from lsy_drone_racing.control.controller_v24_r10 import ControllerV24R10


class ControllerV24R10NoSearch(ControllerV24R10):
    """v24_r10 without the search pass: takeoff -> navigate straight from the nominal layout."""

    def _build_search_curve(self, frame) -> None:
        """No spiral. Zero-length search -> SEARCH is 'done' on entry -> NAVIGATE the same tick.

        The mode loop in ControllerV17.compute_control checks
        ``self._search_t >= self._search_t_total - 1e-3`` (0 >= -1e-3 -> True) and falls through.
        """
        self._search_curve = None
        self._search_t = 0.0
        self._search_t_total = 0.0
