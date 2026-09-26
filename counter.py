"""
counter.py
----------
Implements the virtual counting line. Each unique track ID is counted
exactly once, the moment its centroid crosses the configured line
segment, so a vehicle sitting still/near the line, or moving back and
forth in detection jitter, never gets double-counted.
"""

from collections import defaultdict
from typing import Tuple

from utils import get_logger, has_crossed_line

logger = get_logger(__name__)


class LineCounter:
    """
    line_start / line_end define the virtual counting line in pixel
    coordinates of the (resized) frame, e.g. ((0, 300), (960, 300))
    for a horizontal line roughly mid-frame.
    """

    def __init__(self, line_start: Tuple[int, int], line_end: Tuple[int, int], road_name: str = ""):
        self.line_start = line_start
        self.line_end = line_end
        self.road_name = road_name
        self.total_count = 0
        self.counts_by_class = defaultdict(int)
        self._counted_ids = set()

    def update(self, detections, track_manager):
        """
        Call once per frame. For every detection whose track has at least
        two recorded positions, check whether it just crossed the line;
        if so and it hasn't been counted before, increment counters.
        """
        newly_counted = []
        for det in detections:
            if det.track_id in self._counted_ids:
                continue

            state = track_manager.get_state(det.track_id)
            if state is None or len(state.positions) < 2:
                continue

            prev_point = state.positions[-2]
            curr_point = state.positions[-1]

            if has_crossed_line(prev_point, curr_point, self.line_start, self.line_end):
                self._counted_ids.add(det.track_id)
                self.total_count += 1
                self.counts_by_class[det.class_name] += 1
                newly_counted.append(det)
                logger.info(
                    f"[{self.road_name}] Vehicle #{det.track_id} ({det.class_name}) "
                    f"crossed line. Running total = {self.total_count}"
                )

        return newly_counted

    def reset(self):
        """Reset counts, e.g. at the start of a new signal cycle window."""
        self.total_count = 0
        self.counts_by_class.clear()
        self._counted_ids.clear()

    def summary(self) -> dict:
        return {
            "road": self.road_name,
            "total": self.total_count,
            "by_class": dict(self.counts_by_class),
        }
