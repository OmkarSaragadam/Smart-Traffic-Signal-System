"""
tracker.py
----------
detector.py already assigns a stable track ID to every vehicle each frame
(via Ultralytics' built-in ByteTrack integration). This module consumes
those IDs and keeps a short rolling history of centroid positions per
track — the data structure that both counter.py (line-crossing) and
the accident detector (stationary-vehicle check) build on.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple

import config
from utils import get_logger

logger = get_logger(__name__)

HISTORY_LEN = 60  # ~2 seconds at 30fps; enough for speed/stationary checks


@dataclass
class TrackState:
    class_name: str
    positions: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=HISTORY_LEN))
    last_seen_frame: int = 0
    stationary_frames: int = 0
    counted: bool = False  # has this track already been counted at the line?


class TrackManager:
    """Keeps per-track_id state alive across frames and prunes stale tracks."""

    def __init__(self, max_missed_frames: int = 30):
        self.tracks: Dict[int, TrackState] = {}
        self.max_missed_frames = max_missed_frames
        self.frame_index = 0

    def update(self, detections):
        """Call once per frame with the list of Detection objects seen."""
        self.frame_index += 1
        seen_ids = set()

        for det in detections:
            seen_ids.add(det.track_id)
            state = self.tracks.get(det.track_id)
            if state is None:
                state = TrackState(class_name=det.class_name)
                self.tracks[det.track_id] = state

            prev_centroid = state.positions[-1] if state.positions else None
            state.positions.append(det.centroid)
            state.last_seen_frame = self.frame_index

            if prev_centroid is not None:
                dx = det.centroid[0] - prev_centroid[0]
                dy = det.centroid[1] - prev_centroid[1]
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist < config.STATIONARY_SPEED_THRESHOLD_PX:
                    state.stationary_frames += 1
                else:
                    state.stationary_frames = 0

        self._prune(seen_ids)

    def _prune(self, seen_ids):
        stale = [
            tid for tid, st in self.tracks.items()
            if self.frame_index - st.last_seen_frame > self.max_missed_frames
        ]
        for tid in stale:
            del self.tracks[tid]

    def get_state(self, track_id: int) -> TrackState:
        return self.tracks.get(track_id)

    def stationary_track_ids(self, fps: float):
        """Return track IDs that have been (near-)stationary long enough
        to be considered a possible accident/obstruction."""
        threshold_frames = int(config.STATIONARY_TIME_THRESHOLD_SEC * max(fps, 1))
        return [
            tid for tid, st in self.tracks.items()
            if st.stationary_frames >= threshold_frames
        ]
