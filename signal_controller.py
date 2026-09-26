"""
signal_controller.py
---------------------
State machine for a 4-way intersection (North / South / East / West).
Only one road is ever GREEN at a time; every phase transitions
GREEN -> YELLOW -> (ALL RED clearance) -> next road's GREEN, cycling
through roads in a round-robin order, unless an emergency-vehicle
override forces an immediate switch.
"""

import time
from enum import Enum
from typing import Dict, Optional

import config
from utils import get_logger, calculate_green_time, classify_density

logger = get_logger(__name__)


class SignalState(Enum):
    RED = "Red"
    YELLOW = "Yellow"
    GREEN = "Green"


class RoadSignal:
    """Per-road signal bookkeeping."""

    def __init__(self, name: str):
        self.name = name
        self.state = SignalState.RED
        self.vehicle_count = 0
        self.density = "Low"
        self.green_time = config.MIN_GREEN_TIME
        self.phase_started_at: Optional[float] = None

    def time_remaining(self) -> float:
        if self.phase_started_at is None:
            return 0.0
        duration = {
            SignalState.GREEN: self.green_time,
            SignalState.YELLOW: config.YELLOW_TIME,
            SignalState.RED: 0,
        }[self.state]
        elapsed = time.time() - self.phase_started_at
        return max(0.0, duration - elapsed)

    def to_dict(self) -> dict:
        return {
            "road": self.name,
            "state": self.state.value,
            "vehicle_count": self.vehicle_count,
            "density": self.density,
            "green_time": self.green_time,
            "time_remaining": round(self.time_remaining(), 1),
        }


class IntersectionController:
    """
    Orchestrates the 4 RoadSignal objects. Call `update_counts()` every
    frame/cycle with the latest vehicle counts, and call `tick()`
    periodically (e.g. once per loop iteration or once per second) to
    advance the state machine.
    """

    def __init__(self, roads=None):
        roads = roads or config.ROADS
        self.signals: Dict[str, RoadSignal] = {r: RoadSignal(r) for r in roads}
        self._order = list(roads)
        self._active_index = 0
        self._emergency_road: Optional[str] = None

        # Kick off the first phase
        first_road = self._order[self._active_index]
        self._start_green(first_road)

    # ------------------------------------------------------------------
    def update_counts(self, road: str, vehicle_count: int):
        """Feed the latest vehicle count for a road (called every frame)."""
        signal = self.signals[road]
        signal.vehicle_count = vehicle_count
        signal.density = classify_density(vehicle_count)
        # Only recompute green_time while the road is NOT currently green,
        # so an already-running green phase isn't shortened/lengthened
        # mid-flight in a jarring way.
        if signal.state != SignalState.GREEN:
            signal.green_time = calculate_green_time(vehicle_count)

    def trigger_emergency(self, road: str):
        """Force an immediate green for `road` (ambulance / fire truck / police)."""
        if self._emergency_road == road:
            return
        logger.warning(f"EMERGENCY VEHICLE DETECTED on {road}. Forcing green override.")
        self._emergency_road = road
        self._force_all_red()
        self._start_green(road, is_emergency=True)

    def clear_emergency(self):
        self._emergency_road = None

    # ------------------------------------------------------------------
    def _force_all_red(self):
        for signal in self.signals.values():
            signal.state = SignalState.RED
            signal.phase_started_at = time.time()

    def _start_green(self, road: str, is_emergency: bool = False):
        for name, signal in self.signals.items():
            if name == road:
                signal.state = SignalState.GREEN
                signal.green_time = (
                    config.MAX_GREEN_TIME if is_emergency else calculate_green_time(signal.vehicle_count)
                )
                signal.phase_started_at = time.time()
            else:
                signal.state = SignalState.RED
                signal.phase_started_at = signal.phase_started_at or time.time()
        self._active_index = self._order.index(road)

    def _advance_to_next_road(self):
        self._active_index = (self._active_index + 1) % len(self._order)
        next_road = self._order[self._active_index]
        self._start_green(next_road)

    # ------------------------------------------------------------------
    def tick(self):
        """
        Advance the state machine. Should be called frequently (e.g. every
        loop iteration in main.py / dashboard.py); it is time-based, not
        call-count based, so calling it often is safe and simply a no-op
        until the current phase's timer actually elapses.
        """
        active_road = self._order[self._active_index]
        active_signal = self.signals[active_road]

        if active_signal.state == SignalState.GREEN and active_signal.time_remaining() <= 0:
            active_signal.state = SignalState.YELLOW
            active_signal.phase_started_at = time.time()

        elif active_signal.state == SignalState.YELLOW and active_signal.time_remaining() <= 0:
            active_signal.state = SignalState.RED
            active_signal.phase_started_at = time.time()
            if self._emergency_road == active_road:
                self.clear_emergency()
            self._advance_to_next_road()

    def snapshot(self) -> dict:
        return {name: sig.to_dict() for name, sig in self.signals.items()}
