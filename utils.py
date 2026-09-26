"""
utils.py
--------
Small, dependency-light helper functions used across the project:
logging setup, density classification, green-time calculation,
geometry helpers (line-crossing test), and image-quality helpers
(night / low-light enhancement).
"""

import logging
import os
import time
from datetime import datetime

import cv2
import numpy as np

import config


# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes to both console and a
    rotating daily log file under config.LOGS_DIR."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured, avoid duplicate handlers
        return logger

    logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    log_file = os.path.join(config.LOGS_DIR, f"{datetime.now():%Y-%m-%d}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# ----------------------------------------------------------------------
# TRAFFIC DENSITY
# ----------------------------------------------------------------------
def classify_density(vehicle_count: int) -> str:
    """Map a vehicle count to a density label using config.DENSITY_THRESHOLDS."""
    for label, (low, high) in config.DENSITY_THRESHOLDS.items():
        if low <= vehicle_count <= high:
            return label
    return "Very High"  # fallback, should not normally hit


# ----------------------------------------------------------------------
# SIGNAL TIMING
# ----------------------------------------------------------------------
def calculate_green_time(vehicle_count: int) -> int:
    """
    Green Time = Minimum Time + Vehicle Count * 0.5
    Clamped to [MIN_GREEN_TIME, MAX_GREEN_TIME].
    """
    raw_time = config.MIN_GREEN_TIME + vehicle_count * config.GREEN_TIME_PER_VEHICLE
    clamped = max(config.MIN_GREEN_TIME, min(config.MAX_GREEN_TIME, raw_time))
    return int(round(clamped))


# ----------------------------------------------------------------------
# GEOMETRY: virtual counting line crossing
# ----------------------------------------------------------------------
def point_side_of_line(point, line_start, line_end) -> float:
    """
    Returns a signed value indicating which side of the infinite line
    (line_start -> line_end) the point lies on. 0 means exactly on the line.
    Used to detect when a tracked centroid crosses the virtual counting line
    (sign flips between consecutive frames).
    """
    (x, y) = point
    (x1, y1) = line_start
    (x2, y2) = line_end
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)


def has_crossed_line(prev_point, curr_point, line_start, line_end) -> bool:
    """True if the segment prev_point->curr_point crosses the counting line,
    determined by a sign change of point_side_of_line between frames."""
    prev_side = point_side_of_line(prev_point, line_start, line_end)
    curr_side = point_side_of_line(curr_point, line_start, line_end)
    return prev_side == 0 or curr_side == 0 or (prev_side > 0) != (curr_side > 0)


# ----------------------------------------------------------------------
# IMAGE QUALITY: night / low-light enhancement
# ----------------------------------------------------------------------
def is_low_light(frame: np.ndarray) -> bool:
    """Quick brightness check on a BGR frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < config.NIGHT_BRIGHTNESS_THRESHOLD


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    """
    Improve visibility in dark / foggy / rainy frames using CLAHE
    (Contrast Limited Adaptive Histogram Equalization) on the L channel
    of LAB color space, plus mild denoising. Cheap enough to run every
    frame on CPU.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT, tileGridSize=config.CLAHE_TILE_GRID_SIZE
    )
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 3, 3, 7, 15)
    return enhanced


# ----------------------------------------------------------------------
# TIME HELPERS
# ----------------------------------------------------------------------
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class FPSCounter:
    """Simple rolling FPS counter."""

    def __init__(self, window: int = 30):
        self.window = window
        self._timestamps = []

    def tick(self) -> float:
        self._timestamps.append(time.time())
        if len(self._timestamps) > self.window:
            self._timestamps.pop(0)
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0
