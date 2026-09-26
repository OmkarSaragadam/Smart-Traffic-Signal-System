"""
detector.py
-----------
Wraps the Ultralytics YOLOv8 model to provide vehicle detection with
bounding boxes, class labels and confidence scores. Also exposes a
lightweight hook for emergency-vehicle recognition.

NOTE ON EMERGENCY VEHICLES
The stock COCO-pretrained YOLOv8 model (yolov8n.pt / yolov8s.pt ...) has
no "ambulance" / "fire truck" / "police vehicle" classes - COCO simply
doesn't have them. To really detect emergency vehicles you need a model
fine-tuned on a labeled emergency-vehicle dataset. This module is built
so that dropping such a custom .pt file into MODELS_DIR and pointing
config.YOLO_MODEL_NAME at it "just works" with zero code changes.
Until then, EmergencyClassifier uses a colour/heuristic fallback
(siren light-bar colour detection) so the feature still runs end-to-end.
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

import config
from utils import get_logger, is_low_light, enhance_low_light

logger = get_logger(__name__)


@dataclass
class Detection:
    track_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def centroid(self):
        return (int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2))


class VehicleDetector:
    """Loads a YOLOv8 model and runs detection (+ optional tracking)."""

    def __init__(self, model_path: str = None, device: str = None):
        model_path = model_path or config.YOLO_MODEL_NAME
        self.device = device or config.DEVICE
        logger.info(f"Loading YOLO model: {model_path} on device={self.device}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names  # dict: id -> coco class name

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize + enhance frame for night/fog/rain robustness."""
        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        if is_low_light(frame):
            frame = enhance_low_light(frame)
        return frame

    def detect_and_track(self, frame: np.ndarray) -> (np.ndarray, List[Detection]):
        """
        Run detection + ByteTrack tracking on a single frame.
        Returns the (possibly enhanced) frame and a list of Detection objects
        limited to classes we care about (config.VEHICLE_CLASSES).
        """
        frame = self._preprocess(frame)

        results = self.model.track(
            frame,
            persist=True,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            tracker=config.TRACKER_CONFIG,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return frame, detections

        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            return frame, detections

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        track_ids = result.boxes.id.cpu().numpy().astype(int)

        for box, conf, cls_id, tid in zip(boxes, confs, cls_ids, track_ids):
            class_name = self.class_names.get(cls_id, str(cls_id))
            if class_name not in config.VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = box.astype(int)
            detections.append(
                Detection(
                    track_id=int(tid),
                    class_name=config.VEHICLE_CLASSES[class_name],
                    confidence=float(conf),
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                )
            )

        return frame, detections


class EmergencyClassifier:
    """
    Heuristic fallback emergency-vehicle flag: looks for large saturated
    red/blue blobs (siren light bars) near the top of a detected vehicle's
    bounding box. This is NOT a substitute for a properly trained model —
    it exists so the "Emergency Vehicle Detected" feature is demonstrable
    without requiring a custom dataset. Swap in a fine-tuned YOLO head
    (config.EMERGENCY_CLASSES) for production use.
    """

    def __init__(self):
        # HSV ranges for strong red and blue (typical siren colours)
        self.red_lower1 = np.array([0, 120, 120])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 120, 120])
        self.red_upper2 = np.array([180, 255, 255])
        self.blue_lower = np.array([100, 120, 120])
        self.blue_upper = np.array([130, 255, 255])

    def is_emergency_vehicle(self, frame: np.ndarray, detection: Detection) -> bool:
        if detection.class_name not in ("Truck", "Bus", "Car"):
            return False

        x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
        top_h = max(1, int((y2 - y1) * 0.3))
        roi = frame[y1:y1 + top_h, x1:x2]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        red_mask = (
            cv2.inRange(hsv, self.red_lower1, self.red_upper1)
            | cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        )
        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)

        red_ratio = np.count_nonzero(red_mask) / red_mask.size
        blue_ratio = np.count_nonzero(blue_mask) / blue_mask.size

        return (red_ratio > 0.05 and blue_ratio > 0.02) or (red_ratio + blue_ratio) > 0.15
