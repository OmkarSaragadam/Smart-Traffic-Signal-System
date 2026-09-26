"""
config.py
---------
Central configuration for the Smart Traffic AI system.
Every tunable constant lives here so the rest of the codebase never
hard-codes a "magic number".
"""

import os

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
DATABASE_DIR = os.path.join(BASE_DIR, "database")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

for _dir in (MODELS_DIR, VIDEOS_DIR, DATABASE_DIR, OUTPUTS_DIR, REPORTS_DIR, LOGS_DIR):
    os.makedirs(_dir, exist_ok=True)

DATABASE_PATH = os.path.join(DATABASE_DIR, "traffic.db")

# ----------------------------------------------------------------------
# YOLO MODEL
# ----------------------------------------------------------------------
# Any Ultralytics YOLOv8 checkpoint works. "yolov8n.pt" is downloaded
# automatically by the `ultralytics` package on first run if not present
# in MODELS_DIR.
YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, YOLO_MODEL_NAME)

CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
DEVICE = "cpu"  # set to "0" (string) to use first CUDA GPU if available

# Tracker config shipped with ultralytics (bytetrack.yaml / botsort.yaml)
TRACKER_CONFIG = "bytetrack.yaml"

# ----------------------------------------------------------------------
# VEHICLE CLASSES
# ----------------------------------------------------------------------
# COCO class names (as returned by the stock YOLOv8 model) that we treat
# as "vehicles". Auto-rickshaw has no dedicated COCO class, so unless a
# custom-trained weight file is supplied it is not detected by the base
# model — the hook is left in place for a fine-tuned model.
VEHICLE_CLASSES = {
    "car": "Car",
    "motorcycle": "Motorcycle",
    "bus": "Bus",
    "truck": "Truck",
    "bicycle": "Bicycle",
    "auto_rickshaw": "Auto Rickshaw",  # only present in a custom model
}

# Emergency vehicle classes. The stock COCO model cannot distinguish an
# ambulance from any other van/truck, so this is implemented as a hook
# for a custom-trained model (see detector.py -> EmergencyClassifier).
EMERGENCY_CLASSES = {
    "ambulance": "Ambulance",
    "fire_truck": "Fire Truck",
    "police_vehicle": "Police Vehicle",
}

# ----------------------------------------------------------------------
# TRAFFIC DENSITY THRESHOLDS
# ----------------------------------------------------------------------
DENSITY_THRESHOLDS = {
    "Low": (0, 10),
    "Medium": (11, 25),
    "High": (26, 45),
    "Very High": (46, float("inf")),
}

# ----------------------------------------------------------------------
# SIGNAL TIMING
# ----------------------------------------------------------------------
MIN_GREEN_TIME = 15        # seconds
MAX_GREEN_TIME = 45        # seconds
GREEN_TIME_PER_VEHICLE = 0.5
YELLOW_TIME = 3             # seconds, fixed
ALL_RED_CLEARANCE = 1       # seconds, safety gap between phases

# ----------------------------------------------------------------------
# INTERSECTION
# ----------------------------------------------------------------------
ROADS = ["North", "South", "East", "West"]

# ----------------------------------------------------------------------
# ACCIDENT / STATIONARY-TRAFFIC DETECTION
# ----------------------------------------------------------------------
STATIONARY_SPEED_THRESHOLD_PX = 3.0     # px/frame movement considered "not moving"
STATIONARY_TIME_THRESHOLD_SEC = 60      # how long before we flag "Possible Accident"

# ----------------------------------------------------------------------
# NIGHT / LOW-LIGHT DETECTION
# ----------------------------------------------------------------------
NIGHT_BRIGHTNESS_THRESHOLD = 60   # mean grayscale pixel value below this => "night"
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID_SIZE = (8, 8)

# ----------------------------------------------------------------------
# MISC
# ----------------------------------------------------------------------
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
LOG_LEVEL = "INFO"
