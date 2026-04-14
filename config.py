"""
SairAI — Global Configuration
=============================
All tuneable knobs in one place. Override via environment variables or
by editing this file directly.
"""

import os
import json
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
OUTPUT_DIR = DATA_DIR / "output"
LANES_DIR = DATA_DIR / "lanes"

# ──────────────────────────────────────────────
# Detection (YOLOv8 + SAHI)
# ──────────────────────────────────────────────
YOLO_MODEL = os.getenv("SAIRAI_YOLO_MODEL", "yolov8x.pt")
YOLO_CONFIDENCE = float(os.getenv("SAIRAI_YOLO_CONF", "0.10"))
YOLO_IOU_THRESHOLD = float(os.getenv("SAIRAI_YOLO_IOU", "0.45"))
YOLO_DEVICE = os.getenv("SAIRAI_DEVICE", "mps")
YOLO_IMGSZ = int(os.getenv("SAIRAI_YOLO_IMGSZ", "1280"))

USE_SAHI = os.getenv("SAIRAI_USE_SAHI", "true").lower() == "true"
SAHI_SLICE_SIZE = int(os.getenv("SAIRAI_SAHI_SLICE", "640"))
SAHI_OVERLAP_RATIO = float(os.getenv("SAIRAI_SAHI_OVERLAP", "0.25"))

# COCO class IDs we care about (vehicles only)
VEHICLE_CLASS_IDS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ──────────────────────────────────────────────
# Tracking (ByteTrack via supervision)
# ──────────────────────────────────────────────
TRACKER_MATCH_THRESH = float(os.getenv("SAIRAI_TRACK_THRESH", "0.20"))
TRACKER_LOST_FRAMES = int(os.getenv("SAIRAI_TRACK_LOST", "30"))
TRACKER_MIN_HITS = int(os.getenv("SAIRAI_TRACK_MIN_HITS", "3"))

# ──────────────────────────────────────────────
# Homography / real-world calibration
# ──────────────────────────────────────────────
HOMOGRAPHY_REFERENCE_POINTS = [
    (200, 150, 0.0, 0.0),
    (600, 150, 12.0, 0.0),
    (600, 450, 12.0, 20.0),
    (200, 450, 0.0, 20.0),
]

LANE_WIDTH_M = 3.65

# ──────────────────────────────────────────────
# Traffic metrics
# ──────────────────────────────────────────────
SPEED_STATIONARY_THRESHOLD_KPH = 5.0
METRICS_ROLLING_WINDOW_SEC = 60
FPS_OVERRIDE = None

# ──────────────────────────────────────────────
# Lane congestion
# ──────────────────────────────────────────────
# Ratio of (sum of vehicle bbox areas) / (lane polygon area).
# Above this threshold the lane is "congested" (red).
CONGESTION_THRESHOLD = float(os.getenv("SAIRAI_CONGESTION_THRESH", "0.35"))

# ──────────────────────────────────────────────
# Dashboard / API
# ──────────────────────────────────────────────
API_HOST = os.getenv("SAIRAI_HOST", "0.0.0.0")
API_PORT = int(os.getenv("SAIRAI_PORT", "8000"))
CORS_ORIGINS = ["*"]

# ──────────────────────────────────────────────
# Privacy
# ──────────────────────────────────────────────
BLUR_LICENSE_PLATES = True

# ──────────────────────────────────────────────
# Resize
# ──────────────────────────────────────────────
RESIZE_RATIO = 1

# ──────────────────────────────────────────────
# Lane helpers
# ──────────────────────────────────────────────
def load_lanes(video_name: str) -> list:
    """
    Load lane definitions for a video from data/lanes/<video_name>.json.

    Each lane is a dict:
        {
            "id": "inbound_1",
            "label": "Inbound Lane 1",
            "direction": "inbound" | "outbound",
            "polygon": [[x1,y1], [x2,y2], ...]   # pixel coords
        }
    """
    lanes_file = LANES_DIR / f"{video_name}.json"
    if lanes_file.exists():
        with open(lanes_file) as f:
            return json.load(f)
    return []


def save_lanes(video_name: str, lanes: list):
    """Persist lane definitions for a video."""
    LANES_DIR.mkdir(parents=True, exist_ok=True)
    lanes_file = LANES_DIR / f"{video_name}.json"
    with open(lanes_file, "w") as f:
        json.dump(lanes, f, indent=2)