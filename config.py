"""Project-wide configuration. Override paths via environment variables."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASETS_DIR = PROJECT_ROOT / "datasets"

# General-purpose COCO pretrained model (included in repo)
DEFAULT_YOLO_MODEL = os.environ.get(
    "YOLO_MODEL_PATH",
    str(MODELS_DIR / "yolov8s.pt"),
)

# Custom fine-tuned face detection model (train or download separately)
FACE_YOLO_MODEL = os.environ.get(
    "FACE_MODEL_PATH",
    str(MODELS_DIR / "face_best.pt"),
)

# Camera defaults
CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "720"))
INFERENCE_SIZE = (640, 640)
