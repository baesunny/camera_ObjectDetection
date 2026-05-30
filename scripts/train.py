"""Train a YOLOv8 model on a custom face detection dataset."""

import sys
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASETS_DIR

DEFAULT_DATA_YAML = DATASETS_DIR / "face_detection_6" / "data.yaml"


def main():
    data_yaml = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_YAML
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {data_yaml}\n"
            "Download images from Roboflow or Google Drive (see datasets/README.md)."
        )

    model = YOLO("yolov8n.pt")
    model.train(data=str(data_yaml), epochs=30, lr0=0.01)
    print("Training complete. Weights saved under runs/detect/.")


if __name__ == "__main__":
    main()
