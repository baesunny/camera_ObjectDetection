"""Baseline YOLO detection with CLAHE preprocessing and 3x3 grid overlay."""

import sys
import time
from pathlib import Path

import albumentations as A
import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_YOLO_MODEL

GRID_COLS = ("Right", "Center", "Left")
GRID_ROWS = ("Top", "Mid", "Bottom")


def improve_frame(frame):
    transform = A.Compose([A.CLAHE(clip_limit=4.0, p=1)])
    return transform(image=frame)["image"]


def identify_grid_loc(xmin, ymin, xmax, ymax, width, height):
    grid_width = width // 3
    grid_height = height // 3
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    col = GRID_COLS[min(int(center_x / grid_width), 2)]
    row = GRID_ROWS[min(int(center_y / grid_height), 2)]
    return f"{row}-{col}"


def main():
    model = YOLO(DEFAULT_YOLO_MODEL)
    cap = cv2.VideoCapture(0)
    prev_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        new_frame_time = time.time()
        improved_frame = improve_frame(frame)
        results = model(improved_frame)

        fps = 1 / max(new_frame_time - prev_frame_time, 1e-6)
        prev_frame_time = new_frame_time

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()
            detections = sorted(zip(boxes, scores, labels), key=lambda x: x[1], reverse=True)[:3]

            for box, score, label in detections:
                if score < 0.5:
                    continue
                xmin, ymin, xmax, ymax = map(int, box)
                label_name = model.names[int(label)]
                grid_location = identify_grid_loc(xmin, ymin, xmax, ymax, width, height)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    f"{label_name}: {score:.2f} ({grid_location})",
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )

        for x in range(1, 3):
            cv2.line(frame, (x * width // 3, 0), (x * width // 3, height), (0, 0, 255), 2)
        for y in range(1, 3):
            cv2.line(frame, (0, y * height // 3), (width, y * height // 3), (0, 0, 255), 2)

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Baseline YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
