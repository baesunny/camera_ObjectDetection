"""Real-time face detection with head pose overlay."""

import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import torch
from torchvision.transforms import functional as F
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CAMERA_HEIGHT, CAMERA_WIDTH, FACE_YOLO_MODEL, INFERENCE_SIZE
from src.detection.elements import head_pose

CONFIDENCE_THRESHOLD = 0.6
CLASS_LIST = ["face"]
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(FACE_YOLO_MODEL)
    model.to(device)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    x_scale = CAMERA_WIDTH / INFERENCE_SIZE[0]
    y_scale = CAMERA_HEIGHT / INFERENCE_SIZE[1]

    while True:
        start = time.time()
        success, frame = cap.read()
        if not success:
            print("Camera error")
            break

        pose_text = head_pose(image=frame, face_mesh=face_mesh)
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, pose_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

        frame_resized = cv2.resize(frame, INFERENCE_SIZE)
        frame_tensor = F.to_tensor(frame_resized).unsqueeze(0).to(device)
        detection = model.predict(frame_tensor, conf=CONFIDENCE_THRESHOLD)[0].cpu()

        for data in detection.boxes.data.tolist():
            xmin, ymin, xmax, ymax, conf, label = (
                int(data[0]),
                int(data[1]),
                int(data[2]),
                int(data[3]),
                float(data[4]),
                int(data[5]),
            )
            xmin = int(xmin * x_scale)
            ymin = int(ymin * y_scale)
            xmax = int(xmax * x_scale)
            ymax = int(ymax * y_scale)

            if label != 0:
                continue

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.putText(
                frame,
                f"{CLASS_LIST[label]} {round(conf, 2)}",
                (xmin, ymin - 10),
                cv2.FONT_ITALIC,
                1,
                WHITE,
                2,
            )

        elapsed = time.time() - start
        fps = f"FPS: {1 / elapsed:.2f}"
        cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
