"""Real-time face detection with Korean voice guidance (main demo)."""

import sys
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import pyttsx3
import torch
from torchvision.transforms import functional as F
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CAMERA_HEIGHT, CAMERA_WIDTH, FACE_YOLO_MODEL, INFERENCE_SIZE
from src.detection.elements import detect, head_pose
from src.detection.utils import intersection_over_union

CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.8
BOX_UPDATE_INTERVAL = 5
STABLE_THRESHOLD_TIME = 0.8
CLASS_LIST = ["face"]
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

engine = pyttsx3.init()


def speak(text):
    engine.say(text)
    engine.runAndWait()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(FACE_YOLO_MODEL)
    model.to(device)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=CONFIDENCE_THRESHOLD,
        min_tracking_confidence=CONFIDENCE_THRESHOLD,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    x_scale = CAMERA_WIDTH / INFERENCE_SIZE[0]
    y_scale = CAMERA_HEIGHT / INFERENCE_SIZE[1]

    last_change_time = time.time()
    last_bbox = None
    state_loc_variable = 1
    num_frame = 0

    while True:
        start = time.time()
        success, frame = cap.read()
        num_frame += 1

        if not success:
            print("Camera error")
            break

        pose_text = head_pose(image=frame, face_mesh=face_mesh)
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, pose_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

        frame_resized = cv2.resize(frame, INFERENCE_SIZE)
        frame_tensor = F.to_tensor(frame_resized).unsqueeze(0).to(device)
        detection = model.predict(frame_tensor, conf=CONFIDENCE_THRESHOLD)[0].cpu()

        max_area = 0
        max_box = None
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

            area = (xmax - xmin) * (ymax - ymin)
            if area > max_area and label == 0:
                max_area = area
                max_box = [xmin, ymin, xmax, ymax, conf, label]

        if max_box:
            xmin, ymin, xmax, ymax, conf, label = max_box
            new_bbox = [xmin, ymin, xmax, ymax]
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

            if last_bbox is not None:
                iou = intersection_over_union(last_bbox, new_bbox)
                if iou < IOU_THRESHOLD:
                    last_change_time = time.time()
                    state_loc_variable = 1
                elif (
                    time.time() - last_change_time > STABLE_THRESHOLD_TIME
                    and state_loc_variable == 1
                ):
                    pct, loc = detect(xmin, ymin, xmax, ymax, frame)
                    text = f"현재 얼굴 비중은 {pct}%이고 위치는 {loc}입니다."
                    threading.Thread(target=speak, args=(text,), daemon=True).start()
                    last_change_time = time.time()
                    state_loc_variable = 0

            if num_frame % BOX_UPDATE_INTERVAL == 0:
                last_bbox = new_bbox

        elapsed = time.time() - start
        fps = f"FPS: {1 / elapsed:.2f}"
        cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)
        cv2.imshow("Face Detection with Voice", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
