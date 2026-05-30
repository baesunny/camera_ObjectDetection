"""Real-time COCO object detection with Korean voice guidance."""

import sys
import threading
import time
from pathlib import Path

import cv2
import pyttsx3
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CAMERA_HEIGHT, CAMERA_WIDTH, DEFAULT_YOLO_MODEL
from src.detection.elements import box_to_pct, detect
from src.detection.utils import iou_multiple

CONFIDENCE_THRESHOLD = 0.6
IOU_THRESHOLD = 0.8
BOX_UPDATE_INTERVAL = 10
STABLE_THRESHOLD_TIME = 1.0
CHANGE_THRESHOLD_TIME = 0.5
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

CLASS_LIST_KOREAN = [
    "사람", "자전거", "자동차", "오토바이", "비행기", "버스", "기차", "트럭", "보트", "신호등",
    "소화전", "정지 신호", "주차 미터기", "벤치", "새", "고양이", "개", "말", "양", "소",
    "코끼리", "곰", "얼룩말", "기린", "배낭", "우산", "핸드백", "넥타이", "여행가방", "프리스비",
    "스키", "스노보드", "스포츠 공", "연", "야구 방망이", "야구 글러브", "스케이트보드", "서핑보드",
    "테니스 라켓", "병", "와인 잔", "컵", "포크", "나이프", "숟가락", "그릇", "바나나", "사과",
    "샌드위치", "오렌지", "브로콜리", "당근", "핫도그", "피자", "도넛", "케이크", "의자", "소파",
    "화분", "침대", "식탁", "화장실", "TV", "노트북", "마우스", "리모컨", "키보드", "휴대전화",
    "전자레인지", "오븐", "토스터", "싱크대", "냉장고", "책", "시계", "꽃병", "가위", "테디 베어",
    "헤어 드라이어", "칫솔",
]

engine = pyttsx3.init()


def speak(text):
    local_engine = pyttsx3.init()
    local_engine.say(text)
    local_engine.runAndWait()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(DEFAULT_YOLO_MODEL)
    class_list = model.names

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    last_change_time = time.time()
    last_stable_time = time.time()
    last_bboxes = None
    state_loc_variable = 1
    change_variable = 0
    num_frame = 0

    while True:
        start = time.time()
        success, frame = cap.read()
        num_frame += 1

        if not success:
            print("Camera error")
            break

        frame = cv2.flip(frame, 1)
        results = model.predict(frame, conf=CONFIDENCE_THRESHOLD)

        detections = []
        new_bboxes = []
        if results:
            for data in results:
                boxes = data.boxes.xyxy.cpu().numpy()
                confs = data.boxes.conf.cpu().numpy()
                labels = data.boxes.cls.cpu().numpy()
                pcts = box_to_pct(boxes, CAMERA_WIDTH, CAMERA_HEIGHT)

            detections = sorted(
                zip(boxes, confs, labels, pcts), key=lambda x: x[1], reverse=True
            )
            if len(detections) >= 4:
                detections = detections[:3]
                boxes = boxes[:3]

            for idx, box in enumerate(boxes):
                xmin, ymin, xmax, ymax = map(int, box)
                new_bboxes.append([xmin, ymin, xmax, ymax, labels[idx], pcts[idx]])

        for x in range(1, 3):
            cv2.line(
                frame,
                (x * CAMERA_WIDTH // 3, 0),
                (x * CAMERA_WIDTH // 3, CAMERA_HEIGHT),
                WHITE,
                1,
            )
        for y in range(1, 3):
            cv2.line(
                frame,
                (0, y * CAMERA_HEIGHT // 3),
                (CAMERA_WIDTH, y * CAMERA_HEIGHT // 3),
                WHITE,
                1,
            )

        if len(detections) == 1 and last_bboxes is not None:
            iou = iou_multiple(last_bboxes, new_bboxes)
            box, conf, label, _ = detections[0]
            xmin, ymin, xmax, ymax = map(int, box)
            label_name = class_list[int(label)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label_name} {conf:.2f}",
                (xmin, ymin - 10),
                cv2.FONT_ITALIC,
                1,
                WHITE,
                2,
            )

            if iou < IOU_THRESHOLD:
                last_change_time = time.time()
                if change_variable == 0:
                    change_variable = 1
                elif last_change_time - last_stable_time > CHANGE_THRESHOLD_TIME:
                    state_loc_variable = 1
            elif change_variable == 1:
                last_stable_time = time.time()
                change_variable = 0

            if time.time() - last_change_time > STABLE_THRESHOLD_TIME and state_loc_variable == 1:
                box, conf, label, _ = detections[0]
                xmin, ymin, xmax, ymax = map(int, box)
                label_name_korean = CLASS_LIST_KOREAN[int(label)]
                pct, loc = detect(xmin, ymin, xmax, ymax, frame)
                text = f"현재 {label_name_korean}의 비중은 {pct}%이고 위치는 {loc}입니다."
                threading.Thread(target=speak, args=(text,), daemon=True).start()
                last_stable_time = time.time()
                state_loc_variable = 0

        elif len(detections) > 1 and last_bboxes is not None:
            iou = iou_multiple(last_bboxes, new_bboxes)
            for box, conf, label, _ in detections:
                xmin, ymin, xmax, ymax = map(int, box)
                label_name = class_list[int(label)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label_name} {conf:.2f}",
                    (xmin, ymin - 10),
                    cv2.FONT_ITALIC,
                    1,
                    WHITE,
                    2,
                )

            if iou < IOU_THRESHOLD:
                last_change_time = time.time()
                if change_variable == 0:
                    change_variable = 1
                elif last_change_time - last_stable_time > CHANGE_THRESHOLD_TIME:
                    state_loc_variable = 1
            elif change_variable == 1:
                last_stable_time = time.time()
                change_variable = 0

            if time.time() - last_change_time > STABLE_THRESHOLD_TIME and state_loc_variable == 1:
                text = "현재"
                for box, conf, label, _ in detections:
                    xmin, ymin, xmax, ymax = map(int, box)
                    label_name_korean = CLASS_LIST_KOREAN[int(label)]
                    _, loc = detect(xmin, ymin, xmax, ymax, frame)
                    text += f" {label_name_korean}의 위치는 {loc}입니다."
                threading.Thread(target=speak, args=(text,), daemon=True).start()
                last_stable_time = time.time()
                state_loc_variable = 0

        if new_bboxes and num_frame % BOX_UPDATE_INTERVAL == 0:
            if not (change_variable == 1 and state_loc_variable == 0):
                last_bboxes = new_bboxes

        elapsed = time.time() - start
        fps = f"FPS: {1 / elapsed:.2f}"
        cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)
        cv2.imshow("Real-time YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
