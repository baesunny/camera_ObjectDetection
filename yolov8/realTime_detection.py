# import logging

# # 기존 핸들러 제거
# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)

# # 로그 설정 초기화
# logging.basicConfig(level=logging.ERROR)

import cv2
from ultralytics import YOLO
import numpy as np
import time
from sklearn.model_selection import train_test_split
from src.elements import detect, box_to_pct
from src.utils import iou_multiple
import torch

import pyttsx3
import threading

engine = pyttsx3.init()
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Check if CUDA is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the trained model
model = YOLO("yolov8n.pt")

class_list = model.names
class_list_korean = [
    "사람", "자전거", "자동차", "오토바이", "비행기", "버스", "기차", "트럭", "보트", "신호등", "소화전", "정지 신호", "주차 미터기", "벤치", "새", "고양이", "개", "말", 
    "양", "소", "코끼리", "곰", "얼룩말", "기린", "배낭", "우산", "핸드백", "넥타이", "여행가방", "프리스비", "스키", "스노보드", "스포츠 공", "연", "야구 방망이", 
    "야구 글러브", "스케이트보드", "서핑보드", "테니스 라켓", "병", "와인 잔", "컵", "포크", "나이프", "숟가락", "그릇", "바나나", "사과", "샌드위치", "오렌지", 
    "브로콜리", "당근", "핫도그", "피자", "도넛", "케이크", "의자", "소파", "화분", "침대", "식탁", "화장실", "TV", "노트북", "마우스", "리모컨", "키보드", "휴대전화", 
    "전자레인지", "오븐", "토스터", "싱크대", "냉장고", "책", "시계", "꽃병", "가위", "테디 베어", "헤어 드라이어", "칫솔"
]

# Camera setting
cap = cv2.VideoCapture(0)
width, height = 1280, 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Hyperparameter setting
CONFIDENCE_THRESHOLD = 0.5
iou_threshold = 0.7
stable_threshold_time = 1.7

# Constants
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

last_change_time = time.time()
last_bboxes = None
detection = None

# Debugging
key=0

# 말을 한번만 하기 위해서 상태를 나타내는 변수 도입 -> 1일 때만 말해줄거야
state_loc_variable = 1

while True:
    start = time.time()
    success, frame = cap.read()
    if not success:
        print('Cam Error')
        break
    
    # 좌우 반전
    frame = cv2.flip(frame, 1)
    
    # Perform detection
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD)

    # Extract bounding boxes, labels, and scores
    if results:
        new_bboxes=[]
        # results 안에 결과 하나
        for data in results:
            boxes = data.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
            confs = data.boxes.conf.cpu().numpy()  # Extract confidence scores
            labels = data.boxes.cls.cpu().numpy()  # Extract class labels
            pcts = box_to_pct(boxes, width, height)
        # Combine all detections into a single list, and Sort by Confidence Score
        detections = sorted(zip(boxes, confs, labels, pcts), key=lambda x: x[3], reverse=True)
        
        if len(detections) >= 4:
            detections = detections[:3]
            boxes = boxes[:3]
    
        for idx, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = map(int, box)
            new_bboxes.append([xmin, ymin, xmax, ymax, labels[idx]])
    
    # Display Grid on Frame
    gridColor = WHITE
    gridThickness = 1
    for x in range(1, 3):
        cv2.line(frame, (x*width // 3, 0), (x*width // 3, height), gridColor, gridThickness)
    for y in range(1, 3):
        cv2.line(frame, (0, y*height // 3), (width, y*height//3), gridColor, gridThickness)    
    
    if len(detections) == 1:     
        if last_bboxes is not None:
            iou = iou_multiple(last_bboxes, new_bboxes)
            box, conf, label, pct = detections[0]
            xmin, ymin, xmax, ymax = map(int, box)
            label_name = class_list[int(label)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {conf:.2f}", (xmin, ymin - 10), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)  # WHITE
           
            if iou < iou_threshold:
                last_change_time = time.time()
                state_loc_variable = 1
            elif (time.time() - last_change_time) > stable_threshold_time and state_loc_variable == 1:
                box, conf, label, pct = detections[0]
                xmin, ymin, xmax, ymax = map(int, box)
                label_name_korean = class_list_korean[int(label)]
                
                pct, loc = detect(xmin, ymin, xmax, ymax, frame)
                text = f"현재 {label_name_korean}의 비중은 {pct}%이고 위치는 {loc}입니다."
                threading.Thread(target=speak, args=(text,)).start()
                last_change_time = time.time()
                state_loc_variable = 0
    
    elif len(detections) > 1:
        if last_bboxes is not None:
            iou = iou_multiple(last_bboxes, new_bboxes)
            for j in detections:
                box, conf, label, pct = j
                xmin, ymin, xmax, ymax = map(int, box)
                label_name = class_list[int(label)]
                
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, f"{label_name} {conf:.2f}", (xmin, ymin - 10), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)  # WHITE
            
            if iou < iou_threshold:
                last_change_time = time.time()
                state_loc_variable = 1
            elif (time.time() - last_change_time) > stable_threshold_time and state_loc_variable == 1:
                text="현재"
                for j in detections:
                    box, conf, label, pct = j
                    xmin, ymin, xmax, ymax = map(int, box)
                    label_name_korean = class_list_korean[int(label)]
                    
                    pct, loc = detect(xmin, ymin, xmax, ymax, frame)
                    text += f" {label_name_korean}의 위치는 {loc}입니다."
                threading.Thread(target=speak, args=(text,)).start()
                last_change_time = time.time()
                state_loc_variable = 0
    
    if new_bboxes:
        last_bboxes = new_bboxes       
            
    # FPS calculation
    end = time.time()
    totalTime = end - start
    fps = f'FPS: {1 / totalTime:.2f}'
    print(f'Time to process 1 frame: {totalTime * 1000:.0f} milliseconds')
    print(fps)
    
    cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)
    cv2.imshow('Real-time YOLO Detection', frame)
    
    # key+=1
    # if key == 50:
    #     break
    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
