import time
import numpy as np
import mediapipe as mp
import torch
from torchvision.transforms import functional as F
import cv2
from ultralytics import YOLO
from src.utils import intersection_over_union
from src.elements import face_detect, head_Pose
import pyttsx3
import threading
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Check if CUDA is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# load Detection model & facemesh
coco128 = open('./coco128.txt', 'r')
data = coco128.read()
class_list = data.split('\n')
coco128.close()

model = YOLO('./best.pt')
model.to(device)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# camera setting 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# hyperparameter setting
CONFIDENCE_THRESHOLD = 0.8    # 최소 정확도 이하의 객체는 화면에 출력하지 않음
iou_threshold = 0.8           # 유사도 IoU 기준
stable_threshold_time = 1.5   # 움직임 안정 여부 시간(s) 기준
pct_threshold = 10            # 얼굴 비중 안내 기준

# constants
GREEN = (0, 255, 0)       
WHITE = (255, 255, 255)   
BLUE = (0, 0, 255)

last_change_time = time.time()
last_bbox = None
max_area = 0
max_box = None

original_dim = (1280, 720)
resize_dim = (640, 640)
# Scale factors for coordinates
x_scale = original_dim[0] / resize_dim[0]
y_scale = original_dim[1] / resize_dim[1]

# 말을 한번만 하기 위해서 상태를 나타내는 변수 도입 -> 1일 때만 말해줄거야
state_loc_variable = 1

# DETECTION
while True:
    start = time.time()
    success, frame = cap.read()
    if not success:
        print('Cam Error')
        break

    # 요소 3 : head pose estimation
    headpose = head_Pose(image=frame, face_mesh=face_mesh)
    # 좌우반전
    frame = cv2.flip(frame, 1)
    ##### TODO : 요소3 안내 조건 조정 및 추가 #####
    cv2.putText(frame, headpose, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
    
    # resize and move the frame to the same device as the model
    frame_resized = cv2.resize(frame, (640, 640))
    frame_tensor = F.to_tensor(frame_resized).unsqueeze(0).to(device)

    detection = model.predict(frame,conf=CONFIDENCE_THRESHOLD)[0]
    for data in detection.boxes.data.tolist(): 
        # data : [xmin, ymin, xmax, ymax, confidence_score, class_id]
        xmin, ymin, xmax, ymax, conf, label = int(data[0]), int(data[1]), int(data[2]), int(data[3]), float(data[4]), int(data[5])
        xmin = int(xmin * x_scale)
        ymin = int(ymin * y_scale)
        xmax = int(xmax * x_scale)
        ymax = int(ymax * y_scale)

        if label!=0:
             continue
        
        # 가장 큰 사이즈의 박스만 pass
        area = (xmax-xmin)*(ymax-ymin)
        if area > max_area:
             max_area = area
             max_box = [xmin, ymin, xmax, ymax, conf, label]
    
    if max_box:
        # bbox 표시
        new_bbox = max_box[:3]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.putText(frame, class_list[label]+' '+str(round(conf, 2)), (xmin, ymin-10), cv2.FONT_ITALIC, 1, WHITE, 2)

        if last_bbox is not None:
            iou = intersection_over_union(last_bbox, new_bbox)
            if iou < iou_threshold:
                 last_change_time = time.time()
                 state_loc_variable = 1

            elif (time.time() - last_change_time) > stable_threshold_time and state_loc_variable == 1:

                # 요소 1 & 2 탐지 시작
                pct, loc = face_detect(xmin,ymin,xmax,ymax,frame)
                # (요소1)
                if pct > pct_threshold: # TODO: 요소1 조건 조정
                    pct_text = f"object percentage: {pct}%"
                    cv2.putText(frame, pct_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
                # (요소2)
                loc_text = f"object location : {loc}"
                cv2.putText(frame, loc_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
                # (TTS)
                threading.Thread(target=speak, args=(f"현재 얼굴 비중은 {pct}%이고 얼굴 위치는 {loc}이다.",)).start()

                # 안내 완료 후 타이머 리셋
                last_change_time = time.time()
                # state_loc_variable 리셋
                state_loc_variable = 0 
        
        last_bbox = new_bbox

    max_area = 0
    max_box = None

    # fps 계산
    end = time.time()
    totalTime = (end - start)
    fps = f'FPS: {1 / totalTime:.2f}'
    print(f'Time to process 1 frame: {totalTime * 1000:.0f} milliseconds')
    print(fps)
        
    cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2) # fps 표시
    cv2.imshow('frame', frame)
        
    if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()