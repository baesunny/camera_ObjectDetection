import time
import numpy as np
import mediapipe as mp
import cv2
from ultralytics import YOLO
from src.utils import intersection_over_union
from src.elements import face_detect, head_Pose
import torch
from torchvision.transforms import functional as F

from gtts import gTTS
import pygame
import tempfile
import threading
import os

class VoiceAssistant:
    def __init__(self):
        pygame.mixer.init()
        self.lock = threading.Lock()
        self.is_speaking = False

    def speak(self, text):
        with self.lock:
            if not self.is_speaking:
                self.is_speaking = True
                threading.Thread(target=self._speak_task, args=(text,)).start()

    def _speak_task(self, text):
        # gTTS를 사용하여 오디오 파일 생성
        tts = gTTS(text=text, lang='ko')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)
        temp_file.close()
        
        # 오디오 제생
        pygame.mixer.music.load(temp_file.name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # 재생이 끝날 때까지 대기
            pygame.time.Clock().tick(10)
        
        os.remove(temp_file.name)
        with self.lock:
            self.is_speaking = False

    def stop_speaking(self):
        with self.lock:
            if self.is_speaking:
                pygame.mixer.music.stop()  # 재생 중지
                self.is_speaking = False

    
voice_assistant = VoiceAssistant()

# Check if CUDA is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load Detection model & facemesh
# coco128 = open('camera_ObjectDetection\yolov8\coco128.txt', 'r')
# data = coco128.read()
# class_list = data.split('\n')
# coco128.close()
class_list = ['face']

model = YOLO(r'C:\Users\baseoki\OneDrive\바탕 화면\프로젝트\camera_ObjectDetection\custom_dataset 학습 코드 YOLO\runs\detect\에포크 100 patience 10\weights\best.pt')
model.to(device)  # Use the device set above

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Camera setting
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Hyperparameter setting
CONFIDENCE_THRESHOLD = 0.5
iou_threshold = 0.7
stable_threshold_time = 1.7
pct_threshold = 10

# Constants
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

# Detection loop
while True:
    start = time.time()
    success, frame = cap.read()
    
    if not success:
        print('Cam Error')
        break
    
    # Head pose estimation
    headpose = head_Pose(image=frame, face_mesh=face_mesh)
    # Left/Right Inversion
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, headpose, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
    
    # Convert resized frame to tensor and move to the same device as the model
    frame_resized = cv2.resize(frame, (640, 640))
    frame_tensor = F.to_tensor(frame_resized).unsqueeze(0).to(device)

    # Use the model for prediction
    detection = model.predict(frame_tensor, conf=CONFIDENCE_THRESHOLD)[0].cpu()

    # 모든 박스 검사 -> 최대 박스 선정
    for data in detection.boxes.data.tolist():
        xmin, ymin, xmax, ymax, conf, label = int(data[0]), int(data[1]), int(data[2]), int(data[3]), float(data[4]), int(data[5])
        xmin = int(xmin * x_scale)
        ymin = int(ymin * y_scale)
        xmax = int(xmax * x_scale)
        ymax = int(ymax * y_scale)

        # Calculate area of the box
        area = (xmax - xmin) * (ymax - ymin)

        # Check if this box is bigger than the previously found ones
        if area > max_area and label == 0:
            max_area = area
            max_box = [xmin, ymin, xmax, ymax, conf, label]

    # 조건을 충족하는 가장 큰 박스에서 작업 수행
    if max_box:
        xmin, ymin, xmax, ymax, conf, label = max_box
        new_bbox = [xmin, ymin, xmax, ymax]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # GREEN
        cv2.putText(frame, f"{class_list[label]} {round(conf, 2)}", (xmin, ymin - 10), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)  # WHITE

        if last_bbox is not None:
            iou = intersection_over_union(last_bbox, new_bbox)
            if iou < iou_threshold:
                last_change_time = time.time()
                state_loc_variable = 1
                voice_assistant.stop_speaking()
                # voice_assistant.shutdown()
            elif (time.time() - last_change_time) > stable_threshold_time and state_loc_variable == 1:
                pct, loc = face_detect(xmin, ymin, xmax, ymax, frame)
                #if pct > pct_threshold:
                text = f"현재 얼굴 비중은 {pct}%이고 얼굴 위치는 {loc}입니다."
                voice_assistant.speak(text)
                last_change_time = time.time()
                state_loc_variable = 0

        last_bbox = new_bbox
    
    max_area = 0
    max_box = None
    
    # FPS calculation
    end = time.time()
    totalTime = end - start
    fps = f'FPS: {1 / totalTime:.2f}'
    print(f'Time to process 1 frame: {totalTime * 1000:.0f} milliseconds')
    print(fps)

    cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()