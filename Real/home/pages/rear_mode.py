import cv2
import torch
from torchvision.transforms import functional as F
from ultralytics import YOLO
from PIL import Image

import streamlit as st
import numpy as np
import tempfile
import time
import mediapipe as mp

import sys
import os

# Add the 'home' directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from pages.src.utils import intersection_over_union
from pages.src.elements import detect, head_Pose

import pyttsx3
import threading
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Check if CUDA is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the model
class_list = ['face']
model = YOLO('C:/Users/happy/Desktop/camera_ObjectDetection/전면카메라/home/best.pt')
model.to(device)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# hyperparameter setting
CONFIDENCE_THRESHOLD = 0.5    
iou_threshold = 0.8          
stable_threshold_time = 1.5   
pct_threshold = 10   

# constants
GREEN = (0, 255, 0)       
WHITE = (255, 255, 255)   
BLUE = (0, 0, 255)


def main():
    st.title("후면카메라")
    
    # Initialize the state variables
    if 'capture_button_pressed' not in st.session_state:
        st.session_state.capture_button_pressed = False
    if 'captured_image_path' not in st.session_state:
        st.session_state.captured_image_path = None
    if 'stop_button_pressed' not in st.session_state:
        st.session_state.stop_button_pressed = False

    frame_placeholder = st.empty()
    capture_button = st.button("Capture")
    stop_button = st.button("Stop")

    last_change_time = time.time()
    last_bbox = None
    max_area = 0
    max_box = None
    original_dim = (1280, 720)
    resize_dim = (640, 640)
    # Scale factors for coordinates
    x_scale = original_dim[0] / resize_dim[0]
    y_scale = original_dim[1] / resize_dim[1]

    state_loc_variable = 1

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened() and not st.session_state.stop_button_pressed:
        start = time.time()
        print(start)
        success, frame = cap.read()
        if not success:
            st.write("Camera Error")
            break
      
        # Head pose estimation
        headpose = head_Pose(image=frame, face_mesh=face_mesh)
        frame = cv2.flip(frame, 1)
        ##### TODO : 요소3 안내 조건 조정 및 추가 #####
        cv2.putText(frame, headpose, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        
        # YOLO
        frame_resized = cv2.resize(frame, (640, 640))
        frame_tensor = F.to_tensor(frame_resized).unsqueeze(0).to(device)
        detection = model.predict(frame_tensor,conf=CONFIDENCE_THRESHOLD)[0].cpu()

        for data in detection.boxes.data.tolist(): 
            xmin, ymin, xmax, ymax, conf, label = int(data[0]), int(data[1]), int(data[2]), int(data[3]), float(data[4]), int(data[5])
            xmin = int(xmin * x_scale)
            ymin = int(ymin * y_scale)
            xmax = int(xmax * x_scale)
            ymax = int(ymax * y_scale)

            if label!=0:
                continue
            
            area = (xmax-xmin)*(ymax-ymin)
            if area > max_area:
                max_area = area
                max_box = [xmin, ymin, xmax, ymax, conf, label]
        
        if max_box:
            xmin, ymin, xmax, ymax, conf, label = max_box
            new_bbox = [xmin, ymin, xmax, ymax]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.putText(frame, class_list[label]+' '+str(round(conf, 2)), (xmin, ymin-10), cv2.FONT_ITALIC, 1, WHITE, 2)

            if last_bbox is not None:
                iou = intersection_over_union(last_bbox, new_bbox)
                if iou < iou_threshold:
                    last_change_time = time.time()
                    state_loc_variable = 1

                elif (time.time() - last_change_time) > stable_threshold_time and state_loc_variable == 1:

                    # 요소 1 & 2 탐지 시작
                    pct, loc = detect(xmin,ymin,xmax,ymax,frame)
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
        totalTime = end - start
        fps = f'FPS: {1 / totalTime:.2f}'
        print(f'Time to process 1 frame: {totalTime * 1000:.0f} milliseconds')
        print(fps)
        cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2) # fps 표시

      
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels = "RGB")

        
        if capture_button:
            st.session_state['capture_button_pressed'] = True
            with tempfile.NamedTemporaryFile(delete=False,suffix='png') as tmp_file:
                image = Image.fromarray(frame)
                image.save(tmp_file.name)
                st.session_state.captured_image_path = tmp_file.name
        
            st.success("Image Captured!")
            st.session_state['capture_button_pressed'] = False


        if st.session_state.captured_image_path:
            st.markdown(f"[Download captured image](file://{st.session_state.captured_image_path})")

            st.success("Image Downloaded!")
            st.session_state['captured_image_path'] = None

        if stop_button:
            st.session_state['stop_button_pressed'] = True

        if cv2.waitKey(1) == ord("q") or st.session_state.stop_button_pressed:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()













