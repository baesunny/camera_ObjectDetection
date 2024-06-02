import streamlit as st
import tempfile
import sys
import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import time
import numpy as np
import mediapipe as mp
import torch
from torchvision.transforms import functional as F
import cv2
from PIL import Image
from ultralytics import YOLO

from home.pages.src.utils import intersection_over_union
from home.pages.src.elements import detect, head_Pose
import pyttsx3
import threading

torch.cuda.empty_cache()
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Check if CUDA is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

def main():
    st.title("전면카메란")
    
    # Initialize the state variables
    if 'capture_button_pressed' not in st.session_state:
        st.session_state.capture_button_pressed = False
    if 'captured_image_path' not in st.session_state:
        st.session_state.captured_image_path = None
    if 'image_captured' not in st.session_state:
        st.session_state.image_captured = False

    frame_placeholder = st.empty()
    capture_button = st.button("Capture")

    class_list = ['FACE']
    model = YOLO('C:/Users/happy/Desktop/camera_ObjectDetection/전면카메라/home/best.pt')
    model.to(device)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


    cap = cv2.VideoCapture(0)
    width, height = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # hyperparameter setting
    CONFIDENCE_THRESHOLD = 0.3
    iou_threshold = 0.8
    box_threshold = 5           
    stable_threshold_time = 0.8             

    # constants
    GREEN = (0, 255, 0)       
    WHITE = (255, 255, 255)   
    BLUE = (0, 0, 255)

    last_change_time = time.time()
    last_bbox = None
    max_area = 0
    max_box = None

    original_dim = (width, height)
    resize_dim = (640, 640)
    x_scale = original_dim[0] / resize_dim[0]
    y_scale = original_dim[1] / resize_dim[1]

    state_loc_variable = 1
    num_frame=0

    while cap.isOpened():
        start = time.time()
        success, frame = cap.read()
        num_frame += 1
        if not success:
            st.write("Cam Error")
            break
        
        # Head pose estimation
        headpose = head_Pose(image=frame, face_mesh=face_mesh)
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, headpose, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        
        # Resize and move the frame to the same device as the model
        frame_resized = cv2.resize(frame, (640, 640))
        frame_tensor = F.to_tensor(frame_resized).unsqueeze(0).to(device)
        detection = model.predict(frame_tensor,conf=CONFIDENCE_THRESHOLD)[0].cpu()

        # 모든 박스 검사 - 가장 큰 면적의 bbox 선정
        for data in detection.boxes.data.tolist(): 
            xmin, ymin, xmax, ymax, conf, label = int(data[0]), int(data[1]), int(data[2]), int(data[3]), float(data[4]), int(data[5])
            xmin = int(xmin * x_scale)
            ymin = int(ymin * y_scale)
            xmax = int(xmax * x_scale)
            ymax = int(ymax * y_scale)
            
            # 가장 큰 사이즈의 박스만 pass
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
                    pct, loc = detect(xmin,ymin,xmax,ymax,frame)
                    text = f"현재 얼굴 비중은 {pct}%이고 위치는 {loc}입니다.{headpose}"
                    threading.Thread(target=speak, args=(text,)).start()
                    last_change_time = time.time()
                    state_loc_variable = 0 
            
            if num_frame % box_threshold == 0:
                last_bbox = new_bbox

        max_area = 0
        max_box = None       

        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
        
        if capture_button and not st.session_state.capture_button_pressed:
            st.session_state.capture_button_pressed = True
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                image = Image.fromarray(frame)
                image.save(tmp_file.name)
                st.session_state.captured_image_path = tmp_file.name
                st.session_state.image_captured = True

        if st.session_state.image_captured:
            st.success("Image Captured!")
            st.session_state.image_captured = False
        
        if st.session_state.captured_image_path:
            st.markdown(f"[Download captured image](file://{st.session_state.captured_image_path})")
            st.session_state.captured_image_path = None
        

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()