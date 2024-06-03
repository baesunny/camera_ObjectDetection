# utils
import streamlit as st
import time
import mediapipe as mp
import torch
from torchvision.transforms import functional as F
import cv2
from ultralytics import YOLO

# functions
from pages.src.utils import intersection_over_union, showGrid, save_image
from pages.src.elements import detect, head_Pose

# tts
import pyttsx3
import threading
torch.cuda.empty_cache()

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Check if CUDA is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load finetuned model
class_list = ['FACE']
model = YOLO('./weights/yolo_face.pt')
model.to(device)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def main():
    st.title("전면카메라:camera:")
        
    # Initialize the state variables
    if 'capture_button_pressed' not in st.session_state:
        st.session_state.capture_button_pressed = False
    if 'image_captured' not in st.session_state:
        st.session_state.image_captured = False

    frame_placeholder = st.empty()
    capture_button = st.button("Capture")

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
        headpose_text, headpose_say = head_Pose(image=frame, face_mesh=face_mesh)
        # Left/Right Inversion
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, headpose_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        
        # Resize and move the frame to the same device as the model
        frame_resized = cv2.resize(frame, (640, 640))
        frame_tensor = F.to_tensor(frame_resized).unsqueeze(0).to(device)

        # Model Inference
        detection = model.predict(frame_tensor,conf=CONFIDENCE_THRESHOLD)[0].cpu()

        # Select max bbox 
        for data in detection.boxes.data.tolist(): 
            xmin, ymin, xmax, ymax, conf, label = int(data[0]), int(data[1]), int(data[2]), int(data[3]), float(data[4]), int(data[5])
            xmin = int(xmin * x_scale)
            ymin = int(ymin * y_scale)
            xmax = int(xmax * x_scale)
            ymax = int(ymax * y_scale)
            
            # calculate area of the box
            area = (xmax-xmin)*(ymax-ymin)
            # check if this box is bigger than the previous largest one
            if area > max_area:
                max_area = area
                max_box = [xmin, ymin, xmax, ymax, conf, label]
        
        # TTS w/t the max bbox
        if max_box:
            xmin, ymin, xmax, ymax, conf, label = max_box
            new_bbox = [xmin, ymin, xmax, ymax]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.putText(frame, f"{class_list[label]} {round(conf, 2)}", (xmin, ymin - 10), cv2.FONT_ITALIC, 1, WHITE, 2)

            if last_bbox is not None:
                iou = intersection_over_union(last_bbox, new_bbox)
                print(iou)
                if iou < iou_threshold:
                    last_change_time = time.time()
                    state_loc_variable = 1
                elif (time.time() - last_change_time) > stable_threshold_time and state_loc_variable == 1:
                    pct, loc = detect(xmin,ymin,xmax,ymax,frame)
                    text = f"현재 얼굴 비중은 {pct}%이고 위치는 {loc}입니다.{headpose_say}"
                    threading.Thread(target=speak, args=(text,)).start()
                    last_change_time = time.time()
                    state_loc_variable = 0 
            
            if num_frame % box_threshold == 0:
                last_bbox = new_bbox

        max_area = 0
        max_box = None    

        # fps calculation
        end = time.time()
        totalTime = end-start
        fps = f'FPS: {1 / totalTime:.2f}'
        print(f'Time to process 1 frame: {totalTime * 1000:.0f} milliseconds')
        print(fps)
        cv2.putText(frame, fps, (10,20), cv2.FONT_HERSHEY_SIMPLEX,0.5,BLUE,2)   

        # show frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showGrid(frame)
        frame_placeholder.image(frame, channels="RGB")
        
        # capture 기능
        if capture_button and not st.session_state.capture_button_pressed:
            st.session_state.capture_button_pressed = True
            save_image(frame=frame)

            st.session_state.image_captured=True
            capture_button=False
            st.session_state.capture_button_pressed=False

        if st.session_state.image_captured:
            st.success("Image Captured!")
            st.session_state.image_captured = False

        if cv2.waitKey(1) == ord("q"):
            break

if __name__ == "__main__":
    main()