import cv2
import torch
from torchvision.transforms import functional as F
from ultralytics import YOLO

import streamlit as st
import numpy as np
#import tempfile

original_dim = (1280, 720)
resize_dim = (640, 640)
# Scale factors for coordinates
x_scale = original_dim[0] / resize_dim[0]
y_scale = original_dim[1] / resize_dim[1]

# Check if CUDA is available and set the device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

class_list = ['FACE']

model = YOLO('./best.pt')
model.to(device)  # Use the device set above

# camera setting 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

st.title("Video Capture w/t OpenCV")

frame_placeholder = st.empty()
stop_button_pressed = st.button("Stop")

while cap.isOpened() and not stop_button_pressed:
    suc, frame = cap.read()
    if not suc:
        st.write("Camera Error")
        break

    # Left/Right Inversion
    frame = cv2.flip(frame, 1)
    frame_resized = cv2.resize(frame, (640, 640))
    frame_tensor = F.to_tensor(frame_resized).unsqueeze(0).to(device)

    # Use the model for prediction
    detection = model.predict(frame_tensor, conf=0.5)[0].cpu()

    for data in detection.boxes.data.tolist():
        xmin, ymin, xmax, ymax, conf, label = int(data[0]), int(data[1]), int(data[2]), int(data[3]), float(data[4]), int(data[5])

        xmin = int(xmin * x_scale)
        ymin = int(ymin * y_scale)
        xmax = int(xmax * x_scale)
        ymax = int(ymax * y_scale)
        
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # GREEN
        cv2.putText(frame, f"{class_list[label]} {round(conf, 2)}", (xmin, ymin - 10), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)  # WHITE
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame_placeholder.image(frame, channels = "RGB")

    if cv2.waitKey(1) == ord("q") or stop_button_pressed:
        break

cap.release()
cv2.destroyAllWindows()