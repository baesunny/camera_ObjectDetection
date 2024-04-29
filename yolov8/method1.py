import time
import numpy as np
import mediapipe as mp
import cv2
from ultralytics import YOLO
from src.elements import face_detect, head_Pose
import torch
from torchvision.transforms import functional as F

# Set up device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load Detection model & facemesh
coco128 = open('camera_ObjectDetection\yolov8\coco128.txt', 'r')
data = coco128.read()
class_list = data.split('\n')
coco128.close()

model = YOLO('yolov8\yolov8s.pt')
model.to(device)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Camera setting
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Hyperparameter setting
CONFIDENCE_THRESHOLD = 0.8

# Constants
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)

coor = np.zeros([4, 10])
vars = []
count = 0

original_dim = (1280, 720)
resize_dim = (640, 640)
# Scale factors for coordinates
x_scale = original_dim[0] / resize_dim[0]
y_scale = original_dim[1] / resize_dim[1]


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
    # forward 뒤의 점 없애고 두 개의 상태 나타낼 때 띄어쓰기 넣음.
    headpose = headpose.replace("forward.", "forward")
    second_ind = headpose.find('looking', 10)
    if second_ind != -1:
        headpose = headpose[:second_ind] + " " +headpose[second_ind:]
    
    cv2.putText(frame, headpose, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
    
    # Convert resized frame to tensor and move to the same device as the model
    frame_resized = cv2.resize(frame, (640, 640))
    frame_tensor = F.to_tensor(frame_resized).unsqueeze(0).to(device)
    
    # Make predictions
    detection = model.predict(frame_tensor, conf=CONFIDENCE_THRESHOLD)[0].cpu()
    
    for data in detection.boxes.data.tolist():
        xmin, ymin, xmax, ymax, conf, label = int(data[0]), int(data[1]), int(data[2]), int(data[3]), float(data[4]), int(data[5])
        xmin = int(xmin * x_scale)
        ymin = int(ymin * y_scale)
        xmax = int(xmax * x_scale)
        ymax = int(ymax * y_scale)
        
        if label != 0:
            continue
        
        # Display bbox
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.putText(frame, class_list[label] + ' ' + str(round(conf, 2)), (xmin, ymin - 10), cv2.FONT_ITALIC, 1, WHITE, 2)

        # Variance-based tracking logic (adapted for clarity and correctness)
        # Update and check your tracking and change detection logic here

    count += 1
    
    # FPS calculation
    end = time.time()
    totalTime = (end - start)
    fps = f'FPS: {1 / totalTime:.2f}'
    print(f'Time to process 1 frame: {totalTime * 1000:.0f} milliseconds')
    print(fps)
        
    cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)
    cv2.imshow('frame', frame)
        
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()