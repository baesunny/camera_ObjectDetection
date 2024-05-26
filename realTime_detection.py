import cv2
from ultralytics import YOLO
import numpy as np
import albumentations as A
import time

# def to improve frame
def improve_frame(frame):
    transform = A.Compose([
        A.CLAHE(clip_limit=4.0, p=1),
    ])
    filtered_frame = transform(image = frame)['image']
    return filtered_frame

def identify_grid_loc(xmin, ymin, xmax, ymax, width, height):
    grid_width = width//3
    grid_height = height//3
    
    # get Center grid value
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) /2
    
    # Identifying Grid Loc
    if center_x < grid_width: 
        col = 'Right'
    elif center_x < grid_width * 2:
        col = 'Center'
    else:
        col = 'Left'
    
    if center_y < grid_height:
        row = 'Top'
    elif center_y < grid_height * 2:
        row = 'Mid'
    else:
        row = 'Bottom'
    
    return f"{row}-{col}"
    
# Function for real-time detection
def run_real_time_detection():
    # Load the trained model
    model = YOLO("yolov8n.pt")

    # Start video capture (0 for default webcam, or specify video file path)
    cap = cv2.VideoCapture(0)
    
    # Initialize Frame time 
    prev_frame_time = 0
    new_frame_time = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        new_frame_time = time.time()
        
        # Improve Frame
        improved_frame = improve_frame(frame) # type: ignore
        
        # Perform detection
        results = model(improved_frame)
        
        # Calculate FPS
        fps = 1/(new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Extract bounding boxes, labels, and scores
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Extract bounding boxes
            scores = result.boxes.conf.cpu().numpy()  # Extract confidence scores
            labels = result.boxes.cls.cpu().numpy()  # Extract class labels

            # Combine all detections into a single list, and Sort by Confidence Score
            detections = sorted(zip(boxes, scores, labels), key=lambda x: x[1], reverse=True)
            
            if len(detections) > 4:
                detections = detections[:3]
                
            # Draw bounding boxes and labels
            for box, score, label in detections[:3]:
                if score >= 0.5:  # Only consider detections with confidence >= 0.5
                    xmin, ymin, xmax, ymax = map(int, box)
                    label_name = model.names[int(label)]
                    grid_location = identify_grid_loc(xmin, ymin, xmax, ymax, width, height)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label_name}: {score:.2f} ({grid_location})", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    
                    # Print out Confidence Score
                    print(f"{label_name}: {score:.2f}")
                    
        # Display Grid on Frame
        gridColor = (0, 0, 255)
        gridThickness = 2
        for x in range(1, 3):
            cv2.line(frame, (x*width // 3, 0), (x*width // 3, height), gridColor, gridThickness)
        
        for y in range(1, 3):
            cv2.line(frame, (0, y*height // 3), (width, y*height//3), gridColor, gridThickness)
        
        # Display FPS on the frame?
        print(f"\nFPS: {fps:.2f}")
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        # Display the resulting frame
        cv2.imshow('Real-time YOLO Detection', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Run real-time detection
run_real_time_detection()
