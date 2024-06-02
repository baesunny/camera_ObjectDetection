# utils
import streamlit as st
import time
import torch
import cv2
from ultralytics import YOLO

# functions
from pages.src.utils import iou_multiple, improve_frame, showGrid, save_image
from pages.src.elements import detect, box_to_pct

# tts
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

# Load pretrained model
model = YOLO("./weights/yolov8n.pt")
class_list = model.names
class_list_korean = [
    "사람", "자전거", "자동차", "오토바이", "비행기", "버스", "기차", "트럭", "보트", "신호등", "소화전", "정지 신호", "주차 미터기", "벤치", "새", "고양이", "개", "말", "양", "소", "코끼리", "곰", "얼룩말", "기린", "배낭", "우산", "핸드백", "넥타이", "여행가방", "프리스비", "스키", "스노보드", "스포츠 공", "연", "야구 방망이", "야구 글러브", "스케이트보드", "서핑보드", "테니스 라켓", "병", "와인 잔", "컵", "포크", "나이프", "숟가락", "그릇", "바나나", "사과", "샌드위치", "오렌지", "브로콜리", "당근", "핫도그", "피자", "도넛", "케이크", "의자", "소파", "화분", "침대", "식탁", "화장실", "TV", "노트북", "마우스", "리모컨", "키보드", "휴대전화", "전자레인지", "오븐", "토스터", "싱크대", "냉장고", "책", "시계", "꽃병", "가위", "테디 베어", "헤어 드라이어", "칫솔"
]


def main():
    st.title("후면카메라:camera:")
    
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
    CONFIDENCE_THRESHOLD = 0.6  
    iou_threshold = 0.8   
    box_threshold = 10 # FPS 고려    
    stable_threshold_time = 1.5   
    change_threshold_time = 0.5

    # constants
    GREEN = (0, 255, 0)       
    WHITE = (255, 255, 255)   
    BLUE = (0, 0, 255)

    last_change_time = time.time()
    last_stable_time = time.time()
    last_bboxes = None
    detections = None

    state_loc_variable = 1
    num_frame = 0
    change_variable = 0

    while cap.isOpened():
        start = time.time()
        success, frame = cap.read()
        num_frame += 1

        if not success:
            st.write("Cam Error")
            break

        # Left/Right Inversion
        frame = cv2.flip(frame, 1)

        # Model Inference
        improvedFrame = improve_frame(frame)
        results = model.predict(improvedFrame, conf=CONFIDENCE_THRESHOLD)

        # Extract bounding boxes, labels, and scores
        if results:
            new_bboxes=[]
            for data in results:
                boxes = data.boxes.xyxy.cpu().numpy()
                confs = data.boxes.conf.cpu().numpy()
                labels = data.boxes.cls.cpu().numpy()
                pcts = box_to_pct(boxes, width, height)

            detections = sorted(zip(boxes, confs, labels, pcts), key=lambda x: x[1], reverse=True)
            
            if len(detections) > 3:
                detections = detections[:3]
                boxes = boxes[:3]

        for idx, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = map(int, box)
            new_bboxes.append([xmin, ymin, xmax, ymax, labels[idx], pcts[idx]])  
        
        # Assume 1 new detected object & TTS
        if len(detections) == 1:     
            if last_bboxes is not None:
                iou = iou_multiple(last_bboxes, new_bboxes)
                box, conf, label, pct = detections[0]
                xmin, ymin, xmax, ymax = map(int, box)
                label_name = class_list[int(label)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                cv2.putText(frame, f"{label_name} {conf:.2f}", (xmin, ymin - 10), cv2.FONT_ITALIC, 1, WHITE, 2)
           
                if iou < iou_threshold:
                    last_change_time = time.time()
                    if change_variable == 0:
                        change_variable = 1
                    else:
                        if last_change_time - last_stable_time > change_threshold_time:
                            state_loc_variable = 1
                elif change_variable == 1:
                    last_stable_time = time.time()
                    change_variable = 0
                
                if (time.time() - last_change_time) > stable_threshold_time and state_loc_variable == 1:
                    box, conf, label, pct = detections[0]
                    xmin, ymin, xmax, ymax = map(int, box)
                    label_name_korean = class_list_korean[int(label)]
                
                    pct, loc = detect(xmin, ymin, xmax, ymax, frame)
                    text = f"현재 {label_name_korean}의 비중은 {pct}%이고 위치는 {loc}입니다."
                    threading.Thread(target=speak, args=(text,)).start()
                    last_stable_time = time.time()
                    state_loc_variable = 0

        # Assume several new detected objects & TTS
        elif len(detections) > 1:
            if last_bboxes is not None:
                iou = iou_multiple(last_bboxes, new_bboxes)
                for j in detections:
                    box, conf, label, pct = j
                    xmin, ymin, xmax, ymax = map(int, box)
                    label_name = class_list[int(label)]
                    
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
                    cv2.putText(frame, f"{label_name} {conf:.2f}", (xmin, ymin - 10), cv2.FONT_ITALIC, 1, WHITE, 2)
                
                if iou < iou_threshold:
                    last_change_time = time.time()
                    if change_variable == 0:
                        change_variable = 1
                    else:
                        if last_change_time - last_stable_time > change_threshold_time:
                            state_loc_variable = 1
                elif change_variable == 1:
                    last_stable_time = time.time()
                    change_variable = 0
                    
                if (time.time() - last_change_time) > stable_threshold_time and state_loc_variable == 1:
                    text="현재"
                    for j in detections:
                        box, conf, label, pct = j
                        xmin, ymin, xmax, ymax = map(int, box)
                        label_name = class_list[int(label)]
                        label_name_korean = class_list_korean[int(label)]
                        
                        pct, loc = detect(xmin, ymin, xmax, ymax, frame)
                        text += f" {label_name_korean}의 위치는 {loc}입니다."
                    
                    threading.Thread(target=speak, args=(text,)).start()
                    last_stable_time = time.time()
                    state_loc_variable = 0
    
        if new_bboxes and num_frame % box_threshold == 0:
            if not (change_variable == 1 and state_loc_variable == 0):
                last_bboxes = new_bboxes
            
        
        # fps calculation
        end = time.time()
        totalTime = end - start
        fps = f'FPS: {1 / totalTime:.2f}'
        print(f'Time to process 1 frame: {totalTime * 1000:.0f} milliseconds')
        print(fps)
        cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)

        # show frame 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        showGrid(frame)
        frame_placeholder.image(frame, channels = "RGB")

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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()













