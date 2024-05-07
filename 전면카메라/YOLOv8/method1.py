import time
import numpy as np
import mediapipe as mp
import cv2
from ultralytics import YOLO
from src.elements import face_detect, head_Pose

# load Detection model & facemesh
coco128 = open('./coco128.txt', 'r')
data = coco128.read()
class_list = data.split('\n')
coco128.close()        

model = YOLO('./yolov8s.pt')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# camera setting 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# hyperparameter setting
CONFIDENCE_THRESHOLD = 0.8 # 최소 정확도 이하의 객체는 화면에 출력하지 않음

# constants
GREEN = (0, 255, 0)       
WHITE = (255, 255, 255)   
BLUE = (0, 0, 255)

coor=np.zeros([4,10])
vars=[]
count=0


# DETECTION
while True:
    start = time.time()
    success, frame = cap.read()
    if not success:
        print('Cam Error')
        break
    
    detection = model.predict(frame,conf=CONFIDENCE_THRESHOLD)[0]
    for data in detection.boxes.data.tolist(): 
        # data : [xmin, ymin, xmax, ymax, confidence_score, class_id]
        xmin, ymin, xmax, ymax, conf, label = int(data[0]), int(data[1]), int(data[2]), int(data[3]), float(data[4]), int(data[5])

        if label!=0:
             continue
        
        # bbox 표시
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
        cv2.putText(frame, class_list[label]+' '+str(round(conf, 2)), (xmin, ymin-10), cv2.FONT_ITALIC, 1, WHITE, 2)

        if count==0:
            coor = np.array([[xmin, ymin, xmax, ymax] for _ in range(10)]).T
            var = np.var(coor[0])/np.mean(coor[0])+np.var(coor[1])/np.mean(coor[1])+np.var(coor[2])/np.mean(coor[2])+np.var(coor[3])/np.mean(coor[3])
            vars.append(var)
            print(vars)
        elif count%5 == 0:
            coor[:, (count//5)%10] = [xmin, ymin, xmax, ymax]
            var = np.var(coor[0])/np.mean(coor[0])+np.var(coor[1])/np.mean(coor[1])+np.var(coor[2])/np.mean(coor[2])+np.var(coor[3])/np.mean(coor[3])
            vars.append(var)
            if var>1:
                #print("changed!")
                cv2.putText(frame,"changed!",(20,200),cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
                coor = np.array([[xmin, ymin, xmax, ymax] for _ in range(10)]).T

                # 요소1 & 2
                pct, loc = face_detect(xmin,ymin,xmax,ymax,frame)
                pct_text = f"object percentage: {pct}%"
                loc_text = f"object location : {loc}"
                cv2.putText(frame, pct_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
                cv2.putText(frame, loc_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
            #print(vars)
            #print(coor)
            break
    count+=1

    headpose = head_Pose(image=frame, face_mesh=face_mesh)
    ##### TODO : 요소3 안내 조건 조정 및 추가 #####
    #print(f"현재 얼굴 방향 :{headpose}")
    cv2.putText(frame, headpose, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2) # headpose 표시

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
