# Camera Object Detection

> 시각장애인을 위한 스마트폰·웹캠 사진 촬영 **음성 안내** 프로젝트

웹캠 영상에서 얼굴·객체를 실시간으로 탐지하고, 화면 내 **위치**, **크기(비중)**, **고개 방향**을 계산한 뒤 한국어 TTS로 안내한다.

---

## 프로젝트 배경

- **주제**: 시각장애인을 위한 스마트폰 사진기 음성안내 어플리케이션
- **목표**: OpenCV + Object Detection 전이학습으로 촬영 latency 최소화 및 정확도 향상
- **데이터**: K-FACE(한국인 얼굴) + Roboflow 공개 데이터

---

## 프로젝트 목적

시각장애인이 셀프 카메라로 사진을 찍을 때 **"피사체가 화면 어디에 있는지"** 를 음성으로 듣고 촬영 구도를 맞출 수 있도록 돕는다.


| 단계              | 내용                                                |
| --------------- | ------------------------------------------------- |
| **1. 객체 탐지**    | YOLOv8 fine-tuning으로 얼굴/객체 인식                     |
| **2. 위치·크기 계산** | 3×3 격자 기반 위치 + 바운딩박스 면적 비율(%)                     |
| **3. 헤드포즈**     | MediaPipe Face Mesh + OpenCV `solvePnP`로 고개 방향 추정 |
| **4. 음성 안내**    | pyttsx3 TTS — *"현재 얼굴 비중은 23%이고 위치는 가운데이다."*      |
| **5. 안정화**      | IoU 기반 프레임 추적으로 불필요한 반복 안내 방지                     |


### 핵심 출력 3요소


| 요소        | 설명                  | 예시                             |
| --------- | ------------------- | ------------------------------ |
| **pct**   | 바운딩박스가 화면에서 차지하는 비율 | `23%`                          |
| **loc**   | 3×3 격자 위치 (한국어)     | `가운데`, `왼쪽 위`                  |
| **angle** | 고개 방향               | `looking left`, `looking down` |


---

## 기술 스택


| 영역     | 기술                                                       |
| ------ | -------------------------------------------------------- |
| 객체 탐지  | YOLOv8 (Ultralytics), RetinaNet*, MobileNet-SSD*, YOLOX* |
| 헤드포즈   | MediaPipe Face Mesh, OpenCV `solvePnP`                   |
| 실시간 처리 | OpenCV, PyTorch                                          |
| 음성 안내  | pyttsx3 (오프라인 TTS)                                       |
| 웹 UI   | Streamlit + streamlit-webrtc                             |
| 학습 데이터 | Roboflow face-detection, K-FACE                          |


>  RetinaNet / MobileNet-SSD / YOLOX는 `notebooks/training/`에서 비교 실험 수준이다. **실시간 추론의 메인 파이프라인은 YOLOv8**이다.

---

## 빠른 시작

### 1. 환경 설정

```bash
git clone https://github.com/<your-username>/camera_ObjectDetection.git
cd camera_ObjectDetection
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. 모델 준비


| 모델             | 경로                    | 용도              |
| -------------- | --------------------- | --------------- |
| YOLOv8s (COCO) | `models/yolov8s.pt`   | 범용 객체 탐지 (포함됨)  |
| YOLOv8n        | `models/yolov8n.pt`   | 경량 베이스 모델 (포함됨) |
| 커스텀 얼굴 모델      | `models/face_best.pt` | 얼굴 탐지 (학습 후 배치) |


커스텀 얼굴 모델은 직접 학습하거나, `notebooks/training/yolov8/runs/`의 학습 결과 가중치를 `models/face_best.pt`로 복사한다.

```bash
# 환경변수로 모델 경로 지정 (선택)
set FACE_MODEL_PATH=models\face_best.pt
set YOLO_MODEL_PATH=models\yolov8s.pt
```

### 3. 실행

```bash
# ★ 메인 데모: 얼굴 탐지 + 헤드포즈 + 한국어 음성 안내
python src/apps/face_with_voice.py

# 얼굴 탐지 + 헤드포즈 (음성 없음)
python src/apps/face_detection.py

# COCO 80클래스 객체 탐지 + 음성 안내
python src/apps/object_detection_tts.py

# 베이스라인 (CLAHE 전처리 + 격자 표시)
python scripts/baseline_detection.py

# Streamlit 웹캠 촬영 앱
streamlit run src/apps/streamlit_cam.py
```

웹캠 창에서 `**q**` 키를 누르면 종료된다.

### 4. 모델 학습

```bash
python scripts/train.py datasets/face_detection_6/data.yaml
```

상세 학습 노트북: `notebooks/training/yolov8/yolov8_custom.ipynb`

---

## 파이프라인 흐름

```
웹캠 (1280×720)
    │
    ├─► MediaPipe Face Mesh ──► head_pose() ──► "looking left / forward ..."
    │
    └─► YOLOv8 predict (640×640)
            │
            ├─► 최대 면적 bbox 선택
            ├─► IoU < threshold → 위치 변화 감지
            ├─► stable_threshold(0.8s) 경과 후
            └─► detect() → pct + loc
                    │
                    └─► TTS: "현재 얼굴 비중은 X%이고 위치는 Y이다."
```

---

## 디렉터리 구조

```
camera_ObjectDetection/
├── README.md
├── requirements.txt
├── config.py                      # 모델·카메라 경로 설정
│
├── src/                           # ★ 메인 소스 코드
│   ├── detection/
│   │   ├── elements.py            # detect(), head_pose(), box_to_pct()
│   │   └── utils.py               # IoU, 다중 객체 매칭
│   └── apps/
│       ├── face_with_voice.py     # 메인 데모 (얼굴 + TTS)
│       ├── face_detection.py      # 얼굴 탐지 + 헤드포즈
│       ├── object_detection_tts.py# COCO 객체 + TTS
│       └── streamlit_cam.py       # Streamlit 웹캠 앱
│
├── scripts/
│   ├── baseline_detection.py      # CLAHE + 격자 베이스라인
│   └── train.py                   # YOLOv8 학습 스크립트
│
├── models/
│   ├── yolov8s.pt                 # COCO pretrained
│   ├── yolov8n.pt                 # 경량 pretrained
│   └── face_best.pt               # 커스텀 얼굴 모델 (학습 후)
│
├── head_pose/                     # 헤드포즈 단독 실험
│   ├── headPose.py
│   ├── headPose_multi.py          # 최대 3명
│   └── headPose_woMarks.py
│
├── datasets/                      # 학습 데이터셋
│   ├── face_detection_6/          # 메인 (FACE, 930/60/30)
│   ├── face_cropped/              # 크롭 버전 (PEOPLE, 315/30/15)
│   ├── face_retinanet_annotations/
│   ├── docs/                      # K-FACE 활용계획서, Drive 링크
│   └── README.md
│
├── notebooks/
│   ├── prototypes/                # 초기 기능 검증 (ver_0.0, ver_0.1)
│   └── training/
│       ├── yolov8/                # YOLOv8 fine-tuning + runs/
│       ├── retinanet/             # RetinaNet 실험
│       ├── mobilenet_ssd/         # MobileNet-SSD 실험
│       └── yolox/                 # YOLOX 실험
│
├── docs/                          # 프로젝트 기획 PDF
├── legacy/
│   └── retinanet_prototype/       # (구) YOLO 기반 프로토타입 — RetinaNet 미사용
│
├── realTime_detection.py          # → scripts/baseline_detection.py 래퍼
└── train_model.py                 # → scripts/train.py 래퍼
```

> **레거시 폴더** (`yolov8/`, `headPose/`, `custom_dataset*/`, `test/`, `dataset/`)은 이전 구조의 잔여물이다. `src/`로 이전 완료 후 삭제 가능하다. 각 폴더에 `DEPRECATED.md` 참고.

---

## 데이터셋


| 데이터셋               | 클래스    | 규모                             | 출처                                                                                    |
| ------------------ | ------ | ------------------------------ | ------------------------------------------------------------------------------------- |
| `face_detection_6` | FACE   | train 930 / valid 60 / test 30 | [Roboflow](https://universe.roboflow.com/selfcamtraindata-zo6hf/face-detection-rf4vf) |
| `face_cropped`     | PEOPLE | train 315 / valid 30 / test 15 | 동일 (크롭 전처리)                                                                           |
| K-FACE             | —      | 별도 다운로드                        | [Google Drive](datasets/docs/drive.md)                                                |


자세한 내용: `[datasets/README.md](datasets/README.md)`

---

## 모델 비교 실험


| 모델                | 상태                       | 위치                                  |
| ----------------- | ------------------------ | ----------------------------------- |
| **YOLOv8**        | ✅ 학습 완료 (`runs/detect/`) | `notebooks/training/yolov8/`        |
| **RetinaNet**     | ⚠ 학습 시도                  | `notebooks/training/retinanet/`     |
| **MobileNet-SSD** | ❌ clone만                 | `notebooks/training/mobilenet_ssd/` |
| **YOLOX**         | ❌ clone + 설치만            | `notebooks/training/yolox/`         |


