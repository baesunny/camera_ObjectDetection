# Datasets

얼굴 탐지 모델 fine-tuning에 사용한 Roboflow 데이터셋입니다.

| 폴더 | 클래스 | Train / Valid / Test | 설명 |
|------|--------|----------------------|------|
| `face_detection_6/` | FACE | 930 / 60 / 30 | 메인 학습 데이터셋 |
| `face_cropped/` | PEOPLE | 315 / 30 / 15 | 크롭 전처리 버전 |
| `face_retinanet_annotations/` | — | — | RetinaNet 학습용 CSV 어노테이션 |
| `face_mobilenet_coco/` | — | — | MobileNet-SSD 실험용 메타데이터 |

## 데이터 출처

- [Roboflow face-detection-rf4vf](https://universe.roboflow.com/selfcamtraindata-zo6hf/face-detection-rf4vf) (CC BY 4.0)
- [K-FACE 추가 데이터 (Google Drive)](docs/drive.md)

## 학습 실행

```bash
python scripts/train.py datasets/face_detection_6/data.yaml
```

## 관련 문서

- `docs/data_usage_plan.md` — K-FACE 데이터 활용 계획서
