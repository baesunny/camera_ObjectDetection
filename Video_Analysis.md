## 영상 분석(Video Analysis)?

- 비디오(영상) 분석의 핵심 요소 :
   1. Object Detection 
   2. Object Tracking
   3. Action Classification


> optical tracking 관련 설명 참조
> https://kmhana.tistory.com/20
> deep sort: https://nanonets.com/blog/object-tracking-deepsort/#single-object-tracking
> 
---

**Object Detection / Recognition / Tracking**
- object detection
  : 영상에서 대상을 찾아내는 것
- object recognition
  : 영상에서 찾아낸 대상이 무엇인지를 식별하는 것 (일종의 identification 개념)
- object tracking
  : tracking은 비디오 영상에서 특정 대상의 위치 변화(움직임)를 추적하는 것
     인접한 영상 프레읾들 사이에서의 시공간적 유사성(대상의 위치, 크기, 형태 등이 유사함) 존재 >> 다양한 history 정보 반영

> 우리는 객체가 움직이는 상황에서 탐지를 해야하고, 해당 객체가 무엇인지에 대한 classification 역시 불필요하기 때문에 tracking이 가장 적절해보임.

---
**객체 추적 기술**
1. 첫 번째 프레임에서 객체 검출
      - 객체를 검출하여 객체의 위치(bounding box)와 종류를 판별
2. 특징 추출
      - 첫 프레임에서 검출한 객체 영역의 특징을 추출
      - 이때 특징은 객체의 경계선이나 색상 등으로 그 객체를 표현할 수 있는 고유한 특징
3. 유사한 특징을 가진 영역 검색
      - 다음 프레임에서 이전에 추출한 객체의 특징과 유사한 영역을 검색
4. 객체 위치 추정
      - 다음 프레임에서 검색된 영역 중에서 이전 프레임에서 검출된 객체와 가장 유사한 영역을 찾고 이를 통해 객체의 위치를 추정
5. 객체 추적
      - 추정된 객체를 이용하여 다음 프레임에서 객체를 추적
      - 추적된 객체의 위치를 다시 한번 검증하고, 이전 프레임과 현재 프레임에서 객체의 위치를 비교하여 속도와 가속도 등을 계산

> 배경과 객체가 시각적으로 완전히 구분되며, 유사한 객체가 없는 경우 >> 아주 쉽게 객체추적 가능

---

**공부하다가 생각난 추가 아이디어**
> 포즈 추천은 어떨까?

1. 사전에 여러 가지 포즈를 입력해둠. (브이, 따봉, 꽃받침 등등)
2. keypoints tracking으로 포즈 감지 > 분류 (https://fritz.ai/computer-vision-from-image-to-video-analysis/)
3. 취하지 않은 포즈를 순차적으로 안내멘트
   ex: 브이 포즈를 취한 것이 감지되었다 >> 다음 안내멘트는 "휴대폰을 들고있는 반대손의 엄지를 들어올려보세요."

---

**관련모델정리**
1. object tracking
   Deep SORT: https://nanonets.com/blog/object-tracking-deepsort/#single-object-tracking
2. 얼굴의 각도 구도 파악
  https://suy379.tistory.com/92

