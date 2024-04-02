## BITAmin project (#2nd. Computer Vision)

**주제 선정: 시각장애인을 위한 스마트폰 사진기 앱**


  - **전체 파이프라인/방향성**
    - https://www.figma.com/file/MsDkrxFuI04DjDSyjFTOh1/Untitled?type=whiteboard&node-id=0%3A1&t=yYXuY3KpLKQYLxBO-1

- 서비스 목표
    - 사용자가 상황별로 사진을 찍을 때 사진을 더 편리하게 찍을 수 있도록 음성으로 안내하는 서비스를 제공한다.  
    - opencv와 object detection/ object tracking model을 이용하여 우리의 task에 맞도록 전이학습시킨다. 
    - 모델의 학습을 통해 성능 및 정확도를 향상시킨다. 
    - 알고리즘의 최적화를 통해 latency를 최소화한다.


**생각해 볼 포인트**

- 구도가 문제냐 각도가 문제냐  ⇒  **구도를 제안**해주자! (잘 나온 각도의 기준이 애매함)
- 유사도 측정?
    - 잘 나온 사진의 배경과 실제 사진의 유사도를 측정 → 유사도가 가장 높게 나온 사진에서의 피사체 위치를 참고하여 구도 제안
- 얼굴이 잘 나왔는지에 대한 정보는? (눈을 감았는지, 표정, 각도 등의 고려 여부)
    - 사진 결과물에 대한 보완이 아닌, 사진을 찍을 당시의 실시간 제안이 핵심!
- *Object Detection* → 대상이 잘리지 않고 화면 안에 잘 들어오는지가 포인트.
- 각도가 정상적인지 → 눈이나 코, 머리카락 등의 끝 지점을 기준으로 그 사이의 거리나 크기 등을 고려하여 파악해야 함. 어려울 수 있을 듯.
- 셀프 카메라 모드 → 얼굴의 눈코입 바운딩 박스의 좌표 파악
- 후면카메라 → 피사체 바운딩박스 좌표 파악 + 배경 대조 후 구도 제안
- 얼굴 전체가 나오게 하는건 디폴트 + 구도 추천
- 사진 구도 추천의 기준?
    - 어떤 구도가 좋을지 (ex 얼굴의 중앙이 격자 1/3 위치에 오도록 추천)
    - 어떤 위치가 좋을지 (ex 사람이 그냥 중앙에 오도록, 근데 프레임 전체 크기의 어느 정도를 차지하도록) 


**TASK**
- 1차 목표 : 객체 탐지를 어떻게 정확하게 할 것인가? (성능 up)
- 2차 목표 : 구도가 잘 나오는지 안나오는지를 어떻게 파악할 것인가? 
- 3차 목표 : 파악한 내용을 어떻게 전달할 것인가? (tts 연결)


**중간 발표:**
전체적인 파이프라인 및 서비스 세부 기획

        
  
