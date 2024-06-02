import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
from PIL import Image

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        return frame

def main():
    st.title("사진 찍기 앱")
    
    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor, media_stream_constraints={
        "video": True,
        "audio": False,
    })
    
    if st.button("사진 찍기"):
        if ctx.video_processor:
            frame = ctx.video_processor.frame
            if frame is not None:
                img = Image.fromarray(frame)
                img.save("captured_image.png")
                st.image(img, caption="Captured Image")
            else:
                st.write("웹캠에서 프레임을 캡처할 수 없습니다.")
        else:
            st.write("비디오 프로세서가 초기화되지 않았습니다.")

if __name__ == "__main__":
    main()
