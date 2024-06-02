import streamlit as st
import time
import cv2

def save_image(frame):
    current_time = time.strftime("%H-%M-%S", time.localtime())
    file_name = f"image-{current_time}.png"
    file_path = f"./gallery/{file_name}"
    cv2.imwrite(file_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def main():
    st.title("test mode")
    
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

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.write("Cam Error")
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

        if capture_button and not st.session_state.capture_button_pressed:
            st.session_state.capture_button_pressed = True
            save_image(frame=frame)

            st.session_state.image_captured = True
            capture_button = False
            st.session_state.capture_button_pressed = False
           

        if st.session_state.image_captured:
            st.success("Image Captured!")
            st.session_state.image_captured = False


        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()