import streamlit as st
@st.cache_data
def show_icon(emoji: str):
    """Shows an emoji as a Notion-style page icon.

    Args:
        emoji (str): name of the emoji, i.e. ":balloon:"
    """

    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


# UI configurations
st.set_page_config(page_title="Eyes ON Cam",
                   page_icon=":eyes:",
                   layout="wide")
show_icon(":eyes:")

st.markdown("# Eyes :eyes: ON Cam :camera:")
st.write("시각장애인을 위한 스마트폰 사진기 음성안내 애플리케이션")
st.write("Bitamin 2조")

if st.button(":camera: 전면카메라"):
    st.switch_page("pages/front_mode.py")
if st.button(":camera: 후면카메라"):
    st.switch_page("pages/rear_mode.py")
if st.button(":frame_with_picture: Gallery"):
    st.switch_page("pages/gallery.py")