import streamlit as st 
import glob

def load_image():
    image_files = glob.glob("gallery/*.png")
    st.write("총",len(image_files),"장의 사진이 저장되었습니다.")

    manuscripts = []
    for image_file in image_files:
        image_file = image_file.replace("\\","/")
        parts = image_file.split("/")
        if parts[1] not in manuscripts:
            manuscripts.append(parts[1])
    manuscripts.sort()
    
    return image_files, manuscripts


st.title(":frame_with_picture:Gallery:black_heart:")
images, manuscripts = load_image()

view_images = []
for image in images:
    if any(manuscript in image for manuscript in manuscripts):
        view_images.append(image)

n = st.number_input("Select Grid Width", 1, 5, 3)

groups = []
for i in range(0, len(view_images),n):
    groups.append(view_images[i:i+n])

for group in groups:
    cols = st.columns(n)
    for i, image in enumerate(group):
        cols[i].image(image)

