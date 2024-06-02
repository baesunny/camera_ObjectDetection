import streamlit as st

if st.button("전면 Mode"):
    st.switch_page("pages/front_mode.py")
if st.button("후면 Mode"):
    st.switch_page("pages/rear_mode.py")
if st.button("Gallery"):
    st.switch_page("pages/gallery.py")