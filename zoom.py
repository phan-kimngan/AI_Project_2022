import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from PIL import  Image
from plotly.offline import iplot
from streamlit_webrtc import webrtc_streamer
import av
from cam_dl import VideoTransformer_emotion, VideoTransformer_attendance, VideoTransformer_both
import datetime
def app():
    header = st.container()
    web = st.container()
    add = st.container()
    train = st.container()

    with header:
        main_logo_path = r'./img/ZoomClassIllustration.jpg'
        main_logo = Image.open(main_logo_path).resize((600, 400))
        st.title('Zoom Cloud')
        st.image(main_logo)
        st.text("Please choose your option ")
        attendance = st.checkbox('Attendance', key="1")
        emotion = st.checkbox('Emotion', key="2")
        emotion_attendance = st.checkbox('Attendance & Emotion', key="3")

        date = st.date_input('Date')
