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
from datetime import datetime
from recog.load_emb_data import main
from numpy import load
from configparser import ConfigParser
main()
def convert_df(df):
    return df.to_csv().encode('utf-8')
def app():
    header = st.container()
    update = st.container()


    with header:
        main_logo_path = r'./img/ZoomClassIllustration.jpg'
        main_logo = Image.open(main_logo_path).resize((700, 400))
        st.title('WEBCAM')
        st.image(main_logo)
        st.header("Choose option")

        attendance = st.checkbox('Attendance', key="20")
        emotion = st.checkbox('Emotion', key="21")
        emotion_attendance = st.checkbox('Attendance & Emotion', key="22")
        video_trans = None


        if emotion:
            start_time = datetime.now()
            video_trans = VideoTransformer_emotion
        elif attendance:
            start_time = datetime.now()
            video_trans = VideoTransformer_attendance
        elif emotion_attendance:
            start_time = datetime.now()
            video_trans = VideoTransformer_both
        else:
            st.text("Please choose option")

        if video_trans:
            st.header("Face Recognition Process")
            if emotion:
                st.text("Perform an emotion classification task")
            elif attendance:
                st.text("Perform an identification task")
            elif emotion_attendance:
                st.text("Perform identification and emotion classification task")

        web = webrtc_streamer(key="25", video_processor_factory=video_trans)
        update_data = load('./recog/dataset-embeddings.npz')
        if web.video_processor:
            web.video_transformer.update_database(update_data)

        if st.button("Analysis"):
            if web.video_processor:
                df = web.video_transformer.update_csv()
                run_version_training = web.video_transformer.run_version()
            st.dataframe(df)
            output_path_csv = run_version_training + ".csv"
            csv = convert_df(df)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=output_path_csv,
                mime='text/csv',
            )
    with update:
        PAGES = ["Update Database","Video Recoding","Webcam","System Cloud"]
        config_object = ConfigParser()
        config_object.read("config.ini")
        if config_object["UNKNOWN"]["unknown"] == "True":
            st.write("Do you want to add the profile of new students?")
            yes = st.button ("I want", key = "26")
            if yes:
                info = config_object["CLICK"]
                info["click"] = str(0)
                with open('config.ini', 'w') as conf:
                    config_object.write(conf)
