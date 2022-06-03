import streamlit as st
import pandas as pd
from PIL import  Image
import webcams
import zoom
import video_recording_save
import os
import plotly.graph_objects as go
from bokeh.models.widgets import Div
from recog.load_emb_data import main

PAGES = {
    "Video Recoding": video_recording_save,
    "Webcam": webcams,
    "Report": zoom
}
logo_path = r'./img/Zoom-App-Icon-2.png'
st.sidebar.title('Combining Video Class Solution with Face Recognition Solution')
logo = Image.open(logo_path)
resize_logo = logo.resize((200,200))
st.sidebar.image(resize_logo)


log_in = st.sidebar.button("KIM NGAN PHAN")
if log_in:
    js = "window.open('https://zoom.us/signin')"  # New tab or window
    js = "window.location.href = 'https://zoom.us/signin'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)

lists = os.listdir('./recog/students/')
print("LIST", lists)
stu = st.sidebar.selectbox('Database', options = lists)
for j in lists:
    if j != 'unknown':
        if stu == j:
            k = os.listdir('./recog/students/'+j)
            img = Image.open('./recog/students/'+j+"/"+k[0])
            resize_img = img.resize((500,500))
            st.sidebar.image(resize_img)
            st.sidebar.text("There are {} images from the forder. We only show the first image.".format(len(k)))


st.sidebar.title('Option')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
st.sidebar.button("Log Out ")
