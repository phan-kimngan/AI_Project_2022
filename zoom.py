import streamlit as st
import pandas as pd
import os
from PIL import  Image
from streamlit_webrtc import webrtc_streamer
from cam_dl import VideoTransformer_emotion, VideoTransformer_attendance, VideoTransformer_both
from datetime import datetime
def app():
    header = st.container()
    web = st.container()
    add = st.container()
    train = st.container()

    with header:
        main_logo_path = r'./img/cloud.webp'
        main_logo = Image.open(main_logo_path).resize((400, 400))
        st.title('SYSTEM CLOUD')
        st.image(main_logo)
        st.header("Choose option")
        video = st.checkbox('Video Recorder', key="1")
        webcam = st.checkbox('Webcam', key="2")
        date = st.date_input('Date')
        if video:
            org_link = "./data/outputs/video_recorder/"

        if webcam:
            org_link = "./data/outputs/webcam/"

        if video or webcam:
            link_list = os.listdir(org_link)
            folders = {values : None for values in link_list}
            link_list.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d_%H-%M-%S"))
            for folder in link_list:
                k = folder.split("_")[0]
                if k == str(date):
                    folders[folder] = st.checkbox(folder, key = str(folder))
                    if folders[folder]:
                        #Show video
                        items = os.listdir(org_link+folder+"/")
                        if "h264_" + folder + ".mp4" in items:
                            video_file = org_link + folder + "/"+ "h264_" + folder + ".mp4"
                        else:
                            org_video_file = org_link + folder + "/" + folder + ".mp4"
                            video_file = org_link+folder+"/h264_"+ folder + ".mp4"
                            os.system('ffmpeg -i ' + org_video_file + ' -vcodec libx264 -f mp4 '+ video_file)

                        output_file = open(video_file, "rb")
                        output_file = output_file.read()
                        st.video(output_file)
                        #Show csv
                        path_csv = "/home/phankimngan/Downloads/" # You can change path for csv files
                        dl_items = os.listdir(path_csv)
                        if video:
                            if "_data_outputs_" + folder + ".csv" in dl_items:
                                csv_file = path_csv + "_data_outputs_" + folder + ".csv"
                                output_file = pd.read_csv(csv_file,  encoding = 'utf-8')
                                st.write(output_file)
                        if webcam:
                            if folder + ".csv" in dl_items:
                                csv_file = path_csv + folder + ".csv"
                                output_file = pd.read_csv(csv_file,  encoding = 'utf-8')
                                st.write(output_file)



