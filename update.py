import os
from datetime import datetime, timedelta
from PIL import Image
import streamlit as st
from recog.load_emb_data import main
from configparser import ConfigParser

def app():

    config_object = ConfigParser()
    config_object.read("config.ini")
    unknown = config_object["UNKNOWN"]
    unknown["unknown"] = "None"
    with open('config.ini', 'w') as conf:
        config_object.write(conf)

    with open('config.ini', 'w') as conf:
        config_object.write(conf)
    header = st.container()
    update = st.container()
    with header:
        st.title('PROFILE STUDENT')
    with update:

        st.subheader('1. Enter ID of student')
        title = st.text_input('Enter ID of student')
        lists = os.listdir('./recog/students/')

        #upload image from folder
        st.subheader('2.1. Update database from the folder')
        image = st.file_uploader('Upload a picture from the folder')
        if image is not None:
            input_path_img = "./data/inputs/test_img/" + image.name
            img = Image.open(input_path_img)
            st.write("Got an image from the folder:")
            st.image(img)
            col1, col2 = st.columns(2)
            with col1:
                add = st.button ("Add to database", key = "13")
            with col2:
                finish = st.button ("Finish", key = "14")

            if add:
                if title:
                    newpath = r'./recog/students/' + title
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)
                    run_version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    with open (os.path.join("./recog/students/",title + "/" + str(run_version) + ".jpg"),'wb') as file:
                        file.write(image.getbuffer())
                    st.write("Saved picture.")
                else:
                    st.write("Please enter ID of student")
            if finish:
                main()
                st.write("Updated database")

        #upload image from webcam
        st.subheader('2.2. Update database from the webcam')
        picture = st.camera_input("Take a picture from webcam")
        if picture is not None:
            newpath = r'./recog/students/' + title
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            st.write("Got an image from the webcam:")
            st.image(picture)
            col1, col2 = st.columns(2)
            with col1:
                add = st.button ("Add to database", key = "15")
            with col2:
                finish = st.button ("Finish", key = "16")
            if add:
                if title:
                    run_version = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    with open (os.path.join("./recog/students/",title + "/" + str(run_version) + ".jpg"),'wb') as file:
                        file.write(picture.getbuffer())
                    st.write("Saved picture")
                else:
                    st.write("Please enter ID of student")
            if finish:
                #Update database
                main()
                #Done
                st.write("Updated database")
    click = config_object["CLICK"]
    click["click"] = "1"
    with open('config.ini', 'w') as conf:
        config_object.write(conf)
