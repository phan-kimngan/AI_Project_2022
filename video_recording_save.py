import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot
from streamlit_webrtc import VideoTransformerBase
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import webrtc_streamer,  WebRtcMode
import numpy as np
from tensorflow.keras.models import load_model
import time
import cv2
from streamlit_webrtc import VideoTransformerBase
import numpy as np
import streamlit as st
from mtcnn.mtcnn import MTCNN
from recog.inception_resnet_v1 import InceptionResNetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from numpy import load
from sklearn.svm import SVC
from numpy import asarray
from PIL import Image
from numpy import expand_dims
from webcam import webcam
from datetime import datetime, timedelta
from recog.load_emb_data import main
import pandas as pd
from configparser import ConfigParser
from mtcnn.mtcnn import MTCNN

config_object = ConfigParser()
model = load_model('final_model.h5')
#https://github.com/Creatrohit9/Face-Emotion-Recognition
model_reg = InceptionResNetV1(input_shape=(160, 160, 3),
                      classes=128,
                      dropout_keep_prob=0.8,
                      weights_path='facenet_weights.h5')

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def convert_df(df):
    return df.to_csv().encode('utf-8')

def app():
    header = st.container()
    add = st.container()
    start = st.container()
    update = st.container()

    with header:
        main_logo_path = r'./img/ZoomClassIllustration.jpg'
        main_logo = Image.open(main_logo_path).resize((700, 400))
        st.title('VIDEO RECODING')
        st.image(main_logo)
        st.header("Choose option")

        attendance = st.checkbox('Attendance', key="10")
        emotion = st.checkbox('Emotion', key="11")
        emotion_attendance = st.checkbox('Attendance & Emotion', key="12")
    with add:
        st.header("Upload video recoding")
        files = st.file_uploader('Please upload your video recoding')
        check_similar_fre = 0
        if files is not None:
            st.write('**Uploading...**')
            input_path_video = "./data/inputs/" + files.name
            video_file = open(input_path_video, "rb")
            video_bytes = video_file.read()
            st.video(video_bytes)
            st.text("Here is the video recoding you've selected.")# cv2.VideoWriter_fourcc(*'mp4v')

            lists = os.listdir('./recog/students/')
            df = pd.DataFrame(columns = ['Date', 'Join time', 'Duration time', "Attentiveness score(%)"], index = lists)
            dict_delta_time = {values : 0 for values in lists}
            dict_last_time = {values : 0 for values in lists}
            model = load_model('final_model.h5')

            with start:
                if st.button('Start'):
                    st.header("Face Recognition Process")
                    if emotion :
                        op = st.text("Perform an emotion classification task")
                        st.text(" Wait a few minutes...")
                    elif attendance:
                        st.text("Perform an identification task")
                        st.text(" Wait a few minutes...")
                    elif emotion_attendance:
                        st.text("Perform identification and emotion classification task")
                        st.text(" Wait a few minutes...")
                    else:
                        st.text("Please choose option")


                    cap = cv2.VideoCapture(input_path_video)
                    run_version_training = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    if not os.path.exists("./data/outputs/video_recorder/"+run_version_training):
                        os.makedirs("./data/outputs/video_recorder/"+run_version_training)
                        os.makedirs("./data/outputs_test/video_recorder/"+run_version_training)

                    output_path_video = "./data/outputs_test/video_recorder/" + run_version_training +"/" + run_version_training + ".mp4"
                    save_new_path = "./data/outputs/video_recorder/" + run_version_training +"/" + run_version_training + ".mp4"
                    output_path_csv = "./data/outputs/video_recorder/" + run_version_training +"/" + run_version_training + ".csv"
                    save_path = output_path_video

                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    out = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*"mp4v"), 12, (frame_width, frame_height))

                    ids = []
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    total_time = frame_count / fps

                    data = load('./recog/dataset-embeddings.npz')
                    trainx, trainy = data['arr_0'],data['arr_1']
                    in_encode = Normalizer(norm='l2')
                    trainx = in_encode.transform(trainx)
                    out_encode = LabelEncoder()
                    out_encode.fit(trainy)
                    trainy = out_encode.transform(trainy)
                    model_svm =SVC(kernel='linear', probability=True)
                    model_svm.fit(trainx,trainy)
                    if emotion or attendance or emotion_attendance:
                        start_time = datetime.now()
                        num_frames = 0
                        while(cap.isOpened()):
                            ret, frame = cap.read()
                            num_frames = num_frames + 1
                            if num_frames%5 == 0:
                                if ret == True:
                                    class_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
                                    detector = MTCNN()
                                    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                                    faces = detector.detect_faces(frame)
                                    preds_svm_list = []
                                    for i in faces:
                                        x, y, w, h = i['box']
                                        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,255, 0), 2)
                                        face = gray[y:y + h, x:x + w]
                                        face_ = frame[y:y + h, x:x + w]
                                        face_ = Image.fromarray(face_,'RGB')
                                        face_ = face_.resize((160,160))
                                        face_ = asarray(face_)

                                        testx = face_.reshape(-1,160,160,3)
                                        list_testx = list ()
                                        for test_pixels in testx:
                                            testx = test_pixels.astype('float32')
                                            mean = test_pixels.mean()
                                            std  = test_pixels.std()
                                            test_pixels = (test_pixels - mean)/std
                                            samples = expand_dims(test_pixels,axis=0)
                                            yhat = model_reg.predict(samples)
                                            list_testx.append(yhat[0])
                                        list_testx = asarray(list_testx)
                                        list_testx = in_encode.transform(list_testx)

                                        k = [findEuclideanDistance(list_testx, x) for x in trainx]
                                        check_similar = [i for i in k if i<=1]
                                        if len(check_similar) == 0:
                                            check_similar_fre = check_similar_fre + 1
                                            preds_svm = "unknown"

                                        else :
                                            preds_svm = model_svm.predict(list_testx)
                                            preds_svm = out_encode.inverse_transform(preds_svm)[0]

                                        face = cv2.resize(face,(48,48))
                                        face = np.expand_dims(face,axis=0)
                                        face = face/255.0
                                        face = face.reshape(face.shape[0],48,48,1)
                                        preds = model.predict(face)[0]
                                        label = class_labels[preds.argmax()]
                                        font = cv2.FONT_HERSHEY_SIMPLEX
                                        preds_svm_list.append(preds_svm)
                                        if emotion:
                                            cv2.putText(frame, label, (x,y-2), font, 0.8, (0,0,225), 2, cv2.LINE_4)
                                            out.write(frame)

                                        if attendance:
                                            cv2.putText(frame, preds_svm, (x,y-2), font, 0.8, (0,0,225), 2, cv2.LINE_4)
                                            out.write(frame)
                                        if emotion_attendance:
                                            cv2.putText(frame, "Name: " + preds_svm, (x,y-25), font, 0.8, (0,0,225), 2, cv2.LINE_4)
                                            cv2.putText(frame, "Emotion: " + label, (x,y-2), font, 0.8, (0,0,225), 2, cv2.LINE_4)
                                            out.write(frame)

                                    for i in preds_svm_list:
                                        if i not in ids:
                                            ids.append(i)
                                            df.at[i,'Date'] = datetime.now().strftime("%Y/%m/%d")
                                            df.at[i,'Join time'] = datetime.now().strftime("%H:%M:%S")
                                            dict_last_time[i] = datetime.now()
                                            dict_delta_time[i] = timedelta(0)
                                        else:
                                            time_curent = datetime.now()
                                            delta_time = time_curent - dict_last_time[i]
                                            dict_last_time[i] = datetime.now()
                                            dict_delta_time[i] = dict_delta_time[i] + delta_time
                                    print(preds_svm_list)
                                    #for i in list(dict_delta_time.keys()):
                                        #if dict_delta_time[i]!=0:
                                            #print(i)
                                            #print(round(dict_delta_time[i].total_seconds()))
                                else:
                                    end_time = datetime.now()
                                    break

                        cap.release()
                        out.release()
                        #Convert to H264

                        os.system('ffmpeg -i ' + save_path + ' -vcodec libx264 -f mp4 '+ save_new_path)


                        output_file = open(save_new_path, "rb")
                        output_bytes = output_file.read()
                        st.video(output_bytes)

                        training_time = (end_time - start_time).total_seconds()
                        for i in list(dict_delta_time.keys()):
                            df.at[i,'Duration time'] = round(dict_delta_time[i].total_seconds()*total_time/training_time,2) if dict_delta_time[i]!=0 else 0.0
                            df.at[i,'Attentiveness score(%)'] = round(dict_delta_time[i].total_seconds()/training_time*100,2) if dict_delta_time[i]!=0 else 0.0

                        df["Date"] = df["Date"].replace(np.nan, str(0.0000))
                        df["Join time"] = df["Join time"].replace(np.nan, str(0.0000))

                        #st.write("Class has {} participants in duration: {} ".format(len(ids), ", ".join(ids)))
                        #st.write("Video duration is {} seconds".format(total_time))
                        st.dataframe(df)


                        #save df by button
                        csv = convert_df(df)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=output_path_csv,
                            mime='text/csv',
                        )


                        if "unknown" in ids:
                            config_object.read("config.ini")
                            info = config_object["UNKNOWN"]
                            info["unknown"] = str(True)
                            with open('config.ini', 'w') as conf:
                                config_object.write(conf)
    with update:
        config_object.read("config.ini")
        if config_object["UNKNOWN"]["unknown"] == "True":
            st.write("Do you want to add the profile of new students?")
            yes = st.button ("I want", key = "26")
            if yes:

                info = config_object["CLICK"]
                info["click"] = str(0)
                with open('config.ini', 'w') as conf:
                    config_object.write(conf)
