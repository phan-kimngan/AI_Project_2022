import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from PIL import  Image
from plotly.offline import iplot
from streamlit_webrtc import VideoTransformerBase
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import webrtc_streamer,  WebRtcMode
import numpy as np
from tensorflow.keras.models import load_model
import time
import cv2
from streamlit_webrtc import VideoTransformerBase
from tensorflow.keras.models import load_model
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
#import h264decoder
#loading the pre-trained model file
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
    check_similar_fre = 0
    with header:
        main_logo_path = r'./img/ZoomClassIllustration.jpg'
        main_logo = Image.open(main_logo_path).resize((600, 400))
        st.title('Video Recording')
        st.image(main_logo)
        st.header("Choose option")

        attendance = st.checkbox('Attendance', key="10")
        emotion = st.checkbox('Emotion', key="11")
        emotion_attendance = st.checkbox('Attendance & Emotion', key="12")
    with add:
        st.header("Upload video recoding.")
        files = st.file_uploader('Please upload your video recoding')

        check_similar_fre = 0
        if files is not None:
            st.write('**Uploading...**')
            input_path_video = "./data/inputs/" + files.name
            video_file = open(input_path_video, "rb")
            video_bytes = video_file.read()
            st.video(video_bytes)
            st.text("Here is the video recoding you've selected ...")

            cap = cv2.VideoCapture(input_path_video)
            run_version_training = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if not os.path.exists("./data/outputs/"+run_version_training):
                os.makedirs("./data/outputs/"+run_version_training)
            output_path_video = "./data/outputs/" + run_version_training +"/" + run_version_training + "_" + files.name
            output_path_csv = "./data/outputs/" + run_version_training +"/" + run_version_training + "_" + files.name.split(".")[0]+ ".csv"

            save_path = output_path_video
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))


            ids = []
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            total_time = frame_count / fps
            lists = os.listdir('./recog/students/')

            df = pd.DataFrame(columns = ['Date', 'Join time', 'Duration time', "Attentiveness score(%)"], index = lists)
            dicts = {values : 0 for values in lists}

            model = load_model('final_model.h5')
            with start:
                if st.button('Start'):
                    data = load('./recog/dataset-embeddings.npz')
                    trainx, trainy = data['arr_0'],data['arr_1']
                    in_encode = Normalizer(norm='l2')
                    trainx = in_encode.transform(trainx)
                    out_encode = LabelEncoder()
                    out_encode.fit(trainy)
                    trainy = out_encode.transform(trainy)
                    model_svm =SVC(kernel='linear', probability=True)
                    model_svm.fit(trainx,trainy)

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


                    if emotion or attendance or emotion_attendance:
                        start_time = datetime.now()
                        while(cap.isOpened()):
                            ret, frame = cap.read()
                            if ret == True:
                                class_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
                                faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                                faces = faceCascade.detectMultiScale(gray,1.1,4)
                                for(x, y, w, h) in faces:
                                    print(x,y,w,h)
                                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255, 0), 2)
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


                                    if preds_svm:
                                        if preds_svm not in ids:
                                            ids.append(preds_svm)
                                            df.at[preds_svm,'Date'] = datetime.now().strftime("%Y/%m/%d")
                                            df.at[preds_svm,'Join time'] = datetime.now().strftime("%H:%M:%S")
                                            last_time = datetime.now()
                                            dicts[preds_svm] = timedelta(0)
                                        else:
                                            time_curent = datetime.now()
                                            delta_time = time_curent - last_time
                                            last_time = datetime.now()
                                            dicts[preds_svm] = dicts[preds_svm] + delta_time
                                            print(dicts[preds_svm].total_seconds())
                                    print(preds_svm)
                                    if emotion:
                                        cv2.putText(frame, label, (x,y), font, 0.5, (0,225,0), 1, cv2.LINE_4)
                                        out.write(frame)
                                        print(out)
                                    if attendance:
                                        cv2.putText(frame, preds_svm, (x,y), font, 0.5, (0,225,0), 1, cv2.LINE_4)
                                        out.write(frame)
                                    if emotion_attendance:
                                        cv2.putText(frame, "Name: " + preds_svm + "/" + "Emotion: " + label, (x,y), font, 0.5, (0,225,0), 1, cv2.LINE_4)
                                        out.write(frame)

                            else:
                                end_time = datetime.now()
                                break

                        cap.release()
                        out.release()

                        output_file = open(save_path, "rb")
                        output_bytes = output_file.read()
                        st.video(output_bytes)

                        training_time = (end_time - start_time).total_seconds()
                        for i in list(dicts.keys()):
                            df.at[i,'Duration time'] = round(dicts[i].total_seconds()*total_time/training_time,2) if dicts[i]!=0 else 0.0
                            df.at[i,'Attentiveness score(%)'] = round(dicts[i].total_seconds()/training_time*100,2) if dicts[i]!=0 else 0.0

                        df["Date"] = df["Date"].replace(np.nan, str(0.0000))
                        df["Join time"] = df["Join time"].replace(np.nan, str(0.0000))

                        st.write("Found {} faces contain : {}".format(len(faces), preds_svm))
                        st.write("Class has {} participants in duration".format(len(ids)))
                        st.write("Video duration is {} seconds".format(total_time))
                        st.dataframe(df)
                        csv = convert_df(df)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=output_path_csv,
                            mime='text/csv',
                        )






    if 1 > 0:
        with update:
            st.header('Do you want to add a new student?')

            st.subheader('1. Enter ID of student')
            title = st.text_input('Enter ID of student')
            lists = os.listdir('./recog/students/')

            #upload image from folder
            st.subheader('2.1. Update database from the folder')
            image = st.file_uploader('Upload a picture from the folder')
            if image is not None:
                newpath = r'./recog/students/' + title
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                img = Image.open(image)
                st.write("Got an image from the folder:")
                st.image(img)
                col1, col2 = st.columns(2)
                with col1:
                    add = st.button ("Add to database", key = "13")
                with col2:
                    finish = st.button ("Finish", key = "14")
                if add:
                    if title:
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
                #k = 0
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
                    main()
                    st.write("Updated database")




    #if cancel:
        #st.write('**Cancel prediction!**')

    #with model:
        #logo_path = r'./img/bee-icon-free-honey-bee-icon-invertebrate-animal-insect-wasp-transparent-png-879795.png'
        #logo = Image.open(logo_path)
        #resize_logo = logo.resize((100, 100))
        #st.sidebar.image(resize_logo)
        #st.sidebar.selectbox('Select your prediction model:',
                             #('CNN', 'AlexNet', 'LeNet'))
        #thresh = st.sidebar.slider('Select your prediction threshold:', 0.0, 1.0, 0.1)

    #with add:
        #file = st.file_uploader('Upload your image you want to predict')
        #if file:
            #file = r'./img/bee-icon-free-honey-bee-icon-invertebrate-animal-insect-wasp-transparent-png-879795.png'
            #img = Image.open(file)
            #st.title("Here is the image you've selected")
            #resized_image = img.resize((336, 336))
            #st.image(resized_image)

    #st.write('Select your task:')
    #subspecies = st.checkbox('Bee subspecies recognition')
    #health = st.checkbox('Bee health recognition')

    #if subspecies:
        #d1 = [['-1', '75%'], ['1 mixed local stock 2', '10%'], ['carniolan honey bee', '7%'], ['Italian honey bee', '5%'],
             #['Russian honey bee', '1.5%'], ['VHS Italian honey bee', '0.5%'],
             #['Western honey bee', '1%']]
        #df1 = pd.DataFrame(d1, columns=['Predicted subspecies', 'Confident score'])
    #if health:
        #d2 = [['ant problem', '75%'], ['few varrao', '10%'], ['healthy', '7%'],
             #['hive being robbed', '5%'], ['missing queen', '1.5%'], ['Varroa, small hive beetles', '0.5%']]
        #df2 = pd.DataFrame(d2, columns=['Predicted health', 'Confident score'])


    #if predict:
        #if subspecies and health:
            #st.write(df1)
            #st.write(df2)
        #elif subspecies:
            #st.write(df1)
        #elif health:
            #st.write(df2)
        #else:
            #st.write('**no task is selected!**')

