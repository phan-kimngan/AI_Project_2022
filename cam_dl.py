import cv2
from streamlit_webrtc import VideoTransformerBase
from tensorflow.keras.models import load_model
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
import numpy as np
import time
import pandas as pd
import os
from datetime import datetime, timedelta
from configparser import ConfigParser

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance
def update(data):
    data = data
    trainx, trainy = data['arr_0'],data['arr_1']

    trainx = in_encode.transform(trainx)

    out_encode.fit(trainy)
    trainy = out_encode.transform(trainy)

    model_svm.fit(trainx,trainy)
    return trainx, trainy
def save_out(run_version_training):
    if not os.path.exists("./data/outputs/webcam/" + run_version_training):
        os.makedirs("./data/outputs/webcam/" + run_version_training)
    output_path_video = "./data/outputs/webcam/" + run_version_training +"/" + run_version_training + ".mp4"
    out = cv2.VideoWriter(output_path_video, cv2.VideoWriter_fourcc(*'mp4v'), 150, (640, 480))
    return out


model = load_model('final_model.h5')
model_reg = InceptionResNetV1(input_shape=(160, 160, 3),
                      classes=128,
                      dropout_keep_prob=0.8,
                      weights_path='./recog/facenet_weights.h5')

in_encode = Normalizer(norm='l2')
out_encode = LabelEncoder()
model_svm =SVC(kernel='linear', probability=True)

update_data = load('./recog/dataset-embeddings.npz')
trainx, trainy = update(update_data)


class VideoTransformer_emotion(VideoTransformerBase):
        def __init__(self):
            self.trainx = trainx
            self.trainy = trainy
            self.ids = []
            self.lists = os.listdir('./recog/students/')
            self.df = pd.DataFrame(columns = ['Date', 'Join time', 'Duration time', "Attentiveness score(%)"], index = self.lists)
            self.dict_delta_time = {values : 0 for values in self.lists}
            self.dict_last_time = {values : 0 for values in self.lists}
            self.start_time = datetime.now()
            self.run_version_training = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.out = save_out(self.run_version_training)
        def update_database(self, update_data):
            self.trainx, self.trainy = update(update_data)
        def transform(self, frame):
            frame = frame.to_ndarray(format="bgr24")
            class_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
            detector = MTCNN()
            #detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = detector.detect_faces(frame)
            #faces = detector.detectMultiScale(gray,1.6,4)
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

                k = [findEuclideanDistance(list_testx, x) for x in self.trainx]
                check_similar = [i for i in k if i<=1]
                if len(check_similar) == 0:

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
                cv2.putText(frame, "Emotion: " + label, (x ,y - 2), font, 0.8 , (0,0,255), 2, cv2.LINE_4)
                self.out.write(frame)
            for i in preds_svm_list:
                if i not in self.ids:
                    self.ids.append(i)
                    self.df.at[i,'Date'] = datetime.now().strftime("%Y/%m/%d")
                    self.df.at[i,'Join time'] = datetime.now().strftime("%H:%M:%S")
                    self.dict_last_time[i] = datetime.now()
                    self.dict_delta_time[i] = timedelta(0)
                else:
                    time_curent = datetime.now()
                    delta_time = time_curent - self.dict_last_time[i]
                    self.dict_last_time[i] = datetime.now()
                    self.dict_delta_time[i] = self.dict_delta_time[i] + delta_time
            return frame
        def update_csv(self):
            if "unknown" in self.ids:
                config_object = ConfigParser()
                config_object.read("config.ini")
                info = config_object["UNKNOWN"]
                info["unknown"] = str(True)
                with open('config.ini', 'w') as conf:
                    config_object.write(conf)

            self.end_time = datetime.now()
            total_time = (self.end_time - self.start_time).total_seconds()
            for i in list(self.dict_delta_time.keys()):
                self.df.at[i,'Duration time'] = round(self.dict_delta_time[i].total_seconds(),2) if self.dict_delta_time[i]!=0 else 0.0
                self.df.at[i,'Attentiveness score(%)'] = round(self.dict_delta_time[i].total_seconds()/total_time*100,2) if self.dict_delta_time[i]!=0 else 0.0

            self.df["Date"] = self.df["Date"].replace(np.nan, str(0.0000))
            self.df["Join time"] = self.df["Join time"].replace(np.nan, str(0.0000))

            #st.write("Found {} faces contain : {}".format(len(faces), preds_svm))
            #st.write("Class has {} participants in duration: {} ".format(len(self.ids), ", ".join(self.ids)))
            #st.write("Video duration is {} seconds".format(total_time))

            return self.df
        def run_version(self):
            return self.run_version_training

class VideoTransformer_attendance(VideoTransformerBase):
        def __init__(self):
            self.trainx = trainx
            self.trainy = trainy
            self.ids = []
            self.lists = os.listdir('./recog/students/')
            self.df = pd.DataFrame(columns = ['Date', 'Join time', 'Duration time', "Attentiveness score(%)"], index = self.lists)
            self.dict_delta_time = {values : 0 for values in self.lists}
            self.dict_last_time = {values : 0 for values in self.lists}
            self.start_time = datetime.now()
            self.run_version_training = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.out = save_out(self.run_version_training)
        def update_database(self, update_data):
            self.trainx, self.trainy = update(update_data)
        def transform(self, frame):
            frame = frame.to_ndarray(format="bgr24")
            class_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
            detector = MTCNN()
            #detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = detector.detect_faces(frame)
            #faces = detector.detectMultiScale(gray,1.6,4)
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

                k = [findEuclideanDistance(list_testx, x) for x in self.trainx]
                check_similar = [i for i in k if i<=1]
                if len(check_similar) == 0:

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
                cv2.putText(frame, "Name: "+preds_svm, (x ,y - 2), font, 0.8 , (0,0,255), 2, cv2.LINE_4)
                self.out.write(frame)
            for i in preds_svm_list:
                if i not in self.ids:
                    self.ids.append(i)
                    self.df.at[i,'Date'] = datetime.now().strftime("%Y/%m/%d")
                    self.df.at[i,'Join time'] = datetime.now().strftime("%H:%M:%S")
                    self.dict_last_time[i] = datetime.now()
                    self.dict_delta_time[i] = timedelta(0)
                else:
                    time_curent = datetime.now()
                    delta_time = time_curent - self.dict_last_time[i]
                    self.dict_last_time[i] = datetime.now()
                    self.dict_delta_time[i] = self.dict_delta_time[i] + delta_time
            return frame
        def update_csv(self):
            if "unknown" in self.ids:
                config_object = ConfigParser()
                config_object.read("config.ini")
                info = config_object["UNKNOWN"]
                info["unknown"] = str(True)
                with open('config.ini', 'w') as conf:
                    config_object.write(conf)

            self.end_time = datetime.now()
            total_time = (self.end_time - self.start_time).total_seconds()
            for i in list(self.dict_delta_time.keys()):
                self.df.at[i,'Duration time'] = round(self.dict_delta_time[i].total_seconds(),2) if self.dict_delta_time[i]!=0 else 0.0
                self.df.at[i,'Attentiveness score(%)'] = round(self.dict_delta_time[i].total_seconds()/total_time*100,2) if self.dict_delta_time[i]!=0 else 0.0

            self.df["Date"] = self.df["Date"].replace(np.nan, str(0.0000))
            self.df["Join time"] = self.df["Join time"].replace(np.nan, str(0.0000))
            #st.write("Class has {} participants in duration: {} ".format(len(self.ids), ", ".join(self.ids)))
            #st.write("Video duration is {} seconds".format(total_time))

            return self.df
        def run_version(self):
            return self.run_version_training

class VideoTransformer_both(VideoTransformerBase):
        def __init__(self):
            self.trainx = trainx
            self.trainy = trainy
            self.ids = []
            self.lists = os.listdir('./recog/students/')
            self.df = pd.DataFrame(columns = ['Date', 'Join time', 'Duration time', "Attentiveness score(%)"], index = self.lists)
            self.dict_delta_time = {values : 0 for values in self.lists}
            self.dict_last_time = {values : 0 for values in self.lists}
            self.start_time = datetime.now()
            self.run_version_training = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.out = save_out(self.run_version_training)
        def update_database(self, update_data):
            self.trainx, self.trainy = update(update_data)
        def transform(self, frame):
            frame = frame.to_ndarray(format="bgr24")
            class_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
            detector = MTCNN()
            #detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = detector.detect_faces(frame)
            #faces = detector.detectMultiScale(gray,1.6,4)
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

                k = [findEuclideanDistance(list_testx, x) for x in self.trainx]
                check_similar = [i for i in k if i<=1]
                if len(check_similar) == 0:

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
                cv2.putText(frame, "Name: " + preds_svm, (x,y-25), font, 0.8, (0,0,225), 2, cv2.LINE_4)
                cv2.putText(frame, "Emotion: " + label, (x,y-2), font, 0.8, (0,0,225), 2, cv2.LINE_4)
                self.out.write(frame)
            for i in preds_svm_list:
                if i not in self.ids:
                    self.ids.append(i)
                    self.df.at[i,'Date'] = datetime.now().strftime("%Y/%m/%d")
                    self.df.at[i,'Join time'] = datetime.now().strftime("%H:%M:%S")
                    self.dict_last_time[i] = datetime.now()
                    self.dict_delta_time[i] = timedelta(0)
                else:
                    time_curent = datetime.now()
                    delta_time = time_curent - self.dict_last_time[i]
                    self.dict_last_time[i] = datetime.now()
                    self.dict_delta_time[i] = self.dict_delta_time[i] + delta_time
            return frame
        def update_csv(self):
            if "unknown" in self.ids:
                config_object = ConfigParser()
                config_object.read("config.ini")
                info = config_object["UNKNOWN"]
                info["unknown"] = str(True)
                with open('config.ini', 'w') as conf:
                    config_object.write(conf)

            self.end_time = datetime.now()
            total_time = (self.end_time - self.start_time).total_seconds()
            for i in list(self.dict_delta_time.keys()):
                self.df.at[i,'Duration time'] = round(self.dict_delta_time[i].total_seconds(),2) if self.dict_delta_time[i]!=0 else 0.0
                self.df.at[i,'Attentiveness score(%)'] = round(self.dict_delta_time[i].total_seconds()/total_time*100,2) if self.dict_delta_time[i]!=0 else 0.0

            self.df["Date"] = self.df["Date"].replace(np.nan, str(0.0000))
            self.df["Join time"] = self.df["Join time"].replace(np.nan, str(0.0000))

            return self.df
        def run_version(self):
            return self.run_version_training
