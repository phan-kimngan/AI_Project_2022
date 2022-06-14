from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from PIL import Image 
from numpy import savez_compressed
from numpy import asarray
from os import listdir
from mtcnn.mtcnn import MTCNN

#Method to extract Face
def extract_image(image):
    img1 = Image.open(image)            #open the image
    img1 = img1.convert('RGB')          #convert the image to RGB format 
    pixels = asarray(img1)              #convert the image to numpy array
    detector = MTCNN()                  #assign the MTCNN detector
    f = detector.detect_faces(pixels)
    #fetching the (x,y)co-ordinate and (width-->w, height-->h) of the image
    x1,y1,w,h = f[0]['box']             
    x1, y1 = abs(x1), abs(y1)
    x2 = abs(x1+w)
    y2 = abs(y1+h)
    #locate the co-ordinates of face in the image
    store_face = pixels[y1:y2,x1:x2]
    plt.imshow(store_face)
    image1 = Image.fromarray(store_face,'RGB')    #convert the numpy array to object
    image1 = image1.resize((160,160))             #resize the image
    face_array = asarray(image1)                  #image to array
    return face_array


#Method to fetch the face
def load_faces(directory):
    face = []
    i=1
    for filename in listdir(directory):
        path = directory + filename
        faces = extract_image(path)
        face.append(faces)
    return face


#Method to get the array of face data(trainX) and it's labels(trainY)
def load_dataset(directory):
    x, y = [],[]
    i=1
    for subdir in listdir(directory):
        path = directory + subdir + '/'
        #load all faces in subdirectory
        faces = load_faces(path)
        #create labels
        labels = [subdir for _ in range(len(faces))]
        #summarize
        print("%d There are %d images in the class %s:"%(i,len(faces),subdir))
        x.extend(faces)
        y.extend(labels)
        i=i+1
    return asarray(x),asarray(y)  


#load the datasets
trainX,trainY = load_dataset('./students/')
savez_compressed('dataset.npz',trainX,trainY)
