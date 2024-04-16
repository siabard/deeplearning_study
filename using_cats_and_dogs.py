import tensorflow as tf 

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.models import Sequential
from keras.applications import Xception
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D

IMG_WIDTH =224 
IMG_HEIGHT = 224

def create_model():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())

    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2  , activation='softmax'))
    return model 

model = create_model()

model.load_weights('./cats_and_dogs_filtered_Xception_Colab.h5')

# test model with validation image
import cv2 

img_list = ["./data/cat1.jpg", "./data/dog1.jpg", "./data/cat2.jpg", "./data/dog2.jpg"]
src_img_list = []

for i in range(len(img_list)):
    src_img = cv2.imread(img_list[i], cv2.IMREAD_COLOR)
    src_img = cv2.resize(src_img, dsize=(IMG_WIDTH, IMG_HEIGHT))
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    src_img = src_img / 255.5
    src_img_list.append(src_img)


# translate to 4 dimension tensor
src_img_array = np.array(src_img_list) 

pred = model.predict(src_img_array)
print(pred.shape)

for i in range(len(pred)):
    print(img_list[i], pred[i], np.argmax(pred[i]))
