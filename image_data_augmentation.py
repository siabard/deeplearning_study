import tensorflow as tf 
from tensorflow import keras 

import numpy as np
import matplotlib.pyplot as plt


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

gen  = ImageDataGenerator(rotation_range = 30, width_shift_range = 3.0, shear_range = 0.4, horizontal_flip=True)

img_array_list = []
img_names = ['./data/dog1.jpg', './data/dog2.jpg', './data/cat1.jpg', './data/cat2.jpg']

for i in range(len(img_names)):
    loaded_img = load_img(img_names[i], target_size = (100, 100))
    loaded_img_array = img_to_array(loaded_img) / 255.0
    img_array_list.append(loaded_img_array)

plt.figure(figsize=(6, 6))
for i in range(len(img_array_list)):
    plt.subplot(1, len(img_array_list), i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.title(img_names[i])
    plt.imshow(img_array_list[i])
plt.show()

batch_size = 2
data_gen = gen.flow(np.array(img_array_list), batch_size = batch_size)
img = data_gen.next()

plt.figure(figsize=(6, 6))

for i in range(len(img)):
    plt.subplot(1, len(img), i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img[i])
plt.show()