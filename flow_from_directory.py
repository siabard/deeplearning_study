import tensorflow as tf 
from tensorflow import keras 


import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

gen = ImageDataGenerator(rotation_range=30, width_shift_range=0.3, shear_range=0.3, rescale=1. / 255)
data_path = './data/test_dir'
batch_size = 3

data_gen = gen.flow_from_directory(directory=data_path, batch_size=batch_size, shuffle=True, target_size=(100, 100), class_mode='categorical')

img, label = data_gen.next()

plt.figure(figsize=(6, 6))
for i in range(len(img)):
    plt.subplot(1, len(img), i+1)
    plt.xticks([]); plt.yticks([])
    plt.title(str(np.argmax(label[i])))
    plt.imshow(img[i])

plt.show()