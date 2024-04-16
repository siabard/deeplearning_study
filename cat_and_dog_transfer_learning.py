import tensorflow as tf 

from tensorflow import keras 

from keras.models import Sequential
from keras.applications import Xception
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

IMG_WIDTH =224 
IMG_HEIGHT = 224

base_model = Xception(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())

model.add(Dense(16, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2  , activation='softmax'))

train_dir = './data/cats_and_dogs_filtered/train'
test_dir = './data/cats_and_dogs_filtered/validation'

train_data_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1)
test_data_gen = ImageDataGenerator(rescale=1./255)

train_data = train_data_gen.flow_from_directory(train_dir, batch_size=32, color_mode='rgb', shuffle=True, class_mode='categorical', target_size=(IMG_WIDTH, IMG_HEIGHT))
test_data = test_data_gen.flow_from_directory(test_dir, batch_size=32, color_mode='rgb', shuffle=False, class_mode='categorical')

model.compile(optimizer=keras.optimizers.Adam(2e-5), loss='categorical_crossentropy', metrics=['accuracy'])

from datetime import datetime 
from keras.callbacks import ModelCheckpoint, EarlyStopping

save_file_name = './cats_and_dogs_filtered_Xception_Colab.h5'
checkpoint = ModelCheckpoint(save_file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
earlyshooting = EarlyStopping(monitor='val_loss', patience=5)

hist = model.fit(train_data, epochs=30, validation_data=test_data, callbacks=[checkpoint, earlyshooting])
