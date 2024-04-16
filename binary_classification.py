import tensorflow as tf
import numpy as np

from tensorflow import keras 

from keras.models import Sequential
from keras.layers import Flatten, Dense 
from keras.optimizers.legacy import SGD 

x_data = np.array([2,4,6,8,10, 12, 14, 16, 18, 20]).astype('float32')
y_data = np.array([0,0,0,0,0,  1, 1, 1, 1, 1]).astype('float32')

# Model 구성 
model = Sequential()
model.add(Dense(8, input_shape=(1,), activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Model 컴파일 
model.compile(SGD(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Model 학습
model.fit(x_data, y_data, epochs = 500)

# Model 검증

test_data = np.array([0.5, 3.0, 3.5, 11.0, 13.0, 31.0])
sigmoid_value = model.predict(test_data)

logical_value = tf.cast(sigmoid_value > 0.5, dtype = tf.float32)

for i in range(len(test_data)):
    print(test_data[i], sigmoid_value[i], logical_value.numpy()[i])