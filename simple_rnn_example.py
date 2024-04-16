import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras as keras
from keras.layers import SimpleRNN, Dense
from keras import Sequential

def seq2dataset(seq, window, horizon):
    data_x = []
    data_y = []

    for i in range(len(seq) - (window + horizon) + 1):
        x = seq[i:(i + window)]
        y = (seq[i + window + horizon - 1])

        data_x.append(x)
        data_y.append(y)
    return np.array(data_x), np.array(data_y)

def define_model(x_data):
    model = Sequential()
    model.add(SimpleRNN(units=128, activation='tanh', input_shape=x_data[0].shape))
    model.add(Dense(1))
    model.summary()
    return model

# 데이터 구성 
x = np.arange(0, 100, 0.1)
y = 0.5 * np.sin(2* x) - np.cos(x / 2.0)

# RNN 입력에 필수인 3차원 텐서로 바꾸기 위해 
# reshape(-1, 1)을 사용해서 (1000, 1) 형태로 변경함 
seq_data = y.reshape(-1, 1)

X, Y = seq2dataset(seq_data, 3, 1)

split_ratio = 0.8
split = int(split_ratio * len(X))

x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split:]
y_test = Y[split:]

srnn = define_model(x_train)

srnn.compile(loss = 'mse', optimizer='adam', metrics=['mae'])
hist = srnn.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test))

pred = srnn.predict(x_test)

plt.plot(pred, label='prediction)')
plt.plot(y_test, label = 'label')
plt.grid() 
plt.legend(loc='best')
plt.show()