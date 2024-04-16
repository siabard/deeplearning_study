import tensorflow as tf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from tensorflow import keras 
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers.legacy import Adam

raw_df = pd.read_csv('./data/005930.KS_3MA_5MA.csv')

# outlier, missing value 처리
# 거래량이 0 인 경우 NaN으로 처리
raw_df['Volume'] = raw_df['Volume'].replace(0, np.nan)

# 각 Column 에서 0 인값 확인 
for col in raw_df.columns:
    missing_rows = raw_df.loc[raw_df[col] == 0].shape[0]
    print(col + ': ' + str(missing_rows))

# NaN Drop
raw_df = raw_df.dropna()

# 정규화 처리 
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scale_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', '3MA', '5MA', 'Volume']
scaled_df = scaler.fit_transform(raw_df[scale_cols])

scaled_df = pd.DataFrame(scaled_df, columns=scale_cols)

# Feature Column 및 label Column 정의
feature_cols = ['3MA', '5MA', 'Adj Close']
label_cols = ['Adj Close']

label_df = pd.DataFrame(scaled_df, columns = label_cols)
feature_df = pd.DataFrame(scaled_df, columns = feature_cols)

## DataFrame => numpy 변환
label_np = label_df.to_numpy()
feature_np = feature_df.to_numpy()

# 학습데이터 생성
def make_sequence_dataset(feature, label, window_size):
    feature_list = []
    label_list = []

    for i in range(len(feature) - window_size):
        feature_list.append(feature[i:i+window_size])
        label_list.append(label[i+window_size])

    return np.array(feature_list), np.array(label_list)

window_size = 40
X, Y = make_sequence_dataset(feature_np, label_np, window_size)

# 트레이닝 / 테스트 분리
split = -200
x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split:]
y_test = Y[split:]

# 모델 구성
model = Sequential()
model.add(LSTM(128, activation='tanh', input_shape = x_train[0].shape))
model.add(Dense(1, activation='linear'))

# 모델 컴파일 및 학습
from keras.callbacks import EarlyStopping
optimizer = Adam(learning_rate=0.00001)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=5)

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100, batch_size=16, callbacks=[early_stop])

## 주가 예측
pred = model.predict(x_test)

plt.figure(figsize=(12, 6))
plt.title('3MA + 5MA + Adj Close, window_size=40')
plt.xlabel('peroid')
plt.ylabel('adj close')
plt.plot(y_test, label='actual')
plt.plot(pred, label='prediction')
plt.grid()
plt.legend(loc='best')
plt.show()