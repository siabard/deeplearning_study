# 데이터 불러오기 및 확인
import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras.datasets import mnist 
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import utils
from keras.optimizers.legacy import Adam

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("\n train shape =", x_train.shape, ", train label shape = ", y_train.shape)
print("\n test shape =", x_test.shape, ", test label shape = ", y_test.shape)

print("\n train label = ", y_train)
print("\n test label = ", y_test)

# 이미지 25개만 출력해보기 
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))

for index in range(25):
    plt.subplot(5, 5, index + 1)
    plt.imshow(x_train[index], cmap='gray')
    plt.axis('off')
plt.show()

# 전처리 / 정규화 

x_train = (x_train - 0.0) / (255.0 - 0.0)
y_test  = (y_test - 0.0) / (255.0 - 0.0)

# 원 핫 인코딩 
y_train = utils.to_categorical(y_train, num_classes = 10)
y_test = utils.to_categorical(y_test, num_classes = 10) 

# 모델 구축 및 컴파일 
model = Sequential()

model.add(Flatten(input_shape = (28, 28)))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 모델 학습

hist = model.fit(x_train, y_train, epochs = 30, validation_split = 0.3)

# 모델 정확도 평가
model.evaluate(x_test, y_test)

# 손실 및 정확도 

plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')

plt.legend(loc='best')
plt.show()



plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(hist.history['accuracy'], label='train accuracy')
plt.plot(hist.history['val_accuracy'], label='validation accuracy')

plt.legend(loc='best')
plt.show()

# 혼동행렬
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(6,6))
predicted_value = model.predict(x_test)

cm = confusion_matrix(np.argmax(y_test, axis=-1), np.argmax(predicted_value, axis=-1))
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

print(cm)
print('\n')

for i in range(10):
    print(('label = %d\t(%d/%d)\taccuracy = %.3f') % (i, np.max(cm[i]), mp.sum(cm[i]), np.max(cm[i]) / np.sum(cm[i])))