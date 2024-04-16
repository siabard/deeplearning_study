import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Flatten, Dense, Dropout 
from keras.optimizers.legacy import Adam


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

def augment_image(x_train, y_train):
    gen = ImageDataGenerator(rotation_range=20, shear_range=0.2, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
    augment_ratio = 1.5 # 전체 데이터의 150%
    augment_size = int(augment_ratio * x_train.shape[0])

    randidx = np.random.randint(x_train.shape[0], size = augment_size)

    x_augmented = x_train[randidx].copy()
    y_augmented = y_train[randidx].copy()

    x_augmented, y_augmented = gen.flow(x_augmented, y_augmented, batch_size = augment_size, shuffle=False).next()
    x_train = np.concatenate((x_train, x_augmented))
    y_train = np.concatenate((y_train, y_augmented))

    s = np.arange(x_train.shape[0])
    np.random.shuffle(s)

    x_train = x_train[s]
    y_train = y_train[s]

    return x_train, y_train

def read_model():
    # Load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # convert to tensor
    x_train = x_train.reshape((x_train.shape[0], 32, 32, 3))
    x_test = x_test.reshape((x_test.shape[0], 32, 32, 3))

    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return x_train, y_train, x_test, y_test

def build_model():
    cnn = Sequential()

    cnn.add(Conv2D(input_shape=(32, 32, 3), kernel_size=(3,3), filters=32, activation='relu', padding='same'))
    cnn.add(Conv2D(kernel_size=(3,3), filters=32, activation='relu', padding='same'))
    cnn.add(MaxPool2D(pool_size=(2,2)))
    cnn.add(Dropout(0.25))

    cnn.add(Conv2D(kernel_size=(3,3), filters=64, activation='relu', padding='same'))
    cnn.add(Conv2D(kernel_size=(3,3), filters=64, activation='relu', padding='same')) 
    cnn.add(MaxPool2D(pool_size=(2,2)))
    cnn.add(Dropout(0.25))

    cnn.add(Conv2D(kernel_size=(3,3), filters=128, activation='relu', padding='same'))
    cnn.add(MaxPool2D(pool_size=(2,2)))
    cnn.add(Dropout(0.25))
    cnn.add(Conv2D(kernel_size=(3,3), filters=128, activation='relu', padding='same'))
    cnn.add(MaxPool2D(pool_size=(2,2)))
    cnn.add(Dropout(0.25))
    cnn.add(Conv2D(kernel_size=(3,3), filters=256, activation='relu', padding='same'))
    cnn.add(MaxPool2D(pool_size=(2,2)))
    cnn.add(Dropout(0.25))

    cnn.add(Flatten())
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dropout(0.5))
    cnn.add(Dense(10, activation='softmax'))
    cnn.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return cnn

def plot_hist_accuracy(hist):
    import matplotlib.pyplot as plt

    plt.title('Accuracy Trend')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()

    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])

    plt.legend(['train', 'validation'], loc='best')
    plt.show()

def plot_hist_loss(hist):
    import matplotlib.pyplot as plt

    plt.title('Loss Trend')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])

    plt.legend(['train', 'validation'], loc='best')
    plt.show()

def train_model():
    (x_train, y_train, x_test, y_test) = read_model()
    (x_train, y_train) = augment_image(x_train, y_train)
    model = build_model()
    hist = model.fit(x_train, y_train, batch_size=256, epochs=250, validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test)

    plot_hist_accuracy(hist)
    plot_hist_loss(hist)

train_model()

