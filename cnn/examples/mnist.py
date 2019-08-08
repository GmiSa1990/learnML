

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_eval, y_eval) = mnist.load_data()
x_train = np.reshape(x_train, [x_train.shape[0], 28, 28, 1]).astype(np.float32)
x_eval = np.reshape(x_eval, [x_eval.shape[0], 28, 28, 1]).astype(np.float32)
x_train /= 255
x_eval /= 255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_eval = tf.keras.utils.to_categorical(y_eval, 10)
input_shape = (28,28,1)
# print(tf.keras.backend.image_data_format())

class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # When using this layer as the first layer in a model,
        # provide the keyword argument `input_shape`
        self.conv1 = Conv2D(filters=32, kernel_size=(3,3),
                            activation='relu',
                            input_shape=input_shape)
        self.conv2 = Conv2D(filters=64, kernel_size=(2,2),
                            activation='relu')
        self.pool1 = MaxPooling2D(pool_size=(2,2))
        self.fltn = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.drop1 = Dropout(0.5)
        self.dense2 = Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.fltn(x)
        x = self.dense1(x)
        x = self.drop1(x)
        return self.dense2(x)

    def predict(self, inputs):
        pass

batch_size = 128
epochs = 1

model = CNN()

model.compile(optimizer=tf.keras.optimizers.Adadelta(),
                loss=categorical_crossentropy,
                metrics=['accuracy'])

history=model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs)

score = model.evaluate(x=x_eval, y=y_eval)
print('the score is %f' % score[1])
