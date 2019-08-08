
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, LSTM

# enable eager execution mode.
tf.enable_eager_execution()

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# preprocessing training data and evaluation data.
(x_train, y_train), (x_eval, y_eval) = mnist.load_data()
x_train = x_train.astype(np.float32)
x_train /= 255
y_train = y_train.astype(np.int32)
x_eval = x_eval.astype(np.float32)
x_eval /= 255
y_eval = y_eval.astype(np.int32)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 56

class MyModel(tf.keras.models.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(units=128)
        self.lstm1 = LSTM(units=128)
        self.dense2 = Dense(units=10)
    def call(self, inputs):
        x = tf.reshape(inputs, [-1, 28])
        x = self.dense1(x)
        x = tf.reshape(x, [-1, 28, 128])
        x = self.lstm1(x)
        return self.dense2(x)

model = MyModel()
optimizer = tf.train.AdamOptimizer(lr)
model.compile(optimizer=optimizer,
            loss=tf.losses.softmax_cross_entropy,
            metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size,
        validation_data=(x_eval, y_eval))