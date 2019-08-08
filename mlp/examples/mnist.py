
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np

tf.enable_eager_execution()

class DataLoader(object):
    def __init__(self):
        (x_train, y_train), (x_eval, y_eval) = mnist.load_data()
        self.x_train = np.reshape(x_train, (x_train.shape[0],-1)).astype(np.float32)
        self.x_eval = np.reshape(x_eval, (x_eval.shape[0],-1)).astype(np.float32)
        self.x_train /= 255     # normalization.
        self.x_eval  /= 255
        self.y_train = y_train.astype(np.int32)
        self.y_eval = y_eval.astype(np.int32)
        self.y_train = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_eval = tf.keras.utils.to_categorical(self.y_eval, 10)
        print('the number of training sample and evaluation sample are {} and {}.\n'
                .format(self.x_train.shape[0], self.x_eval.shape[0]))
    def get_batch(self, batchsize):
        index = np.random.randint(0,np.shape(self.x_train)[0],batchsize)
        return self.x_train[index,:], self.y_train[index]

class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.regularizer = regularizers.l2(0.01)
        self.dense1 = Dense(units=100, activation=tf.nn.relu) #, kernel_regularizer= self.regularizer
        self.dense2 = Dense(units=10, activation=tf.nn.softmax)

    def call(self, inputs):
        tmp = self.dense1(inputs)
        outputs = self.dense2(tmp)
        return outputs

    def predict(self, inputs):
        return tf.argmax(self(inputs), axis=-1)

# class MLP2(tf.Keras.):
#     def __init__(self):
# input1 = 
MODEL = 2
epoches = 1
batch_size = 150
data_loader = DataLoader()
optimizer = tf.train.AdamOptimizer(0.01)
model = MLP()

if MODEL == 1:
    writer = tf.contrib.summary.create_file_writer("logs/")

    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        for batch_index in range(epoches):
            xs, ys = data_loader.get_batch(batch_size)
            with tf.GradientTape() as tape:
                y_pred = model(tf.convert_to_tensor(xs,dtype=tf.float32))
                loss = tf.losses.sparse_softmax_cross_entropy(ys, y_pred)
                tf.contrib.summary.scalar('loss', loss, step=batch_index)
                print('{}: loss function: {}'.format(batch_index, loss))
            vars_gradient = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(vars_gradient, model.variables))

    y_eval = model.predict(data_loader.x_eval).numpy()
    num_eval = np.shape(data_loader.x_eval)[0]
    print('the accuracy is {}'.format(np.sum(y_eval == data_loader.y_eval)/num_eval))

elif MODEL == 2:
    model.compile(optimizer,
                loss=categorical_crossentropy,
                metrics=['accuracy'])
    history=model.fit(x=data_loader.x_train,
            y=data_loader.y_train,
            batch_size=batch_size,
            epochs=epoches,
            validation_data=(data_loader.x_eval, data_loader.y_eval)) #
    model.summary()
    score = model.evaluate(x=data_loader.x_eval,
                            y=data_loader.y_eval)
    print('score is %f' % score[1])

    y_eval = model.predict(data_loader.x_eval).numpy()
    num_eval = np.shape(data_loader.x_eval)[0]
    y_eval1 = np.argmax(data_loader.y_eval,axis=1).astype(np.int32)
    print('the accuracy is {}'.format(np.sum(y_eval == y_eval1)/num_eval))
