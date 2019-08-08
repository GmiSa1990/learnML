
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.layers import Dense, LSTM
tf.enable_eager_execution()

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28   # MNIST data input (img shape: 28*28)
n_steps = 28    # time steps
n_hidden_units = 128   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)


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
train_op = tf.train.AdamOptimizer(lr)
step = 0
while step * batch_size < training_iters:
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])

    with tf.GradientTape() as tape:
        pred = model(batch_xs)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=batch_ys, logits=pred)
    grads = tape.gradient(loss, model.variables)
    train_op.apply_gradients(grads_and_vars=zip(grads, model.variables))
    if step % 20 == 0:
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(batch_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        print("{:3d}: accuracy: {:.2f}%, loss: {:.3f}".format(step, accuracy*100, loss))
    step += 1

