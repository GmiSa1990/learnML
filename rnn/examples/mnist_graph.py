# python 3
# tensorflow 1.12

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected

# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters
lr = 0.001
n_epochs = 100
batch_size = 150

n_steps = 28    # time steps
n_inputs = 28   # MNIST data input (img shape: 28*28)
n_hidden_units = 128   # neurons in hidden layer
n_classes = 10      # MNIST classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    X_1 = fully_connected(X, n_hidden_units, activation_fn=None)
    # X_in = (128 batch * 28 steps, 128 hidden)
    # X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_1 = tf.reshape(X_1, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)

    # lstm cell is divided into two parts (c_state, h_state)
    #init_state = cell.zero_state(batch_size, dtype=tf.float32)

    _, final_state = tf.nn.dynamic_rnn(cell, X_1, dtype=tf.float32)#, initial_state=init_state)

    # hidden layer for output as the final results
    #############################################
    #results = tf.matmul(final_state[1], weights['out']) + biases['out']
    results = fully_connected(final_state[1], n_classes, activation_fn=None)

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            sess.run([train_op], feed_dict={x:batch_xs, y:batch_ys})
            if iteration % 20 == 0:
                print('epoch: {}, iteration: {}, train acc: {}'.format(
                    epoch, iteration, sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys})))
                print('epoch: {}, iteration: {}, test acc: {}'.format(
                    epoch, iteration, sess.run(accuracy, feed_dict={x:X_test, y:y_test})))

