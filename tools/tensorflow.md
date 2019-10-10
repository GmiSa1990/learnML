[TOC]

# Tensorflow
```python
# based on python-3
import tensorflow as tf
import numpy as np
```
## Graph Execution Mode
### Variables & Scope
**Tensorflow adopts the data type float32 only!**
```python
# define tf variables.
with tf.name_scope("a_name_scope") as scope:
    initializer = tf.constant_initializer(value=3)
    # Important: remember to define data type.
    var1 = tf.get_variable(name='var1',
                            shape=[1],
                            dtype=tf.float32,
                            initializer=initializer)
    var2 = tf.Variable(name='var2',
                        initial_value=[2],
                        dtype=tf.float32)

# define initializer.
init = tf.global_variables_initializer()

# sess.run()
with tf.Session() as sess:
    # Important: if you define any variables, remember to initialize them.
    sess.run(init)
    print(var1.name)
    print(sess.run(var1))
    print(var2.name)
    print(sess.run(var2))
```

### Placeholder
If we want to send the 'outside' data into neural network, we have to use *placeholder* as container.

- example 1: basic multiplication

```python
# tensorflow adopts the float32 data only!
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # employ feed_dict to designate the vars that will be sent in.
    res = sess.run(output, feed_dict={input1:[7.], input2:[2.]})
    print(res)
```

- example 2: basic RNN

```python
n_inputs = 3
n_neurons = 5

# here, 'None' means that batch size is not determined yet until now.
X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

# initial_value is defined with random_noraml function.
Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons], dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons], dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons], dtype=tf.float32))

Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

# define initializer.
init = tf.global_variables_initializer()

```

### Define Layer

A fully connected layer is implemented here.
```python
def add_layer(xdata, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal(in_size,out_size),dtype=tf.float32)
    biases = tf.Variable(tf.zeros(1,out_size)+0.1)
    Wx_plus_b = tf.add(tf.matmul(xdata, weights),biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
```

### Session

```python
with tf.Session() as sess:
	# method 'run'
    sess.run(fetches,			# a single graph element, or a list of graph elements,...
            feed_dict=None,		# dict to map graph elements to values.
            options=None,		# a [RunOptions] protocol buffer
            run_metadata=None)  # a [RunMetadata] protocol buffer  
```

### One example

Similar as one in reference 2.
```python

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def add_layer(xdata, in_size, out_size, activation_function=None):
    weights = tf.Variable(tf.random_normal([in_size,out_size]),dtype=tf.float32)
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.add(tf.matmul(xdata, weights),biases)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

xdata = np.linspace(0, np.pi, 100)[:,np.newaxis]
noise = np.random.normal(0, 0.05, xdata.shape)
ydata = np.sin(xdata) - 0.5

# shape argument is used to match the size the training data.
x_data = tf.placeholder(tf.float32,shape=(None, 1))
y_data = tf.placeholder(tf.float32,shape=(None, 1))

# create neural network.
l1 = add_layer(x_data, 1, 100, activation_function=tf.nn.relu)
l2 = add_layer(l1, 100, 10, activation_function=tf.nn.relu)
prediction = add_layer(l2, 10, 1)


# loss function and train step.
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction),axis=1))
# loss = tf.nn.l2_loss(y_data- prediction)
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdamOptimizer().minimize(loss)

# visualization.
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(xdata,ydata)
plt.ion()
plt.show()

epoch = 3000
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for item in range(epoch):
        sess.run(train_step, feed_dict={x_data:xdata,y_data:ydata})
        if item%50 == 0:
            result = sess.run(prediction, feed_dict={x_data:xdata})
            try:
                ax.lines.remove(line1[0])
            except Exception:
                pass
            line1 = ax.plot(xdata,result, 'r-', lw=5)
            print('epoch = {}:'.format(item))
            print(sess.run(loss,feed_dict={x_data:xdata,y_data:ydata}))
            plt.pause(0.1)

```

### Useful function

- `tf.argmax(input, axis=None, output_type=tf.dtypes.int64)`

  If `axis` equal 0, then compute along column axis; if equal 1, then compute along row axis.

```python
x = tf.convert_to_tensor(np.arange(10).reshape(2,5), dtype=tf.float32)

tf.argmax(x, axis=0)
# equivalent to tf.argmax(x)
# output: shape=(5,), numpy=array([1, 1, 1, 1, 1],dtype=int64)

tf.argmax(x, axis=1)
# output: shape=(2,), dtype=int64, numpy=array([4, 4])
```

- `tf.cast(x, dtype)`

  do the type conversion, such as *True* -> 1

```python
x = tf.constant([1.0], dtype=tf.float32)
# shape=(1,), dtype=float32, numpy=array([1.], dtype=float32)
y = tf.cast(x, tf.int32)
# shape=(1,), dtype=int32, numpy=array([1])
```

- 

## Eager Execution Mode



## Visualization

### using matplotlib
For intuitive visualization of results, we can generate some plots. Refer to [Matplotlib](#matplotlib).

### tensorboard
#### Steps to use tensorboard
- define the names for placeholder, variables, biases and loss, ...
```python
input1 = tf.placeholder(tf.float32, name='input1')
var1   = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
```
- create `tf.Session()` and use `tf.summary.FileWriter()` to save the graph.
```python
sess = tf.Session()
writer = tf.summary.FileWriter("logs/",sess.graph)
```
- go back to terminal and run the following command.
```cmd
tensorboard --logdir = "logs/"
```
- copy the web address displayed on your terminal to the web browser and then you can see the Tensorboard.(Chrome is recommended!)

#### Contents to be displayed in Tensorboard
- graph

used to show the structure of neural network.
![graph-tf](images/graph_run=.png)
- histogram

restore the run-time result of parameters( weights, biases) inside nueral layer.
```python
    tf.summary.histogram('weights', weight)
```
- scalar

restore the tun-time result of loss function.
```python
    tf.summary.scalar('loss', loss)
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/', sess.graph)
        sess.run(init) # initialize all the tf.variables
        for i in range(epoch):
            sess.run(train_step, feed_dict={})
            if i%50 == 0:
                rs = sess.run(merged, feed_dict={})
                writer.add_summary(rs, i)

```
## Loss Function

- `sparse_softmax_cross_entropy_with_logits`  &  `softmax_cross_entropy_with_logits_v2`

```python
tf.nn.softmax_cross_entropy_with_logits_v2(labels, logits)
# `labels`: shape: [batch_size, num_classes],
#			each row of labels[i] must be a valid probability distribution.
# `logits`: unscaled log probabilities.

tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
# `labels`: shape: [batch_size,], each element must be an index in [0, num_classes]
# `logits`: unscaled log probabilities.
```







# Reference
1. [tensorflow intro](https://docs.google.com/presentation/d/1zkmVGobdPfQgsjIw6gUqJsjB8wvv9uBdT7ZHdaCjZ7Q/edit#slide=id.g1d0ac6f6ba_0_0)
2. [Movan python](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/4-1-tensorboard1/)
3. 