- [Model](#model)
  - [model creation](#model-creation)
    - [Model class](#model-class)
    - [Sequential class](#sequential-class)
  - [save/load model](#saveload-model)
  - [compile](#compile)
  - [fit](#fit)
- [Components](#components)
  - [Metrics](#metrics)
    - [Usage of metrics](#usage-of-metrics)
    - [Frequently used metrics](#frequently-used-metrics)
  - [Loss function](#loss-function)
  - [Optimizer](#optimizer)
  - [Callbacks](#callbacks)

# Keras

## Model

### model creation
Two ways to instantiate a **Model**:
### Model class
start from Input, chain layer calls to specify the model's forward pass, and finally create the model from inputs and outputs
```python
input = tf.keras.Input(shape(3,))
x = tf.keras.layers.Dense(4, activation=tf.keras.activation.relu)(input)
output = tf.keras.Dense(2, activation = tf.keras.activation.softmax)(x)
model = tf.keras.Model(inputs = input, outputs=output)
```
### Sequential class

### save/load model
```python
"""
save a model to hdf5 file.
model -> keras model instance to be saved.
"""
tf.keras.models.save_model(
    model,
    filepath,
    overwrite=True,
    include_optimizer=True
)

"""
loads a model saved via save_model
filepath -> h5py file
compile -> If an optimizer was found as part of the saved model, the model is already compiled. Otherwise, the model is uncompiled and a warning will be displayed. When compile is set to False, the compilation is omitted without any warning.
"""
tf.keras.models.load_model(
    filepath,
    custom_objects=None,
    compile=True
)

```
### compile
```python
complie(
    optimizer,
    ...
)
```
### fit
```python
fit(
    x,
    y,
    ...
)
```



## Metrics

#### Usage of metrics
A metric is a function that is used to judge the performance of your model. Metric functions are to be supplied in the metrics parameter when a model is compiled.

A metric function is similar to a loss function, except that the results from evaluating a metric are not used when training the model. You may use any of the loss functions as a metric function.

#### Frequently used metrics
- categorical accuracy


## Loss function

### Built-in function



### Customized function



```python
from tensorflow.keras import backend as K
# most of functions in K are for element-wise operation.

K.binary_crossentropy(target, output)

```



## Optimizer

- Adam

- SGD

- RMSprop

  usually used for RNN model training.


## Callbacks
- LambdaCallback
  For creating simple, custom callbacks on-the-fly.This callback is constructed with anonymous functions that will be called at the appropriate time. Note that the callbacks expects positional arguments, as:
  - on_epoch_begin and on_epoch_end expect two positional arguments: epoch, logs
  - on_batch_begin and on_batch_end expect two positional arguments: batch, logs
  - on_train_begin and on_train_end expect one positional argument: logs
