
## Word2vector

1-of-N-encoding

## Categorical cross entropy


## Long Short-Term Memory (LSTM)
- schematic diagram
  - Three gate: Input Gate, Forget Gate and Output Gate.
  - One input and One output here.
![lstm](./images/lstm.PNG)




# Language Model

Estimate the probability of word sequence.

- word sequence: w1, w2, w3, ..., wn
- P(w1,w2,w3,...,wn)
- application-1: speech recognition, different word sequences might have the same pronunciation.
- application-2: sentence generation ...

## Traditional LM -- N-gram

how to estimate P(w1,w2,...,wn)?
- collect a large amount of text data as training data, but the word sequence w1,w2,...,wn might not appear in the training data.
- 2-gram LM:
  P(w1,w2,...,wn)=P(w1|START)P(w2|w1)...P(wn|wn-1).
  It is easy to generalize to 3-gram, 4-gram, ...

## NN-based LM


# CNN

## Flatten Layer
`tf.keras.layers.Flatten(data_format=None)`

Flattens the input. Does not affect the batch size.

```python
model = Sequential()
model.add(Conv2D(64, (3, 3),
                 input_shape=(3, 32, 32), padding='same',))
# now: model.output_shape == (None, 64, 32, 32)

model.add(Flatten())
# now: model.output_shape == (None, 65536)
```

## Dropout
`tf.keras.layers.Dropout(rate, noise_shape=None, seed=None)`

rate: fraction of the input units to drop.

![dropout](images/dropout1.PNG)

Using dropout regularization randomly disables some portion of neurons in a hidden layer. In the Keras library, you can add dropout after any hidden layer, and you can specify a dropout rate, which determines the percentage of disabled neurons in the preceding layer. In the original paper that proposed dropout layers, by [Hinton (2012)](https://arxiv.org/pdf/1207.0580.pdf), dropout (with p=0.5) was used on each of the fully connected (dense) layers before the output; it was not used on the convolutional layers. This became the most commonly used configuration.

## Convolutional Layer




## Pooling Layer

# RNN







## LSTM



## LSTMCell

LSTM is a **recurrent layer**, while LSTMCell is an object **used by LSTM layer** that contains the calculation logic for one step.
A recurrent layer contains a cell object. The cell contains the core code for the calculations of each step, while the recurrent layer commands the cell and performs the actual recurrent calculations.

Usually, people use LSTM layers in their code. Or they use RNN layers containing LSTMCell. Both things are almost the same. An LSTM layer is a RNN layer using an LSTMCell, as you can check out in the source code.

**About the number of cells**: Alghout it seems, because of its name, that LSTMCell is a single cell, it is actually an object that manages all the units/cells as we may think. In the same code mentioned, you can see that the units argument is used when creating an instance of LSTMCell.

## GRU

## Embedding
this layer can only be used as the first layer in a model.
```python

```





# Batch Normalization

Normalize the activations of the previous layer at each batch, i.e. applies transformation that maintains the mean activation close to 0 while the activation standard deviation close to 1.



```python
from tensorflow.keras.layers import BatchNormalization
BatchNormalization()
```

