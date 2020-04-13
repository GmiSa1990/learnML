# Reference: https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, LSTM, GRU, Input)
from tensorflow.keras import utils, callbacks
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.preprocessing.sequence import pad_sequences


np.random.seed(7)

def get_callbacks():
    cbs = []
    # stop training when no improvements are made
    cbs.append(callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=50,
        mode='min'))
    # visualization with tensorboard
    cbs.append(callbacks.TensorBoard(
        log_dir=r"c:/python/tmp_tensorboard/state_batchsize1",
        histogram_freq=100,
        batch_size=1))
    return cbs

# define the raw dataset
alphabet = r"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# create mapping of characters to integers (0-25) and the reverse
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

# prepare the dataset of input to output pairs encoded as integers
seq_length = 1
dataX = []
dataY = []
for i in range(0, len(alphabet) - seq_length, 1):
	seq_in = alphabet[i:i + seq_length]
	seq_out = alphabet[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
	# print(seq_in, '->', seq_out)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (len(dataX), seq_length, 1))

# normalize
X = X/float(len(alphabet))

# one hot encode the output labels
y = utils.to_categorical(dataY)



# create and fit the model
# model_input = Input(shape=(X.shape[1], X.shape[2]), name='Input')
model_input = Input(batch_shape=(1, X.shape[1], X.shape[2]), name='Input')
tensor = LSTM(16, name='LSTM',kernel_regularizer=l2(1e-4), stateful=True)(model_input)
model_output = Dense(y.shape[1], activation='softmax',name='DENSE')(tensor)
model = Model(inputs=model_input, outputs=model_output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, batch_size=1, epochs=1000, verbose=2, callbacks=get_callbacks(), validation_data=(X,y), shuffle=False)

model.summary()

scores = model.evaluate(X, y, verbose=0, batch_size=1)
print("\nmodel accuracy: %.3f%%" % (scores[1]*100))

print('\n\n\n')

for pattern in dataX:
    X = np.reshape(pattern, (1,len(pattern),1))
    X = X / float(len(alphabet))
    res = model.predict(X, verbose=0)
    index = np.argmax(res)
    res_ch = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, "->", res_ch)
# model.save(r"c:/python/tmp_tensorboard/1.h5", overwrite=True,include_optimizer=True)