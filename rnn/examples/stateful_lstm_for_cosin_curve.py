import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import (Dense, LSTM, GRU, Input)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import callbacks


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
        log_dir=r"c:/python/tmp_tensorboard/cos_1_64",
        histogram_freq=20,
        batch_size=1))
    return cbs

lookBack = 5
dataLen = 1000
validationRatio = 0.2
predNum = 300
# data generation
data = np.cos(np.arange(dataLen)*(20*np.pi/1000))
data = (data + 1) / 2
# plt.plot(dataX)
# plt.show()
# X = data[0:-1]
# y = data[1:]
# X = np.reshape(X, (len(X),1,1))
# y = np.reshape(y, (len(y),1))
# windowing data
X = []
y = []
for i in range(dataLen-lookBack):
    X.append(data[i:i+lookBack])
    y.append(data[i+lookBack])
X = np.array(X)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))
y = np.array(y)
y = np.reshape(y, (X.shape[0],1))

trainCnt = int(X.shape[0] * (1-validationRatio))
XTrain, XVal = X[0:trainCnt], X[trainCnt:]
yTrain, yVal = y[0:trainCnt], y[trainCnt:]


# model
modelInput = Input(batch_shape=(1,lookBack,1), name='input-scalar')
tensor = LSTM(64, stateful=True, kernel_regularizer=l2(1e-5), name='LSTM')(modelInput)
modelOutput = Dense(y.shape[1], activation='tanh', name='DENSE-scalar')(tensor)
model = Model(inputs=modelInput, outputs=modelOutput)

model.compile(optimizer='adam', loss='mse')
epochs = 100
for i in range(epochs):
    model.fit(XTrain, yTrain, batch_size=1, epochs=1, verbose=2, callbacks=get_callbacks(), validation_data=(XVal,yVal), shuffle=False)
    model.reset_states()

# model.fit(XTrain, yTrain, batch_size=1, epochs=100, verbose=2, callbacks=get_callbacks(), validation_data=(XVal,yVal), shuffle=False)

model.summary()

# inVal = X[0]
inVal = np.vstack((X[-1,1:],y[-1]))
prediction = []
for i in range(predNum):
    x = np.reshape(inVal, (1,lookBack,1))
    pred = model.predict(x, batch_size=1)
    prediction.append(pred.squeeze())
    inVal = np.vstack((inVal[1:],pred))
plt.figure()
prediction = np.array(prediction) * 2 - 1
plt.plot(prediction, label='prediction')
cos_y = np.cos(np.arange(dataLen, dataLen+predNum)*(20*np.pi/1000))
plt.plot(cos_y, label='original')
plt.legend()
plt.show()

