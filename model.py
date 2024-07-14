import numpy as np
import tensorflow as tf
import pandas as pd
import packaging as pack
import netCDF4 as cdf
import xarray as xr
import matplotlib.pyplot as plt
import keras
from keras import layers
import random
####################
# Sequential version
####################


# model = keras.Sequential([
#     layers.Dense(2, activation='sigmoid', name='layer1'),
#     layers.Dense(2, activation='sigmoid', name='layer2'),
#     layers.Dense(1, activation='softmax', name='output')
# ])

# x = np.array([[1, 1]])
# print(x)
# print(model(x))

# model.summary()

####################
# Functional version
####################

print('Testing the functional version now')
def createUncompiledModel():
    firstInput = keras.Input(shape=(1,))
    secondInput = keras.Input(shape=(1,))
    x = layers.concatenate([firstInput, secondInput])
    x = layers.Dense(2, activation='tanh', name='layer1')(x)
    x = layers.Dense(2, activation='tanh', name='layer2')(x)
    x = layers.Dense(2, activation='tanh', name='layer3')(x)
    outputs = layers.Dense(1, activation='tanh', name='output')(x)

    model = keras.Model(inputs=[firstInput, secondInput], outputs=outputs)
    return model

def createCompiledModel():
    model = createUncompiledModel()

    model.compile(optimizer=keras.optimizers.SGD(), 
                  loss=keras.losses.mean_squared_error,
                  metrics=[keras.metrics.mean_squared_error, keras.metrics.Accuracy]
    )

    return model

def createDataset(datasetSize):
    x = np.array([[0, 0]]).reshape(1, 2)
    y = np.array([[0]]).reshape(1, 1)
    for i in range(datasetSize):
        val = random.randint(0, 3)
        if val == 0:
            x = np.append(x, np.array([[0, 0]]), axis=0)
            y = np.append(y,  np.array([[0]]), axis=0)
        elif val == 1:
            x = np.append(x,  np.array([[0, 1]]), axis=0)
            y = np.append(y,  np.array([[1]]), axis=0)
        elif val == 2:
            x = np.append(x,  np.array([[1, 0]]), axis=0)
            y = np.append(y,  np.array([[1]]), axis=0)
        elif val == 3:
            x = np.append(x,  np.array([[1, 1]]), axis=0)
            y = np.append(y,  np.array([[0]]), axis=0)
    return x, y

model = createCompiledModel()

datasetSize = 20000
batchSize = 2
epochs = 4

x_train, y_train = createDataset(datasetSize)

x_test, y_test = createDataset(int(datasetSize / 10))

history = model.fit(x_train, y_train, batch_size=batchSize, epochs=epochs, validation_split=0.2)

print(history.history)

result = model.evaluate(x_test, y_test, batch_size=batchSize)

print(result)

predictions = model.predict(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))

print(predictions)
