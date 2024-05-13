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

XTRAIN_DATA_PATH = 'D:/Documents/Code/research/data/testData.nc'
YTRAIN_DATA_PATH = 'D:/Documents/Code/research/data/testData.nc'
DATA_VAR = 'temperature'
BATCH_SIZE = 32
EPOCHS = 4
VALIDATION_SPLIT = 0.2
SAVE_MODEL = False
KERAS_MODELS_PATH = 'D:/Documents/Code/research/keras-models'

# Creates a model with one input of variable length
def createModel(inputShape):
    input = keras.Input(shape=inputShape)
    x = layers.Flatten()(input)
    x = layers.Dense(x.shape[1], activation='tanh', name='layer1')(x)
    x = layers.Dense(x.shape[1], activation='tanh', name='layer2')(x)
    output = layers.Dense(x.shape[1])(x) # Applies linear activation
    
    return keras.Model(inputs=input, outputs=output)

def createCompiledModel(inputDim):
    model = createModel(inputDim)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.mean_squared_error,
        metrics=[keras.metrics.mean_squared_error, keras.metrics.Accuracy]
    )

    return model


# Gather the data

dsX = xr.open_dataset(XTRAIN_DATA_PATH)
dsY = xr.open_dataset(YTRAIN_DATA_PATH)

dataX = dsX[DATA_VAR].to_numpy()

dataY = dsY[DATA_VAR].to_numpy()
dataY = dataY.reshape(dataY.shape[0], dataY.shape[1] * dataY.shape[2])

# Normalize the data

maxValue = np.amax(dataX)
dataX = dataX / maxValue

maxValue = np.amax(dataY)
dataY = dataY / maxValue

# Fit the data

model = createCompiledModel((dataX.shape[1],dataX.shape[2]))
history = model.fit(
    dataX, 
    dataY,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    verbose=1
)

print(history.history)

if SAVE_MODEL:
    model.save(f'{KERAS_MODELS_PATH}/weatherModel.keras')

