import fnmatch
import os
from math import prod
from copy import deepcopy

import numpy as np
import tensorflow as tf
import pandas as pd
import packaging as pack
import netCDF4 as cdf
import xarray as xr
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.utils import Sequence
import random

XTRAIN_DATA_PATH = 'D:/Documents/Code/research/data/timed-data/'
YTRAIN_DATA_PATH = 'D:/Documents/Code/research/data/actual-timed-data/'
DATA_VAR = 'temperature'
BATCH_SIZE = 2
STEPS_PER_EPOCH = int(1000 / 32)  # random guess; updated later
EPOCHS = 3
VALIDATION_SPLIT = 0.2
VALIDATION_STEP = 5  # random guess; updated later
LEARNING_RATE = 0.0008 # 0.001 is default
SAVE_MODEL = False
KERAS_MODELS_PATH = 'D:/Documents/Code/research/keras-models'


class CustomModel(keras.Model):
    def train_step(self, data):
        x1, x2, y = data

        with tf.GradientTape() as tape:
            yPred = self([x1, x2], training=True)

            loss = self.compute_loss(y=y, y_pred=yPred)

        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Updating the metrics might not be correct
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, yPred)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x1, x2, y = data

        yPred = self([x1, x2], training=True)

        loss = self.compute_loss(y=y, y_pred=yPred)

        # This might not be correct
        for metric in self.metrics:
            if metric.name != 'loss':
                metric.update_state(y, yPred)
            else:
                metric.update_state(loss)

        return {m.name: m.result() for m in self.metrics}

# Creates a model with one input of variable length
def createModel(inputShape):
    input1 = keras.Input(shape=inputShape, name='input1') # Takes one time step
    input2 = keras.Input(shape=inputShape, name='input2') # Takes the second time step
    x1 = layers.Flatten()(input1)
    # x2 = layers.Flatten()(input2)
    # x1 = layers.Dense(x1.shape[1], activation='linear', name='layer01')(x1)
    # x2 = layers.Dense(x2.shape[1], activation='linear', name='layer02')(x2)
    # x = layers.Concatenate()([x1, x2])
    x = layers.Dense(x1.shape[1], activation='linear', name='layer1')(x1)
    output = layers.Dense((inputShape[0] * inputShape[1]))(x) # Applies linear activation
    
    return CustomModel(inputs=[input1, input2], outputs=output)

def createCompiledModel(inputDim):
    model = createModel(inputDim)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.mean_squared_error,
        metrics=[keras.metrics.mean_squared_error]
    )

    return model

def getInputDim():
    ds = xr.open_dataset(f'{XTRAIN_DATA_PATH}data0.nc')
    fileCount = len(fnmatch.filter(os.listdir(XTRAIN_DATA_PATH), '*.nc'))
    dim = (fileCount,) + ds[DATA_VAR].shape
    ds.close()
    return dim

# Gather the data by loading everything in np arrays
def getData():

    dsX = xr.open_dataset(XTRAIN_DATA_PATH)
    dsY = xr.open_dataset(YTRAIN_DATA_PATH)

    dataX = dsX[DATA_VAR].to_numpy()

    dataY = dsY[DATA_VAR].to_numpy()
    # Essentially flattens the data, so it can be compared to the output
    dataY = dataY.reshape(dataY.shape[0], dataY.shape[1] * dataY.shape[2])

    # Normalize the data

    maxValue = np.amax(dataX)
    dataX = dataX / maxValue

    maxValue = np.amax(dataY)
    dataY = dataY / maxValue

    return dataX, dataY

# Gets data using an implementation of sequence
# More memory efficient however slower.
class WeatherSequence(Sequence):
    def __init__(self, xFilename, yFilename, dataVar, batchSize):
        super().__init__()
        self.xFilename = xFilename
        self.yFilename = yFilename
        self.dataVar = dataVar
        self.batchSize = batchSize

        ds = xr.open_dataset(self.xFilename)
        self.dataLength = ds[self.dataVar].shape[0]
        self.length = int(np.floor(self.dataLength / float(self.batchSize)))
        ds.close()

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        dsX = xr.open_dataset(self.xFilename)
        data = dsX[self.dataVar]
        batchStartX = data[index * self.batchSize : (index + 1) * self.batchSize]
        batchEndX = data[(index * self.batchSize) + 1 : ((index + 1) * self.batchSize) + 1]
        batchStartX = batchStartX.to_numpy()
        batchEndX = batchEndX.to_numpy()
        batchStartX = batchStartX / np.amax(batchStartX)
        batchEndX = batchEndX / np.amax(batchEndX)

        dsY = xr.open_dataset(self.yFilename)
        data = dsY[self.dataVar]
        batchY = data[index * self.batchSize: (index + 1) * self.batchSize]
        batchY = batchY.to_numpy()
        batchY = batchY / np.amax(batchY)
        # Flattens the y data so it in the correct output shape
        batchY = batchY.reshape(batchY.shape[0], batchY.shape[1] * batchY.shape[2])

        dsX.close()
        dsY.close()
        return batchStartX, batchEndX, batchY

    def getValidationData(self):
        pass

# Creates a generator to get data
class WeatherGenerator:
    def __init__(self, xFilename, yFilename, dataVar, batchSize, epochs, validationSplit, dim):
        self.xFilename = xFilename
        self.yFilename = yFilename
        self.dataVar = dataVar
        self.batchSize = batchSize
        self.epochs = epochs
        self.validationSplit = validationSplit
        self.dim = dim
        stepsPerEpoch = int(np.floor((self.dim[0] * (1 - self.validationSplit)) / self.batchSize)) - 1
        self.length = stepsPerEpoch
        self.startIndex = int(self.dim[0] * validationSplit)
        self.trainOnX = True
        self.index = 0

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        batchedX1Data = None
        batchedX2Data = None
        batchedYData = None
        for i in range(self.startIndex + (index * self.batchSize), self.startIndex + ((index + 1) * self.batchSize)):
            dsX1 = xr.open_dataset(f'{self.xFilename}data{i}.nc')
            data = dsX1[self.dataVar]
            data = data.to_numpy()
            #data = data / np.amax(data)
            if batchedX1Data is None:
                batchedX1Data = deepcopy(data)
            elif len(batchedX1Data.shape) == 2:
                batchedX1Data = np.array([batchedX1Data, deepcopy(data)])
            else:
                batchedX1Data = np.append(batchedX1Data, [deepcopy(data)], axis=0)
            dsX1.close()

            dsX2 = xr.open_dataset(f'{self.xFilename}data{i + 1}.nc')
            data = dsX2[self.dataVar]
            data = data.to_numpy()
            #data = data / np.amax(data)
            if batchedX2Data is None:
                batchedX2Data = deepcopy(data)
            elif len(batchedX2Data.shape) == 2:
                batchedX2Data = np.array([batchedX2Data, deepcopy(data)])
            else:
                batchedX2Data = np.append(batchedX2Data, [deepcopy(data)], axis=0)
            dsX2.close()

            dsY = xr.open_dataset(f'{self.yFilename}data{i}.nc')
            data = dsY[self.dataVar]
            data = data.to_numpy()
            #data = data / np.amax(data)
            if batchedYData is None:
                batchedYData = deepcopy(data)
            elif len(batchedYData.shape) == 2:
                batchedYData = np.array([batchedYData, deepcopy(data)])
            else:
                batchedYData = np.append(batchedYData, [deepcopy(data)], axis=0)
            dsY.close()

        # Makes the y data flat
        batchedYData = batchedYData.reshape(batchedYData.shape[0], prod(batchedYData.shape[1:]))

        return batchedX1Data, batchedX2Data, batchedYData

        # dsX = xr.open_dataset(self.xFilename)
        # data = dsX[self.dataVar]
        # batchStartX = data[self.startIndex + (index * self.batchSize) : self.startIndex + ((index + 1) * self.batchSize)]
        # batchEndX = data[self.startIndex + (index * self.batchSize) + 1: self.startIndex + ((index + 1) * self.batchSize) + 1]
        # batchStartX = self.normalizeData(batchStartX)
        # batchEndX = self.normalizeData(batchEndX)
        # dsX.close()
        #
        # dsY = xr.open_dataset(self.yFilename)
        # data = dsY[self.dataVar]
        # batchY = data[self.startIndex + (index * self.batchSize) : self.startIndex + ((index + 1) * self.batchSize)]
        # batchY = self.normalizeData(batchY)
        #
        # # Flattens the y data so it in the correct output shape
        # batchY = batchY.reshape(batchY.shape[0], batchY.shape[1] * batchY.shape[2])
        # dsY.close()
        # return batchStartX, batchEndX, batchY
    
    def normalizeData(self, data):
        data = data.to_numpy()
        return data / np.amax(data)
    
    def __call__(self):
        while True:
            yield self.__getitem__(self.index % self.__len__())
            self.index += 1

class ValidationData:

    def __init__(self, xFilename, yFilename, dataVar, validationSplit, validationSteps, batchSize, dim):
        self.xFilename = xFilename
        self.yFilename = yFilename
        self.dataVar = dataVar
        self.validationSplit = validationSplit
        self.validationSteps = validationSteps
        self.batchSize = batchSize
        self.dim = dim
        self.startIndex = int(self.dim[0] * validationSplit)
        self.index = 0

    def __getitem__(self, index):
        return self.getValidationData(index)

    def __len__(self):
        return self.validationSteps
    
    def __call__(self):
        while True:
            # Essientally wraps the data around so it repeats up to
            # the data we seperated just for validation
            yield self.__getitem__(self.index % self.__len__())
            self.index += 1

    # Very similar to __getitem__() in weatherGenerator
    # Possibly incorporate this into just __getitem__()? 
    def getValidationData(self, index):
        batchedX1Data = None
        batchedX2Data = None
        batchedYData = None
        for i in range((index * self.batchSize), ((index + 1) * self.batchSize)):
            dsX1 = xr.open_dataset(f'{self.xFilename}data{i}.nc')
            data = dsX1[self.dataVar]
            data = data.to_numpy()
            #data = data / np.amax(data)
            if batchedX1Data is None:
                batchedX1Data = data
            elif len(batchedX1Data.shape) == 2:
                batchedX1Data = np.array([batchedX1Data, data])
            else:
                batchedX1Data = np.append(batchedX1Data, [data], axis=0)
            dsX1.close()

            dsX2 = xr.open_dataset(f'{self.xFilename}data{i + 1}.nc')
            data = dsX2[self.dataVar]
            data = data.to_numpy()
            #data = data / np.amax(data)
            if batchedX2Data is None:
                batchedX2Data = data
            elif len(batchedX2Data.shape) == 2:
                batchedX2Data = np.array([batchedX2Data, data])
            else:
                batchedX2Data = np.append(batchedX2Data, [data], axis=0)
            dsX2.close()

            dsY = xr.open_dataset(f'{self.yFilename}data{i}.nc')
            data = dsY[self.dataVar]
            data = data.to_numpy()
            #data = data / np.amax(data)
            if batchedYData is None:
                batchedYData = data
            elif len(batchedYData.shape) == 2:
                batchedYData = np.array([batchedYData, data])
            else:
                batchedYData = np.append(batchedYData, [data], axis=0)
            dsY.close()

        batchedYData = batchedYData.reshape(batchedYData.shape[0], prod(batchedYData.shape[1:]))

        return batchedX1Data, batchedX2Data, batchedYData

        # dsX = xr.open_dataset(self.xFilename)
        # data = dsX[self.dataVar]
        # batchStartX = data[(self.batchSize * index) : (self.batchSize * (index + 1))]
        # batchEndX = data[(self.batchSize * index) + 1:(self.batchSize * (index + 1)) + 1] # Grabs data one ahead
        # batchStartX = self.normalizeData(batchStartX)
        # batchEndX = self.normalizeData(batchEndX)
        #
        # dsY = xr.open_dataset(self.yFilename)
        # data = dsY[self.dataVar]
        # batchY = data[(self.batchSize * index) : (self.batchSize * (index + 1))]
        # batchY = self.normalizeData(batchY)
        # # Flattens the y data so it in the correct output shape
        # batchY = batchY.reshape(batchY.shape[0], batchY.shape[1] * batchY.shape[2])
        #
        # dsX.close()
        # dsY.close()
        # return batchStartX, batchEndX, batchY
    
    def normalizeData(self, data):
        data = data.to_numpy()
        return data / np.amax(data)

def fitModelWithGenerator():
    dim = getInputDim()

    # Starts from zero
    # Subtract one to give some headway for data
    STEPS_PER_EPOCH = int(np.floor((dim[0] * (1 - VALIDATION_SPLIT)) / BATCH_SIZE)) - 1

    VALIDATION_STEP = int(np.floor((dim[0]*VALIDATION_SPLIT) / BATCH_SIZE))

    datasetGenerator = WeatherGenerator(XTRAIN_DATA_PATH, YTRAIN_DATA_PATH, DATA_VAR, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, dim)

    dataset = tf.data.Dataset.from_generator(
        datasetGenerator,
        output_signature=
            (tf.TensorSpec(shape=(BATCH_SIZE,) + dim[1:], dtype=tf.float32),
                tf.TensorSpec(shape=(BATCH_SIZE,) + dim[1:], dtype=tf.float32),
                tf.TensorSpec(shape=(BATCH_SIZE, prod(dim[1:])), dtype=tf.float32))
    )

    validationGenerator = ValidationData(XTRAIN_DATA_PATH, YTRAIN_DATA_PATH, DATA_VAR, VALIDATION_SPLIT, VALIDATION_STEP, BATCH_SIZE, dim)

    validationDataset = tf.data.Dataset.from_generator(
        validationGenerator,
        output_signature=
            (tf.TensorSpec(shape=(BATCH_SIZE,) + dim[1:], dtype=tf.float32),
                tf.TensorSpec(shape=(BATCH_SIZE,) + dim[1:], dtype=tf.float32),
                tf.TensorSpec(shape=(BATCH_SIZE, prod(dim[1:])), dtype=tf.float32))
    )

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = createCompiledModel(dim[1:])

    history = model.fit(
        dataset,
        batch_size=BATCH_SIZE,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=validationDataset,
        validation_steps=VALIDATION_STEP,
        verbose=1
    )

    return history, model


# Do not recommend using method because it does not
# implement a proper validation_data scheme or
# strategy for faster processing.
def fitModelWithSequence():
    dim = getInputDim()

    sequence = WeatherSequence(XTRAIN_DATA_PATH, YTRAIN_DATA_PATH, DATA_VAR, BATCH_SIZE)

    # I do not want to write another class just to handle validationData but 
    # with sequence so it only considers the first batch of validationData
    validationData = ValidationData(XTRAIN_DATA_PATH, YTRAIN_DATA_PATH, DATA_VAR, VALIDATION_SPLIT, VALIDATION_STEP, BATCH_SIZE, dim)

    model = createCompiledModel(dim[1:])

    history = model.fit(
        sequence,
        epochs=EPOCHS,
        validation_data=validationData.getValidationData(0),
        verbose=1
    )

    return history, model

# dim = getInputDim()
#
# datasetGenerator = WeatherGenerator(XTRAIN_DATA_PATH, YTRAIN_DATA_PATH, DATA_VAR, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, dim)
#
# dataset = tf.data.Dataset.from_generator(
#     datasetGenerator,
#     output_signature=
#         (tf.TensorSpec(shape=(BATCH_SIZE,) + dim[1:], dtype=tf.float32),
#             tf.TensorSpec(shape=(BATCH_SIZE,) + dim[1:], dtype=tf.float32),
#             tf.TensorSpec(shape=(BATCH_SIZE, prod(dim[1:])), dtype=tf.float32))
# )
#
# iterator = iter(dataset)
#
# print(next(iterator))


history, model = fitModelWithGenerator()

print(history.history)

x0 = np.arange(4 * 4).reshape(4, 4) + 500
x0 = np.expand_dims(x0, axis=0) #/ np.amax(x0)

x1 = np.arange(4 * 4).reshape(4, 4) + 502
x1 = np.expand_dims(x1, axis=0) #/ np.amax(x1)

y = np.arange(4 * 4).reshape(4, 4) + 501

prediction = model.predict([x0, x1]).reshape(4, 4)
print(prediction)
#(prediction * 16)
print(y)
print(abs((prediction) - y))
print((prediction - (y))**2)
#print(abs((prediction*16) - y))
#print((prediction - (y/16))**2)



if SAVE_MODEL:
    model.save(f'{KERAS_MODELS_PATH}/weatherModel.keras')