import fnmatch
import os
from math import prod
from copy import deepcopy
import faulthandler

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

XTRAIN_DATA_PATH = 'D:/Documents/Code/research/wrfout'
YTRAIN_DATA_PATH = 'D:/Documents/Code/research/wrfout'
DATA_VAR = 'T'
BATCH_SIZE = 4
STEPS_PER_EPOCH = int(1000 / 32)  # random guess; updated later
EPOCHS = 10
VALIDATION_SPLIT = 0.2
VALIDATION_STEP = 5  # random guess; updated later
LEARNING_RATE = 0.001 # 0.001 is default
SAVE_MODEL = False
KERAS_MODELS_PATH = 'D:/Documents/Code/research/keras-models'
NORMALIZE_DATA = True
# All temp for later when using optimizedDataset

CUBE_SIZE = 6


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
    input1 = keras.Input(shape=inputShape, name='input1')  # Takes one time step
    input2 = keras.Input(shape=inputShape, name='input2')  # Takes the second time step
    x1 = layers.Flatten()(input1)
    x2 = layers.Flatten()(input2)
    # x1 = layers.Dense(x1.shape[1], activation='linear', name='layer01')(x1)
    # x2 = layers.Dense(x2.shape[1], activation='linear', name='layer02')(x2)
    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(x.shape[1], activation='linear', name='layer1')(x)
    output = layers.Dense(prod(inputShape), name='output')(x)  # Applies linear activation
    
    return CustomModel(inputs=[input1, input2], outputs=output)

def createCompiledModel(inputDim):
    model = createModel(inputDim)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=keras.losses.mean_squared_error,
        metrics=[keras.metrics.mean_squared_error]
    )

    return model

# Both the x and y path are the same
def getFilePath(index, dim):
    day = '0' if index != (dim[0]-1) else '1'
    minute = str((index % 12) * 5).zfill(2)
    hour = str(index // 12).zfill(2)

    if day == '1':  # Edge case
        hour = '00'

    return f'{XTRAIN_DATA_PATH}/wrfout_d02_2023-06-2{day}_{hour}%3A{minute}%3A00'

def getInputDim():
    #ds = xr.open_dataset(f'{XTRAIN_DATA_PATH}/wrfout_d02_2023-06-20_00%3A00%3A00')
    fileCount = len(fnmatch.filter(os.listdir(XTRAIN_DATA_PATH), '*'))
    #dim = (fileCount,) + ds[DATA_VAR].shape[1:]  # First value is a time dimension of 1
    dim = (fileCount, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
    #ds.close()
    return dim
DIM = getInputDim()

def getDataDim():
    ds = xr.open_dataset(f'{XTRAIN_DATA_PATH}/wrfout_d02_2023-06-20_00%3A00%3A00')
    dim = ds[DATA_VAR].shape
    ds.close()
    # Here we change the size of dim to limit the amount of data that gets fed in
    dim = (dim[0], dim[1] // 5, dim[2] // 10, dim[3] // 10)
    return dim
DS_DIM = getDataDim()

def getGlobalRange():
    dim = getInputDim()
    maxValue = 0
    minValue = 0
    for i in range(dim[0]):
        path = getFilePath(i, dim)
        ds = xr.open_dataset(path)
        currentMax = float(ds[DATA_VAR].max())
        currentMin = float(ds[DATA_VAR].min())
        if currentMax > maxValue:
            maxValue = currentMax
        if currentMin < minValue:
            minValue = currentMin
        ds.close()
    return (minValue, maxValue)
DATA_RANGE = getGlobalRange()

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

class optimizeDataset(tf.data.Dataset):

    def __new__(cls):
        # Taken from getInputDim, but this is faster because we don't
        # need the first value of the tuple
        dim = (0, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        generator = WeatherGenerator(XTRAIN_DATA_PATH, YTRAIN_DATA_PATH, DATA_VAR, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, DIM, DS_DIM, DATA_RANGE)
        return tf.data.Dataset.from_generator(
            generator,
            output_signature=
            (tf.TensorSpec(shape=(BATCH_SIZE,) + dim[1:], dtype=tf.float32),
             tf.TensorSpec(shape=(BATCH_SIZE,) + dim[1:], dtype=tf.float32),
             tf.TensorSpec(shape=(BATCH_SIZE, prod(dim[1:])), dtype=tf.float32))
        )


# Creates a generator to get data
class WeatherGenerator:
    def __init__(self, xFilename, yFilename, dataVar, batchSize, epochs, validationSplit, dim, dsDim, dataRange):
        self.xFilename = xFilename
        self.yFilename = yFilename
        self.dataVar = dataVar
        self.batchSize = batchSize
        self.epochs = epochs
        self.validationSplit = validationSplit
        self.dim = dim
        self.dsDim = dsDim
        self.dataRange = dataRange
        stepsPerEpoch = int(np.floor((self.dim[0] * (1 - self.validationSplit)) / self.batchSize)) - 1
        self.length = stepsPerEpoch
        self.startIndex = int(self.dim[0] * validationSplit)
        self.trainOnX = True
        self.index = 0
        self.regionIndex = (0, 0, 0)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index, regionIndex):
        batchedX1Data = None
        batchedX2Data = None
        batchedYData = None
        for i in range(self.startIndex + (index * self.batchSize), self.startIndex + ((index + 1) * self.batchSize)):
            path = getFilePath(i, self.dim)
            try:
                dsX1 = xr.open_dataset(path)
                data = dsX1[self.dataVar][0][regionIndex[0]: regionIndex[0]+CUBE_SIZE,
                                             regionIndex[1]: regionIndex[1]+CUBE_SIZE,
                                             regionIndex[2]: regionIndex[2]+CUBE_SIZE]
                data = np.array([self.normalizeData(data)])  # Expand dims by 1
                #data = data / np.amax(data)
                if batchedX1Data is None:
                    batchedX1Data = deepcopy(data)
                else:
                    batchedX1Data = np.append(batchedX1Data, deepcopy(data), axis=0)
                dsX1.close()
            except Exception as e:
                raise Exception(f"{i} BatchX1Data error: {e}")


            path = getFilePath(i+2, self.dim)
            try:
                dsX2 = xr.open_dataset(path)
                data = dsX2[self.dataVar][0][regionIndex[0]: regionIndex[0]+CUBE_SIZE,
                                             regionIndex[1]: regionIndex[1]+CUBE_SIZE,
                                             regionIndex[2]: regionIndex[2]+CUBE_SIZE]
                data = np.array([self.normalizeData(data)])  # Expand dims by 1
                #data = data / np.amax(data)
                if batchedX2Data is None:
                    batchedX2Data = deepcopy(data)
                else:
                    batchedX2Data = np.append(batchedX2Data, deepcopy(data), axis=0)
                dsX2.close()
            except Exception as e:
                raise Exception(f"{i} BatchX2Data error: {e}")

            path = getFilePath(i+1, self.dim)
            try:
                dsY = xr.open_dataset(path)
                data = dsY[self.dataVar][0][regionIndex[0]: regionIndex[0]+CUBE_SIZE,
                                            regionIndex[1]: regionIndex[1]+CUBE_SIZE,
                                            regionIndex[2]: regionIndex[2]+CUBE_SIZE]
                data = np.array([self.normalizeData(data)])  # Expand dims by 1
                #data = data / np.amax(data)
                if batchedYData is None:
                    batchedYData = deepcopy(data)
                else:
                    batchedYData = np.append(batchedYData, deepcopy(data), axis=0)
                dsY.close()
            except Exception as e:
                raise Exception(f"{i} BatchYData error: {e}")

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
        if NORMALIZE_DATA:
            return data
        return (data - self.dataRange[0]) / (self.dataRange[1] - self.dataRange[0])


    def __call__(self):
        while True:
            region = (self.regionIndex[0] % self.dsDim[1],
                      self.regionIndex[1] % self.dsDim[2],
                      self.regionIndex[2] % self.dsDim[3])
            yield self.__getitem__(self.index % self.__len__(), region)
            self.index += 1
            self.regionIndex = (self.index // self.__len__(),
                                self.regionIndex[0] // self.dsDim[1],
                                self.regionIndex[1] // self.dsDim[2])



class ValidationData:

    def __init__(self, xFilename, yFilename, dataVar, validationSplit, validationSteps, batchSize, dim, dsDim, dataRange):
        self.xFilename = xFilename
        self.yFilename = yFilename
        self.dataVar = dataVar
        self.validationSplit = validationSplit
        self.validationSteps = validationSteps
        self.batchSize = batchSize
        self.dim = dim
        self.dsDim = dsDim
        self.dataRange = dataRange
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
            path = getFilePath(i, self.dim)
            try:
                dsX1 = xr.open_dataset(path)
                data = dsX1[self.dataVar][0][:CUBE_SIZE, :CUBE_SIZE, :CUBE_SIZE]
                data = np.array([self.normalizeData(data)])
                #data = data / np.amax(data)
                if batchedX1Data is None:
                    batchedX1Data = data
                else:
                    batchedX1Data = np.append(batchedX1Data, data, axis=0)
                dsX1.close()
            except Exception as e:
                raise Exception(f"{i} Validation BatchX1Data: {e}")

            path = getFilePath(i+2, self.dim)
            try:
                dsX2 = xr.open_dataset(path)
                data = dsX2[self.dataVar][0][:CUBE_SIZE, :CUBE_SIZE, :CUBE_SIZE]
                data = np.array([self.normalizeData(data)])
                #data = data / np.amax(data)
                if batchedX2Data is None:
                    batchedX2Data = data
                else:
                    batchedX2Data = np.append(batchedX2Data, data, axis=0)
                dsX2.close()
            except Exception as e:
                raise Exception(f"{i} Validation BatchX2Data: {e}")

            path = getFilePath(i+1, self.dim)
            try:
                dsY = xr.open_dataset(path)
                data = dsY[self.dataVar][0][:CUBE_SIZE, :CUBE_SIZE, :CUBE_SIZE]
                data = np.array([self.normalizeData(data)])
                #data = data / np.amax(data)
                if batchedYData is None:
                    batchedYData = data
                else:
                    batchedYData = np.append(batchedYData, data, axis=0)
                dsY.close()
            except Exception as e:
                raise Exception(f"{i} Validation BatchYData: {e}")

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
        if NORMALIZE_DATA:
            return data
        return (data - self.dataRange[0]) / (self.dataRange[1] - self.dataRange[0])


def fitModelWithGenerator():
    dim = getInputDim()

    dsDim = getDataDim()

    dataRange = getGlobalRange()

    # Starts from zero
    # Subtract one to give some headway for data
    STEPS_PER_EPOCH = int(((np.floor((dim[0] * (1 - VALIDATION_SPLIT)) / BATCH_SIZE)) - 1)
                          * (dsDim[1] // CUBE_SIZE)
                          * (dsDim[2] // CUBE_SIZE)
                          * (dsDim[3] // CUBE_SIZE))

    # VALIDATION_STEP = int(np.floor((dim[0]*VALIDATION_SPLIT) / BATCH_SIZE))

    # datasetGenerator = WeatherGenerator(XTRAIN_DATA_PATH, YTRAIN_DATA_PATH, DATA_VAR, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, dim, dsDim, dataRange)
    #
    # dataset = tf.data.Dataset.from_generator(
    #     datasetGenerator,
    #     output_signature=
    #         (tf.TensorSpec(shape=(BATCH_SIZE,) + dim[1:], dtype=tf.float32),
    #             tf.TensorSpec(shape=(BATCH_SIZE,) + dim[1:], dtype=tf.float32),
    #             tf.TensorSpec(shape=(BATCH_SIZE, prod(dim[1:])), dtype=tf.float32))
    # )

    dataset = optimizeDataset().prefetch(tf.data.AUTOTUNE)

    # validationGenerator = ValidationData(XTRAIN_DATA_PATH, YTRAIN_DATA_PATH, DATA_VAR, VALIDATION_SPLIT, VALIDATION_STEP, BATCH_SIZE, dim, dsDim, dataRange)

    # validationDataset = tf.data.Dataset.from_generator(
    #     validationGenerator,
    #     output_signature=
    #         (tf.TensorSpec(shape=(BATCH_SIZE,) + dim[1:], dtype=tf.float32),
    #             tf.TensorSpec(shape=(BATCH_SIZE,) + dim[1:], dtype=tf.float32),
    #             tf.TensorSpec(shape=(BATCH_SIZE, prod(dim[1:])), dtype=tf.float32))
    # )

    # The strategy stuff does not work because we are not running on multiple gpus
    # Tensorflow by itself combines all CPUs in a single device and runs multiple threads on that

    strategy = tf.distribute.MirroredStrategy()

    dataset = strategy.experimental_distribute_dataset(dataset)

    with strategy.scope():
        model = createCompiledModel(dim[1:])

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

    history = model.fit(
        dataset,
        batch_size=BATCH_SIZE,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        #validation_data=validationDataset,
        #validation_steps=VALIDATION_STEP,
        callbacks=[callback],
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

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

    history = model.fit(
        sequence,
        epochs=EPOCHS,
        validation_data=validationData.getValidationData(0),
        callbacks=[callback],
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

def main():
    history, model = fitModelWithGenerator()

    print(history.history)

    dataRange = getGlobalRange()

    ds = xr.open_dataset(f'{XTRAIN_DATA_PATH}/wrfout_d02_2023-06-20_00%3A00%3A00')
    data1 = ds[DATA_VAR][0][:CUBE_SIZE, :CUBE_SIZE, :CUBE_SIZE].to_numpy()
    data1 = np.expand_dims(data1, axis=0)

    ds.close()

    ds = xr.open_dataset(f'{XTRAIN_DATA_PATH}/wrfout_d02_2023-06-20_00%3A10%3A00')
    data2 = ds[DATA_VAR][0][:CUBE_SIZE, :CUBE_SIZE, :CUBE_SIZE].to_numpy()
    data2 = np.expand_dims(data2, axis=0)

    ds.close()

    if NORMALIZE_DATA:
        data1 = (data1 - dataRange[0]) / (dataRange[1] - dataRange[0])
        data2 = (data2 - dataRange[0]) / (dataRange[1] - dataRange[0])

    prediction = model.predict([data1, data2]).reshape(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

    #print(prediction)

    average = (data1 + data2) / 2

    ds = xr.open_dataset(f'{XTRAIN_DATA_PATH}/wrfout_d02_2023-06-20_00%3A05%3A00')
    dataY = ds[DATA_VAR][0][:CUBE_SIZE, :CUBE_SIZE, :CUBE_SIZE].to_numpy()

    if NORMALIZE_DATA:
        dataY = (dataY - dataRange[0]) / (dataRange[1] - dataRange[0])

    ds.close()

    average_error = abs(dataY - average)

    prediction_error = abs(dataY - prediction)

    print("###########Errors##########")

    print(average_error)
    print(prediction_error)

    print("###########Stats###########")

    print(f"Average summed error: {np.sum(average_error)}")
    print(f"Prediction summed error: {np.sum(prediction_error)}")

    print(f"Average (max, min) error: {np.max(average_error), np.min(average_error)}")

    print(f"Prediction (max, min) error: {np.max(prediction_error), np.min(prediction_error)}")

    print(f"Average mean error: {np.mean(average_error)}")
    print(f"Prediction mean error: {np.mean(prediction_error)}")


    if SAVE_MODEL:
        model.save(f'{KERAS_MODELS_PATH}/weatherModel.keras')

if __name__ == '__main__':
    faulthandler.enable()
    main()
