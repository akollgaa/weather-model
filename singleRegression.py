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

XTRAIN_DATA_PATH = 'D:/Documents/Code/research/data/testData.nc'
YTRAIN_DATA_PATH = 'D:/Documents/Code/research/data/testData.nc'
DATA_VAR = 'temperature'
BATCH_SIZE = 32
STEPS_PER_EPOCH = int(1000 / 32) # random guess; updated later
EPOCHS = 4
VALIDATION_SPLIT = 0.2
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
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, yPred)

        return {m.name: m.result() for m in self.metrics}

# Creates a model with one input of variable length
def createModel(inputShape):
    input1 = keras.Input(shape=inputShape, name='input1') # Takes one time step
    input2 = keras.Input(shape=inputShape, name='input2') # Takes the second time step
    x1 = layers.Flatten()(input1)
    x2 = layers.Flatten()(input2)
    x = layers.Concatenate()([x1, x2])
    x = layers.Dense(x.shape[1], activation='tanh', name='layer1')(x)
    output = layers.Dense((inputShape[0] * inputShape[1]))(x) # Applies linear activation
    
    return CustomModel(inputs=[input1, input2], outputs=output)

def createCompiledModel(inputDim):
    model = createModel(inputDim)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.mean_squared_error,
        metrics=[keras.metrics.mean_squared_error, keras.metrics.Accuracy]
    )

    return model

def getInputDim():
    ds = xr.open_dataset(XTRAIN_DATA_PATH)
    dim = (ds[DATA_VAR].shape[0], ds[DATA_VAR].shape[1], ds[DATA_VAR].shape[2])
    ds.close()
    return dim

# Gather the data by loading everything in np arrays
def getData():

    dsX = xr.open_dataset(XTRAIN_DATA_PATH)
    dsY = xr.open_dataset(YTRAIN_DATA_PATH)

    dataX = dsX[DATA_VAR].to_numpy()

    dataY = dsY[DATA_VAR].to_numpy()
    # Essientally flattens the data so it can be compared to the output
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
        self.length = int(np.ceil(ds[self.dataVar].shape[0] / float(self.batchSize)))
        ds.close()

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        dsX = xr.open_dataset(self.xFilename)
        data = dsX[self.dataVar]
        batchX = data[index * self.batchSize : (index + 1) * self.batchSize]
        batchX = batchX.to_numpy()
        batchX = batchX / np.amax(batchX)

        dsY = xr.open_dataset(self.yFilename)
        data = dsY[self.dataVar]
        batchY = data[index * self.batchSize: (index + 1) * self.batchSize]
        batchY = batchY.to_numpy()
        batchY = batchY / np.amax(batchY)
        # Flattens the y data so it in the correct output shape
        batchY = batchY.reshape(batchY.shape[0], batchY.shape[1] * batchY.shape[2])

        dsX.close()
        dsY.close()
        return batchX, batchY

    def getValidationData(self):
        pass

# Creates a generator to get data
class WeatherGenerator:
    def __init__(self, xFilename, yFilename, dataVar, batchSize, epochs, validationSplit, dim):
        super().__init__()
        self.xFilename = xFilename
        self.yFilename = yFilename
        self.dataVar = dataVar
        self.batchSize = batchSize
        self.epochs = epochs
        self.validationSplit = validationSplit
        self.dim = dim
        stepsPerEpoch = int(np.floor(((self.dim[0] * (1 - self.validationSplit)) / self.epochs) / self.batchSize)) - 1
        self.length = ((stepsPerEpoch + 1) * self.epochs)
        self.startIndex = int(self.dim[0] * validationSplit)
        self.trainOnX = True

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        dsX = xr.open_dataset(self.xFilename)
        data = dsX[self.dataVar]
        batchStartX = data[self.startIndex + (index * self.batchSize) : self.startIndex + ((index + 1) * self.batchSize)]
        batchEndX = data[self.startIndex + (index * self.batchSize) + 1: self.startIndex + ((index + 1) * self.batchSize) + 1]
        batchStartX = self.normalizeData(batchStartX)
        batchEndX = self.normalizeData(batchEndX)
        dsX.close()
        #return [batchStartX, batchEndX]
    
        dsY = xr.open_dataset(self.yFilename)
        data = dsY[self.dataVar]
        batchY = data[self.startIndex + (index * self.batchSize) : self.startIndex + ((index + 1) * self.batchSize)]
        batchY = self.normalizeData(batchY)

        # Flattens the y data so it in the correct output shape
        batchY = batchY.reshape(batchY.shape[0], batchY.shape[1] * batchY.shape[2])
        dsY.close()
        return batchStartX, batchEndX, batchY
    
    def normalizeData(self, data):
        data = data.to_numpy()
        return data / np.amax(data)
    
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    # Very similar to __getitem__
    def getValidationData(self):
        dsX = xr.open_dataset(self.xFilename)
        data = dsX[self.dataVar]
        batchX = data[:self.startIndex]
        batchX = batchX.to_numpy()
        batchX = batchX / np.amax(batchX)

        dsY = xr.open_dataset(self.yFilename)
        data = dsY[self.dataVar]
        batchY = data[:self.startIndex]
        batchY = batchY.to_numpy()
        batchY = batchY / np.amax(batchY)
        # Flattens the y data so it in the correct output shape
        batchY = batchY.reshape(batchY.shape[0], batchY.shape[1] * batchY.shape[2])

        dsX.close()
        dsY.close()
        return batchX, batchY


def fitModelWithGenerator():
    dim = getInputDim()

    # Starts from zero
    # Subtract one to give some headway for data
    STEPS_PER_EPOCH = int(np.floor(((dim[0] * (1 - VALIDATION_SPLIT)) / EPOCHS) / BATCH_SIZE)) - 1

    datasetGenerator = WeatherGenerator(XTRAIN_DATA_PATH, YTRAIN_DATA_PATH, DATA_VAR, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, dim)

    dataset = tf.data.Dataset.from_generator(
        datasetGenerator,
        output_signature =
            (tf.TensorSpec(shape=(BATCH_SIZE, dim[1], dim[2]), dtype=tf.int32),
                tf.TensorSpec(shape=(BATCH_SIZE, dim[1], dim[2]), dtype=tf.int32),
                tf.TensorSpec(shape=(BATCH_SIZE, dim[1] * dim[2]), dtype=tf.int32))
    )

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = createCompiledModel((dim[1], dim[2]))

    #validX, validY = dataGenerator.getValidationData()

    history = model.fit(
        dataset,
        batch_size=BATCH_SIZE,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        verbose=1
    )

    return history, model


# Does not implement strategy
def fitModelWithSequence():
    dim = getInputDim()

    sequence = WeatherSequence(XTRAIN_DATA_PATH, YTRAIN_DATA_PATH, DATA_VAR, BATCH_SIZE)

    model = createCompiledModel((dim[1], dim[2]))

    history = model.fit(
        sequence,
        epochs=EPOCHS,
        verbose=1
    )

    return history, model

history, model = fitModelWithGenerator()

print(history.history)

if SAVE_MODEL:
    model.save(f'{KERAS_MODELS_PATH}/weatherModel.keras')