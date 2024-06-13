import fnmatch
import os
import tensorflow as tf
import xarray as xr
import numpy as np
import keras
from math import prod

X_TRAIN_DATA_PATH = 'D:/Documents/Code/research/data/timed-data'
Y_TRAIN_DATA_PATH = 'D:/Documents/Code/research/data/actual-timed-data'
KERAS_MODELS_PATH = 'D:/Documents/Code/research/keras-models'
EPOCHS = 4
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SAVE_MODEL = False


class CustomModel(keras.Model):
    def train_step(self, data):
        x1, x2, y = data

        with tf.GradientTape() as tape:
            y_pred = self([x1, x2], training=True)
            loss = self.compute_loss(y=y, y_pred=y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


# A more specific version of an append function for numpy
# Allows the array to grow in a new dimension continuously
def append_to_array(array1, array2):
    if array1 is None or array2 is None:
        return array2
    elif len(array1.shape) == len(array2.shape):
        return np.array([array1, array2])
    return np.append(array1, [array2], axis=0)


class WeatherDataGenerator:
    def __init__(self, dim, steps_per_epoch):
        self.dim = dim
        self.steps_per_epoch = steps_per_epoch
        self.index = 0
        self.start_index = self.dim[0] * VALIDATION_SPLIT

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        start_batch_x = None
        end_batch_x = None
        batch_y = None
        for i in range(int(self.start_index + (index * BATCH_SIZE)), int(self.start_index + ((index + 1) * BATCH_SIZE))):
            current_start_x_batch = self.gather_batches(X_TRAIN_DATA_PATH, i)
            start_batch_x = append_to_array(start_batch_x, current_start_x_batch)

            current_end_x_batch = self.gather_batches(X_TRAIN_DATA_PATH, i + 1)
            end_batch_x = append_to_array(end_batch_x, current_end_x_batch)

            current_batch_y = self.gather_batches(Y_TRAIN_DATA_PATH, i)
            batch_y = append_to_array(batch_y, current_batch_y)

        batch_y = np.reshape(batch_y, (batch_y.shape[0], -1))  # Flattens the array\
        return start_batch_x, end_batch_x, batch_y

    def gather_batches(self, file_path, index):
        ds = xr.open_dataset(f'{file_path}/data{index}.nc')
        combined_data = None
        for data in ds:
            data_array = ds[data]
            data_array = self.normalize(data_array)  # Note: converts to a numpy array
            combined_data = append_to_array(combined_data, data_array)
        ds.close()
        return combined_data

    @staticmethod
    def normalize(data_array):
        data_array = data_array.to_numpy()
        return data_array / np.amax(data_array)

    def __call__(self):
        # We repeat the data over itself if it reaches the end
        while True:
            x1, x2, y = self.__getitem__(self.index % self.__len__())
            yield x1, x2, y
            self.index += 1


def create_model(input_shape):
    start_time = keras.Input(shape=input_shape, name='start')
    end_time = keras.Input(shape=input_shape, name='end')
    start_time_flatten = keras.layers.Flatten()(start_time)
    end_time_flatten = keras.layers.Flatten()(end_time)
    x = keras.layers.Concatenate()([start_time_flatten, end_time_flatten])
    x = keras.layers.Dense(x.shape[1], activation='tanh', name='layer1')(x)
    output = keras.layers.Dense(prod(input_shape), name='output')(x)
    model = CustomModel(inputs=[start_time, end_time], outputs=output)
    return model


def create_compiled_model(input_shape):
    model = create_model(input_shape)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.mean_squared_error,
        metrics=['mae']
    )

    return model


def fit_model_with_generator(data_size, input_shape):
    model = create_compiled_model(input_shape)

    steps_per_epoch = int((data_size * (1 - VALIDATION_SPLIT)) // BATCH_SIZE) - 1

    dataset_generator = WeatherDataGenerator(input_shape, steps_per_epoch)

    dataset = tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE, *input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE, *input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE, prod(input_shape)), dtype=tf.float32)
        )
    )

    history = model.fit(
        dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        batch_size=BATCH_SIZE,
        verbose=1
    )

    return history, model


def calculate_input_shape():
    ds = xr.open_dataset(f'{X_TRAIN_DATA_PATH}/data0.nc')
    shape = ()
    counter = 1
    for data in ds:
        shape = (counter, *ds[data].shape)
        counter += 1
    return shape


def calculate_data_size():
    ds = xr.open_dataset(f'{X_TRAIN_DATA_PATH}/data0.nc')
    file_count = len(fnmatch.filter(os.listdir(X_TRAIN_DATA_PATH), '*.nc'))
    return file_count


data_shape = calculate_input_shape()

data_count = calculate_data_size()

hist, mod = fit_model_with_generator(data_count, data_shape)

print(hist.history)

if SAVE_MODEL:
    mod.save(f'{KERAS_MODELS_PATH}/weatherModel.keras')
