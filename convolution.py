import tensorflow as tf
import xarray as xr
import numpy as np
import os
import glob
import keras
import random
from math import prod

DATA_PATH = "D:/Documents/Code/research/wrfout"
ALL_FILES = [os.path.basename(f) for f in glob.glob(DATA_PATH + "/wrfout*")][:10]
INPUT_SHAPE = (20, 20, 20)  # This is a subset of the larger data to be processed
BATCH_SIZE = 16
DATA_VARIABLES = [
    'T'
]

# These are precalculated values to speed up computation
DATA_VARS_MIN = {'T': -8.417083740234375,
                 'P': -50.28515625,
                 'PB': 5420.28076171875,
                 'QVAPOR': 0.0,
                 'QRAIN': -1.3538986555136634e-12,
                 'QSNOW': -3.2677986886402296e-15,
                 'QGRAUP': 0.0,
                 'U': -14.949191093444824,
                 'V': -28.35787010192871,
                 'W': -5.611952781677246,
                 'QCLOUD': -7.121892919154105e-12}

DATA_VARS_MAX = {'T': 192.04437255859375,
                 'P': 1313.625,
                 'PB': 97532.5546875,
                 'QVAPOR': 0.02198491431772709,
                 'QRAIN': 0.0021312725730240345,
                 'QSNOW': 0.0001476061879657209,
                 'QGRAUP': 0.0,
                 'U': 26.568021774291992,
                 'V': 17.940385818481445,
                 'W': 8.062047958374023,
                 'QCLOUD': 0.002458590315654874}


@keras.saving.register_keras_serializable(package="customModel")
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

    def test_step(self, data):
        x1, x2, y = data

        y_pred = self([x1, x2], training=False)

        loss = self.compute_loss(y=y, y_pred=y_pred)

        for metric in self.metrics:
            if metric.name != 'loss':
                metric.update_state(y, y_pred)
            else:
                metric.update_state(loss)

        return {m.name: m.result() for m in self.metrics}


def normalize(var, data) -> np.array:
    data = data.to_numpy()
    return (data - DATA_VARS_MIN[var]) / (DATA_VARS_MAX[var] - DATA_VARS_MIN[var])


# Non-mutating function
def append_to_array(array1, array2):
    array2 = np.expand_dims(array2, axis=0)
    if array1 is None:
        return array2
    return np.append(array1, array2, axis=0)


def load_data():
    print("\rLoading data...", end='')
    data = {var: None for var in DATA_VARIABLES}
    for i in range(len(ALL_FILES)):
        ds = xr.open_dataset(f"{DATA_PATH}/{ALL_FILES[i]}")
        print(f"\rLoading data {i}/{len(ALL_FILES)}", end='')
        for var in DATA_VARIABLES:
            arr = normalize(var, ds[var])
            data[var] = append_to_array(data[var], arr)
        ds.close()
    print("\rLoading data complete")
    return data


class DataGenerator:
    def __init__(self, data):
        self.index = 0
        self.data = data
        self.data_shape = (len(self.data[DATA_VARIABLES[0]]) - 4,
                           len(self.data[DATA_VARIABLES[0]][0][0]) - INPUT_SHAPE[2],
                           len(self.data[DATA_VARIABLES[0]][0][0][0]) - INPUT_SHAPE[1],
                           len(self.data[DATA_VARIABLES[0]][0][0][0][0]) - INPUT_SHAPE[0])
    def __len__(self):
        return prod(self.data_shape)

    def __getitem__(self):
        start_batch_x, end_batch_x, batch_y = None, None, None
        for _ in range(BATCH_SIZE):
            column = self.index % self.data_shape[3]
            row = (self.index // self.data_shape[3]) % self.data_shape[2]
            height = (self.index // (self.data_shape[3] * self.data_shape[2])) % self.data_shape[1]
            time = (self.index // (self.data_shape[3] * self.data_shape[2] * self.data_shape[1])) % self.data_shape[0]

            batch_x1, batch_x2, batch_y1 = None, None, None
            for var in DATA_VARIABLES:
                batch_x1 = append_to_array(batch_x1, self.get_batch(var, column, row, height, time))
                batch_x2 = append_to_array(batch_x2, self.get_batch(var, column, row, height, time + 4))
                batch_y1 = append_to_array(batch_y1, self.get_batch(var, column, row, height, time + 2))
            start_batch_x = append_to_array(start_batch_x, batch_x1)
            end_batch_x = append_to_array(end_batch_x, batch_x2)
            batch_y = append_to_array(batch_y, batch_y1)
            self.index += random.randint(0, self.__len__()-1)
            self.index %= self.__len__()

        batch_y = np.reshape(batch_y, newshape=(batch_y.shape[0], batch_y.shape[1], -1))
        return start_batch_x, end_batch_x, batch_y

    def get_batch(self, var, column, row, height, time):
        # Indexing the zero is because the data shape has a size of 1
        batch = self.data[var][time][0][
            height: height + INPUT_SHAPE[2],
            row: row + INPUT_SHAPE[1],
            column: column + INPUT_SHAPE[0]
        ]
        return batch

    def __call__(self):
        while True:
            yield self.__getitem__()
            self.index = random.randint(0, self.__len__()-1)
            self.index %= self.__len__()


def create_model(input_shape):
    input1 = keras.layers.Input(shape=input_shape, name="start")
    input2 = keras.layers.Input(shape=input_shape, name="end")

    input = keras.layers.Concatenate()([input1, input2])

    x = keras.layers.ConvLSTM3D

    x1 = keras.layers.Conv3D(filters=32, kernel_size=7, strides=1, data_format="channels_first", name="1Conv3D1")(input1)
    x1 = keras.layers.MaxPool3D(pool_size=2, strides=2, name="1MaxPool1")(x1)
    x1 = keras.layers.Conv3D(filters=64, kernel_size=5, strides=1, data_format="channels_first", name="1Conv3D2")(x1)
    x1 = keras.layers.MaxPool3D(pool_size=2, strides=2, name="1MaxPool2")(x1)
    x1 = keras.layers.Flatten()(x1)

    x2 = keras.layers.Conv3D(filters=32, kernel_size=7, strides=1, data_format="channels_first", name="2Conv3D1")(input2)
    x2 = keras.layers.MaxPool3D(pool_size=2, strides=2, name="2MaxPool1")(x2)
    x2 = keras.layers.Conv3D(filters=64, kernel_size=5, strides=1, data_format="channels_first", name="2Conv3D2")(x2)
    x2 = keras.layers.MaxPool3D(pool_size=2, strides=2, name="2MaxPool2")(x2)
    x2 = keras.layers.Flatten()(x2)

    x = keras.layers.Concatenate()([x1, x2])
    x = keras.layers.Dense(prod(x.shape[1:]), activation="relu", name="dense1")(x)
    output = keras.layers.Dense(prod(input_shape), name="output")(x)

    return CustomModel(inputs=[input1, input2], outputs=output)


# TODO: Check if this gets aliased
def compile_model(model):
    model.compile(
        optimizer="adam",
        loss=keras.losses.mean_squared_error,
        weighted_metrics=["mae", "msle"]
    )
    return model


def get_prediction_data(data, var, time):
    height, row, column = 0, 0, 0
    return data[var][time][0][
        height: height + INPUT_SHAPE[2],
        row: row + INPUT_SHAPE[1],
        column: column + INPUT_SHAPE[0]
    ]


def main():
    model = create_model((len(DATA_VARIABLES), *INPUT_SHAPE))
    model = compile_model(model)

    data = load_data()

    dataset = DataGenerator(data)

    dataset = tf.data.Dataset.from_generator(
        dataset,
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE, len(DATA_VARIABLES), *INPUT_SHAPE), dtype='float32'),
            tf.TensorSpec(shape=(BATCH_SIZE, len(DATA_VARIABLES), *INPUT_SHAPE), dtype='float32'),
            tf.TensorSpec(shape=(BATCH_SIZE, len(DATA_VARIABLES), prod(INPUT_SHAPE)), dtype='float32'),
        )
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=2)

    history = model.fit(
        dataset,
        epochs=10,
        steps_per_epoch=10,
        batch_size=BATCH_SIZE,
        callbacks=[callback],
        verbose=1
    )

    data1 = np.expand_dims([get_prediction_data(data, DATA_VARIABLES[0], 0)], axis=0)
    data2 = np.expand_dims([get_prediction_data(data, DATA_VARIABLES[0], 4)], axis=0)
    actual_data = get_prediction_data(data, DATA_VARIABLES[0], 2)

    prediction = model.predict([data1, data2]).reshape(INPUT_SHAPE)
    average = (data1 + data2) / 2

    avg_error = np.mean(abs(average - actual_data))
    prediction_error = np.mean(abs(prediction - actual_data))

    avg_error_max = np.max(abs(average - actual_data))
    prediction_error_max = np.max(abs(prediction - actual_data))

    avg_error_min = np.min(abs(average - actual_data))
    prediction_error_min = np.min(abs(prediction - actual_data))

    print(f"Average error: {avg_error}\n" +
          f"Prediction error: {prediction_error}\n" +
          f"Average max error: {avg_error_max}\n" +
          f"Prediction max error: {prediction_error_max}\n" +
          f"Average min error: {avg_error_min}\n" +
          f"Prediction min error: {prediction_error_min}\n")


if __name__ == "__main__":
    main()