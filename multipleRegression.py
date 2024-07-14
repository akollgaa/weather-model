import fnmatch
import os
import tensorflow as tf
import xarray as xr
import numpy as np
import keras
from math import prod
from datetime import datetime

X_TRAIN_DATA_PATH = 'D:/Documents/Code/research/wrfout'
KERAS_MODELS_PATH = 'D:/Documents/Code/research/keras-models'
DATA_OUT_PATH = 'D:/Documents/Code/research/output'
BATCH_SIZE = (16, ) * 14
EPOCHS = (25,) * len(BATCH_SIZE)
VALIDATION_SPLIT = 0.2
SAVE_MODEL = (True, ) * len(BATCH_SIZE)
CALCULATE_STATS = True
NORMALIZE = (True,) * len(BATCH_SIZE)  # False implies standardization will be used instead
CUBE_SIZE = (1, 2, 3, 4,) + (4,) * len(BATCH_SIZE)
SHAPE_REDUCE = ([6, 7, 7],) * len(BATCH_SIZE)
TOTAL_ITERATIONS = len(BATCH_SIZE)
LOAD_DATA = True
DATA_VARS = [
    'T',
    'P',
    'PB',
    'QVAPOR',
    'QRAIN',
    'QSNOW',
    'U',
    'V',
    'W',
    'QCLOUD'
]
#VARIABLE_TEST = [{data: True for data in DATA_VARS}] * len(BATCH_SIZE)
# Funky code, but we'll live
VARIABLE_TEST = [{data: True for data in DATA_VARS}] * 4
for i in range(len(DATA_VARS)):
    new_dict = {data: True if j <= i else False for j, data in enumerate(DATA_VARS)}
    VARIABLE_TEST.append(new_dict)

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

DATA_VARS_MEAN = {'T': 35.421511191397805,
                  'P': 402.6549568572259,
                  'PB': 56756.6796875,
                  'QVAPOR': 0.005870880280334541,
                  'QRAIN': 5.818143079671262e-08,
                  'QSNOW': 9.50368343477367e-08,
                  'QGRAUP': 0.0,
                  'U': 1.3379405083128326,
                  'V': -0.38019444416422127,
                  'W': 0.002123257230844196,
                  'QCLOUD': 2.2856711307277937e-06}

DATA_VARS_STD = {'T': 12.86922902730058,
                 'P': 113.14531431220384,
                 'PB': 9284.244562274176,
                 'QVAPOR': 0.0018998037632266662,
                 'QRAIN': 3.8898476392190537e-07,
                 'QSNOW': 6.546815799049338e-07,
                 'QGRAUP': 0.0,
                 'U': 1.890336787350396,
                 'V': 2.3707455801620534,
                 'W': 0.041930900232612055,
                 'QCLOUD': 1.0485764700858156e-05}


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

        self.compute_loss(y=y, y_pred=y_pred)

        for metric in self.metrics:
            if metric.name != 'loss':
                metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}


# A more specific version of an append function for numpy
# Allows the array to grow in a new dimension continuously
def append_to_array(array1, array2):
    array2 = np.expand_dims(array2, axis=0)
    if array1 is None:
        return array2
    return np.append(array1, array2, axis=0)


# Both the x and y path are the same
def get_file_path(index, data_size):
    day = '0' if index != (data_size - 1) else '1'
    minute = str((index % 12) * 5).zfill(2)
    hour = str(index // 12).zfill(2)

    if day == '1':  # Edge case
        hour = '00'

    return f'{X_TRAIN_DATA_PATH}/wrfout_d02_2023-06-2{day}_{hour}%3A{minute}%3A00'


# Check what is the most amount of variables we are going to ever test and load those
def get_max_true_variables():
    most_vars = 0
    most_true = 0
    for i in range(len(VARIABLE_TEST)):
        true_count = 0
        for var in VARIABLE_TEST[i]:
            if VARIABLE_TEST[i][var]:
                true_count += 1
        if most_true < true_count:
            most_vars = i
            most_true = true_count
    return most_vars


# Takes all the necessary data and loads it into a numpy array.
def load_data(data_size, iteration, data_stats):
    all_data = []
    length = int(data_size * (1 - VALIDATION_SPLIT))
    most_vars = get_max_true_variables()
    for index in range(length):
        print(f"\rLoading data index: {index}/{length - 1}", end='')
        ds = xr.open_dataset(get_file_path(index, data_size))
        combined_data = {}
        for data in DATA_VARS:
            if not VARIABLE_TEST[most_vars][data]:  # Skip variables that we won't directly test
                continue
            data_array = ds[data]
            if NORMALIZE[iteration]:
                data_array = normalize(data_array, data, data_stats)
            else:
                data_array = standardize(data_array, data, data_stats)
            combined_data[data] = data_array
        ds.close()
        all_data.append(combined_data)
    print('\rLoading data index: Complete')
    return all_data


def normalize(data_array, data, data_stats) -> np.array:
    data_array = data_array.to_numpy()
    return (data_array - data_stats[0][data]) / (data_stats[1][data] - data_stats[0][data])


def standardize(data_array, data, data_stats) -> np.array:
    data_array = data_array.to_numpy()
    return (data_array - data_stats[2][data]) / data_stats[3][data]


class WeatherDataGenerator:
    def __init__(self, data_size, dim, length, data_dim, iteration, all_data, start_index, data_stats):
        self.data_size = data_size
        self.dim = dim
        self.length = length
        self.data_dim = data_dim
        self.iteration = iteration
        self.index = 0
        self.region_index = (0, 0, 0)
        self.start_index = start_index
        self.data_stats = data_stats
        if LOAD_DATA:
            self.all_data = all_data

    def __len__(self):
        return self.length

    def __getitem__(self, index, region_index):
        start_batch_x = None
        end_batch_x = None
        batch_y = None
        start = int(self.start_index + (index * BATCH_SIZE[self.iteration]))
        end = int(self.start_index + ((index + 1) * BATCH_SIZE[self.iteration]))
        for i in range(start, end):
            current_start_x_batch = self.gather_batches(i, region_index)
            start_batch_x = append_to_array(start_batch_x, current_start_x_batch)

            current_end_x_batch = self.gather_batches(i + 2, region_index)
            end_batch_x = append_to_array(end_batch_x, current_end_x_batch)

            current_batch_y = self.gather_batches(i + 1, region_index)
            batch_y = append_to_array(batch_y, current_batch_y)

        batch_y = np.reshape(batch_y, (batch_y.shape[0], -1))  # Flattens the array
        return start_batch_x, end_batch_x, batch_y

    def gather_batches(self, index, region_index):
        if LOAD_DATA:
            ds = self.all_data[index - self.start_index]
        else:
            path = get_file_path(index, self.data_size)
            ds = xr.open_dataset(path)
        combined_data = None
        for data in DATA_VARS:
            if not VARIABLE_TEST[self.iteration][data]:  # Skip variables that we won't directly test
                continue
            data_array = ds[data][0][region_index[0]: region_index[0] + CUBE_SIZE[self.iteration],
                                     region_index[1]: region_index[1] + CUBE_SIZE[self.iteration],
                                     region_index[2]: region_index[2] + CUBE_SIZE[self.iteration]]
            if not LOAD_DATA:
                if NORMALIZE[self.iteration]:
                    data_array = normalize(data_array, data, self.data_stats)
                else:
                    data_array = standardize(data_array, data, self.data_stats)
            combined_data = append_to_array(combined_data, data_array)
        if not LOAD_DATA:
            ds.close()
        return combined_data

    def __call__(self):
        # We repeat the data over itself if it reaches the end
        while True:
            region = (self.region_index[0] % self.data_dim[1],
                      self.region_index[1] % self.data_dim[2],
                      self.region_index[2] % self.data_dim[3])
            x1, x2, y = self.__getitem__(self.index % self.__len__(), region)
            yield x1, x2, y
            self.index += 1
            self.region_index = (self.index // self.__len__(),
                                 self.region_index[0] // self.data_dim[1],
                                 self.region_index[1] // self.data_dim[2])


def create_model(input_shape):
    start_time = keras.Input(shape=input_shape, name='start')
    end_time = keras.Input(shape=input_shape, name='end')

    start_time_flatten = keras.layers.Flatten()(start_time)
    end_time_flatten = keras.layers.Flatten()(end_time)

    x = keras.layers.Concatenate()([start_time_flatten, end_time_flatten])
    x = keras.layers.Dense(x.shape[1], activation='linear', name='layer1')(x)
    x = keras.layers.Dense(x.shape[1], activation='linear', name='layer2')(x)

    output = keras.layers.Dense(prod(input_shape), name='output')(x)
    model = CustomModel(inputs=[start_time, end_time], outputs=output)
    return model


def create_compiled_model(input_shape):
    model = create_model(input_shape)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.mean_squared_error,
        weighted_metrics=['mae', 'msle']
    )

    return model


def fit_model_with_generator(model, data_size, input_shape, data_dim, iteration, all_data, data_stats):
    steps_per_epoch = int((((data_size * (1 - VALIDATION_SPLIT)) // BATCH_SIZE[iteration]) - 1)
                          * (data_dim[1] // CUBE_SIZE[iteration])
                          * (data_dim[2] // CUBE_SIZE[iteration])
                          * (data_dim[3] // CUBE_SIZE[iteration]))

    length = int(((data_size * (1 - VALIDATION_SPLIT)) // BATCH_SIZE[iteration]) - 1)

    start_index = int(data_size * VALIDATION_SPLIT)

    dataset_generator = WeatherDataGenerator(data_size, input_shape, length, data_dim,
                                             iteration, all_data, start_index, data_stats)

    dataset = tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE[iteration], *input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE[iteration], *input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE[iteration], prod(input_shape)), dtype=tf.float32)
        )
    )

    start_index = 0

    length = int(((data_size * VALIDATION_SPLIT) // BATCH_SIZE[iteration]) - 1)

    validation_generator = WeatherDataGenerator(data_size, input_shape, length, data_dim,
                                                iteration, all_data, start_index, data_stats)

    validation_dataset = tf.data.Dataset.from_generator(
        validation_generator,
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE[iteration], *input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE[iteration], *input_shape), dtype=tf.float32),
            tf.TensorSpec(shape=(BATCH_SIZE[iteration], prod(input_shape)), dtype=tf.float32)
        )
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

    history = model.fit(
        dataset,
        epochs=EPOCHS[iteration],
        validation_data=validation_dataset,
        validation_steps=length,
        steps_per_epoch=steps_per_epoch,
        batch_size=BATCH_SIZE[iteration],
        callbacks=[callback],
        verbose=1
    )

    return history, model


def calculate_input_shape(iteration):
    length = 0
    for var in VARIABLE_TEST[iteration]:
        if VARIABLE_TEST[iteration][var]:
            length += 1
    shape = (length,) + (CUBE_SIZE[iteration],) * 3
    return shape


def calculate_data_shape(iteration):
    ds = xr.open_dataset(f'{X_TRAIN_DATA_PATH}/wrfout_d02_2023-06-20_00%3A00%3A00')
    shape = ds[DATA_VARS[0]].shape
    # Here we compute shape alterations that reduces the amount of data
    shape = (shape[0],
             shape[1] // SHAPE_REDUCE[iteration][0],
             shape[2] // SHAPE_REDUCE[iteration][1],
             shape[3] // SHAPE_REDUCE[iteration][2])
    ds.close()
    return shape


def calculate_data_size():
    file_count = len(fnmatch.filter(os.listdir(X_TRAIN_DATA_PATH), '*'))
    return file_count


def calculate_data_range(data_size, iteration):
    max_value = {}
    min_value = {}
    for i in range(data_size):
        path = get_file_path(i, data_size)
        ds = xr.open_dataset(path)
        for data in DATA_VARS:
            if not VARIABLE_TEST[iteration][data]:
                continue
            current_max = float(ds[data].max())
            current_min = float(ds[data].min())
            if data not in max_value or current_max > max_value[data]:
                max_value[data] = current_max
            if data not in min_value or current_min < min_value[data]:
                min_value[data] = current_min
        ds.close()
    return min_value, max_value


def calculate_data_mean_std(data_size, iteration):
    mean_value = {}
    std_value = {}
    n = 0
    for i in range(data_size):
        path = get_file_path(i, data_size)
        ds = xr.open_dataset(path)
        for data in DATA_VARS:
            if not VARIABLE_TEST[iteration][data]:
                continue
            current_mean = float(ds[data].mean())
            current_std = ((float(ds[data].std()) ** 2) * ds[data].size)
            mean_value[data] = mean_value.get(data, 0) + current_mean
            std_value[data] = std_value.get(data, 0) + current_std
            n += ds[data].size
        ds.close()
    for data in DATA_VARS:
        if not VARIABLE_TEST[iteration][data]:
            continue
        mean_value[data] = mean_value[data] / data_size
        std_value[data] = (std_value[data] / n) ** 0.5
    return mean_value, std_value


def get_prediction_data(index, data_size, iteration, data_stats) -> np.array:
    path = get_file_path(index, data_size)
    ds = xr.open_dataset(path)
    data = None
    for d in DATA_VARS:
        if not VARIABLE_TEST[iteration][d]:
            continue
        raw_data = ds[d][0][:CUBE_SIZE[iteration], :CUBE_SIZE[iteration], :CUBE_SIZE[iteration]]
        normalized_data = normalize(raw_data, d, data_stats)
        data = append_to_array(data, normalized_data)
    ds.close()
    return data


def main():

    # Not proud of this but was a quick solution.
    data_size = calculate_data_size()

    data_stats = [DATA_VARS_MIN, DATA_VARS_MAX, DATA_VARS_MEAN, DATA_VARS_STD]

    if CALCULATE_STATS:
        max_vars = get_max_true_variables()
        data_range = calculate_data_range(data_size, max_vars)
        data_mean_std = calculate_data_mean_std(data_size, max_vars)
        # Reset stats as local variables.
        data_stats = [data_range[0], data_range[1], data_mean_std[0], data_mean_std[1]]
    
    if LOAD_DATA:
        all_data = load_data(data_size, 0, data_stats)
    else:
        all_data = None

    for i in range(TOTAL_ITERATIONS):

        input_shape = calculate_input_shape(i)

        data_shape = calculate_data_shape(i)

        model = create_compiled_model(input_shape)

        hist, mod = fit_model_with_generator(model, data_size, input_shape, data_shape, i, all_data, data_stats)

        # Now that the model is trained we can do predictions to see how well it works

        data1 = np.expand_dims(get_prediction_data(0, data_size, i, data_stats), axis=0)
        data2 = np.expand_dims(get_prediction_data(2, data_size, i, data_stats), axis=0)
        actual_data = get_prediction_data(1, data_size, i, data_stats)

        prediction = mod.predict([data1, data2]).reshape(input_shape)

        average = (data1 + data2) / 2

        average_error = abs(actual_data - average)

        prediction_error = abs(prediction - average)

        try:
            file = open(f'{DATA_OUT_PATH}/output.txt', 'a+')
            now = str(datetime.now())
            file.write(f"Ran model-{i} at {now} \n")
            file.write(f"Batch size: {BATCH_SIZE[i]} | Cube size: {CUBE_SIZE[i]} | " +
                       f"Variables tested: {VARIABLE_TEST[i]} | Shape reduction: {SHAPE_REDUCE[i]} | " +
                       f"Normalization: {NORMALIZE} | Epochs Trained: {hist.epoch}\n")
            file.write(f"{hist.history} \n")
            file.write(f"Average summed error: {np.sum(average_error)} \n")
            file.write(f"Prediction summed error: {np.sum(prediction_error)} \n")
            file.write(f"Average (max, min) error: {np.max(average_error), np.min(average_error)} \n")
            file.write(f"Prediction (max, min) error: {np.max(prediction_error), np.min(prediction_error)} \n")
            file.write(f"Average mean error: {np.mean(average_error)} \n")
            file.write(f"Prediction mean error: {np.mean(prediction_error)} \n")
            if SAVE_MODEL[i]:
                file.write(f"Saving model to {KERAS_MODELS_PATH} \n")
            file.write(f"\n")
            file.close()
        except Exception as e:
            raise Exception(f'{i} - File could not be opened: {e}')

        if SAVE_MODEL[i]:
            mod.save(f'{KERAS_MODELS_PATH}/weatherModel-{i}.keras')

        # Check if we need to load different data
        if i+1 < TOTAL_ITERATIONS and NORMALIZE[i] != NORMALIZE[i + 1]:
            all_data = load_data(data_size, i + 1)



if __name__ == '__main__':
    main()
