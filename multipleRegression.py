import tensorflow as tf
import xarray as xr
import numpy as np
import keras

X_TRAIN_DATA_PATH = 'D:/Documents/Code/research/data/largeTestData.nc'
Y_TRAIN_DATA_PATH = 'D:/Documents/Code/research/data/largeTestData.nc'
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


class WeatherDataGenerator:
    def __init__(self, dim, steps_per_epoch):
        self.dim = dim
        self.steps_per_epoch = steps_per_epoch
        self.index = 0
        self.start_index = self.dim[0] * VALIDATION_SPLIT

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        ds = xr.open_dataset(X_TRAIN_DATA_PATH)
        start_batch_x = self.gather_batches(ds, self.start_index, index)
        end_batch_x = self.gather_batches(ds, self.start_index + 1, index)
        ds.close()

        ds = xr.open_dataset(Y_TRAIN_DATA_PATH)
        batch_y = self.gather_batches(ds, self.start_index, index)
        ds.close()

        # We must flatten y so it matches the output shape
        batch_y = batch_y.reshape(batch_y.shape[0], batch_y.shape[1] * batch_y.shape[2])

        return start_batch_x, end_batch_x, batch_y

    def gather_batches(self, dataset, start_index, index):
        batch = None
        for data in dataset:
            # Slices through the time component of the data
            data_array = dataset[data][int(start_index + (index * BATCH_SIZE)):
                                       int(start_index + (index + 1) * BATCH_SIZE)]
            data_array = self.normalize(data_array)
            if batch is None:  # On the first try it creates an array
                batch = np.array(data_array)
            else:  # After it appends to the array
                batch = np.append(batch, data_array, axis=1)
        return batch

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
    output = keras.layers.Dense((input_shape[0] * input_shape[1]), name='output')(x)
    model = CustomModel(inputs=[start_time, end_time], outputs=output)
    return model


def create_compiled_model(input_shape):
    model = create_model(input_shape)

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy', 'mean_squared_error']
    )

    return model


def fit_model_with_generator(input_shape):
    model = create_compiled_model((input_shape[1], input_shape[2]))

    steps_per_epoch = int((input_shape[0] * (1-VALIDATION_SPLIT)) // BATCH_SIZE) - 1

    dataset_generator = WeatherDataGenerator(input_shape, steps_per_epoch)

    dataset = tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature=(
            tf.TensorSpec(shape=(BATCH_SIZE, input_shape[1], input_shape[2]), dtype=tf.int32),
            tf.TensorSpec(shape=(BATCH_SIZE, input_shape[1], input_shape[2]), dtype=tf.int32),
            tf.TensorSpec(shape=(BATCH_SIZE, input_shape[1] * input_shape[2]), dtype=tf.int32)
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
    ds = xr.open_dataset(X_TRAIN_DATA_PATH)
    shape = None
    for data in ds:
        if shape is None:
            shape = ds[data].shape
        else:
            # We add data to the second column of data
            # We can add it this way because we know all data arrays have the same shape
            shape = (shape[0], shape[1] + ds[data].shape[1], shape[2])
    return shape


data_shape = calculate_input_shape()

hist, mod = fit_model_with_generator(data_shape)

print(hist.history)

if SAVE_MODEL:
    mod.save(f'{KERAS_MODELS_PATH}/weatherModel.keras')
