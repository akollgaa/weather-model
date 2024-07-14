import fnmatch
import os

import numpy as np
import pandas as pd
import packaging as pack
import netCDF4 as cdf
import xarray as xr
import random

# ds = xr.open_dataset("D:/Documents/Code/research/data/sresa1b_ncar_ccsm3-example.nc")

# There is a lat, lon, plev(Z-axis(height) measured in pressure) at a single time
# ua is windspeed
# tas is temperature
# pr is precipitation flux

# print(ds)

# #arr = ds['ua'].to_numpy()

# print(ds['ua'].to_numpy().shape)
# print(ds['tas'].to_numpy().shape)
# print(ds['pr'].to_numpy().shape[0])

# ds.close()

DATA_VARS = [
    'T',
    'P',
    'PB',
    'QVAPOR',
    'QRAIN',
    'QSNOW',
    'QGRAUP',
    'U',
    'V',
    'W',
    'QCLOUD'
]

X_TRAIN_DATA_PATH = 'D:/Documents/Code/research/wrfout'

#ds = xr.open_dataset(f'D:/Documents/Code/research/wrfout/wrfout_d02_2023-06-20_00%3A00%3A00')

#print(ds["T"].shape)

def calculate_data_range(data_size):
    max_value = {}
    min_value = {}
    for i in range(data_size):
        path = get_file_path(i, data_size)
        ds = xr.open_dataset(path)
        for data in DATA_VARS:
            current_max = float(ds[data].max())
            current_min = float(ds[data].min())
            if data not in max_value or current_max > max_value[data]:
                max_value[data] = current_max
            if data not in min_value or current_min < min_value[data]:
                min_value[data] = current_min
        ds.close()
    return min_value, max_value

def calculate_data_mean_std(data_size):
    mean_value = {}
    std_value = {}
    n = 0
    for i in range(data_size):
        path = get_file_path(i, data_size)
        ds = xr.open_dataset(path)
        for data in DATA_VARS:
            current_mean = float(ds[data].mean())
            current_std = ((float(ds[data].std()) ** 2) * ds[data].size)
            mean_value[data] = mean_value.get(data, 0) + current_mean
            std_value[data] = std_value.get(data, 0) + current_std
            n += ds[data].size
        ds.close()
    for data in DATA_VARS:
        mean_value[data] = mean_value[data] / data_size
        std_value[data] = (std_value[data] / n) ** 0.5
    return mean_value, std_value


def get_file_path(index, data_size):
    day = '0' if index != (data_size-1) else '1'
    minute = str((index % 12) * 5).zfill(2)
    hour = str(index // 12).zfill(2)

    if day == '1':  # Edge case
        hour = '00'

    return f'{X_TRAIN_DATA_PATH}/wrfout_d02_2023-06-2{day}_{hour}%3A{minute}%3A00'

def calculate_data_size():
    file_count = len(fnmatch.filter(os.listdir(X_TRAIN_DATA_PATH), '*'))
    return file_count

#print(calculate_data_range(calculate_data_size()))

#print(calculate_data_mean_std(calculate_data_size()))



# temp = ds['T']
#
# value = float(temp.max())
# value = float(temp.min())
# value = float(temp.median())
# value = float(temp.mean())
#
# print(f'The max is {value}')
#
# print(temp)
# val = temp[0][:2, :2, :2].to_numpy()
#
# print(val)
#
# ds.close()

# XTRAIN_DATA_PATH = 'D:/Documents/Code/research/wrfout'
# VALIDATION_SPLIT = 0.2
# BATCH_SIZE = 4
# fileCount = len(fnmatch.filter(os.listdir(XTRAIN_DATA_PATH), '*'))
#
# dim = (fileCount,)
# length = int(np.floor((dim[0] * (1 - VALIDATION_SPLIT)) / BATCH_SIZE)) - 1
# length = int(dim[0] * (1 - VALIDATION_SPLIT))
# def getFilePath(index):
#     day = '0' if index != (dim[0] - 1) else '1'
#     minute = str((index % 12) * 5).zfill(2)
#     hour = str(index // 12).zfill(2)
#
#     if day == '1':  # Edge case
#         hour = '00'
#
#     return f'{XTRAIN_DATA_PATH}/wrfout_d02_2023-06-2{day}_{hour}%3A{minute}%3A00'
#
# for i in range(fileCount):
#     path = getFilePath(i)
#     try:
#         ds = xr.open_dataset(path)
#         temp = ds['T'][0][:2, :4, :3].to_numpy()
#         ds.close()
#     except Exception as e:
#         print(e)




# for x in range(1000):
#
#     temperature = np.arange(4 * 4).reshape(4, 4) + (2 * x)
#     precipitation = np.random.rand(4, 4)
#     humidity = np.random.rand(4, 4)
#     lon = np.arange(4)
#     lat = np.arange(4)
#
#     ds = xr.Dataset(
#         data_vars=dict(
#             temperature=(['lon', 'lat'], temperature),
#             precipitation=(['lon', 'lat'], precipitation),
#             humidity=(['lon', 'lat'], humidity)
#         ),
#         coords=dict(
#             lon=('lon', lon),
#             lat=('lat', lat),
#         ),
#         attrs=dict(description="Random test data for a regression model")
#     )
#
#     #print(data[0:36].shape)
#
#     ds.to_netcdf(f'D:/Documents/Code/research/data/timed-data/data{x}.nc')

# arr = np.array([[[1, 2, 3, 4], 
#                  [5, 6, 7, 8], 
#                  [9, 10, 11, 12], 
#                  [13, 14, 15, 16]], 
#                  [[1, 1, 1, 1], 
#                  [1, 1, 1, 1], 
#                  [1, 1, 1, 1], 
#                  [1, 1, 1, 1]]])

# print(arr)

# print(arr.shape)

# tensor = tf.convert_to_tensor(arr)

# arr = arr.reshape(arr.shape[0], arr.shape[1] * arr.shape[2])

# tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1))

# print(arr.shape)

# print(arr)

# print(tensor)

