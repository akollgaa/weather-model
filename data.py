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

# ds = xr.open_dataset(f'D:/Documents/Code/research/wrfout/wrfout_d02_2023-06-20_00%3A00%3A00')
# temp = ds['T']
#
# print(temp)
# val = temp[0][:2, :2, :2].to_numpy()
#
# print(val)
#
# ds.close()

XTRAIN_DATA_PATH = 'D:/Documents/Code/research/wrfout'
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 4
fileCount = len(fnmatch.filter(os.listdir(XTRAIN_DATA_PATH), '*'))

dim = (fileCount,)
length = int(np.floor((dim[0] * (1 - VALIDATION_SPLIT)) / BATCH_SIZE)) - 1
length = int(dim[0] * (1 - VALIDATION_SPLIT))
def getFilePath(index):
    day = '0' if index != (dim[0] - 1) else '1'
    minute = str((index % 12) * 5).zfill(2)
    hour = str(index // 12).zfill(2)

    if day == '1':  # Edge case
        hour = '00'

    return f'{XTRAIN_DATA_PATH}/wrfout_d02_2023-06-2{day}_{hour}%3A{minute}%3A00'

for i in range(fileCount):
    path = getFilePath(i)
    try:
        ds = xr.open_dataset(path)
        temp = ds['T'][0][:2, :4, :3].to_numpy()
        ds.close()
    except Exception as e:
        print(e)




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

