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

ds = xr.open_dataset(f'D:/Documents/Code/research/data/timed-data/data1.nc')

print(len(ds))

for data in ds:
    print(ds[data].shape)



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

