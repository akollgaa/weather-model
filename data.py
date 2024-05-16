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

temperature = np.random.randn(1000, 4, 4)
precipitation = np.random.rand(1000, 4, 4)
humidity = np.random.rand(1000, 4, 4)
lon = np.arange(4)
lat = np.arange(4)
time = np.arange(1000)

ds = xr.Dataset(
    data_vars=dict(
        temperature=(['time', 'lon', 'lat'], temperature),
        precipitation=(['time', 'lon', 'lat'], precipitation),
        humidity=(['time', 'lon', 'lat'], humidity)
    ),
    coords=dict(
        lon=('lon', lon),
        lat=('lat', lat),
        time=time
    ),
    attrs=dict(description="Random test data for a regression model")
)

#data = ds['temperature']

#print(data[0:36].shape)

ds.to_netcdf('D:/Documents/Code/research/data/largeTestData.nc')

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

