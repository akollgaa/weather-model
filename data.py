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
lon = np.random.randn(4)
lat = np.random.randn(4)
time = np.random.randn(1000)

ds = xr.Dataset(
    data_vars=dict(
        temperature=(['time', 'lon', 'lat'], temperature)
    ),
    coords=dict(
        lon=('lon', lon),
        lat=('lat', lat),
        time=time
    ),
    attrs=dict(description="Random test data for the single regression model")
)

ds.to_netcdf('D:/Documents/Code/research/data/testData.nc')

