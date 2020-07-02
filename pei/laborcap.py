import xarray as xr
import numpy as np

# Load WBT data
ds = xr.open_mfdataset('../data/processed/WBTdaily/WBTdailyens*.nc',combine='nested',concat_dim='ensemble',chunks={'time':1095})

# Adjust for upper and lower bound of WBGT-capacity interpolation
ds = ds.where(ds['WBT']>25,25)
ds = ds.where(ds['WBT']<33,33)

# Convert from WBGT to capacity
ds_daily = 100 - 25*((ds['WBT']-25)**(2/3))

# Annual average capacity
ds_yearly = ds_daily.resample(time='1Y').mean()

ds_yearly.to_netcdf('../data/processed/labor_productivity.nc')