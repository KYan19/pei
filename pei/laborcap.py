import xarray as xr
import numpy as np

# Load WBT data
ds = xr.open_mfdataset('../data/processed/WBTdaily/WBTdailyens*.nc',combine='nested',concat_dim='ensemble',chunks={'time':1095})

ds_daily = 100 - 25*((ds['WBT']-25)**(2/3))
ds_daily = ds_daily.where(np.isfinite(ds_daily),100)

ds_yearly = ds_daily.resample(time='1Y').mean()

ds_yearly.to_netcdf('../data/processed/labor_productivity.nc')