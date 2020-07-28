import xarray as xr
import numpy as np
import pei.laborfunctions as lf

wbt1 = xr.open_dataset('../data/raw/GFDL/wbt_mean_1950-2100_ens1-3.nc',chunks={'time':1095})
wbt2 = xr.open_mfdataset('../data/processed/GFDL/WBT/wbt_mean_stull*',combine='nested',concat_dim='ensemble',chunks={'time':1095})
wbt = xr.concat([wbt1,wbt2],dim='ensemble')

tref1 = xr.open_dataset('../data/raw/GFDL/t_ref_mean_1950-2100_ens1-3.nc',chunks={'time':1095})
tref2 = xr.open_mfdataset('/local/ss23/GFDL_LEs/ATM/RCP85/t_ref.rcp85*',combine='nested',concat_dim='ensemble',chunks={'time':1095})
tref = xr.concat([tref1,tref2],dim='ensemble')

tref_seasonal = tref.resample(time='Q-NOV').mean(['time','ensemble'])

wbgt = 0.7*wbt['WBT'] + 0.3*(tref['t_ref']-273.15)
wbgt_seasonal = wbgt.resample(time='Q-NOV').mean(['time','ensemble'])

tref_seasonal.to_netcdf('../data/processed/GFDL/Map_Data/tref_seasonal.nc')
wbgt_seasonal.to_netcdf('../data/processed/GFDL/Map_Data/wbgt_seasonal.nc')