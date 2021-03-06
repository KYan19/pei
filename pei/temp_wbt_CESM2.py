import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import pei.laborfunctions as lf

wbt = xr.open_mfdataset('../data/processed/CESM2/WBTdailymean/*',combine='nested',concat_dim='ensemble',chunks={'time':1095})
tref = xr.open_mfdataset('../data/processed/CESM2/TSAdailymean/*',combine='nested',concat_dim='ensemble',chunks={'time':1095})

tref_seasonal = tref.resample(time='Q-NOV').mean(['time'])

wbgt = 0.7*wbt['WBT'] + 0.3*(tref['TSA']-273.15)
wbgt_seasonal = wbgt.resample(time='Q-NOV').mean(['time'])

tref_seasonal.to_netcdf('../data/processed/CESM2/Map_Data/tref_seasonal_ens1-30.nc')
wbgt_seasonal.to_netcdf('../data/processed/CESM2/Map_Data/wbgt_seasonal_ens1-30.nc')