import xarray as xr
import numpy as np

# Load WBT and temp data
wbt = xr.open_mfdataset('../data/processed/WBTdailymax/*',combine='nested',concat_dim='ensemble',chunks={'time':1095})
t_ref = xr.open_mfdataset('../data/processed/CESM2/TSAdailymax/*',combine='nested',concat_dim='ensemble',chunks={'time':1095})

# Calculate WBGT
wbgt = 0.7*wbt['WBT'] + 0.3*(t_ref['TSA']-273.15)

# Adjust for upper and lower bound of WBGT-capacity interpolation
wbgt = wbgt.where(wbgt>25,25)
wbgt = wbgt.where(wbgt<33,33)

# Convert from WBGT to capacity
cap_daily = 100 - 25*((wbgt-25)**(2/3))

# Go from daily capacity to annual capacity
cap_yearly = cap_daily.resample(time='1Y').mean()

# Save as data file
cap_yearly.to_netcdf('../data/processed/CESM2/labor_max_ens1-5.nc')