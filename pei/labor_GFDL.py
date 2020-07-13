import xarray as xr
import numpy as np

# Load WBT and temp data
wbt = xr.open_mfdataset('../data/processed/GFDL/WBT/*',combine='nested',concat_dim='ensemble',chunks={'time':1095})
t_ref = xr.open_mfdataset('/local/ss23/GFDL_LEs/ATM/RCP85/t_ref.rcp85.ens*.1950_2100.nc',combine='nested',concat_dim='ensemble',chunks={'time':1095})

# Calculate WBGT
wbgt = 0.7*wbt['WBT'] + 0.3*(t_ref['t_ref']-273.15)

# Adjust for upper and lower bound of WBGT-capacity interpolation
wbgt = wbgt.where(wbgt>25,25)
wbgt = wbgt.where(wbgt<33,33)

# Convert from WBGT to capacity
cap_daily = 100 - 25*((wbgt-25)**(2/3))

# Go from daily capacity to monthly capacity
cap_monthly = cap_daily.resample(time='1M').mean()

# Save as data files
for index in range(0,27,3):
    ds = cap_monthly.isel(ensemble=slice(index,index+3))
    filename = '../data/processed/GFDL/Monthly_Capacity/monthly_capacity_ens'+str(index+4)+'-'+str(index+6)+'_1950-2100.nc'
    ds.to_netcdf(filename)