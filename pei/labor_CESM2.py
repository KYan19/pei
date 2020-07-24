import xarray as xr
import numpy as np

# Load WBT data
paths_wbt = ['../data/processed/CESM2/WBTdailymean/WBT1281ens006.nc','../data/processed/CESM2/WBTdailymean/WBT1281ens007.nc','../data/processed/CESM2/WBTdailymean/WBT1281ens008.nc',
        '../data/processed/CESM2/WBTdailymean/WBT1281ens010.nc']
wbt = xr.open_mfdataset(paths_wbt,combine='nested',concat_dim='ensemble',chunks={'time':1095})

# Load temp data
paths_tsa = ['../data/processed/CESM2/TSAdailymean/TSA1281ens006.nc','../data/processed/CESM2/TSAdailymean/TSA1281ens007.nc','../data/processed/CESM2/TSAdailymean/TSA1281ens008.nc',
        '../data/processed/CESM2/TSAdailymean/TSA1281ens010.nc']
t_ref = xr.open_mfdataset(paths_tsa,combine='nested',concat_dim='ensemble',chunks={'time':1095})

# Calculate WBGT
wbgt = 0.7*wbt['WBT'] + 0.3*(t_ref['TSA']-273.15)

# Adjust for upper and lower bound of WBGT-capacity interpolation
wbgt = wbgt.where(wbgt>25,25)
wbgt = wbgt.where(wbgt<33,33)

# Convert from WBGT to capacity
cap_daily = 100 - 25*((wbgt-25)**(2/3))

# Go from daily capacity to monthly capacity
cap_monthly = cap_daily.resample(time='1M').mean()

# Save as data file
cap_monthly.to_netcdf('../data/processed/CESM2/Monthly_Capacity/monthly_capacity_1281_ens6-7-8-10_1980-2100.nc')