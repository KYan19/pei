import xarray as xr
import numpy as np

# Load WBT and temp data for ens1-3
wbt = xr.open_dataset('../data/raw/GFDL/wbt_mean_1950-2100_ens1-3.nc',chunks={'time':1095})
t_ref = xr.open_dataset('../data/raw/GFDL/t_ref_mean_1950-2100_ens1-3.nc',chunks={'time':1095})

# Calculate WBGT
wbgt = 0.7*wbt['WBT'] + 0.3*(t_ref['t_ref']-273.15)
wbgt_monthly = wbgt.resample(time='1M').mean()

# Save monthly WBGT for ens1-3
wbgt_monthly.to_netcdf('../data/processed/GFDL/Monthly_WBGT/monthly_wbgt_ens001-003_1950-2100.nc')

# Load WBT and temp data for ens3-30
wbt = xr.open_mfdataset('../data/processed/GFDL/WBT/*',combine='nested',concat_dim='ensemble',chunks={'time':1095})
t_ref = xr.open_mfdataset('/local/ss23/GFDL_LEs/ATM/RCP85/t_ref.rcp85.ens*.1950_2100.nc',combine='nested',concat_dim='ensemble',chunks={'time':1095})

# Calculate WBGT
wbgt = 0.7*wbt['WBT'] + 0.3*(t_ref['t_ref']-273.15)
wbgt_monthly = wbgt.resample(time='1M').mean()

# Save monthly WBGT for ens3-30
for index in range(0,27,3):
    ds = wbgt_monthly.isel(ensemble=slice(index,index+3))
    filename = '../data/processed/GFDL/Monthly_WBGT/monthly_wbgt_ens'+str(index+4)+'-'+str(index+6)+'_1950-2100.nc'
    ds.to_netcdf(filename)