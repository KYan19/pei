import xarray as xr
import numpy as np

# Load WBT data
paths_wbt = ['../data/processed/CESM2/WBTdailymean/WBT1281ens001.nc','../data/processed/CESM2/WBTdailymean/WBT1281ens002.nc','../data/processed/CESM2/WBTdailymean/WBT1281ens003.nc',
        '../data/processed/CESM2/WBTdailymean/WBT1281ens004.nc','../data/processed/CESM2/WBTdailymean/WBT1281ens005.nc']
wbt = xr.open_mfdataset(paths_wbt,combine='nested',concat_dim='ensemble',chunks={'time':1095})

# Load temp data
paths_tsa = ['../data/processed/CESM2/TSAdailymean/TSA1281ens001.nc','../data/processed/CESM2/TSAdailymean/TSA1281ens002.nc','../data/processed/CESM2/TSAdailymean/TSA1281ens003.nc',
        '../data/processed/CESM2/TSAdailymean/TSA1281ens004.nc','../data/processed/CESM2/TSAdailymean/TSA1281ens005.nc']
t_ref = xr.open_mfdataset(paths_tsa,combine='nested',concat_dim='ensemble',chunks={'time':1095})

# Calculate WBGT
wbgt = 0.7*wbt['WBT'] + 0.3*(t_ref['TSA']-273.15)

# Go from daily WBGT to monthly WBGT
wbgt_monthly = wbgt.resample(time='1M').mean()

# Save as data file
wbgt_monthly.to_netcdf('../data/processed/CESM2/Monthly_WBGT/monthly_wbgt_1281_ens1-5_1980-2100.nc')

# Load WBT data
paths_wbt = ['../data/processed/CESM2/WBTdailymean/WBT1281ens006.nc','../data/processed/CESM2/WBTdailymean/WBT1281ens007.nc','../data/processed/CESM2/WBTdailymean/WBT1281ens008.nc',
        '../data/processed/CESM2/WBTdailymean/WBT1281ens009.nc','../data/processed/CESM2/WBTdailymean/WBT1281ens010.nc']
wbt = xr.open_mfdataset(paths_wbt,combine='nested',concat_dim='ensemble',chunks={'time':1095})

# Load temp data
paths_tsa = ['../data/processed/CESM2/TSAdailymean/TSA1281ens006.nc','../data/processed/CESM2/TSAdailymean/TSA1281ens007.nc','../data/processed/CESM2/TSAdailymean/TSA1281ens008.nc',
        '../data/processed/CESM2/TSAdailymean/TSA1281ens009.nc','../data/processed/CESM2/TSAdailymean/TSA1281ens010.nc']
t_ref = xr.open_mfdataset(paths_tsa,combine='nested',concat_dim='ensemble',chunks={'time':1095})

# Calculate WBGT
wbgt = 0.7*wbt['WBT'] + 0.3*(t_ref['TSA']-273.15)

# Go from daily WBGT to monthly WBGT
wbgt_monthly = wbgt.resample(time='1M').mean()

# Save as data file
wbgt_monthly.to_netcdf('../data/processed/CESM2/Monthly_WBGT/monthly_wbgt_1281_ens6-10_1980-2100.nc')