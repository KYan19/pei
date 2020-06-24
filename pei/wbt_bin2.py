import xarray as xr
import numpy as np
import pei.myfunctions as mf
from xhistogram.xarray import histogram

# Load WBT data
ds = xr.open_mfdataset('../data/processed/WBTdaily/WBTdailyens*.nc',combine='nested',concat_dim='ensemble',chunks={'time':1095})

# Load area data
land_area = xr.open_dataset('../data/processed/wbt.land_area')
land_mask = np.isfinite(land_area)

# Reduce WBT data
ds_adj = ds['WBT'].where(land_mask,drop=True).rename({'__xarray_dataarray_variable__':'WBT'})

# Get WBT
var = ds_adj['WBT']

# Bins to divide data at each point in time
bins = np.arange(-52,35,0.1)

# Loop through years
for year in range(2070,2090):
    # Get data for specific year
    var_year = var.where(var['time.year']==year,drop=True)
    
    # Histogram in time dimension for this year
    dist = histogram(var_year,bins=[bins],dim=['time'])
    
    # Save this year's histogram data as netCDF
    dist.to_netcdf('../data/processed/WBTyearly/WBT'+str(year)+'.nc')