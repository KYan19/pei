import xarray as xr
import numpy as np
import pei.myfunctions as mf
from xhistogram.xarray import histogram

# Load WBT data
ds = xr.open_mfdataset('../data/processed/WBTdaily/WBTdailyens*.nc',combine='nested',concat_dim='ensemble',chunks={'time':1095})

# Load area data
land_area = xr.open_dataset('../data/processed/wbt.land_area')
land_mask = np.isfinite(land_area)

# Replace NaN with 0 
land_area_adj = land_area.where(land_mask,0).rename({'__xarray_dataarray_variable__':'land_area'})

# Dictionary to store binned DataArrays
ds_dict = {}

regions = ['Northern North America','Central North America','Southern North America',
          'Central America','Northern South America','Southern South America',
          'Scandinavia','Central Europe','Southern Europe','European Russia',
          'Middle East','India','Southeast Asia','Northern China','Southern China',
          'Northern Oceania','Southern Oceania',
          'Northern Africa','Central Africa','Southern Africa']

for region in regions:
    # Isolate WBT and area data for region
    ds_region = mf.slice_region(ds,region)
    area_region = mf.slice_region(land_area_adj,region)

    # Specify histogram variable
    var = ds_region['WBT']

    # Bins to divide data at each point in time
    bins = np.arange(-55,40,0.1)

    # Specify area-weight variable
    area_weights = area_region['land_area']

    # Histogram in lat and lon dimensions
    dist = histogram(var,bins=[bins],dim=['lat','lon'],weights=area_weights)

    # Sum distributions within each year
    dist_annual = dist.groupby('time.year').sum('time')
    
    # Add to dictionary
    ds_dict[region] = dist_annual

ds_byregion = xr.Dataset(ds_dict)
ds_byregion.to_netcdf('../data/processed/WBT_binned.nc')