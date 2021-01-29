import xarray as xr
import numpy as np
import pei.laborfunctions as lf

# Get hottest 3 months from each year -> ""summertime average" WBGT
# Load ESM2M WBGT data
ds_esm2m = xr.open_mfdataset('../data/processed/GFDL/Monthly_WBGT/monthly_wbgt_ens*_1950-2100.nc',combine='nested',concat_dim='ensemble',chunks={'time':1095})
ds_esm2m = ds_esm2m.rename({'__xarray_dataarray_variable__':'WBGT'})

# Shorten WBGT dataset to 1980-2100
ds_esm2m = ds_esm2m.sel(time=slice('1980-01-31','2099-12-31'))['WBGT']

# Calculate annual mean of three hottest months
ds_hottest_esm2m = xr.apply_ufunc(lf.wbgt_max_avg,ds_esm2m,input_core_dims=[['time']],output_core_dims=[['year']],exclude_dims=set(('time',)),vectorize=True,dask='allowed',kwargs={'years':120})
ds_hottest_esm2m = ds_hottest_esm2m.assign_coords(year=range(1980,2100))

# Save output
ds_hottest_esm2m.to_netcdf('../data/processed/GFDL/esm2m_summer_wbgt_1980-2100.nc')

# Load CESM2 WBGT data
ds_cesm2 = xr.open_mfdataset('../data/processed/CESM2/Monthly_WBGT/*',combine='nested',concat_dim='ensemble',chunks={'time':1000}).rename({'__xarray_dataarray_variable__':'WBGT'})

# Shorten WBGT dataset to 1980-2020
ds_cesm2 = ds_cesm2.sel(time=slice('1980-01-31','2099-12-31'))['WBGT']

# Calculate annual mean of three hottest months
ds_hottest_cesm2 = xr.apply_ufunc(lf.wbgt_max_avg,ds_cesm2,input_core_dims=[['time']],output_core_dims=[['year']],exclude_dims=set(('time',)),vectorize=True,dask='allowed',kwargs={'years':120})
ds_hottest_cesm2 = ds_hottest_cesm2.assign_coords(year=range(1980,2100))

# Save output
ds_hottest_cesm2.to_netcdf('../data/processed/CESM2/cesm2_summer_wbgt_1980-2100.nc')