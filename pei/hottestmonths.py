import xarray as xr
import numpy as np
import pei.laborfunctions as lf

# Get hottest 3 months from each year -> ""summertime average" labor capacity

# Load ESM2M labor capacity data
ds_esm2m = xr.open_mfdataset('../data/processed/GFDL/Monthly_Capacity/monthly_capacity_ens*_1950-2100.nc',combine='nested',concat_dim='ensemble',chunks={'time':1095})
ds_esm2m = ds_esm2m.rename({'__xarray_dataarray_variable__':'capacity'})

# Shorten capacity dataset to 2000-2100
ds_esm2m = ds_esm2m.sel(time=slice('2000-01-31','2100-12-31'))['capacity']

# Calculate annual mean of three hottest months
ds_hottest_esm2m = xr.apply_ufunc(lf.max_avg,ds_esm2m,input_core_dims=[['time']],output_core_dims=[['year']],exclude_dims=set(('time',)),vectorize=True,dask='allowed',kwargs={'years':101})
ds_hottest_esm2m = ds_hottest_esm2m.assign_coords(year=range(2000,2101))

# Save output
ds_hottest_esm2m.to_netcdf('../data/processed/GFDL/esm2m_future_hottestmonths.nc')

# Load ESM2M labor capacity data
ds_cesm2 = xr.open_mfdataset('../data/processed/CESM2/Monthly_Capacity/*',combine='nested',concat_dim='ensemble',chunks={'time':1000}).rename({'__xarray_dataarray_variable__':'capacity'})

# Shorten capacity dataset to 2000-2100
ds_cesm2 = ds_cesm2.sel(time=slice('2000-01-31','2100-12-31'))['capacity']

# Calculate annual mean of three hottest months
ds_hottest_cesm2 = xr.apply_ufunc(lf.max_avg,ds_cesm2,input_core_dims=[['time']],output_core_dims=[['year']],exclude_dims=set(('time',)),vectorize=True,dask='allowed',kwargs={'years':101})
ds_hottest_cesm2 = ds_hottest_cesm2.assign_coords(year=range(2000,2101))

# Save output
ds_hottest_cesm2.to_netcdf('../data/processed/CESM2/cesm2_future_hottestmonths.nc')