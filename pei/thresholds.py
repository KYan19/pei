import xarray as xr
import numpy as np

years = np.arange(2000,2101)
freqs1 = {}
freqs2 = {}
freqs3 = {}

for year in years:
    ds = xr.open_dataset('../data/processed/WBTyearly/WBT'+str(year)+'.nc')
    ds_region = ds.sel(lat=slice(-40,40))
    
    # Days greater than heat stress threshold
    thres1 = (ds_region.sel(WBT_bin=slice(25,None)).sum(['WBT_bin']))
    thres2 = (ds_region.sel(WBT_bin=slice(27.9,None)).sum(['WBT_bin']))
    thres3 = (ds_region.sel(WBT_bin=slice(30,None)).sum(['WBT_bin']))
    
    # Add to dictionary
    freqs1[year]=thres1['histogram_WBT']
    freqs2[year]=thres2['histogram_WBT']
    freqs3[year]=thres3['histogram_WBT']
    
ds_freq1 = xr.Dataset(freqs1).to_array('year','Frequencies')
ds_freq1.to_netcdf('../data/processed/WBTthresholds/thres25.nc')

ds_freq2 = xr.Dataset(freqs2).to_array('year','Frequencies')
ds_freq2.to_netcdf('../data/processed/WBTthresholds/thres27.9.nc')

ds_freq3 = xr.Dataset(freqs3).to_array('year','Frequencies')
ds_freq3.to_netcdf('../data/processed/WBTthresholds/thres30.nc')