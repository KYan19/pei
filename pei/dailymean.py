import xarray as xr

# List of ensembles members to process
ensembles = ['006','007','008','009','010']

for ens in ensembles:
    # WBT data crunching
    # Path to data files for this ensemble member
    paths_WBT = '/local/ss23/CESM2_LE/ATM/RCP85/WBT/b.e21.*.f09_g17.LE2-1231.' + ens + '.clm2.h7.WBT.*.nc'
    
    # Concat along time dimension
    ds = xr.open_mfdataset(paths_WBT,combine='by_coords',coords=['time'],chunks={'time':1095})['WBT']
    
    # Calculate daily max
    ds_mean = ds.resample(time='1D').mean()
    
    # Save as new data file
    ds_mean.to_netcdf('../data/processed/CESM2/WBTdailymean/WBTens'+ens+'.nc')
    
    # TSA data crunching
    # Path to data files for this ensemble member
    paths_TSA = '/local/ss23/CESM2_LE/ATM/RCP85/TSA/b.e21.*.f09_g17.LE2-1231.' + ens + '.clm2.h7.TSA.*.nc'
    
    # Concat along time dimension
    ds = xr.open_mfdataset(paths_TSA,combine='by_coords',coords=['time'],chunks={'time':1095})['TSA']
    
    # Calculate daily max
    ds_mean = ds.resample(time='1D').mean()
    
    # Save as new data file
    ds_mean.to_netcdf('../data/processed/CESM2/TSAdailymean/TSAens'+ens+'.nc')