import xarray as xr

# List of ensembles members to process
ensembles = ['001','002','003','004','005']

for ens in ensembles:
    # Path to data files for this ensemble member
    paths = '/local/ss23/CESM2_LE/ATM/RCP85/WBT/b.e21.*.f09_g17.LE2-1231.' + ens + '.clm2.h7.WBT.*.nc'
    
    # Concat along time dimension
    ds = xr.open_mfdataset(paths,combine='by_coords',coords=['time'],chunks={'time':1095})['WBT']
    
    # Calculate daily max
    ds_max = ds.resample(time='1D').max()
    
    # Save as new data file
    ds_max.to_netcdf('../data/processed/CESM2/WBTdailymax/WBTens'+ens+'.nc')