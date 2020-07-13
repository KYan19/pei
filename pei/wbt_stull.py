import xarray as xr
import numpy as np
from pei import thermodynamics as td

ensembles = range(104,107)
rootdir = '/local/ss23/GFDL_LEs/ATM/RCP85/'

for ens in ensembles:
    # Point to model data
    suffix = '.rcp85.ens' + str(ens) + '.1950_2100.nc'
    variables = ['ps','t_ref']
    ds = xr.Dataset()
    for var in variables:
        ds[var] = xr.open_dataarray(rootdir+var+suffix,chunks={'time':1000}).squeeze()
    ds['sphum_k24'] = xr.open_dataarray('../data/raw/GFDL/sphum_k24'+suffix,chunks={'time':1000}).squeeze()

    wbt = td.calc_wbt_from_tref_sh_p(ds['t_ref'], ds['sphum_k24'], ds['ps']/100, method='Stull')
    wbt.name = 'WBT'
    wbt.attrs = {'units':'degC',
                 'long_name':'Wet bulb temperature at 2m, from daily mean absolute temperature, specific humidity and pressure, calculated using Stull (2011) J. Appl. Meteor. Climatol.'}

    newdir = '../data/processed/GFDL/WBT/'

    wbt.to_netcdf(newdir+'wbt_mean_stull'+suffix)