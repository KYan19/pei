# Functions to use while working with labor capacity

import xarray as xr
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import numpy as np

# Path to sample data file, CESM2
path_CESM2 = '../data/processed/CESM2/population_regrid_cesm2.nc'
# Load data from matching file
ds_CESM2 = xr.open_dataset(path_CESM2)

# Path to sample data file, GFDL
path_GFDL = '../data/processed/GFDL/population_regrid_esm2m.nc'
# Load data from matching file
ds_GFDL = xr.open_dataset(path_GFDL)

# Populates dictionary of regional masks based on gridding of sample dataset
def fill_mask(ds,masks):
    lon = ds['longitude']
    lat = ds['latitude']
    
    masks['Global'] = [lon.values, lat.values]
    masks['Northern North America'] = [lon.where((190<=lon)&(lon<=310),drop=True).values,lat.where((45<=lat)&(lat<=75),drop=True).values]
    masks['Central North America'] = [lon.where((230<=lon)&(lon<=310),drop=True).values,lat.where((35<=lat)&(lat<=45),drop=True).values]
    masks['Southern North America'] = [lon.where((230<=lon)&(lon<=300),drop=True).values,lat.where((20<=lat)&(lat<=35),drop=True).values]
    masks['Central America'] = [lon.where((250<=lon)&(lon<=315),drop=True).values,lat.where((7<=lat)&(lat<=20),drop=True).values]
    masks['Northern South America'] = [lon.where((270<=lon)&(lon<=330),drop=True).values,lat.where((-23.5<=lat)&(lat<=7),drop=True).values]
    masks['Southern South America'] = [lon.where((270<=lon)&(lon<=330),drop=True).values,lat.where((-57<=lat)&(lat<=-23.5),drop=True).values]

    masks['Scandinavia'] = [lon.where((3<=lon)&(lon<=30),drop=True).values,lat.where((55<=lat)&(lat<=70),drop=True).values]
    lon_west = lon.where(lon>=345,drop=True)
    lon_east = lon.where(lon<=35,drop=True)
    lon_ceur = xr.concat((lon_west,lon_east),dim='longitude').values
    masks['Central Europe'] = [lon_ceur,lat.where((43<=lat)&(lat<=55),drop=True).values]
    lon_west = lon.where(lon>=350,drop=True)
    lon_east = lon.where(lon<=24,drop=True)
    lon_seur = xr.concat((lon_west,lon_east),dim='longitude').values
    masks['Southern Europe'] = [lon_seur,lat.where((36<=lat)&(lat<=43),drop=True).values]

    masks['Northern China'] = [lon.where((75<=lon)&(lon<=135),drop=True).values,lat.where((32<=lat)&(lat<=50),drop=True).values]
    masks['Southern China'] = [lon.where((98<=lon)&(lon<=125),drop=True).values,lat.where((22<=lat)&(lat<=32),drop=True).values]
    masks['India'] = [lon.where((68<=lon)&(lon<=90),drop=True).values,lat.where((8<=lat)&(lat<=30),drop=True).values]
    masks['Southeast Asia'] = [lon.where((92<=lon)&(lon<=140),drop=True).values,lat.where((0<=lat)&(lat<=25),drop=True).values]
    masks['Middle East'] = [lon.where((25<=lon)&(lon<=60),drop=True).values,lat.where((10<=lat)&(lat<=40),drop=True).values]
    masks['European Russia'] = [lon.where((43<=lon)&(lon<=70),drop=True).values,lat.where((50<=lat)&(lat<=75),drop=True).values]

    masks['Northern Oceania'] = [lon.where((100<=lon)&(lon<=180),drop=True).values,lat.where((-23.5<=lat)&(lat<=0),drop=True).values]
    masks['Southern Oceania'] = [lon.where((100<=lon)&(lon<=180),drop=True).values,lat.where((-50<=lat)&(lat<=-23.5),drop=True).values]
    masks['Indonesia'] = [lon.where((95<=lon)&(lon<=142),drop=True).values,lat.where((-10<=lat)&(lat<=5),drop=True).values]
    masks['Philippines'] = [lon.where((115<=lon)&(lon<=130),drop=True).values,lat.where((5<=lat)&(lat<=20),drop=True).values]

    lon_west = lon.where(lon>=340,drop=True)
    lon_east = lon.where(lon<=55,drop=True)
    lon_cafrica = xr.concat((lon_west,lon_east),dim='longitude').values
    masks['Central Africa'] = [lon_cafrica,lat.where((-10<=lat)&(lat<=10),drop=True).values]
    lon_west = lon.where(lon>=340,drop=True)
    lon_east = lon.where(lon<=25,drop=True)
    lon_nafrica = xr.concat((lon_west,lon_east),dim='longitude').values
    masks['Northern Africa'] = [lon_nafrica,lat.where((10<=lat)&(lat<=38),drop=True).values]
    masks['Southern Africa'] = [lon.where((9<=lon)&(lon<=52),drop=True).values,lat.where((-35<=lat)&(lat<=-10),drop=True).values]
    
# Call fill_mask() on GFDL and CESM2 datasets
masks_GFDL = {}
masks_CESM2 = {}
fill_mask(ds_GFDL,masks_GFDL)
fill_mask(ds_CESM2,masks_CESM2)

# Function to isolate data for a particular region
def slice_region(ds, region, model):
    if model == 'GFDL':
        return ds.sel(lon=masks_GFDL[region][0],lat=masks_GFDL[region][1])
    elif model == 'CESM2':
        return ds.sel(lon=masks_CESM2[region][0],lat=masks_CESM2[region][1])
    else:
        raise ValueError('Model name not valid')

# Function to plot capacity over time for a particular region
# Ensemble members + ensemble average
def capacity(ds,ds_pop,region,model,ax,color='royalblue'):
    # Get yearly capacity data for grid cells in region
    ds_region = slice_region(ds,region,model)
    
    # Get population for grid cells
    pop_region = slice_region(ds_pop,region,model)
    
    # Calculate total area-weighted capacity per year
    capacity = ds_region.weighted(pop_region).mean(['lon','lat'])
    
    # Loop through ensemble members
    for ens in capacity['ensemble']:
        capacity.sel(ensemble=ens).plot(ax=ax,color=color,alpha=0.25)

    # Ensemble average labor capacity
    capacity_avg = capacity.mean(dim='ensemble')
    capacity_avg.plot(ax=ax,color=color,linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Labor Capacity, %')
    ax.set_title(region)

# Function to create a contour plot of labor capacity (new axis)
def contour_plot(ds,title,levels=10,cmap='Reds',label='Labor Capacity, %'):
    # Specify projection
    crs = ccrs.PlateCarree()

    # Create figure and axes
    fig,ax = plt.subplots(figsize=(10,10),subplot_kw={'projection':crs})

    # Specify variables
    X = ds['lon']
    Y = ds['lat']
    Z = ds.squeeze()
    Z, X = add_cyclic_point(Z,coord=X)

    # Create contour plot
    im = ax.contourf(X,Y,Z,levels=levels,transform=crs,cmap=cmap,extend='both')

    # Add coastlines and ocean mask
    ax.coastlines()
    ax.add_feature(cfeature.OCEAN,zorder=10,facecolor='lightskyblue')
    
    # Add national boundaries
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),edgecolor='silver')

    # Set colorbar, title
    cbar = plt.colorbar(im,ax=ax,orientation='horizontal',fraction=0.05,pad=0.05)
    cbar.set_label(label,fontsize=12)
    plt.title(title)
    
# Function to create a contour plot of labor capacity (axis as parameter)
def contour(ds,title,ax,levels=10,cmap='Reds',label='Labor Capacity, %'):
    # Specify projection
    crs = ccrs.PlateCarree()
    
    # Specify variables
    X = ds['lon']
    Y = ds['lat']
    Z = ds.squeeze()
    Z, X = add_cyclic_point(Z,coord=X)

    # Create contour plot
    im = ax.contourf(X,Y,Z,levels=levels,transform=crs,cmap=cmap,extend='both')
    im.cmap.set_over('darkgray')

    # Add coastlines and ocean mask
    ax.coastlines()
    ax.add_feature(cfeature.OCEAN,zorder=10,facecolor='lightskyblue')
    
    # Add national boundaries
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),edgecolor='silver')

    # Set title
    ax.set_title(title,fontsize=14)
    
    return im

# Function finds ToE as number of years before 2100
def emergence(ds,thres=90):
    # Array indices where capacity < threshold
    ds_thres = (ds<thres).nonzero()
    
    # If non-empty, return ToE as number of years before 2100
    if len(ds_thres[0]) > 0:
        return 150-ds_thres[0][0].item()
    
    # If empty, return 0
    return 0

# Function finds number of years in 21st century after ToE
def thres_years(ds,thres=90):
    # Array indices where capacity < threshold
    ds_thres = (ds<thres).nonzero()
    
    # If non-empty, return number of years b/w 2000 and 2100 after ToE
    if len(ds_thres[0]) > 0:
        return min(100,150-ds_thres[0][0].item())
    
    # If empty, return 0
    return 0

def spatial_toe(ds,title,thres1=90,thres2=80,thres3=70):
    # Calculate ToE for different thresholds
    ds_90 = xr.apply_ufunc(emergence,ds,input_core_dims=[['time']],vectorize=True,kwargs={'thres':thres1})
    ds_80 = xr.apply_ufunc(emergence,ds,input_core_dims=[['time']],vectorize=True,kwargs={'thres':thres2})
    ds_70 = xr.apply_ufunc(emergence,ds,input_core_dims=[['time']],vectorize=True,kwargs={'thres':thres3})
    
    # Specify projection
    crs = ccrs.PlateCarree()

    # Create figure and axes
    fig, axs = plt.subplots(ncols=4,nrows=2,figsize=(22,8),subplot_kw={'projection':crs},gridspec_kw={'width_ratios': [0.5,3,3,3]})
    levels = [2000,2010,2020,2030,2040,2050,2060,2070,2080,2090,2099]

    # Plots of ToE: earliest among ensemble members
    contour(2100-ds_90.max(dim='ensemble'),'10% Reduction',axs[0][1],levels=levels,cmap ='Reds_r',label='Year')
    contour(2100-ds_80.max(dim='ensemble'),'20% Reduction',axs[0][2],levels=levels,cmap ='Reds_r',label='Year')
    contour(2100-ds_70.max(dim='ensemble'),'30% Reduction',axs[0][3],levels=levels,cmap ='Reds_r',label='Year')

    # Plots of ToE: mean among ensemble members
    contour(2100-ds_90.mean(dim='ensemble'),None,axs[1][1],levels=levels,cmap ='Reds_r',label='Year')
    contour(2100-ds_80.mean(dim='ensemble'),None,axs[1][2],levels=levels,cmap ='Reds_r',label='Year')
    im = contour(2100-ds_70.mean(dim='ensemble'),None,axs[1][3],levels=levels,cmap ='Reds_r',label='Year')

    # Annotating text
    axs[0][0].text(0.5,0.5,'Ensemble Earliest',fontsize=14,horizontalalignment='center',verticalalignment='center');
    axs[0][0].set_frame_on(False)
    axs[1][0].text(0.5,0.5,'Ensemble Mean',fontsize=14,horizontalalignment='center',verticalalignment='center');
    axs[1][0].set_frame_on(False)

    # Single colorbar for all plots
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.3, 0.07, 0.4, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax,orientation='horizontal');
    cbar.set_label('Year',fontsize=14)

    # Overall figure title
    fig.suptitle(title,fontsize=16);