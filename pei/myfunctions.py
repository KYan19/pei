import xarray as xr
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import numpy as np
from xhistogram.xarray import histogram

# Functions to be loaded into notebooks

# Path to data file
path = '/local/ss23/CESM2_LE/ATM/RCP85/WBT/b.e21.BSSP370cmip6.f09_g17.LE2-1231.001.clm2.h7.WBT.2095010100-2100123100.nc'

# Load data from matching file
ds = xr.open_dataset(path)
    
# Get longitude and latitude values from data
lon = ds['lon']
lat = ds['lat']

# Dictionary of region masks, represented by lon and lat arrays
masks = {}
masks['Global'] = [lon, lat]
masks['Northern North America'] = [lon.where((190<=lon)&(lon<=310),drop=True).values,lat.where((45<=lat)&(lat<=75),drop=True).values]
masks['Central North America'] = [lon.where((230<=lon)&(lon<=310),drop=True).values,lat.where((35<=lat)&(lat<=45),drop=True).values]
masks['Southern North America'] = [lon.where((230<=lon)&(lon<=300),drop=True).values,lat.where((20<=lat)&(lat<=35),drop=True).values]
masks['Central America'] = [lon.where((250<=lon)&(lon<=315),drop=True).values,lat.where((7<=lat)&(lat<=20),drop=True).values]
masks['Northern South America'] = [lon.where((270<=lon)&(lon<=330),drop=True).values,lat.where((-23.5<=lat)&(lat<=7),drop=True).values]
masks['Southern South America'] = [lon.where((270<=lon)&(lon<=330),drop=True).values,lat.where((-57<=lat)&(lat<=-23.5),drop=True).values]

masks['Scandinavia'] = [lon.where((3<=lon)&(lon<=30),drop=True).values,lat.where((55<=lat)&(lat<=70),drop=True).values]
lon_west = lon.where(lon>=345,drop=True)
lon_east = lon.where(lon<=35,drop=True)
lon_ceur = xr.concat((lon_west,lon_east),dim='lon').values
masks['Central Europe'] = [lon_ceur,lat.where((43<=lat)&(lat<=55),drop=True)]
lon_west = lon.where(lon>=350,drop=True)
lon_east = lon.where(lon<=24,drop=True)
lon_seur = xr.concat((lon_west,lon_east),dim='lon').values
masks['Southern Europe'] = [lon_seur,lat.where((36<=lat)&(lat<=43),drop=True)]

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
lon_cafrica = xr.concat((lon_west,lon_east),dim='lon').values
masks['Central Africa'] = [lon_cafrica,lat.where((-10<=lat)&(lat<=10),drop=True).values]
lon_west = lon.where(lon>=340,drop=True)
lon_east = lon.where(lon<=25,drop=True)
lon_nafrica = xr.concat((lon_west,lon_east),dim='lon').values
masks['Northern Africa'] = [lon_nafrica,lat.where((10<=lat)&(lat<=38),drop=True).values]
masks['Southern Africa'] = [lon.where((9<=lon)&(lon<=52),drop=True).values,lat.where((-35<=lat)&(lat<=-10),drop=True).values]

# Function to calculate area-weighted annual mean temperature for a region
def area_weighted(temp_data,area_data,region):
    # Get area, temperature data for region
    area = area_data.sel(lon=masks[region][0],lat=masks[region][1])
    temp = temp_data.sel(lon=masks[region][0],lat=masks[region][1])
    
    # Calculate area-weighted average
    area_mean = (temp*area).sum(['lon','lat'])/area.sum(['lon','lat'])
    return area_mean

# Function to caluclate annual average across ensemble members
def annual_average(ds, var):
    # Average across ensemble members
    mean_data = ds.mean(dim='ensemble')
    
    # Average by year
    mean_annual = mean_data[var].groupby('time.year').mean()
    return mean_annual

# Function to isolate data for a particular region
def slice_region(ds, region):
    ds_region = ds.sel(lon=masks[region][0],lat=masks[region][1])
    return ds_region

# Function to get names of available region masks
def get_regions():
    return masks.keys()

# Function to map area encompassed by a particular regional mask
def map_region(region):
    # Specify projection
    crs = ccrs.PlateCarree()
    
    fig, ax = plt.subplots(subplot_kw={'projection':crs})
    
    # Get longitude and latitude bounds for mask
    if region != 'Global':
        xmin = masks[region][0][0]
        if xmin > 180:
            xmin -= 360
        xmax = masks[region][0][-1]
        if xmax > 180:
            xmax -= 360
        ymin = masks[region][1][0]
        ymax = masks[region][1][-1]
        ax.set_extent([xmin,xmax,ymin,ymax],crs=crs)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    ax.add_feature(cfeature.OCEAN, color='skyblue')
    ax.add_feature(cfeature.LAND, color='lightgrey')
    
# Function to plot temperature using cartopy
def contour_plot(ds,region,title,cmap,borders=False,label='$^\circ\,K$'):
    # Specify projection
    crs = ccrs.PlateCarree()

    # Create figure and axes
    fig,ax = plt.subplots(figsize=(10,10),subplot_kw={'projection':crs})

    # Specify variables
    X = ds['lon']
    Y = ds['lat']
    Z = ds.squeeze()
    Z, X = add_cyclic_point(Z,coord=X)

    # Create contour plot of SST
    im = ax.contourf(X,Y,Z,levels=10,transform=crs,cmap=cmap)
    
    # Adjust longitude and latitude bounds for region
    if region != 'Global':
        xmin = masks[region][0][0]
        if xmin > 180:
            xmin -= 360
        xmax = masks[region][0][-1]
        if xmax > 180:
            xmax -= 360
        ymin = masks[region][1][0]
        ymax = masks[region][1][-1]
        ax.set_extent([xmin,xmax,ymin,ymax],crs=crs)

    # Add grid lines, coastlines
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.left_labels = False
    ax.coastlines()
    ax.add_feature(cfeature.OCEAN,zorder=10,facecolor='lightskyblue')
    
    # Add national boundaries
    if borders:
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),edgecolor='silver')

    # Set colorbar, title
    cbar = plt.colorbar(im,ax=ax,orientation='horizontal',fraction=0.05,pad=0.05)
    cbar.set_label(label,fontsize=12)
    plt.title(title)
    #plt.savefig(region+'freq.png')

# Function to generate a histogram for a data array
def hist(ds, area, region, ax, upper=40, lower=-20, numbins = 60, denom = 50):
    # Histogram bins
    bins = np.linspace(lower,upper,numbins)

    # Get WBT data for region
    ds_region = slice_region(ds,region)['WBT']
    
    # Get area weights for grid cells in region
    area_region = slice_region(area,region)
    area_weights = area_region/(area_region.sum(['lon','lat']))
    
    # Generate histogrma and get average distribution
    total_hist = histogram(ds_region,bins=[bins],weights=area_weights['land_area'],density=True,block_size=1)
    hist_avg = total_hist/denom
    hist_avg.plot(ax=ax)

# Function to plot two histograms on the same axes (comparing time periods)
def hist2(ds1, ds2, area, region):
    fig, ax = plt.subplots()
    hist(ds1,area,region,ax)
    hist(ds2,area,region,ax)
    ax.set_xlabel('WBT (Celsius)')
    ax.set_ylabel('Average Frequency (Days Per Year)')
    ax.set_title(region + ': Annual Distributions of Maximum Daily WBT')
    ax.legend(['1980-1990','2085-2095'], loc='upper left')
    fig.savefig(region+'WBTmax.png')
    
# Function to plot histograms for all ensemble members on the same axes
def ensemble_hist(ds,area,region,upper=40,lower=-20,numbins=60):
    fig, ax = plt.subplots()
    hist(ds,area,region,ax,lower=lower,numbins=numbins)
    for i in range(0,5):
        ds_ens = ds_2085_max.isel(ensemble=i)
        hist(ds_ens,area.isel(ensemble=0),region,ax,denom=10,lower=lower,numbins=numbins)
    ax.set_xlabel('WBT (Celsius)')
    ax.set_ylabel('Average Frequency (Days Per Year)')
    ax.set_title(region+': Ensemble Distributions of Max Daily WBT, 2085-95')
    ax.legend(['Average','Member 1', 'Member 2', 'Member 3', 'Member 4', 'Member 5'], loc='upper left') 
    fig.savefig(region+'_ensemble_hist.png')