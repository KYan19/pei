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
masks['South-Central America'] = [lon.where((230<=lon)&(lon<=330),drop=True).values,lat.where((-30<=lat)&(lat<=35),drop=True).values]
masks['Southern South America'] = [lon.where((270<=lon)&(lon<=330),drop=True).values,lat.where((-60<=lat)&(lat<=-30),drop=True).values]
masks['Central America'] = [lon.where((250<=lon)&(lon<=315),drop=True).values,lat.where((7<=lat)&(lat<=20),drop=True).values]
masks['Northern South America'] = [lon.where((270<=lon)&(lon<=330),drop=True).values,lat.where((-23.5<=lat)&(lat<=10),drop=True).values]
masks['China'] = [lon.where((75<=lon)&(lon<=135),drop=True).values,lat.where((22.5<=lat)&(lat<=50),drop=True).values]
masks['Southern China'] = [lon.where((98<=lon)&(lon<=125),drop=True).values,lat.where((22<=lat)&(lat<=32),drop=True).values]
masks['India'] = [lon.where((68<=lon)&(lon<=90),drop=True).values,lat.where((8<=lat)&(lat<=30),drop=True).values]
masks['Oceania'] = [lon.where((100<=lon)&(lon<=180),drop=True).values,lat.where((-50<=lat)&(lat<=0),drop=True).values]
masks['Northern Oceania'] = [lon.where((100<=lon)&(lon<=180),drop=True).values,lat.where((-23.5<=lat)&(lat<=0),drop=True).values]
masks['Southern Oceania'] = [lon.where((100<=lon)&(lon<=180),drop=True).values,lat.where((-50<=lat)&(lat<=-23.5),drop=True).values]
masks['Russia'] = [lon.where((30<=lon)&(lon<=180),drop=True).values,lat.where((50<=lat)&(lat<=75),drop=True).values]
masks['Scandinavia'] = [lon.where((3<=lon)&(lon<=30),drop=True).values,lat.where((55<=lat)&(lat<=70),drop=True).values]
lon_west = lon.where(lon>=345,drop=True)
lon_east = lon.where(lon<=30,drop=True)
lon_eur = xr.concat((lon_west,lon_east),dim='lon').values
masks['Europe'] = [lon_eur,lat.where((35<=lat)&(lat<=55),drop=True)]
lon_west = lon.where(lon>=350,drop=True)
lon_east = lon.where(lon<=24,drop=True)
lon_seur = xr.concat((lon_west,lon_east),dim='lon').values
masks['Southern Europe'] = [lon_seur,lat.where((36<=lat)&(lat<=44),drop=True)]
lon_west = lon.where(lon>=355,drop=True)
lon_east = lon.where(lon<=10,drop=True)
lon_france = xr.concat((lon_west,lon_east),dim='lon').values
masks['France'] = [lon_france,lat.where((40<=lat)&(lat<=52),drop=True)]
masks['Middle East'] = [lon.where((25<=lon)&(lon<=60),drop=True).values,lat.where((10<=lat)&(lat<=40),drop=True).values]
masks['Southeast Asia'] = [lon.where((92<=lon)&(lon<=140),drop=True).values,lat.where((-10<=lat)&(lat<=25),drop=True).values]
lon_west = lon.where(lon>=340,drop=True)
lon_east = lon.where(lon<=55,drop=True)
lon_safrica = xr.concat((lon_west,lon_east),dim='lon').values
masks['Southern Africa'] = [lon_safrica,lat.where((-40<=lat)&(lat<=10),drop=True).values]
masks['Central Africa'] = [lon_safrica,lat.where((-10<=lat)&(lat<=10),drop=True).values]

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
def contour_plot(ds,region,title,cmap,borders=False):
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
    cbar.set_label('$^\circ\,K$',fontsize=12)
    plt.title(title)

def hist(ds, area, region, ax, upper=40, lower=-20, numbins = 60, denom = 50, alpha=1):
    bins = np.linspace(lower,upper,numbins)

    ds_region = slice_region(ds,region)['WBT']
    
    area_region = slice_region(area,region)
    area_weights = area_region/(area_region.sum(['lon','lat']))
    
    total_hist = histogram(ds_region,bins=[bins],weights=area_weights['land_area'],density=True,block_size=1)
    hist_avg = total_hist/denom
    hist_avg.plot(ax=ax, alpha=alpha)