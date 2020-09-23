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
    #gl = ax.gridlines(draw_labels=True)
    #gl.top_labels = False
    #gl.left_labels = False
    ax.coastlines()
    ax.add_feature(cfeature.OCEAN,zorder=10,facecolor='lightskyblue')
    
    # Add national boundaries
    if borders:
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),edgecolor='silver')

    # Set colorbar, title
    cbar = plt.colorbar(im,ax=ax,orientation='horizontal',fraction=0.05,pad=0.05)
    cbar.set_label(label,fontsize=12)
    plt.title(title)

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
    
# OLD EMERGENCE FUNCTIONS -- FOR ANNUAL LABOR CAPACITY; SINCE REPLACED BY MONTHLY
def capacity(ds,ds_pop,region,model,ax,color='royalblue'):
    '''Function to plot capacity over time for a particular region
        Ensemble members + ensemble average'''
    # Get yearly capacity data for grid cells in region
    ds_region = slice_region(ds,region,model)
    
    # Get population for grid cells
    pop_region = slice_region(ds_pop,region,model)
    
    # Calculate total area-weighted capacity per year
    capacity = ds_region.weighted(pop_region).mean(['lon','lat'])
    
    # Plot individual ensemble members
    capacity.plot.line(hue='ensemble',color=color,alpha=0.25,ax=ax,add_legend=False)

    # Ensemble average labor capacity
    capacity_avg = capacity.mean(dim='ensemble')
    capacity_avg.plot(ax=ax,color=color,linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Labor Capacity, %')
    ax.set_title(region)
    
def emergence(ds,start_year,thres):
    '''Function finds first year with labor capacity < threshold'''
    # Array indices where capacity < threshold
    ds_thres = (ds<thres).nonzero()
    
    # If non-empty, index + startyear = ToE
    if len(ds_thres[0]) > 0:
        return start_year+(ds_thres[0][0].item())
    
    # If empty, return year after 2100
    return 2101

def thres_years(ds,thres=90,length=150):
    '''Function finds number of years in 21st century after ToE'''
    # Array indices where capacity < threshold
    ds_thres = (ds<thres).nonzero()
    
    # If non-empty, return number of years b/w 2000 and 2100 after ToE
    # Length refers to number of years in dataset's time dimension
    if len(ds_thres[0]) > 0:
        return min(100,length-ds_thres[0][0].item())
    
    # If empty, return 0
    return 0

def spatial_toe(ds,title,thres1=90,thres2=80,thres3=70):
    '''Plot spatial map of ToE for all grid cells (global)'''
    # Calculate ToE for different thresholds
    start_year = ds['time.year'][0].item()
    ds_90 = xr.apply_ufunc(emergence,ds,input_core_dims=[['time']],vectorize=True,kwargs={'thres':thres1,'start_year':start_year})
    ds_80 = xr.apply_ufunc(emergence,ds,input_core_dims=[['time']],vectorize=True,kwargs={'thres':thres2,'start_year':start_year})
    ds_70 = xr.apply_ufunc(emergence,ds,input_core_dims=[['time']],vectorize=True,kwargs={'thres':thres3,'start_year':start_year})
    
    # Specify projection
    crs = ccrs.PlateCarree()

    # Create figure and axes
    fig, axs = plt.subplots(ncols=4,nrows=2,figsize=(22,8),subplot_kw={'projection':crs},gridspec_kw={'width_ratios': [0.5,3,3,3]})
    levels = np.linspace(2000,2100,21)
    cmap = 'magma'

    # Plots of ToE: earliest among ensemble members
    im = contour(ds_90.min(dim='ensemble'),'10% Reduction',axs[0][1],levels=levels,cmap =cmap,label='Year')
    contour(ds_80.min(dim='ensemble'),'20% Reduction',axs[0][2],levels=levels,cmap =cmap,label='Year')
    contour(ds_70.min(dim='ensemble'),'30% Reduction',axs[0][3],levels=levels,cmap =cmap,label='Year')

    # Plots of ToE: mean among ensemble members
    contour(ds_90.mean(dim='ensemble'),None,axs[1][1],levels=levels,cmap =cmap,label='Year')
    scatter(ds_90,axs[1][1])
    contour(ds_80.mean(dim='ensemble'),None,axs[1][2],levels=levels,cmap =cmap,label='Year')
    scatter(ds_80,axs[1][2])
    contour(ds_70.mean(dim='ensemble'),None,axs[1][3],levels=levels,cmap =cmap,label='Year')
    scatter(ds_70,axs[1][3])

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
    
def spatial_toe_diff(ds,title,thres1=90,thres2=80,thres3=70):
    '''Plot ToE range for all grid cells (global)'''
    start_year = ds['time.year'][0].item()
    # Calculate ToE for different thresholds
    ds_90 = xr.apply_ufunc(emergence,ds,input_core_dims=[['time']],vectorize=True,kwargs={'thres':thres1,'start_year':start_year})
    ds_80 = xr.apply_ufunc(emergence,ds,input_core_dims=[['time']],vectorize=True,kwargs={'thres':thres2,'start_year':start_year})
    ds_70 = xr.apply_ufunc(emergence,ds,input_core_dims=[['time']],vectorize=True,kwargs={'thres':thres3,'start_year':start_year})
    
    # Specify projection
    crs = ccrs.PlateCarree()

    # Create figure and axes
    fig, axs = plt.subplots(ncols=3,figsize=(20,5),subplot_kw={'projection':crs})
    levels = np.linspace(1,60,60)
    cmap = 'YlOrBr'

    # Plots of ToE range: max ToE - min ToE
    im = contour(ds_90.max(dim='ensemble')-ds_90.min(dim='ensemble'),'10% Reduction',axs[0],levels=levels,cmap =cmap,label='Year',under='white',over=None)
    contour(ds_80.max(dim='ensemble')-ds_80.min(dim='ensemble'),'20% Reduction',axs[1],levels=levels,cmap =cmap,label='Year',under='white',over=None)
    contour(ds_70.max(dim='ensemble')-ds_70.min(dim='ensemble'),'30% Reduction',axs[2],levels=levels,cmap =cmap,label='Year',under='white',over=None)

    # Single colorbar for all plots
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.3, 0.15, 0.4, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax,orientation='horizontal');
    cbar.set_label('Years',fontsize=14)

    # Overall figure title
    fig.suptitle(title,fontsize=16);
    
def average_toe_bar(ds,ds_pop,model,title):
    '''Bar graph showing portion of 21st century spent after ToE
        Uses average ToE across grid cells in regions'''
    length = len(ds['time.year'].values) - 1
    # Calculate # of >ToE years for different thresholds
    ds_90 = xr.apply_ufunc(thres_years,ds,input_core_dims=[['time']],vectorize=True,kwargs={'thres':90,'length':length})
    ds_80 = xr.apply_ufunc(thres_years,ds,input_core_dims=[['time']],vectorize=True,kwargs={'thres':80,'length':length})
    ds_70 = xr.apply_ufunc(thres_years,ds,input_core_dims=[['time']],vectorize=True,kwargs={'thres':70,'length':length})
    
    # Regions to plot
    regions = ['India','Central America','Northern South America','Southeast Asia','Central Africa','Northern Oceania']

    fig, ax = plt.subplots(figsize=(20,10))

    # Location of region labels; width of bars
    locs = np.arange(len(regions))
    width = 0.2

    for x in locs:
        region = regions[x]
        # Get number of >ToE years for grid cells in region
        ds_90_region = slice_region(ds_90,region,model)
        ds_80_region = slice_region(ds_80,region,model)
        ds_70_region = slice_region(ds_70,region,model)
        pop_region = slice_region(ds_pop,region,model)

        # Population-weighted average number of >ToE years
        years_90 = ds_90_region.weighted(pop_region).mean(['lon','lat'])
        years_80 = ds_80_region.weighted(pop_region).mean(['lon','lat'])
        years_70 = ds_70_region.weighted(pop_region).mean(['lon','lat'])

        # [[Lower range],[upper range]] -- range of values across ensemble members
        errors_90 = [[years_90.max('ensemble')-years_90.mean('ensemble')],[years_90.mean('ensemble')-years_90.min('ensemble')]]
        ax.bar(x - width, -years_90.mean('ensemble'), width, label='10%', bottom=2100, yerr=errors_90, error_kw={'capsize':5}, color='royalblue')
        errors_80 = [[years_80.max('ensemble')-years_80.mean('ensemble')],[years_80.mean('ensemble')-years_80.min('ensemble')]]
        ax.bar(x, -years_80.mean('ensemble'), width, label='20%', bottom=2100, yerr=errors_80, error_kw={'capsize':5}, color='green')
        errors_70 = [[years_70.max('ensemble')-years_70.mean('ensemble')],[years_70.mean('ensemble')-years_70.min('ensemble')]]
        ax.bar(x + width, -years_70.mean('ensemble'), width, label='30%',bottom=2100, yerr=errors_70, error_kw={'capsize':5}, color='orange')

    # Region labels on top
    ax.xaxis.set_tick_params(labeltop=True,labelbottom=False,bottom=False)
    ax.set_xticks(locs)
    ax.set_xticklabels(regions)
    ax.set_ylim([2000,2100]);

    # Legend for different thresholds
    blue_patch = mpatches.Patch(color='royalblue', label='10% reduction')
    green_patch = mpatches.Patch(color='green', label='20% reduction')
    orange_patch = mpatches.Patch(color='orange', label='30% reduction')
    ax.legend(handles=[blue_patch,green_patch,orange_patch], loc=(0.25,0.01));
    ax.set_title(title,fontsize=14);
    
def toe_bar(ds,ds_pop,model,title):
    '''Bar graph showing portion of 21st century spent after ToE
        Uses ToE for average labor capacity across grid cells in regions'''
    
    # Regions to plot
    regions = ['India','Central America','Northern South America','Southeast Asia','Central Africa','Northern Oceania']

    fig, ax = plt.subplots(figsize=(20,10))

    # Location of region labels; width of bars
    locs = np.arange(len(regions))
    width = 0.2

    for x in locs:
        region = regions[x]

        # Population-weighted regional labor capacity
        ds_region = slice_region(ds,region,model)
        pop_region = slice_region(ds_pop,region,model)
        capacity = ds_region.weighted(pop_region).mean(['lat','lon'])
        
        length = len(ds['time.year'].values) - 1
        # Number of >ToE years using regional average
        years_90 = xr.apply_ufunc(thres_years,capacity['capacity'],input_core_dims=[['time']],vectorize=True,kwargs={'thres':90,'length':length})
        years_80 = xr.apply_ufunc(thres_years,capacity['capacity'],input_core_dims=[['time']],vectorize=True,kwargs={'thres':80,'length':length})
        years_70 = xr.apply_ufunc(thres_years,capacity['capacity'],input_core_dims=[['time']],vectorize=True,kwargs={'thres':70,'length':length})

        # [[Lower range],[upper range]] -- range of values across ensemble members
        errors_90 = [[years_90.max('ensemble')-years_90.mean('ensemble')],[years_90.mean('ensemble')-years_90.min('ensemble')]]
        ax.bar(x - width, -years_90.mean('ensemble'), width, label='10%', bottom=2100, yerr=errors_90, error_kw={'capsize':5}, color='royalblue')
        errors_80 = [[years_80.max('ensemble')-years_80.mean('ensemble')],[years_80.mean('ensemble')-years_80.min('ensemble')]]
        ax.bar(x, -years_80.mean('ensemble'), width, label='20%', bottom=2100, yerr=errors_80, error_kw={'capsize':5}, color='green')
        errors_70 = [[years_70.max('ensemble')-years_70.mean('ensemble')],[years_70.mean('ensemble')-years_70.min('ensemble')]]
        ax.bar(x + width, -years_70.mean('ensemble'), width, label='30%',bottom=2100, yerr=errors_70, error_kw={'capsize':5}, color='orange')

    # Region labels on top
    ax.xaxis.set_tick_params(labeltop=True,labelbottom=False,bottom=False)
    ax.set_xticks(locs)
    ax.set_xticklabels(regions)
    ax.set_ylim([2000,2100]);

    # Legend for different thresholds
    blue_patch = mpatches.Patch(color='royalblue', label='10% reduction')
    green_patch = mpatches.Patch(color='green', label='20% reduction')
    orange_patch = mpatches.Patch(color='orange', label='30% reduction')
    ax.legend(handles=[blue_patch,green_patch,orange_patch], loc=(0.42,0.01));
    ax.set_title(title,fontsize=14);
    
def area_emerge(ds,ds_area,thres,start_year):
    '''Find area that has emerged'''

    # Get ToE for all grid cells
    ds_toe = xr.apply_ufunc(emergence,ds,input_core_dims=[['time']],vectorize=True,kwargs={'thres':thres,'start_year':start_year})
    
    # DataArray of years
    year = xr.ones_like(ds)*ds['time.year']
    
    # Is ToE <= current year? Sum up area for grid cells marked as True
    ds_emerged = ((ds_toe<=year)*ds_area).sum(['lat','lon'])
    
    # Return area emerged
    return ds_emerged
    
def area_emerge_plot(ds,ds_area,title,ylabel,ax):
    '''Plot time series for area emerged'''
    # Calculate total area
    total_area = ds_area.sum(['lat','lon'])
    
    start_year = ds['time.year'][0].item()
    # Calculate fraction of area that has emerged
    ds_90 = area_emerge(ds,ds_area,90,start_year)/total_area
    ds_80 = area_emerge(ds,ds_area,80,start_year)/total_area
    ds_70 = area_emerge(ds,ds_area,70,start_year)/total_area
    
    # Plot ensemble members + ensemble average for 90% threshold
    ds_90.plot.line(hue='ensemble',ax=ax,color='royalblue',alpha=0.25,add_legend=False)
    ds_90.mean(dim='ensemble').plot(ax=ax,color='royalblue')
    
    # Plot ensemble members + ensemble average for 80% threshold
    ds_80.plot.line(hue='ensemble',ax=ax,color='green',alpha=0.25,add_legend=False)
    ds_80.mean(dim='ensemble').plot(ax=ax,color='green')
    
    # Plot ensemble members + ensemble average for 70% threshold
    ds_70.plot.line(hue='ensemble',ax=ax,color='orange',alpha=0.25,add_legend=False)
    ds_70.mean(dim='ensemble').plot(ax=ax,color='orange')
    
    # Set labels, limits
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.set_ylim([0,1])
    ax.set_title(title)
    
    # Set legend
    blue_line = mlines.Line2D([], [], color='royalblue', label='10% reduction')
    green_line = mlines.Line2D([], [], color='green', label='20% reduction')
    orange_line = mlines.Line2D([], [], color='orange', label='30% reduction')
    ax.legend(handles=[blue_line,green_line,orange_line]);

def spatial_ensemble_esm2m(ds,title):
    # Specify projection
    crs = ccrs.PlateCarree()

    # Create figure and axes
    fig, axs = plt.subplots(ncols=3,figsize=(20,4),subplot_kw={'projection':crs})
    levels = np.linspace(1950,2100,31)

    cmap = 'magma'
    im = contour(ds.isel(ensemble=0),'Member 0',axs[0],levels=levels,cmap =cmap,label='Year')
    contour(ds.isel(ensemble=1),'Member 1',axs[1],levels=levels,cmap =cmap,label='Year')
    contour(ds.isel(ensemble=2),'Member 2',axs[2],levels=levels,cmap =cmap,label='Year')

    # Single colorbar for all plots
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.3, 0.07, 0.4, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax,orientation='horizontal');
    cbar.set_label('Year',fontsize=14)
    
    fig.suptitle(title,fontsize=16)

def spatial_ensemble_cesm2(ds,title):
    # Specify projection
    crs = ccrs.PlateCarree()

    # Create figure and axes
    fig, axs = plt.subplots(ncols=3,nrows=2,figsize=(20,8),subplot_kw={'projection':crs})
    levels = np.linspace(1980,2100,25)

    cmap = 'magma'
    im = contour(ds.isel(ensemble=0),'Member 0',axs[0][0],levels=levels,cmap =cmap,label='Year')
    contour(ds.isel(ensemble=1),'Member 1',axs[0][1],levels=levels,cmap =cmap,label='Year')
    contour(ds.isel(ensemble=2),'Member 2',axs[0][2],levels=levels,cmap =cmap,label='Year')
    contour(ds.isel(ensemble=2),'Member 3',axs[1][0],levels=levels,cmap =cmap,label='Year')
    contour(ds.isel(ensemble=2),'Member 4',axs[1][1],levels=levels,cmap =cmap,label='Year')

    # Single colorbar for all plots
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.3, 0.07, 0.4, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax,orientation='horizontal');
    cbar.set_label('Year',fontsize=14)
    
    fig.suptitle(title,fontsize=16)

def contour_plot(ds,title,levels=10,cmap='Reds',label='Labor Capacity, %'):
    '''Function to create a contour plot of labor capacity (new axis)'''
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
    im = ax.contourf(X,Y,Z,levels=levels,transform=crs,cmap=cmap)

    # Add coastlines and ocean mask
    ax.coastlines()
    ax.add_feature(cfeature.OCEAN,zorder=10,facecolor='lightskyblue')
    
    # Add national boundaries
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),edgecolor='silver')

    # Set colorbar, title
    cbar = plt.colorbar(im,ax=ax,orientation='horizontal',fraction=0.05,pad=0.05)
    cbar.set_label(label,fontsize=12)
    plt.title(title)

def toe_bar(ds,ds_base,ds_pop,model,title,labor_thres,freq_thres):
    '''Bar graph showing portion of 21st century spent after ToE
        Uses ToE for average labor capacity across grid cells in regions'''
    
    # Regions to plot
    regions = ['India','Central America','Northern South America','Southeast Asia','Central Africa','Northern Oceania']

    fig, ax = plt.subplots(figsize=(20,10))

    # Location of region labels; width of bars
    locs = np.arange(len(regions))
    width = 0.2

    for x in locs:
        region = regions[x]

        # Population-weighted regional labor capacity
        ds_region = slice_region(ds,region,model)
        capacity = ds_region.weighted(ds_pop).mean(['lat','lon'])
        
        # Population-weighted regional baseline
        base_region = slice_region(ds_base,region,model)
        base = base_region.weighted(ds_pop).mean(['lat','lon'])
        
        # ToEs using regional average
        ds_toe = toe(capacity,base,labor_thres,freq_thres)
        
        # Calculate # of >ToE years for different thresholds
        years_90 = 2100-ds_toe['0.9']
        years_90 = years_90.where(years_90>=0,0)
        years_80 = 2100-ds_toe['0.8']
        years_80 = years_80.where(years_80>=0,0)
        years_70 = 2100-ds_toe['0.7']
        years_70 = years_70.where(years_70>=0,0)

        # [[Lower range],[upper range]] -- range of values across ensemble members
        errors_90 = [[years_90.max('ensemble')-years_90.mean('ensemble')],[years_90.mean('ensemble')-years_90.min('ensemble')]]
        ax.bar(x - width, -years_90.mean('ensemble'), width, label='10%', bottom=2100, yerr=errors_90, error_kw={'capsize':5}, color='royalblue')
        errors_80 = [[years_80.max('ensemble')-years_80.mean('ensemble')],[years_80.mean('ensemble')-years_80.min('ensemble')]]
        ax.bar(x, -years_80.mean('ensemble'), width, label='20%', bottom=2100, yerr=errors_80, error_kw={'capsize':5}, color='green')
        errors_70 = [[years_70.max('ensemble')-years_70.mean('ensemble')],[years_70.mean('ensemble')-years_70.min('ensemble')]]
        ax.bar(x + width, -years_70.mean('ensemble'), width, label='30%',bottom=2100, yerr=errors_70, error_kw={'capsize':5}, color='orange')

    # Region labels on top
    ax.xaxis.set_tick_params(labeltop=True,labelbottom=False,bottom=False)
    ax.set_xticks(locs)
    ax.set_xticklabels(regions)
    ax.set_ylim([2000,2100]);

    # Legend for different thresholds
    blue_patch = mpatches.Patch(color='royalblue', label='10% reduction')
    green_patch = mpatches.Patch(color='green', label='20% reduction')
    orange_patch = mpatches.Patch(color='orange', label='30% reduction')
    ax.legend(handles=[blue_patch,green_patch,orange_patch], loc='lower right');
    ax.set_title(title,fontsize=14);
    
# OLD EMERGENCE FUNCTIONS -- BEFORE IMPLEMENTATION OF SUMMERTIME MONTHS
def emergence(ds,start_year):
    '''Function finds first year with labor capacity < threshold'''
    # Array indices where capacity < threshold
    ds_thres = ds.nonzero()
    
    # If non-empty, index + startyear = ToE
    if len(ds_thres[0]) > 0:
        return start_year+ds_thres[0][0].item()
    
    # If empty, return year after 2100
    return 2101

def toe(ds,ds_base,labor_thres):
    '''Return dataset of ToEs based on various inputted thresholds (any 3 months below threshold)'''
    # First year of the dataset
    start_year = ds['time.year'][0].item()

    # Dataset for ToEs
    ds_toe = xr.Dataset()

    # Loop through inputted thresholds
    for thres in labor_thres:
        # See if each month's capacity is below threshold
        ds_thres = ds < (thres*ds_base.sel(month=ds['time.month']))
        
        # See if enough months in each year are below threshold
        ds_thres = ds_thres.groupby('time.year').sum() >= 3
        
        # Get first year with enough months below threshold
        ds_toe[str(thres)] = xr.apply_ufunc(emergence,ds_thres,input_core_dims=[['year']],vectorize=True,dask='allowed',kwargs={'start_year':start_year})
    return ds_toe
