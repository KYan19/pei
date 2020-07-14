# Functions to use while working with labor capacity

import xarray as xr
import matplotlib as mpl
from matplotlib import pyplot as plt
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

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
    masks['Northern India'] = [lon.where((68<=lon)&(lon<=90),drop=True).values,lat.where((23<=lat)&(lat<=30),drop=True).values]
    masks['Southern India'] = [lon.where((68<=lon)&(lon<=90),drop=True).values,lat.where((8<=lat)&(lat<=23),drop=True).values]
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

def slice_region(ds, region, model):
    '''Function to isolate data for a particular region'''
    if model == 'GFDL':
        return ds.sel(lon=masks_GFDL[region][0],lat=masks_GFDL[region][1])
    elif model == 'CESM2':
        return ds.sel(lon=masks_CESM2[region][0],lat=masks_CESM2[region][1])
    else:
        raise ValueError('Model name not valid')

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
    
def contour(ds,title,ax,levels,cmap='magma',label='Labor Capacity, %',under=None,over='darkgray',extend='both'):
    '''Function to create a contour plot of labor capacity (axis as parameter)'''
    # Specify projection
    crs = ccrs.PlateCarree()
    
    # Specify variables
    X = ds['lon']
    Y = ds['lat']
    Z = ds.squeeze()
    Z, X = add_cyclic_point(Z,coord=X)

    # In case cmap default over/under colors need to be reset
    colormap = plt.cm.get_cmap(cmap)
    N = colormap.N

    # Create contour plot
    im = ax.contourf(X,Y,Z,levels=levels,transform=crs,cmap=cmap,extend=extend)
    
    # Set over/under colors for cmap
    if over == None:
        colormap.set_over(colormap(N-1))
    else:
        im.cmap.set_over(over)
        
    if under == None:
        colormap.set_under(colormap(1))
    else:
        im.cmap.set_under(under)

    # Add coastlines and ocean mask
    ax.coastlines()
    ax.add_feature(cfeature.OCEAN,zorder=10,facecolor='lightskyblue')
    
    # Add national boundaries
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),edgecolor='silver')

    # Set title
    ax.set_title(title,fontsize=14)
    
    return im

def scatter(ds,ax):
    '''Place markers over grid cells that emerge in some ensemble members but not all'''
    # Number of ensemble members
    ens_num = len(ds['ensemble'].values)
    
    # Number of emergences per grid cell
    ds_gray = (ds>2100).sum(dim='ensemble').stack(xy=('lon','lat'))
    
    # Is number of emergences a non-zero patrial of number of ensemble members?
    # Keep grid cells where this is true
    ds_cusp = ds_gray.where((ds_gray<ens_num)&(ds_gray>0),drop=True)
    
    # Get x,y for these grid cells
    X = [x[0] for x in ds_cusp['xy'].values]
    Y = [x[1] for x in ds_cusp['xy'].values]
    
    # Mark grid cells
    crs = ccrs.PlateCarree()
    ax.scatter(X,Y,transform=crs,zorder=1,marker='o',s=0.1,facecolors='none', edgecolors='black',alpha=0.75)
    
def calc_baseline(ds):
    '''Calculates 1980-2000 baseline capacity, by month (mean - 2*std)'''
    # Slice 1980-2000 data
    ds_hist  = ds.sel(time=slice('1980-01-31','1999-12-31')).groupby('time.month')

    # Calculate mean and stdev
    ds_hist_mean = ds_hist.mean(['time','ensemble'])
    ds_dev = 2*ds_hist.std(['time','ensemble'])
    
    # Return baseline as the lower bound of "envelope" around mean 
    ds_base = ds_hist_mean - ds_dev
    return ds_base['capacity']

def emergence(ds,start_year):
    '''Function finds first year with labor capacity < threshold'''
    # Array indices where capacity < threshold
    ds_thres = ds.nonzero()
    
    # If non-empty, index + startyear = ToE
    if len(ds_thres[0]) > 0:
        return start_year+ds_thres[0][0].item()
    
    # If empty, return year after 2100
    return 2101

def emergence2(ds,start_year):
    '''Function finds first year with labor capacity < threshold'''
    # Array indices where capacity < threshold
    ds_thres = ds.nonzero()
    
    # If non-empty, index + startyear = ToE
    if len(ds_thres[0]) > 0:
        return start_year+int(ds_thres[0][0].item()/4)
    
    # If empty, return year after 2100
    return 2101

def toe2(ds,ds_base,labor_thres):
    '''Return dataset of ToEs based on various inputted thresholds'''
    # First year of the dataset
    start_year = ds['time.year'][0].item()

    # Dataset for ToEs
    ds_toe = xr.Dataset()

    # Loop through inputted thresholds
    for thres in labor_thres:
        # See if each month's capacity is below threshold
        ds_thres = ds < (thres*ds_base.sel(month=ds['time.month']))
        
        # Sum by (freq_thres)-month chunks
        ds_chunk = ds_thres.resample(time='Q-NOV').sum()
        
        # Find occurrences of (freq-thres) consecutive hot months
        ds_chunk = (ds_chunk == 3)
        
        # Get first year with enough months below threshold
        ds_toe[str(thres)] = xr.apply_ufunc(emergence2,ds_chunk,input_core_dims=[['time']],vectorize=True,dask='allowed',kwargs={'start_year':start_year})
    return ds_toe

def toe(ds,ds_base,labor_thres,freq_thres):
    '''Return dataset of ToEs based on various inputted thresholds'''
    # First year of the dataset
    start_year = ds['time.year'][0].item()

    # Dataset for ToEs
    ds_toe = xr.Dataset()

    # Loop through inputted thresholds
    for thres in labor_thres:
        # See if each month's capacity is below threshold
        ds_thres = ds < (thres*ds_base.sel(month=ds['time.month']))
        
        # See if enough months in each year are below threshold
        ds_thres = ds_thres.groupby('time.year').sum() >= freq_thres
        
        # Get first year with enough months below threshold
        ds_toe[str(thres)] = xr.apply_ufunc(emergence,ds_thres,input_core_dims=[['year']],vectorize=True,dask='allowed',kwargs={'start_year':start_year})
    return ds_toe

def spatial_toe(ds,title):
    '''Plot spatial map of ToE for all grid cells (global)'''
    # Specify projection
    crs = ccrs.PlateCarree()

    # Create figure and axes
    fig, axs = plt.subplots(ncols=4,nrows=2,figsize=(22,8),subplot_kw={'projection':crs},gridspec_kw={'width_ratios': [0.5,3,3,3]})
    levels = np.linspace(2000,2100,21)
    cmap = 'magma'

    # Plots of ToE: earliest among ensemble members
    im = contour(ds['0.9'].min(dim='ensemble'),'10% Reduction',axs[0][1],levels=levels,cmap =cmap,label='Year',extend='max')
    contour(ds['0.8'].min(dim='ensemble'),'20% Reduction',axs[0][2],levels=levels,cmap =cmap,label='Year',extend='max')
    contour(ds['0.7'].min(dim='ensemble'),'30% Reduction',axs[0][3],levels=levels,cmap =cmap,label='Year',extend='max')

    # Plots of ToE: mean among ensemble members
    contour(ds['0.9'].mean(dim='ensemble'),None,axs[1][1],levels=levels,cmap=cmap,label='Year',extend='max')
    contour(ds['0.8'].mean(dim='ensemble'),None,axs[1][2],levels=levels,cmap=cmap,label='Year',extend='max')
    contour(ds['0.7'].mean(dim='ensemble'),None,axs[1][3],levels=levels,cmap=cmap,label='Year',extend='max')

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
    
def spatial_toe_diff(ds,title):
    '''Plot ToE range for all grid cells (global)'''
    # Specify projection
    crs = ccrs.PlateCarree()

    # Create figure and axes
    fig, axs = plt.subplots(ncols=3,figsize=(20,5),subplot_kw={'projection':crs})
    levels = np.linspace(1,60,60)
    cmap = 'YlOrBr'

    # Plots of ToE range: max ToE - min ToE
    im = contour(ds['0.9'].max(dim='ensemble')-ds['0.9'].min(dim='ensemble'),'10% Reduction',axs[0],levels=levels,cmap=cmap,label='Year',under='white',over=None)
    scatter(ds['0.9'],axs[0])
    contour(ds['0.8'].max(dim='ensemble')-ds['0.8'].min(dim='ensemble'),'20% Reduction',axs[1],levels=levels,cmap=cmap,label='Year',under='white',over=None)
    scatter(ds['0.8'],axs[1])
    contour(ds['0.7'].max(dim='ensemble')-ds['0.7'].min(dim='ensemble'),'30% Reduction',axs[2],levels=levels,cmap=cmap,label='Year',under='white',over=None)
    scatter(ds['0.7'],axs[2])

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
    # Calculate # of >ToE years for different thresholds
    ds_90 = 2100-ds['0.9']
    ds_90 = ds_90.where(ds_90>=0,0)
    ds_80 = 2100-ds['0.8']
    ds_80 = ds_80.where(ds_80>=0,0)
    ds_70 = 2100-ds['0.7']
    ds_70 = ds_70.where(ds_70>=0,0)
    
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

        # Population-weighted average number of >ToE years
        years_90 = ds_90_region.weighted(ds_pop).mean(['lon','lat'])
        years_80 = ds_80_region.weighted(ds_pop).mean(['lon','lat'])
        years_70 = ds_70_region.weighted(ds_pop).mean(['lon','lat'])

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

def frac_emerge(ds_toe,ds_area):
    '''Find area that has emerged'''
    # DataArray of years
    years = range(2000,2101)
    year = ds_toe.expand_dims({'year':years})['year']
    
    # Is ToE <= current year? Sum up area for grid cells marked as True
    ds_emerged = ((ds_toe<=year)*ds_area).sum(['lat','lon'])
    
    # Return area emerged
    return ds_emerged

def frac_emerge_plot(ds,ds_area,title,ylabel,ax):
    '''Plot time series for area/population emerged'''
    # Calculate total area
    total_area = ds_area.sum(['lat','lon'])
    
    # Calculate fraction of area that has emerged
    ds_frac = frac_emerge(ds,ds_area)/total_area
    
    # Plot ensemble members + ensemble average for 90% threshold
    ds_frac['0.9'].plot.line(hue='ensemble',ax=ax,color='royalblue',alpha=0.25,add_legend=False)
    ds_frac['0.9'].mean(dim='ensemble').plot(ax=ax,color='royalblue')
    
    # Plot ensemble members + ensemble average for 80% threshold
    ds_frac['0.8'].plot.line(hue='ensemble',ax=ax,color='green',alpha=0.25,add_legend=False)
    ds_frac['0.8'].mean(dim='ensemble').plot(ax=ax,color='green')
    
    # Plot ensemble members + ensemble average for 70% threshold
    ds_frac['0.7'].plot.line(hue='ensemble',ax=ax,color='orange',alpha=0.25,add_legend=False)
    ds_frac['0.7'].mean(dim='ensemble').plot(ax=ax,color='orange')
    
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

def frac_emerge_all(ds_toe,ds_pop,model,ylabel,title):
    '''Plot time series of fraction area/population emerged for all regions'''
    
    regions = ['Global','Northern North America','Central North America','Southern North America',
          'Central America','Northern South America','Southern South America',
          'Scandinavia','Central Europe','Southern Europe',
          'Middle East','India','Southeast Asia','Northern China','Southern China',
          'Northern Oceania','Southern Oceania',
          'Northern Africa','Central Africa','Southern Africa']

    # Create figure and axes
    fig, axs = plt.subplots(figsize=(20,20),nrows=5,ncols=4,constrained_layout=True)

    # Running counter to determine axis
    index = 0

    # Loop through regions
    for region in regions:
        # Get correct axis
        ax = axs[int(index/4)][index%4]
        index+=1

        # Get data for region
        ds_region = slice_region(ds_toe,region,model)
        pop_region = slice_region(ds_pop,region,model)

        frac_emerge_plot(ds_region,pop_region,region,ylabel,ax)

    fig.suptitle(title,fontsize=16)