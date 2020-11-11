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
import matplotlib.ticker as mticker

# Path to sample data file, CESM2
path_CESM2 = '../data/processed/CESM2/population_regrid_cesm2_2.nc'
# Load data from matching file
ds_CESM2 = xr.open_dataset(path_CESM2)

# Path to sample data file, GFDL
path_GFDL = '../data/processed/GFDL/population_regrid_esm2m_2.nc'
# Load data from matching file
ds_GFDL = xr.open_dataset(path_GFDL)

# Populates dictionary of regional masks based on gridding of sample dataset
def fill_mask(ds,masks):
    lon = ds['lon']
    lat = ds['lat']
    
    masks['Global'] = [lon.values, lat.values]
    masks['Northern North America'] = [lon.where((190<=lon)&(lon<=310),drop=True).values,lat.where((45<=lat)&(lat<=75),drop=True).values]
    masks['Central North America'] = [lon.where((230<=lon)&(lon<=310),drop=True).values,lat.where((35<=lat)&(lat<=45),drop=True).values]
    masks['Southern North America'] = [lon.where((230<=lon)&(lon<=300),drop=True).values,lat.where((20<=lat)&(lat<=35),drop=True).values]
    masks['Central America'] = [lon.where((250<=lon)&(lon<=315),drop=True).values,lat.where((7<=lat)&(lat<=20),drop=True).values]
    masks['Northern South America'] = [lon.where((270<=lon)&(lon<=330),drop=True).values,lat.where((-23.5<=lat)&(lat<=12),drop=True).values]
    masks['Southern South America'] = [lon.where((270<=lon)&(lon<=330),drop=True).values,lat.where((-57<=lat)&(lat<=-23.5),drop=True).values]
    masks['Southeastern US'] = [lon.where((260<=lon)&(lon<=285),drop=True).values,lat.where((25<=lat)&(lat<=37),drop=True).values]

    masks['Scandinavia'] = [lon.where((3<=lon)&(lon<=30),drop=True).values,lat.where((55<=lat)&(lat<=70),drop=True).values]
    lon_west = lon.where(lon>=345,drop=True)
    lon_east = lon.where(lon<=35,drop=True)
    lon_ceur = xr.concat((lon_west,lon_east),dim='lon').values
    masks['Central Europe'] = [lon_ceur,lat.where((43<=lat)&(lat<=55),drop=True).values]
    lon_west = lon.where(lon>=350,drop=True)
    lon_east = lon.where(lon<=24,drop=True)
    lon_seur = xr.concat((lon_west,lon_east),dim='lon').values
    masks['Southern Europe'] = [lon_seur,lat.where((36<=lat)&(lat<=43),drop=True).values]

    masks['Northern China'] = [lon.where((75<=lon)&(lon<=135),drop=True).values,lat.where((32<=lat)&(lat<=50),drop=True).values]
    masks['Southern China'] = [lon.where((103<=lon)&(lon<=125),drop=True).values,lat.where((23<=lat)&(lat<=35),drop=True).values]
    masks['India'] = [lon.where((68<=lon)&(lon<=90),drop=True).values,lat.where((8<=lat)&(lat<=30),drop=True).values]
    masks['Northern India'] = [lon.where((68<=lon)&(lon<=90),drop=True).values,lat.where((23<=lat)&(lat<=30),drop=True).values]
    masks['Southern India'] = [lon.where((68<=lon)&(lon<=90),drop=True).values,lat.where((8<=lat)&(lat<=23),drop=True).values]
    masks['Southeast Asia'] = [lon.where((92<=lon)&(lon<=130),drop=True).values,lat.where((0<=lat)&(lat<=23),drop=True).values]
    masks['Middle East'] = [lon.where((40<=lon)&(lon<=61),drop=True).values,lat.where((13<=lat)&(lat<=30),drop=True).values]
    masks['European Russia'] = [lon.where((43<=lon)&(lon<=70),drop=True).values,lat.where((50<=lat)&(lat<=75),drop=True).values]

    masks['Northern Oceania'] = [lon.where((100<=lon)&(lon<=160),drop=True).values,lat.where((-23.5<=lat)&(lat<=0),drop=True).values]
    masks['Southern Oceania'] = [lon.where((100<=lon)&(lon<=160),drop=True).values,lat.where((-50<=lat)&(lat<=-23.5),drop=True).values]
    masks['Indonesia'] = [lon.where((95<=lon)&(lon<=142),drop=True).values,lat.where((-10<=lat)&(lat<=5),drop=True).values]
    masks['Philippines'] = [lon.where((115<=lon)&(lon<=130),drop=True).values,lat.where((5<=lat)&(lat<=20),drop=True).values]

    lon_west = lon.where(lon>=340,drop=True)
    lon_east = lon.where(lon<=17,drop=True)
    lon_cafrica = xr.concat((lon_west,lon_east),dim='lon').values
    masks['West Africa'] = [lon_cafrica,lat.where((0<=lat)&(lat<=26),drop=True).values]
    lon_west = lon.where(lon>=340,drop=True)
    lon_east = lon.where(lon<=25,drop=True)
    lon_nafrica = xr.concat((lon_west,lon_east),dim='lon').values
    masks['Northern Africa'] = [lon_nafrica,lat.where((20<=lat)&(lat<=38),drop=True).values]
    masks['Southern Africa'] = [lon.where((9<=lon)&(lon<=52),drop=True).values,lat.where((-35<=lat)&(lat<=-5),drop=True).values]
    
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

def map_region(region,model):
    # Specify projection
    crs = ccrs.PlateCarree()
    
    fig, ax = plt.subplots(subplot_kw={'projection':crs})
    
    # Get correct masks based on model
    if model == 'GFDL':
        masks = masks_GFDL
    elif model == 'CESM2':
        masks = masks_CESM2
    else:
        raise ValueError('Model name not valid')
    
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
        # Shrink plot to focus on region
        ax.set_extent([xmin,xmax,ymin,ymax],crs=crs)

    # Add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))
    ax.add_feature(cfeature.OCEAN, color='skyblue')
    ax.add_feature(cfeature.LAND, color='lightgrey')
    
def contour(ds,title,ax,levels,cmap='magma',label='Labor Capacity, %',under=None,over='darkgray',extend='both',crop=False,colors=None,titleloc='center'):
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
    im = ax.contourf(X,Y,Z,levels=levels,transform=crs,colors=colors,extend=extend)
    
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
    
    # Crop polar regions if necessary
    if crop:
        ax.set_extent([-140,160,-50,50],crs=crs)

    # Set title
    ax.set_title(title,loc=titleloc)
    
    return im

def scatter(ds,ax,s,reduce=False):
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

    if reduce:
        X = X[::3]
        Y = Y[::3]
    
    # Mark grid cells
    crs = ccrs.PlateCarree()
    ax.scatter(X,Y,transform=crs,zorder=1,s=s,marker='.',c='black')

def box(region,ax):
    '''Draw a box around a region on plot'''
    crs = ccrs.PlateCarree()
    
    # Get coordinates of lower left corner
    X = masks_GFDL[region][0][0]
    Y = masks_GFDL[region][1][0]
    
    # Get width, height
    dx = masks_GFDL[region][0][-1] - X
    dy = masks_GFDL[region][1][-1] - Y
    
    # Account for regions that span lon=0
    if dx < 0:
        dx += 360
    
    # Add box
    ax.add_patch(mpatches.Rectangle((X, Y), dx, dy,transform=crs,facecolor='none',edgecolor='black',linestyle='--',zorder=20))
    
def grid(ax,label=False):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), zorder=20, draw_labels=True,linewidth=0.75, color='gray', alpha=0.4, linestyle='--')
    gl.left_labels = False
    gl.bottom_labels = False
    gl.top_labels = False
    if label==False:
        gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-135,-90,-45,0,45,90,135])
    gl.ylocator = mticker.FixedLocator([-45,-30,-15,0,15,30,45,60])

def year_max(ds):
    '''Gets the mean labor capacity in hottest three months of a given year'''
    ds_sorted=np.partition(ds,3)
    return ds_sorted[0:3].mean()
    
def max_avg(ds,years=20):
    '''Splits time dimension into years and passes on to year_max to get yearly baseline'''
    ds_yearly = np.array_split(ds,years)
    ds_max = np.apply_along_axis(year_max,1,ds_yearly)
    return ds_max
    
def calc_baseline(ds):
    '''Calculates 1980-2000 baseline capacity, by month (mean - 2*std)'''
    # Slice 1980-2000 data
    ds_hist  = ds.sel(time=slice('1980-01-31','1999-12-31'))

    # Calculate mean and stdev
    ds_hist = xr.apply_ufunc(max_avg,ds_hist['capacity'],input_core_dims=[['time']],output_core_dims=[['year']],exclude_dims=set(('time',)),vectorize=True,dask='allowed')
    ds_hist_mean = ds_hist.mean(['ensemble','year'])
    ds_hist_dev = ds_hist.std(['ensemble','year'])
    
    # Return baseline as the lower bound of "envelope" around mean 
    ds_base = ds_hist_mean - 2*ds_hist_dev
    return ds_base

'''def calc_baseline(ds):
    # Calculates 1980-2000 baseline capacity, by month (mean - 2*std)
    # Slice 1980-2000 data
    ds_hist  = ds.sel(time=slice('1980-01-31','1999-12-31')).groupby('time.month')

    # Calculate mean and stdev
    ds_hist_mean = ds_hist.mean(['time','ensemble'])
    ds_dev = 2*ds_hist.std(['time','ensemble'])
    
    # Return baseline as the lower bound of "envelope" around mean 
    ds_base = ds_hist_mean - ds_dev
    return ds_base['capacity']'''

def emergence_summer(ds,start_year):
    '''Function finds first year with entire summer season below threshold'''
    # Array indices where all three summer months are below threshold
    ds_thres = ds.nonzero()
    
    # If non-empty, index + startyear = ToE
    if len(ds_thres[0]) > 0:
        return start_year+ds_thres[0][0].item()
    
    # If empty, return year after 2100
    return 2101

def toe_summer(ds,ds_base,labor_thres):
    start_year = ds['year'][0].item()
    
     # Dataset for ToEs
    ds_toe = xr.Dataset()
    
    for thres in labor_thres:
        # Check if capacity is below threshold
        ds_thres = ds < thres*ds_base 
        
        toe = xr.apply_ufunc(emergence_summer,ds_thres,input_core_dims=[['year']],vectorize=True,dask='allowed',kwargs={'start_year':start_year})
        ds_toe[str(thres)] = toe
    return ds_toe

'''def toe_summer(ds,ds_base,labor_thres):
    Return dataset of ToEs based on various inputted thresholds (all 3 summer months below threshold)
    # First year of the dataset
    start_year = ds['time.year'][0].item()

    # Dataset for ToEs
    ds_toe = xr.Dataset()
    
    # Loop through inputted thresholds
    for thres in labor_thres:
        # Check if capacity is below threshold for each month
        ds_thres = ds < (thres*ds_base.sel(month=ds['time.month']))
        
        # Split into seasons and sum number of truth values
        # Resampled month index = last month in season
        ds_thres = ds_thres.resample(time='Q-NOV').sum()
        
        # Split into hemispheres and isolate summer season (JJA Northern; DJF Southern)
        ds_north = ds_thres.where(ds['lat']>0,drop=True)
        ds_north = ds_north.sel(time=(ds_north['time.month']==8))

        ds_south = ds_thres.where(ds['lat']<0,drop=True)
        ds_south = ds_south.sel(time=(ds_south['time.month']==2))
        
        # Get first year with entire summer season below threshold
        north_toe = xr.apply_ufunc(emergence_summer,ds_north,input_core_dims=[['time']],vectorize=True,dask='allowed',kwargs={'start_year':start_year})
        south_toe = xr.apply_ufunc(emergence_summer,ds_south,input_core_dims=[['time']],vectorize=True,dask='allowed',kwargs={'start_year':start_year})
        
        # Combine data for two hemispheres
        ds_toe[str(thres)] = xr.concat([south_toe,north_toe],dim='lat')
    return ds_toe'''

def calc_range(ds):
    #90th percentile - 10th percentile ToEs
    ds_range = ds.quantile(0.9,'ensemble',interpolation='lower') - ds.quantile(0.1,'ensemble',interpolation='lower')
    
    #take average for grid cells that emerge (i.e. range>0)
    avg_range_25 = ds_range['0.75'].where(ds_range['0.75']>0,np.nan).mean('lon',skipna=True)
    avg_range_50 = ds_range['0.5'].where(ds_range['0.5']>0,np.nan).mean('lon',skipna=True)
    
    #combine into one dataset
    avg_range = xr.Dataset()
    avg_range['0.75'] = avg_range_25
    avg_range['0.5'] = avg_range_50
    
    #return gridcell range + latitudinal average range
    return [ds_range,avg_range]

def range_plot(ds,ax,label=False):
    #plot on axes
    ds['0.75'].plot(ax=ax,y='lat',ylim=[-90,90],xlim=[0,40],color='royalblue')
    ds['0.5'].plot(ax=ax,y='lat',ylim=[-90,90],xlim=[0,40],color='orange')
    ax.set_aspect(0.4)
    ax.grid('True')
    ax.set_xticks(range(0,50,10))

    if label:
        ax.set_xlabel(u'Δ Years')
        ax.set_xticklabels(['0','','20','','40'])
    else:
        ax.set_xticklabels(['','','','',''])
        blue_line = mlines.Line2D([], [], color='royalblue', label='25%')
        orange_line = mlines.Line2D([], [], color='orange', label='50%')
        ax.legend(handles=[blue_line,orange_line], bbox_to_anchor=(0.5,1.1,0.5,0.25),fontsize='x-large');

def stipple(ds,ax,s,reduce=False):
    '''Place markers over grid cells with large ensemble range'''
    # Adjust non-emerging grid cells (so that partially emerging grid cells are marked)
    ds = ds.where(ds<2101,other=2200)
    
    #10-90 ensemble range
    ds_range = (ds.quantile(0.9,'ensemble',interpolation='lower') - ds.quantile(0.1,'ensemble',interpolation='lower')).stack(xy=('lon','lat'))
    #Grid cells with range > 50 years
    stipples = ds_range.where(ds_range>50,drop=True)
    
    # Get x,y for these grid cells
    X = [x[0] for x in stipples['xy'].values]
    Y = [x[1] for x in stipples['xy'].values]
    
    # Prevent overcrowding of stipples
    if reduce:
        X = X[::3]
        Y = Y[::3]
    
    # Mark grid cells
    crs = ccrs.PlateCarree()
    ax.scatter(X,Y,transform=crs,zorder=1,s=s,marker='.',c='black')
        
def spatial_toe(ds_esm2m,ds_cesm2,range_esm2m,range_cesm2,title,thres):
    '''Plot spatial map of ToE for all grid cells (global)'''
    # Specify projection
    crs = ccrs.Robinson()

    # Create figure and axes
    fig, axs = plt.subplots(ncols=4,nrows=2,figsize=(21,7),subplot_kw={'projection':crs},gridspec_kw={'width_ratios': [0.3,3,3,0.8]})
    levels = np.linspace(2000,2100,21)
    cmap = 'magma'

    # Plots of ToE: ESM2M
    im = contour(ds_esm2m[thres[0]].mean(dim='ensemble'),'25% Reduction',axs[0][1],levels=levels,cmap=cmap,label='Year',extend='max',crop=True)
    grid(axs[0][1])
    stipple(ds_esm2m[thres[0]],axs[0][1],0.5,False)
    contour(ds_esm2m[thres[1]].mean(dim='ensemble'),'50% Reduction',axs[0][2],levels=levels,cmap =cmap,label='Year',extend='max',crop=True)
    grid(axs[0][2])
    stipple(ds_esm2m[thres[1]],axs[0][2],0.5,False)

    # Plots of ToE: CESM2
    contour(ds_cesm2[thres[0]].mean(dim='ensemble'),None,axs[1][1],levels=levels,cmap=cmap,label='Year',extend='max',crop=True)
    grid(axs[1][1])
    stipple(ds_cesm2[thres[0]],axs[1][1],0.5,True)
    contour(ds_cesm2[thres[1]].mean(dim='ensemble'),None,axs[1][2],levels=levels,cmap=cmap,label='Year',extend='max',crop=True)
    grid(axs[1][2])
    stipple(ds_cesm2[thres[1]],axs[1][2],0.5,True)
    
    #Plots of ToE range by latitude
    range_plot(range_esm2m,axs[0][3])
    range_plot(range_cesm2,axs[1][3],label=True)
    
    # Box selected regions
    regions = ['Middle East','India','Southeast Asia','Northern Oceania','West Africa','Southeastern US','Southern China']
    for region in regions:
        box(region,axs[0][1])

    # Annotating text
    axs[0][0].text(0.5,0.5,'ESM2M',fontsize=22,horizontalalignment='right',verticalalignment='center');
    axs[0][0].set_frame_on(False)
    axs[1][0].text(0.5,0.5,'CESM2',fontsize=22,horizontalalignment='right',verticalalignment='center');
    axs[1][0].set_frame_on(False)

    # Single colorbar for all plots
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.3, 0.125, 0.4, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax,orientation='horizontal');
    cbar.set_label('Year',fontsize=22)
    cbar.set_ticks(np.linspace(2000,2120,7))
    cbar.set_ticklabels(['2000','2020','2040','2060','2080','2100+'])
    fig.subplots_adjust(wspace=.05,hspace=.05)
    
    plt.figtext(0.16,0.83,'a',fontweight='bold',fontsize=22)
    plt.figtext(0.16,0.45,'b',fontweight='bold',fontsize=22)
    plt.figtext(0.485,0.83,'c',fontweight='bold',fontsize=22)
    plt.figtext(0.485,0.45,'d',fontweight='bold',fontsize=22)
    plt.figtext(0.81,0.83,'e',fontweight='bold',fontsize=22)
    plt.figtext(0.81,0.45,'f',fontweight='bold',fontsize=22)

    # Overall figure title
    fig.suptitle(title,fontweight='bold');

'''def spatial_toe(ds,title,thres):
    Plot spatial map of ToE for all grid cells (global)
    # Specify projection
    crs = ccrs.Robinson()

    # Create figure and axes
    fig, axs = plt.subplots(ncols=4,nrows=2,figsize=(22,7),subplot_kw={'projection':crs},gridspec_kw={'width_ratios': [0.3,3,3,3]})
    levels = np.linspace(2000,2100,21)
    cmap = 'magma'

    # Plots of ToE: earliest among ensemble members
    im = contour(ds[thres[0]].min(dim='ensemble'),'10% Reduction',axs[0][1],levels=levels,cmap =cmap,label='Year',extend='max',crop=True)
    contour(ds[thres[1]].min(dim='ensemble'),'25% Reduction',axs[0][2],levels=levels,cmap =cmap,label='Year',extend='max',crop=True)
    contour(ds[thres[2]].min(dim='ensemble'),'50% Reduction',axs[0][3],levels=np.linspace(2000,2100),cmap =cmap,label='Year',extend='max',crop=True)

    # Plots of ToE: mean among ensemble members
    contour(ds[thres[0]].mean(dim='ensemble'),None,axs[1][1],levels=levels,cmap=cmap,label='Year',extend='max',crop=True)
    contour(ds[thres[1]].mean(dim='ensemble'),None,axs[1][2],levels=levels,cmap=cmap,label='Year',extend='max',crop=True)
    contour(ds[thres[2]].mean(dim='ensemble'),None,axs[1][3],levels=levels,cmap=cmap,label='Year',extend='max',crop=True)
    
    # Box selected regions
    regions = ['Northern South America','India','Southeast Asia','Northern Oceania','West-Central Africa']
    for region in regions:
        box(region,axs[0][1])

    # Annotating text
    axs[0][0].text(0.5,0.5,'Ensemble\n Earliest',fontsize=22,horizontalalignment='right',verticalalignment='center');
    axs[0][0].set_frame_on(False)
    axs[1][0].text(0.5,0.5,'Ensemble\n Mean',fontsize=22,horizontalalignment='right',verticalalignment='center');
    axs[1][0].set_frame_on(False)

    # Single colorbar for all plots
    fig.subplots_adjust(bottom=0.2)
    cbar_ax = fig.add_axes([0.3, 0.125, 0.4, 0.05])
    cbar = fig.colorbar(im, cax=cbar_ax,orientation='horizontal');
    cbar.set_label('Year',fontsize=22)
    cbar.set_ticks(np.linspace(2000,2120,7))
    cbar.set_ticklabels(['2000','2020','2040','2060','2080','2100+'])
    fig.subplots_adjust(wspace=.05,hspace=.05)

    # Overall figure title
    fig.suptitle(title,fontweight='bold');'''
    
def spatial_toe_diff(ds,title,thres,s=0.3,reduce=False):
    '''Plot ToE range for all grid cells (global)'''
    # Specify projection
    crs = ccrs.Robinson()

    # Create figure and axes
    fig, axs = plt.subplots(ncols=3,figsize=(22,4.5),subplot_kw={'projection':crs})
    levels = np.linspace(1,60,30)
    cmap = 'YlOrBr'

    # Plots of ToE range: max ToE - min ToE
    im = contour(ds[thres[0]].max(dim='ensemble')-ds[thres[0]].min(dim='ensemble'),'10% Reduction',axs[0],levels=levels,cmap=cmap,label='Year',under='white',over=None,crop=True)
    scatter(ds[thres[0]],axs[0],s,reduce)
    contour(ds[thres[1]].max(dim='ensemble')-ds[thres[1]].min(dim='ensemble'),'25% Reduction',axs[1],levels=levels,cmap=cmap,label='Year',under='white',over=None,crop=True)
    scatter(ds[thres[1]],axs[1],s,reduce)
    contour(ds[thres[2]].max(dim='ensemble')-ds[thres[2]].min(dim='ensemble'),'50% Reduction',axs[2],levels=levels,cmap=cmap,label='Year',under='white',over=None,crop=True)
    scatter(ds[thres[2]],axs[2],s,reduce)

    # Single colorbar for all plots
    fig.subplots_adjust(bottom=0.225)
    cbar_ax = fig.add_axes([0.3, 0.15, 0.4, 0.075])
    cbar = fig.colorbar(im, cax=cbar_ax,orientation='horizontal');
    cbar.set_label('Years',fontsize=22)
    cbar.set_ticks([1,10,20,30,40,50,60])
    fig.subplots_adjust(wspace=.05,hspace=.05)

    # Overall figure title
    fig.suptitle(title,fontweight='bold');
    
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