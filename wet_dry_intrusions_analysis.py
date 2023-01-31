import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import rcParams
from warnings import warn
import datetime as dt
from scipy import interpolate
import matplotlib as mpl
import os.path
import itertools
import os.path
from random import randrange
pd.set_option('display.max_rows', 500)


# path and file name of dry wet info and their lat lon data
path_DC_work = '/net/ostro/ML_work_DC/'
wet_dry_info = xr.open_dataset(path_DC_work+'intrusions1_info.nc')
path_lat_lon = path_DC_work+'lat_lon_dry_wet/'

# path and filename of ERA5 data
path_era5 = '/data/mod/era/era5/eurec4a'

# read on era5 data and print max min lat lons
file_a_caso = xr.open_dataset(path_era5+'/2020/01/20/profb_presslev_20200120T0100.nc')
print(np.nanmax(file_a_caso.latitude.values), np.nanmin(file_a_caso.latitude.values))
print(np.nanmax(file_a_caso.longitude.values), np.nanmin(file_a_caso.longitude.values))
strasuka

# defining path out where to store the data
path_out = path_DC_work+'/era5_dry_wet/'


n_images = len(wet_dry_info.index.values)
print(n_images)

ind_start = len('/p/scratch/deepacf/kiste/DC/dataset/barbados/barbados_leif/1/')
for ind_images in range(n_images):

    # read from location variable the image file name
    id_image = wet_dry_info.location.values[ind_images][ind_start:ind_start+14]
    print(id_image)
    print(ind_images, n_images)

    # reading image type (wet or dry)
    im_type = wet_dry_info.dry_wet.values[ind_images]

    # reading year, month, day to build name of the image and find corresponding era5 data
    datetime_value = pd.to_datetime(wet_dry_info.datetime.values[ind_images])

    yy = str(datetime_value.year)
    mm = str(datetime_value.month)
    dd = str(datetime_value.day)
    hh = str(datetime_value.hour)
    mn = str(datetime_value.minute)

    # reading lat/lon
    # building era5 path
    if len(dd) == 1:
        dd = '0'+dd
    if len(hh) == 1:
        hh = '0'+hh
    if len(mm) == 1:
        mm = '0'+mm
    if len(mn) == 1:
        mn = '0'+mn

    # constructing date of the selected image (needed for era5)
    date = yy+mm+dd+hh+mn+'00'

    # reading lats/lons for the id_image
    lat_data = np.load(glob.glob(path_lat_lon+id_image+'_lat_*.npy')[0])
    lon_data = np.load(glob.glob(path_lat_lon+id_image+'_lon_*.npy')[0])
    print(np.shape(lat_data))
    print(lat_data)
    lat_max = np.nanmax(lat_data)
    lat_min = np.nanmin(lat_data)
    lon_min = np.nanmin(lon_data)
    lon_max = np.nanmax(lon_data)

    # assigning path for era5 based on the date
    era5_day_path = path_era5+'/'+yy+'/'+mm+'/'+dd+'/'

    # reading era
    profile_era5 = xr.open_dataset(era5_day_path+'profb_presslev_'+yy+mm+dd+'T'+hh+'00.nc')
    surface_era5 = xr.open_dataset(era5_day_path+'surfskinvarb_'+yy+mm+dd+'T'+hh+'00.nc')

    # selecting the area corresponding to the crop
    surface_crop = surface_era5.where((surface_era5.latitude > lat_min)*(surface_era5.latitude <= lat_max) \
                                    * (surface_era5.longitude > lon_min) *(surface_era5.longitude <= lon_max))

    profiles_crop = profile_era5.where((profile_era5.latitude > lat_min)*(profile_era5.latitude <= lat_max) \
                                    * (profile_era5.longitude > lon_min)*(profile_era5.longitude <= lon_max))

    # define output data for the ensemble of crops
    height = profile_era5.level.values
    SST = np.zeros(len(wet_dry_info.ind_crop.values))
    TCWV = np.zeros(len(wet_dry_info.ind_crop.values))
    TCC = np.zeros(len(wet_dry_info.ind_crop.values))
    TCLW = np.zeros(len(wet_dry_info.ind_crop.values))
    TCRW = np.zeros(len(wet_dry_info.ind_crop.values))
    CBH = np.zeros(len(wet_dry_info.ind_crop.values))
    q_profile = np.zeros((len(wet_dry_info.ind_crop.values), len(height)))
    t_profile = np.zeros((len(wet_dry_info.ind_crop.values), len(height)))

    SST_std = np.zeros(len(wet_dry_info.ind_crop.values))
    TCWV_std = np.zeros(len(wet_dry_info.ind_crop.values))
    TCC_std = np.zeros(len(wet_dry_info.ind_crop.values))
    TCLW_std = np.zeros(len(wet_dry_info.ind_crop.values))
    TCRW_std = np.zeros(len(wet_dry_info.ind_crop.values))
    CBH_std = np.zeros(len(wet_dry_info.ind_crop.values))
    q_profile_std = np.zeros((len(wet_dry_info.ind_crop.values), len(height)))
    t_profile_std = np.zeros((len(wet_dry_info.ind_crop.values), len(height)))


    # making a plot of the era5 domain and the selected box
    fig, axs = plt.subplots(figsize=(10,10), constrained_layout=True)
    surface_era5.tcwv.plot(x='longitude', y='latitude')
    rect=mpatches.Rectangle((lon_max, lat_min),lon_min-lon_max,lat_max-lat_min,
                    fill = False,
                    color = "purple",
                    linewidth = 2)
    plt.gca().add_patch(rect)
    fig.savefig(path_out+id_image+'_'+im_type+'.png')

    # calculating mean profiles and mean surface variables in the selected domain
    SST[ind_images] = surface_crop.sst.mean(skipna='True')
    SST_std[ind_images] = surface_crop.sst.std(skipna='True')

    TCWV[ind_images] = surface_crop.tcwv.mean(skipna='True')
    TCWV_std[ind_images] = surface_crop.tcwv.std(skipna='True')

    TCC[ind_images] = surface_crop.tcc.mean(skipna='True')
    TCC_std[ind_images] = surface_crop.tcc.std(skipna='True')

    TCLW[ind_images] = surface_crop.tclw.mean(skipna='True')
    TCLW_std[ind_images] = surface_crop.tclw.std(skipna='True')

    TCRW[ind_images] = surface_crop.tcrw.mean(skipna='True')
    TCRW_std[ind_images] = surface_crop.tcrw.std(skipna='True')

    CBH[ind_images] = surface_crop.cbh.mean(skipna='True')
    CBH_std[ind_images] = surface_crop.cbh.std(skipna='True')

    q_profile[ind_images,:] = profiles_crop.q.mean(dim=('longitude', 'latitude'), skipna='True')
    q_profile_std[ind_images,:] = profiles_crop.q.std(dim=('longitude', 'latitude'), skipna='True')
    t_profile[ind_images,:] = profiles_crop.t.mean(dim=('longitude', 'latitude'), skipna='True')
    t_profile_std[ind_images,:] = profiles_crop.t.std(dim=('longitude', 'latitude'), skipna='True')


    # saving a ncdf file for each group of crops
    crop_data = xr.Dataset(
        data_vars={
            'im_names': (('n_crops',), id_image, {'long_name': 'Names of image', 'units':''}),
            "SST": (('n_crops',), SST, {'long_name': 'Sea surface temperature ', 'units':'K', "standard_name": "SST"}),
            'SST_std':(('n_crops',), SST_std, {'long_name': 'Sea suface temperature standard deviation', 'units':'K', "standard_name": "SST_standard_dev"}),
            "TCWV": (('n_crops',), TCWV, {'long_name': 'Total column water vapor ', 'units':'kg m**-2', "standard_name": "TCWV"}),
            'TCWV_std':(('n_crops',), TCWV_std, {'long_name': 'Total column water vapor standard deviation', 'units':'kg m**-2', "standard_name": "TCWV_standard_dev"}),
            "TCC": (('n_crops',), TCC, {'long_name': 'Total cloud cover', 'units':'', "standard_name": "TCC"}),
            'TCC_std':(('n_crops',), TCC_std, {'long_name': 'Total cloud cover standard deviation', 'units':'', "standard_name": "TCC_standard_dev"}),
            "TCLW": (('n_crops',), TCLW, {'long_name': 'Total column cloud liquid water', 'units':'Kg m**-2', "standard_name": "TCLW"}),
            'TCLW_std':(('n_crops',), TCLW_std, {'long_name': 'Total column cloud liquid water standard deviation', 'units':'Kg m**-2', "standard_name": "TCLW_standard_dev"}),
            "TCRW": (('n_crops',), TCRW, {'long_name': 'Total column rain water', 'units':'Kg m**-2', "standard_name": "TCRW"}),
            'TCRW_std':(('n_crops',), TCRW_std, {'long_name': 'Total column rain water standard deviation', 'units':'Kg m**-2', "standard_name": "TCRW_standard_dev"}),

            "CBH": (('n_crops',), CBH, {'long_name': 'Cloud base height', 'units':'m', "standard_name": "CBH"}),
            'CBH_std':(('n_crops',), CBH_std, {'long_name': 'Cloud base height standard deviation', 'units':'m', "standard_name": "CBH_standard_dev"}),
            'q':(('n_crops','levels'), q_profile, {'long_name': 'Specific humidity', 'units':'kg kg**-1',}),
            'q_std':(('n_crops','levels'), q_profile_std, {'long_name': 'Specific humidity standard deviation', 'units':'kg kg**-1',}),
            't_std':(('n_crops','levels'), t_profile_std, {'long_name': 'Temperature standard deviation', 'units':'K'}),
            't':(('n_crops','levels'), t_profile, {'long_name': 'Temperature', 'standard_name':'Temperature', 'units':'K'}),
        },
        coords={
            "n_crops": (('n_crops',), np.arange(n_images) ,), # leave units intentionally blank, to be defined in the encoding
            "levels": (('levels',), height, {"axis": "pressure_level","positive": "up","units": "millibars", "long_name":'pressure_level'}),
        },
        attrs={'CREATED_BY'     : 'Claudia Acquistapace',
                        'CREATED_ON'       : str(datetime.now()),
                        'FILL_VALUE'       : 'NaN',
                        'IMAGE_NAME'       : im_name,
                        'PI_NAME'          : 'Claudia Acquistapace',
                        'PI_AFFILIATION'   : 'University of Cologne (UNI), Germany',
                        'PI_ADDRESS'       : 'Institute for geophysics and meteorology, Pohligstrasse 3, 50969 Koeln',
                        'PI_MAIL'          : 'cacquist@meteo.uni-koeln.de',
                        'DATA_DESCRIPTION' : 'ERA5 variables for all the crops of the selected satellite position ',
                        'DATA_DISCIPLINE'  : 'Atmospheric Physics - Remote Sensing Radar Profiler',
                        'DATA_GROUP'       : 'Model: reanalysis',
                        'DATA_LOCATION'    : 'Atlantic Ocean - Eurec4a campaign domain',
                        'DATA_SOURCE'      : 'ERA5',
                        'DATA_PROCESSING'  : 'https://github.com/ClauClouds/',
                        'COMMENT'          : '' }
    )




    # assign additional attributes following CF convention
    crop_data = crop_data.assign_attrs({
            "Conventions": "CF-1.8",
            "title": crop_data.attrs["DATA_DESCRIPTION"],
            "institution": crop_data.attrs["PI_AFFILIATION"],
            "history": "".join([
                "source: " + crop_data.attrs["DATA_SOURCE"] + "\n",
                "processing: " + crop_data.attrs["DATA_PROCESSING"] + "\n",
                " adapted to enhance CF compatibility\n",
            ]),  # the idea of this attribute is that each applied transformation is appended to create something like a log
            "featureType": "trajectoryProfile",
        })

    # storing ncdf data
    crop_data.to_netcdf(path_out+id_image+'_'+im_type+'_era5.nc')