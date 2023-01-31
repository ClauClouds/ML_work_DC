#!/usr/bin/env python
# coding: utf-8

# In[1]:



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

# path and file name of crops info
path_DC_work = '/net/ostro/ML_work_DC/'
crop_file = path_DC_work+'crops_info.nc'
crops_info = xr.open_dataset(crop_file)

# path and filename of ERA5 data
path_era5 = '/data/mod/era/era5/eurec4a'

# defining path out where to store the data
path_out = path_DC_work


# In[2]:


crops_info.im_name.values


# In[3]:


len(crops_info.ind_crop.values)


# In[4]:


# specify the location of (left,bottom),width,height
import matplotlib.patches as mpatches

# loop on image names to produce ncdf files containing data of era5 for each of the groups of crops

# reading time stamps from the crops
time_stamps = pd.to_datetime(crops_info.time_start.values)

for ind_images in range(len(time_stamps)):
    
    # reading crop name
    im_name = crops_info.im_name.values[ind_images]
    
    print('processing '+im_name)
    

    # reading year, month, day for reading era5 data
    yy = str(time_stamps[ind_images].year)
    mm = '0'+str(time_stamps[ind_images].month)
    dd = str(time_stamps[ind_images].day)
    hh = str(time_stamps[ind_images].hour)
    mn = str(time_stamps[ind_images].minute)
    ss = str(time_stamps[ind_images].second)

    # reading lat/lon 
    # building era5 path
    if len(dd) == 1:
        dd = '0'+dd
    if len(hh) == 1:
        hh = '0'+hh

    # assigning path for era5 based on the date
    era5_day_path = path_era5+'/'+yy+'/'+mm+'/'+dd+'/'  

    # reading era
    profile_era5 = xr.open_dataset(era5_day_path+'profb_presslev_'+yy+mm+dd+'T'+hh+'00.nc')
    surface_era5 = xr.open_dataset(era5_day_path+'surfskinvarb_'+yy+mm+dd+'T'+hh+'00.nc')

    # define output data for the ensemble of crops
    height = profile_era5.level.values
    SST = np.zeros(len(crops_info.ind_crop.values))
    TCWV = np.zeros(len(crops_info.ind_crop.values))
    TCC = np.zeros(len(crops_info.ind_crop.values))
    TCLW = np.zeros(len(crops_info.ind_crop.values))
    TCRW = np.zeros(len(crops_info.ind_crop.values))
    CBH = np.zeros(len(crops_info.ind_crop.values))
    q_profile = np.zeros((len(crops_info.ind_crop.values), len(height)))
    t_profile = np.zeros((len(crops_info.ind_crop.values), len(height)))

    SST_std = np.zeros(len(crops_info.ind_crop.values))
    TCWV_std = np.zeros(len(crops_info.ind_crop.values))
    TCC_std = np.zeros(len(crops_info.ind_crop.values))
    TCLW_std = np.zeros(len(crops_info.ind_crop.values))
    TCRW_std = np.zeros(len(crops_info.ind_crop.values))
    CBH_std = np.zeros(len(crops_info.ind_crop.values))
    q_profile_std = np.zeros((len(crops_info.ind_crop.values), len(height)))
    t_profile_std = np.zeros((len(crops_info.ind_crop.values), len(height)))

    
    # loop on the image crops associated with the image selected
    for ind_crop in range(len(crops_info.ind_crop.values)):
        
        # reading lat/lon box from the crop 
        lat_max = crops_info.latitude_max.values[ind_images,ind_crop]
        lat_min = crops_info.latitude_min.values[ind_images,ind_crop]
        lon_max = crops_info.longitude_max.values[ind_images,ind_crop]
        lon_min = crops_info.longitude_min.values[ind_images,ind_crop]
        print(im_name)
        print('ind_crop, lat_max, lat_min, lon_max, lon_min')
        print(ind_crop, lat_max, lat_min, lon_max, lon_min)
        print(lon_min-lon_max,lat_max-lat_min)
        
        # selecting the area corresponding to the crop
        surface_crop = surface_era5.where((surface_era5.latitude > lat_min)*(surface_era5.latitude <= lat_max) *                                          (surface_era5.longitude > lon_max)*(surface_era5.longitude <= lon_min))

        profiles_crop = profile_era5.where((profile_era5.latitude > lat_min)*(profile_era5.latitude <= lat_max) *                                          (profile_era5.longitude > lon_max)*(profile_era5.longitude <= lon_min))

        # making a plot of the era5 domain and the selected box
        fig, axs = plt.subplots(figsize=(10,10), constrained_layout=True)
        surface_era5.sst.plot(x='longitude', y='latitude')
        rect=mpatches.Rectangle((lon_max, lat_min),lon_min-lon_max,lat_max-lat_min, 
                        fill = False,
                        color = "purple",
                        linewidth = 2)
        plt.gca().add_patch(rect)
        fig.savefig(path_out+im_name+'_'+str(ind_crop)+'_domains.png')
        
        # calculating mean profiles and mean surface variables in the selected domain
        SST[ind_crop] = surface_crop.sst.mean(skipna='True')
        SST_std[ind_crop] = surface_crop.sst.std(skipna='True')

        TCWV[ind_crop] = surface_crop.tcwv.mean(skipna='True')
        TCWV_std[ind_crop] = surface_crop.tcwv.std(skipna='True')

        TCC[ind_crop] = surface_crop.tcc.mean(skipna='True')
        TCC_std[ind_crop] = surface_crop.tcc.std(skipna='True')

        TCLW[ind_crop] = surface_crop.tclw.mean(skipna='True')
        TCLW_std[ind_crop] = surface_crop.tclw.std(skipna='True')

        TCRW[ind_crop] = surface_crop.tcrw.mean(skipna='True')
        TCRW_std[ind_crop] = surface_crop.tcrw.std(skipna='True')

        CBH[ind_crop] = surface_crop.cbh.mean(skipna='True')
        CBH_std[ind_crop] = surface_crop.cbh.std(skipna='True')

        q_profile[ind_crop,:] = profiles_crop.q.mean(dim=('longitude', 'latitude'), skipna='True')
        q_profile_std[ind_crop,:] = profiles_crop.q.std(dim=('longitude', 'latitude'), skipna='True')
        t_profile[ind_crop,:] = profiles_crop.t.mean(dim=('longitude', 'latitude'), skipna='True')
        t_profile_std[ind_crop,:] = profiles_crop.t.std(dim=('longitude', 'latitude'), skipna='True')


    # saving a ncdf file for each group of crops
    crop_data = xr.Dataset(
        data_vars={
            'crop_names': (('n_crops',), crops_info.crop_names.values[ind_images,:], {'long_name': 'Names of crops associated with the main image', 'units':''}),
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
            "n_crops": (('n_crops',), np.arange(len(crops_info.crop_names.values[ind_images,:])) ,), # leave units intentionally blank, to be defined in the encoding
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
    crop_data.to_netcdf(path_out+im_name+'_era5.nc')


# In[125]:


crops_info.im_name.values


# In[70]:


surface_crop.sst.plot(x='longitude', y='latitude')
#


# In[62]:


profile_era5


# In[95]:


surface_era5


# In[ ]:




