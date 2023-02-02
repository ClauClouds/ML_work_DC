#!/usr/bin/env python
# coding: utf-8

# Code to plot data from era5 associated with the different images and their near crops
# 

# In[8]:


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

# read era files
file_list = glob.glob(path_DC_work+'*_era5.nc')
print(file_list)

data_list = []
for ind, filename in enumerate(file_list):
    data_list.append(xr.open_dataset(filename))


# In[9]:


data = xr.concat(data_list, dim='images')


# In[10]:


data['images'] = crops_info.im_name.values


# In[11]:


data


# In[19]:


dict_plot_settings = {
    'plot_ticks'   :32,
    'labelsizeaxes':32,
    'fontSizeTitle':32,
    'fontSizeX'    :32,
    'fontSizeY'    :32,
    'cbarAspect'   :32,
    'fontSizeCbar' :32,
    'rcparams_font':['Tahoma'],
    'savefig_dpi'  :100,
    'font_size'    :32,
    'grid'         :True}

colors_arr = ['deepskyblue', 'steelblue', 'violet', 'slateblue', 'navy']

# plots settings defined by user at the top
labelsizeaxes   = dict_plot_settings['labelsizeaxes']
fontSizeTitle   = dict_plot_settings['fontSizeTitle']
fontSizeX       = dict_plot_settings['fontSizeX']
fontSizeY       = dict_plot_settings['fontSizeY']
cbarAspect      = dict_plot_settings['cbarAspect']
fontSizeCbar    = dict_plot_settings['fontSizeCbar']
rcParams['font.sans-serif'] = dict_plot_settings['rcparams_font']
matplotlib.rcParams['savefig.dpi'] = dict_plot_settings['savefig_dpi']
plt.rcParams.update({'font.size':dict_plot_settings['font_size']})
grid = dict_plot_settings['grid']
matplotlib.rc('xtick', labelsize=dict_plot_settings['plot_ticks'])  # sets dimension of ticks in the plots
matplotlib.rc('ytick', labelsize=dict_plot_settings['plot_ticks'])  # sets dimension of ticks in the plots

image_names = data.images.values
# produce quicklook plot of the masked dataset
fig, axs = plt.subplots(6,1, figsize=(20,30), sharex=True, constrained_layout=True)

for ind_image in range(len(image_names)):
    print(image_names[ind_image])
    print(data.SST.values[ind_image, :])
    mesh1 = axs[0].plot(np.arange(len(data.SST.values[ind_image, :])),                         data.SST.values[ind_image, :], color=colors_arr[ind_image], linewidth=4)
    axs[0].fill_between(np.arange(len(data.SST.values[ind_image, :])),                        data.SST.values[ind_image, :]-data.SST_std.values[ind_image, :],                        data.SST.values[ind_image, :]+data.SST_std.values[ind_image, :], color=colors_arr[ind_image], alpha=0.2)
    mesh1 = axs[1].plot(np.arange(len(data.TCWV.values[ind_image, :])),                         data.TCWV.values[ind_image, :], color=colors_arr[ind_image], linewidth=4)
    axs[1].fill_between(np.arange(len(data.SST.values[ind_image, :])),                        data.TCWV.values[ind_image, :]-data.TCWV_std.values[ind_image, :],                        data.TCWV.values[ind_image, :]+data.TCWV_std.values[ind_image, :], color=colors_arr[ind_image], alpha=0.2)
    mesh1 = axs[2].plot(np.arange(len(data.TCC.values[ind_image, :])),                         data.TCC.values[ind_image, :], color=colors_arr[ind_image], linewidth=4)    
    axs[2].fill_between(np.arange(len(data.TCC.values[ind_image, :])),                        data.TCC.values[ind_image, :]-data.TCC_std.values[ind_image, :],                        data.TCC.values[ind_image, :]+data.TCC_std.values[ind_image, :], color=colors_arr[ind_image], alpha=0.2)

    mesh1 = axs[3].plot(np.arange(len(data.TCLW.values[ind_image, :])),                         data.TCLW.values[ind_image, :], color=colors_arr[ind_image], linewidth=4) 
    axs[3].fill_between(np.arange(len(data.TCLW.values[ind_image, :])),                        data.TCLW.values[ind_image, :]-data.TCLW_std.values[ind_image, :],                        data.TCLW.values[ind_image, :]+data.TCLW_std.values[ind_image, :], color=colors_arr[ind_image], alpha=0.2)

    mesh1 = axs[4].plot(np.arange(len(data.TCRW.values[ind_image, :])),                         data.TCRW.values[ind_image, :], color=colors_arr[ind_image], linewidth=4,label=str(image_names[ind_image])) 
    axs[4].fill_between(np.arange(len(data.TCRW.values[ind_image, :])),                        data.TCRW.values[ind_image, :]-data.TCRW_std.values[ind_image, :],                        data.TCRW.values[ind_image, :]+data.TCRW_std.values[ind_image, :], color=colors_arr[ind_image], alpha=0.2)

    mesh1 = axs[5].plot(np.arange(len(data.CBH.values[ind_image, :])),                         data.CBH.values[ind_image, :], color=colors_arr[ind_image], linewidth=4) 
    axs[5].fill_between(np.arange(len(data.CBH.values[ind_image, :])),                        data.CBH.values[ind_image, :]-data.CBH_std.values[ind_image, :],                        data.CBH.values[ind_image, :]+data.CBH_std.values[ind_image, :], color=colors_arr[ind_image], alpha=0.2)

#axs[0].scatter(time, radar_data.cloud_base.values, color='grey', label='cloud base height', s=2., alpha=0.45)
#axs[0].scatter(time, radar_data.cloud_top_height.values, color='black', label='cloud top height', s=2., alpha=0.45)
axs[4].legend(frameon=False)
axs[0].set_ylim(299.,301.)
axs[0].set_ylabel('SST [K]', fontsize=fontSizeX)

axs[1].set_ylim(25.,50.)
axs[1].set_ylabel('TCWV [kg m$^{-2}$]', fontsize=fontSizeX)

axs[2].set_ylim(0.,1.)
axs[2].set_ylabel('TCC []', fontsize=fontSizeX)

axs[3].set_ylim(0.,0.1)
axs[3].set_ylabel('TCLW [kg m$^{-2}$]]', fontsize=fontSizeX)

axs[4].set_ylim(0.,0.1)
axs[4].set_ylabel('TCRW [kg m$^{-2}$]]', fontsize=fontSizeX)

axs[5].set_ylim(0.,3500.)
axs[5].set_ylabel('CBH [m]]', fontsize=fontSizeX)

axs[5].set_xlabel('crops []', fontsize=fontSizeX)
for ax, l in zip(axs[:].flatten(), ['(a) SST'+image_names[0],
                                    '(b) CBH'+image_names[1], 
                                    '(c) T'+image_names[2],  
                                    '(d)'+image_names[3], 
                                    '(e)'+image_names[4],
                '']):
    #ax.set_xlim(SST_binned_arr[0]-0.1, SST_binned_arr[-1]+0.1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(which='minor', length=5, width=2)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(axis='both', labelsize=32)
fig.savefig(path_DC_work+'era5_surface_variables.png', format='png')



# In[20]:


fig, axs = plt.subplots(6,1, figsize=(20,30), sharex=True, constrained_layout=True)

for ind_image in range(len(image_names)):
    print(image_names[ind_image])
    print(data.SST.values[ind_image, :])
    mesh1 = axs[0].scatter(np.arange(len(data.SST.values[ind_image, :])),                         data.SST.values[ind_image, :], color=colors_arr[ind_image], linewidth=4)
    mesh1 = axs[1].scatter(np.arange(len(data.TCWV.values[ind_image, :])),                         data.TCWV.values[ind_image, :], color=colors_arr[ind_image], linewidth=4)
    mesh1 = axs[2].scatter(np.arange(len(data.TCC.values[ind_image, :])),                         data.TCC.values[ind_image, :], color=colors_arr[ind_image], linewidth=4)    
    mesh1 = axs[3].scatter(np.arange(len(data.TCLW.values[ind_image, :])),                         data.TCLW.values[ind_image, :], color=colors_arr[ind_image], linewidth=4) 
    mesh1 = axs[4].scatter(np.arange(len(data.TCRW.values[ind_image, :])),                         data.TCRW.values[ind_image, :], color=colors_arr[ind_image], linewidth=4,label=str(image_names[ind_image])) 
    mesh1 = axs[5].scatter(np.arange(len(data.CBH.values[ind_image, :])),                         data.CBH.values[ind_image, :], color=colors_arr[ind_image], linewidth=4) 
    
#axs[0].scatter(time, radar_data.cloud_base.values, color='grey', label='cloud base height', s=2., alpha=0.45)
#axs[0].scatter(time, radar_data.cloud_top_height.values, color='black', label='cloud top height', s=2., alpha=0.45)
axs[4].legend(frameon=False)
axs[0].set_ylim(299.,301.)
axs[0].set_ylabel('SST [K]', fontsize=fontSizeX)

axs[1].set_ylim(25.,50.)
axs[1].set_ylabel('TCWV [kg m$^{-2}$]', fontsize=fontSizeX)

axs[2].set_ylim(0.,1.)
axs[2].set_ylabel('TCC []', fontsize=fontSizeX)

axs[3].set_ylim(0.,0.1)
axs[3].set_ylabel('TCLW [kg m$^{-2}$]]', fontsize=fontSizeX)

axs[4].set_ylim(0.,0.1)
axs[4].set_ylabel('TCRW [kg m$^{-2}$]]', fontsize=fontSizeX)

axs[5].set_ylim(0.,3500.)
axs[5].set_ylabel('CBH [m]]', fontsize=fontSizeX)

axs[5].set_xlabel('crops []', fontsize=fontSizeX)
for ax, l in zip(axs[:].flatten(), ['(a) SST'+image_names[0],
                                    '(b) CBH'+image_names[1], 
                                    '(c) T'+image_names[2],  
                                    '(d)'+image_names[3], 
                                    '(e)'+image_names[4],
                '']):
    #ax.set_xlim(SST_binned_arr[0]-0.1, SST_binned_arr[-1]+0.1)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(which='minor', length=5, width=2)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(axis='both', labelsize=32)
fig.savefig(path_DC_work+'era5_surface_variables_scatter.png', format='png')



# In[42]:


np.shape(data.q.values)
levels
import metpy.calc as mpcalc
from metpy.units import units

import metpy.calc as mpcalc
from metpy.units import units

press = levels * units.hPa
temp = t * units.degC
mpcalc.thickness_hydrostatic(press, temp)


# In[52]:


levels = data.levels.values
q = data.q.values
t = data.t.values

strlev = []
for ind, lev in enumerate(levels):
    strlev.append(str(lev))

# plotting now profiles of q and t
fig, axs = plt.subplots(5,1, figsize=(20,30), sharex=True, constrained_layout=True)

for ind in range(10):
    axs[0].plot(q[0,ind,:],np.arange(len(levels)), color=colors_arr[0], label=str(image_names[ind_image]))
for ind in range(10):
    axs[1].plot(q[1,ind,:],np.arange(len(levels)), color=colors_arr[1], label=str(image_names[ind_image]))

for ind in range(10):
    axs[2].plot(q[2,ind,:],np.arange(len(levels)), color=colors_arr[2], label=str(image_names[ind_image]))
    
    
for ind in range(10):
    axs[3].plot(q[3,ind,:],np.arange(len(levels)), color=colors_arr[3], label=str(image_names[ind_image]))

for ind in range(10):
    axs[4].plot(q[4,ind,:],np.arange(len(levels)), color=colors_arr[4], label=str(image_names[ind_image]))
    
axs[4].set_xlabel('Specific humidity [kg m$^{-2}$]')

for ax, l in zip(axs[:].flatten(), ['(a)'+image_names[0],
                                    '(b)'+image_names[1], 
                                    '(c)'+image_names[2],  
                                    '(d)'+image_names[3], 
                                    '(e)'+image_names[4]]):
    #ax.set_xlim(SST_binned_arr[0]-0.1, SST_binned_arr[-1]+0.1)
    ax.text(-0.05, 1.1, l,  fontweight='black', fontsize=fontSizeX, transform=ax.transAxes)
    ax.set_ylabel('pressure levels')
    ax.set_yticklabels(strlev)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(which='minor', length=5, width=2)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(axis='both', labelsize=32)
fig.savefig(path_out+'q_profiles.png', format='png')


# In[50]:



# plotting now profiles of q and t
fig, axs = plt.subplots(5,1, figsize=(20,30), sharex=True, constrained_layout=True)

for ind in range(10):
    axs[0].plot(t[0,ind,:],np.arange(len(levels)), color=colors_arr[0], label=str(image_names[ind_image]))
for ind in range(10):
    axs[1].plot(t[1,ind,:],np.arange(len(levels)), color=colors_arr[1], label=str(image_names[ind_image]))

for ind in range(10):
    axs[2].plot(t[2,ind,:],np.arange(len(levels)), color=colors_arr[2], label=str(image_names[ind_image]))
    
    
for ind in range(10):
    axs[3].plot(t[3,ind,:],np.arange(len(levels)), color=colors_arr[3], label=str(image_names[ind_image]))

for ind in range(10):
    axs[4].plot(t[4,ind,:],np.arange(len(levels)), color=colors_arr[4], label=str(image_names[ind_image]))
    
axs[4].set_xlabel('temperature [K]')

for ax, l in zip(axs[:].flatten(), ['(a)'+image_names[0],
                                    '(b)'+image_names[1], 
                                    '(c)'+image_names[2],  
                                    '(d)'+image_names[3], 
                                    '(e)'+image_names[4]]):
    #ax.set_xlim(SST_binned_arr[0]-0.1, SST_binned_arr[-1]+0.1)
    ax.text(-0.05, 1.1, l,  fontweight='black', fontsize=fontSizeX, transform=ax.transAxes)
    ax.set_ylabel('pressure levels')
    ax.set_yticklabels(strlev)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(3)
    ax.spines["left"].set_linewidth(3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(which='minor', length=5, width=2)
    ax.tick_params(which='major', length=7, width=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.tick_params(axis='both', labelsize=32)
fig.savefig(path_out+'t_profiles.png', format='png')


# In[31]:


data.SST.values-data.SST_std.values


# In[ ]:




