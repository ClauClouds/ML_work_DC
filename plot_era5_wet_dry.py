
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
era5_data = xr.open_dataset('/net/ostro/ML_work_DC/era5_dry_wet/dry_wet_era5.nc')

#selecting wet and dry datasets
data_dry = era5_data.where(era5_data.im_type==1)
data_wet = era5_data.where(era5_data.im_type==0)

print(data_dry)
