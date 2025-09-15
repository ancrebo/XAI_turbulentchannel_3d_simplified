# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
main_statisticsSHAP.py
-------------------------------------------------------------------------------------------------------------------------
Created on Wed Mar 27 08:23:08 2024

@author: Andres Cremades Botella

File to create the statistics of the flow fields: mean, rms, normalization. The file requires to set the following
variables:
    - folder_def  : (str) name of the folder containing the files for configuring the case of analysis.
    - chd_str     : (str) name of the file containing the data of the channel.
    - folders_str : (str) name of the file containing the folders and files used in the problem.
    - st_data_str : (str) name of the file containing the information required for the statistics.
"""
# -----------------------------------------------------------------------------------------------------------------------
# Define the names of the files containing the definitios of the parameters
# - folder_def : folder containing the files with the definitions required in the problem
# - chd_str    : file containing the data of the channel
# - folders    : file containing the folder and file structures
# - st_data    : file containing the data of the statistics
# -----------------------------------------------------------------------------------------------------------------------
folder_def  = "configuration"
chd_str     = "channel_data"
folders_str = "folders"
st_data_str = "stats_data_shap"
sh_data_str = "shap_data"

# -----------------------------------------------------------------------------------------------------------------------
# Import Packages
# -----------------------------------------------------------------------------------------------------------------------
from py_bin.py_functions.shapmean import calc_SHAPmean
from py_bin.py_functions.shaprms import calc_rms,calc_rms_nomean,calc_rms_nomean
from py_bin.py_class.flow_field import flow_field
import os

# -----------------------------------------------------------------------------------------------------------------------
# Unlock the h5 files for avoiding problems in some clusters
# -----------------------------------------------------------------------------------------------------------------------
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# -----------------------------------------------------------------------------------------------------------------------
# Import information files
# -----------------------------------------------------------------------------------------------------------------------
exec("from "+folder_def+" import "+chd_str+" as chd")
exec("from "+folder_def+" import "+folders_str+" as folders")
exec("from "+folder_def+" import "+st_data_str+" as st_data")
exec("from "+folder_def+" import "+sh_data_str+" as sh_data")

# -----------------------------------------------------------------------------------------------------------------------
# Data for the statistics:
#     - field_ini           : index of the initial field
#     - field_fin           : index of the final field
#     - folder              : folder of the flow field data
#     - file                : file of the flow field data
#     - savefile            : boolean to decide if the calculations are stored in a file or variable
#                             (True: file, False: variable)
#     - SHAPmean_file       : file to save the mean velocity
#     - data_folder         : folder to store the calculated data
#     - dx                  : downsampling in x
#     - dy                  : downsampling in y
#     - dz                  : downsampling in z
#     - L_x                 : length of the channel in the streamwise direction
#     - L_y                 : half-width of the channel in the wall-normal direction
#     - L_z                 : length of the channel in the spanwise direction
#     - SHAPrms_file        : file to save the rms of the shap with the mean (second order momentum)
#     - unorm_file          : file for saving the normalization
#     - rey                 : Friction Reynolds number
#     - utau                : Friction velocity
# -----------------------------------------------------------------------------------------------------------------------
field_ini           = st_data.field_ini
field_fin           = st_data.field_fin
field_delta         = st_data.field_delta
folder              = folders.uvw_folder
file                = folders.uvw_file
folder_shap         = folders.shap_folder
file_shap           = folders.shap_file
nsamples            = sh_data.nsamples
file_shap           = file_shap.replace("$NSAMPLES$",str(nsamples))
save_file           = st_data.save_file
SHAPmean_file       = folders.SHAPmean_file
data_folder         = folders.data_folder
dx                  = chd.dx
dy                  = chd.dy
dz                  = chd.dz
L_x                 = chd.L_x
L_y                 = chd.L_y
L_z                 = chd.L_z
SHAPrms_file        = folders.SHAPrms_file
rey                 = chd.rey
utau                = chd.utau
mean_norm           = bool(st_data.mean_norm)

if mean_norm:
    from py_bin.py_functions.normalization_normaldist import calc_norm
else:
    from py_bin.py_functions.normalization import calc_norm

# -----------------------------------------------------------------------------------------------------------------------
# Create the folder to store the data
# -----------------------------------------------------------------------------------------------------------------------
try:
    os.mkdir(data_folder)
except:
    pass

# -----------------------------------------------------------------------------------------------------------------------
# Obtain the flow characteristics
# -----------------------------------------------------------------------------------------------------------------------
Data_flow={"folder":folder,"file":file,"down_x":dx,"down_y":dy,"down_z":dz,"L_x":L_x,"L_y":L_y,"L_z":L_z,
            "rey":rey,"utau":utau}
flow = flow_field(data_in=Data_flow)
flow.shape_tensor()

# -----------------------------------------------------------------------------------------------------------------------
# Calculate the mean values of the velocity
# -----------------------------------------------------------------------------------------------------------------------
Data_shapmean={"field_ini":field_ini,"field_fin":field_fin,"field_delta":field_delta,"folder":folder_shap,
                "file":file_shap,"save_file":save_file,"SHAPmean_file":SHAPmean_file,"data_folder":data_folder,
                "shpx":flow.shpx,"shpy":flow.shpy,"shpz":flow.shpz}
calc_SHAPmean(data_in=Data_shapmean)

# -----------------------------------------------------------------------------------------------------------------------
# Calculate the RMS of the velocity
# -----------------------------------------------------------------------------------------------------------------------
data_rms={"field_ini":field_ini,"field_fin":field_fin,"field_delta":field_delta,"SHAPmean_file":SHAPmean_file,
          "data_folder":data_folder,"file":file_shap,"folder":folder_shap,"dx":dx,"dy":dy,"dz":dz,
          "shpx":flow.shpx,"shpy":flow.shpy,"shpz":flow.shpz,"save_file":save_file,"SHAPrms_file":SHAPrms_file}
calc_rms_nomean(data_in=data_rms)

# -----------------------------------------------------------------------------------------------------------------------
# Calculate the RMS of the velocity
# -----------------------------------------------------------------------------------------------------------------------
data_rms={"field_ini":field_ini,"field_fin":field_fin,"field_delta":field_delta,"SHAPmean_file":SHAPmean_file,
          "data_folder":data_folder,"file":file_shap,"folder":folder_shap,"dx":dx,"dy":dy,"dz":dz,
          "shpx":flow.shpx,"shpy":flow.shpy,"shpz":flow.shpz,"save_file":save_file,"SHAPrms_file":SHAPrms_file}
calc_rms(data_in=data_rms)