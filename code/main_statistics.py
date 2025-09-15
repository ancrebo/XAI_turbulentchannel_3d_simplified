# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
main_statistics.py
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
st_data_str = "stats_data"

# -----------------------------------------------------------------------------------------------------------------------
# Import Packages
# -----------------------------------------------------------------------------------------------------------------------
from py_bin.py_functions.umean import calc_Umean
from py_bin.py_functions.urms import calc_rms
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

# -----------------------------------------------------------------------------------------------------------------------
# Data for the statistics:
#     - field_ini   : index of the initial field
#     - field_fin   : index of the final field
#     - folder      : folder of the flow field data
#     - file        : file of the flow field data
#     - savefile    : boolean to decide if the calculations are stored in a file or variable
#                     (True: file, False: variable)
#     - umean_file  : file to save the mean velocity
#     - data_folder : folder to store the calculated data
#     - dx          : downsampling in x
#     - dy          : downsampling in y
#     - dz          : downsampling in z
#     - L_x         : length of the channel in the streamwise direction
#     - L_y         : half-width of the channel in the wall-normal direction
#     - L_z         : length of the channel in the spanwise direction
#     - urms_file   : file to save the rms of the velocity
#     - unorm_file  : file for saving the normalization
#     - rey         : Friction Reynolds number
#     - utau        : Friction velocity
# -----------------------------------------------------------------------------------------------------------------------
field_ini   = st_data.field_ini
field_fin   = st_data.field_fin
folder      = folders.uvw_folder
file        = folders.uvw_file
save_file   = st_data.save_file
umean_file  = folders.umean_file
data_folder = folders.data_folder
dx          = chd.dx
dy          = chd.dy
dz          = chd.dz
L_x         = chd.L_x
L_y         = chd.L_y
L_z         = chd.L_z
unorm_file  = folders.unorm_file
urms_file   = folders.urms_file
rey         = chd.rey
utau        = chd.utau
mean_norm   = bool(st_data.mean_norm)

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
Data_umean={"field_ini":field_ini,"field_fin":field_fin,"folder":folder,"file":file,"save_file":save_file,
            "umean_file":umean_file,"data_folder":data_folder,"shpx":flow.shpx,"shpy":flow.shpy,"shpz":flow.shpz}
calc_Umean(data_in=Data_umean)

# -----------------------------------------------------------------------------------------------------------------------
# Calculate the RMS of the velocity
# -----------------------------------------------------------------------------------------------------------------------
data_rms={"field_ini":field_ini,"field_fin":field_fin,"umean_file":umean_file,"data_folder":data_folder,
          "file":file,"folder":folder,"dx":dx,"dy":dy,"dz":dz,"shpx":flow.shpx,"shpy":flow.shpy,"shpz":flow.shpz,
          "save_file":save_file,"urms_file":urms_file}
calc_rms(data_in=data_rms)

# -----------------------------------------------------------------------------------------------------------------------
# Calculate the normalization values for the dataset
# -----------------------------------------------------------------------------------------------------------------------
data_norm={"field_ini":field_ini,"field_fin":field_fin,"data_folder":data_folder,"umean_file":umean_file,
            "dx":dx,"dy":dy,"dz":dz,"folder":folder,"file":file,"shpx":flow.shpx,"shpy":flow.shpy,"shpz":flow.shpz,
            "save_file":save_file,"unorm_file":unorm_file}
calc_norm(data_in=data_norm)