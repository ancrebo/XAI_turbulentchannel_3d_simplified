# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plot_coinc_pairs_uv_streak_chong_2_shap.py
-------------------------------------------------------------------------------------------------------------------------
Created on Mon Sep 30 19:14:16 2024

@author: Andres Cremades Botella

Function to plot the coincidence of the pairs uv-shap, streak-shap and chong-shap structures.
Requieres to set the following parameters
    - folder_def  : (str) name of the folder containing the files for configuring the case of analysis.
    - chd_str     : (str) name of the file containing the data of the channel.
    - folders_str : (str) name of the file containing the folders and files used in the problem.
    - st_data_str : (str) name of the file containing the information required for the statistics.
In addition the following variables need to be set:
    - fig_xlabel      : label of the x axis for the figure 
    - fig_ylabel      : label of the y axis for the figure 
    - fig_zlabel      : label of the z axis for the figure 
    - fig_fontsize    : size of the text in the figure 
    - fig_figsize_x   : size of the figure in axis x
    - fig_figsize_y   : size of the figure in axis y
    - fig_colormap    : colormap used in the figure 
    - fig_colornum    : number of levels required in the colormap of figure 
    - fig_name        : name of the figure after saving
    - fig_dpi         : dots per inch of the figure 
"""
# -----------------------------------------------------------------------------------------------------------------------
# Define the variables required for the plot
#     - fig_xlabel      : label of the x axis in figure 4
#     - fig_ylabel      : label of the y axis in figure 4
#     - fig_ylabel2     : label of the second y axis in figure 4
#     - fig_fontsize    : size of the text in the figure 4
#     - fig_figsize_x   : size of the figure 4 in axis x
#     - fig_figsize_y   : size of the figure 4 in axis y
#     - fig_colormap    : colormap used in the plot in figure 4
#     - fig_colornum    : number of levels required in the colormap in figure 4
#     - fig_name        : name of the figure 4 after saving
#     - fig_dpi         : dots per inch of the figure 4
#     - fig_struc1_lab  : label of the structure 1 of the figure 4
#     - fig_struc2_lab  : label of the structure 2 of the figure 4
# -----------------------------------------------------------------------------------------------------------------------
fig_xlabel      = "$y^+$"
fig_ylabel      = "$V/V_{tot}$"
fig_ylabel2     = "Coincidence (%)"
fig_fontsize    = 24
fig_figsize_x   = 16
fig_figsize_y   = 5
fig_colormap    = "viridis"
fig_colornum    = 4
fig_name        = "coinc_pairs_uv_streak_chong_2_shap"
fig_dpi         = 400
fig_struc1a_lab = "SHAP"
fig_struc1b_lab = "Qs"
fig_struc2a_lab = "SHAP"
fig_struc2b_lab = "Streaks"
fig_struc3a_lab = "SHAP"
fig_struc3b_lab = "Vortices"
fig_linewidth   = 3



# -----------------------------------------------------------------------------------------------------------------------
# Define the names of the files containing the definitios of the parameters
# - folder_def  : folder containing the files with the definitions required in the problem
# - chd_str     : file containing the data of the channel
# - folders     : file containing the folder and file structures
# - st_data_str : file containing the data of the statistics
# -----------------------------------------------------------------------------------------------------------------------
folder_def  = "configuration"
chd_str     = "channel_data"
folders_str = "folders"
st_data_str = "stats_data_shap"

# -----------------------------------------------------------------------------------------------------------------------
# Import packages
# -----------------------------------------------------------------------------------------------------------------------
from py_bin.py_class.shap_structure import shap_structure
from py_bin.py_class.uv_structure import uv_structure
from py_bin.py_class.streak_structure import streak_structure
from py_bin.py_class.chong_structure import chong_structure
import os
import matplotlib
from py_bin.py_plots.plot_histuvw_y import plot_histuvw_y
from py_bin.py_plots.plotstruc3d import plotstruc3d, plotstruc3d_separe
from py_bin.py_class.flow_field import flow_field
from py_bin.py_plots.plot_coinc import plot_coinc_3coinc
from py_bin.py_functions.read_velocity import read_velocity
import numpy as np

try:
    os.mkdir(plot_folder)
except:
    print("Folder is created",flush=True)

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
# Define the information of the data
#     - data_folder : folder containing the training data
#     - plot_folder : folder to save the figures
#     - file        : name of the file containing the mean velocity information
#     - dy          : downsampling of the wall-normal direction
#     - dx          : downsampling of the streamwise direction
#     - dz          : downsampling of the spanwise direction
#     - uvw_folder  : folder of the flow fields
#     - uvw_file    : file of the flow fields
#     - L_x         : dimension of the channel in the streamwise direction
#     - L_y         : dimension of the channel in the wall-normal direction
#     - L_z         : dimension of the channel in the spanwise direction
#     - file_trj    : file containing the data of Torroja
# -----------------------------------------------------------------------------------------------------------------------
data_folder = folders.data_folder
plot_folder = folders.plot_folder
file_uv     = folders.uv_shap_file
file_streak = folders.streak_shap_file
file_chong  = folders.chong_shap_file

# -----------------------------------------------------------------------------------------------------------------------
# Create the plot
# -----------------------------------------------------------------------------------------------------------------------
plot_format_data = {"file_1":file_uv,"file_2":file_streak,"file_3":file_chong,"folder":data_folder,
                    "plot_folder":plot_folder,"xlabel":fig_xlabel,"ylabel":fig_ylabel,"ylabel2":fig_ylabel2,
                    "fontsize":fig_fontsize,"figsize_x":fig_figsize_x,"figsize_y":fig_figsize_y,
                    "colormap":fig_colormap,"colornum":fig_colornum,"fig_name":fig_name,"dpi":fig_dpi,
                    "struc1a_lab":fig_struc1a_lab,"struc1b_lab":fig_struc1b_lab,"struc2a_lab":fig_struc2a_lab,
                    "struc2b_lab":fig_struc2b_lab,"struc3a_lab":fig_struc3a_lab,"struc3b_lab":fig_struc3b_lab,
                    "linewidth":fig_linewidth}
plot_coinc_3coinc(data_in=plot_format_data)