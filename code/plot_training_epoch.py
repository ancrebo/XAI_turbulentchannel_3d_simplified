# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plot_training_epoch.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 13:38:39 2024

@author: Andres Cremades Botella

Create the plot of the training epochs. The file requires to set the following paths:
    - folder_def  : (str) name of the folder containing the files for configuring the case of analysis.
    - folders_str : (str) name of the file containing the folders and files used in the problem.
In addition the following variables need to be set:
    - xlabel      : label of the x axis
    - ylabel      : label of the y axis
    - fontsize    : size of the text in the figure
    - figsize_x   : size of the figure in axis x
    - figsize_y   : size of the figure in axis y
    - colormap    : colormap used in the plot
    - colornum    : number of levels required in the colormap
    - fig_name    : name of the figure after saving
    - dpi         : dots per inch of the figure
"""
# -----------------------------------------------------------------------------------------------------------------------
# Define the names of the files containing the definitios of the parameters
# - folder_def : folder containing the files with the definitions required in the problem
# - folders    : file containing the folder and file structures
# -----------------------------------------------------------------------------------------------------------------------
folder_def  = "configuration"
folders_str = "folders"

# -----------------------------------------------------------------------------------------------------------------------
# Define the variables required for the plot
#     - xlabel      : label of the x axis
#     - ylabel      : label of the y axis
#     - fontsize    : size of the text in the figure
#     - figsize_x   : size of the figure in axis x
#     - figsize_y   : size of the figure in axis y
#     - colormap    : colormap used in the plot
#     - colornum    : number of levels required in the colormap
#     - fig_name    : name of the figure after saving
#     - dpi         : dots per inch of the figure
# -----------------------------------------------------------------------------------------------------------------------
xlabel      = "Epoch"
ylabel      = "Loss function (-)"
fontsize    = 18
figsize_x   = 7
figsize_y   = 5
colormap    = "viridis"
colornum    = 3
fig_name    = "training_info"
dpi         = 200

# -----------------------------------------------------------------------------------------------------------------------
# Import packages
# -----------------------------------------------------------------------------------------------------------------------
from py_bin.py_plots.plottrain import plottrain
import os

# ----------------------------------------------------------------------------------------------------------------------
# Unlock the h5 files for avoiding problems in some clusters
# ----------------------------------------------------------------------------------------------------------------------
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# ----------------------------------------------------------------------------------------------------------------------
# Import information files
# ----------------------------------------------------------------------------------------------------------------------
exec("from "+folder_def+" import "+folders_str+" as folders")

# -----------------------------------------------------------------------------------------------------------------------
# Define the information of the data read from the files
#     - data_folder : folder containing the training data
#     - plot_folder : folder to save the figures
#     - file        : name of the file containing the training data
# -----------------------------------------------------------------------------------------------------------------------
data_folder = folders.data_folder
plot_folder = folders.plot_folder
file        = folders.hist_file

# -----------------------------------------------------------------------------------------------------------------------
# Create the plot
# -----------------------------------------------------------------------------------------------------------------------
plot_format_data = {"file":file,"folder":data_folder,"plot_folder":plot_folder,"xlabel":xlabel,\
                    "ylabel":ylabel,"fontsize":fontsize,"figsize_x":figsize_x,"figsize_y":figsize_y,\
                        "colormap":colormap,"colornum":colornum,"fig_name":fig_name,"dpi":dpi}
plottrain(plot_format_data=plot_format_data)