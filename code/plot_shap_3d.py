# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plot_shap_3d.py
-------------------------------------------------------------------------------------------------------------------------
Created on Tue Jun 18 11:30:49 2024

@author: Andres Cremades Botella

Function to plot the 3d Reynolds stress structures:
    - folder_def  : (str) name of the folder containing the files for configuring the case of analysis.
    - chd_str     : (str) name of the file containing the data of the channel.
    - folders_str : (str) name of the file containing the folders and files used in the problem.
    - st_data_str : (str) name of the file containing the information required for the statistics.
For more information about the tangential Reynolds stress structures:
    - Lozano-Durán, A., Flores, O., & Jiménez, J. (2012). The three-dimensional structure of momentum transfer in
      turbulent channels. Journal of Fluid Mechanics, 694, 100-130.
"""
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
xlabel      = "$x^+$"
ylabel      = "$z^+$"
zlabel      = "$y^+$"
fontsize    = 14
figsize_x   = 8
figsize_y   = 7
colormap    = "viridis"
colornum    = 4
fig_name    = "shap3d_83pi"
dpi         = 200
padtext_x   = 50
padtext_y   = 10
padtext_z   = 7

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
sh_data_str = "shap_data"
tr_data_str = "training_data"

# -----------------------------------------------------------------------------------------------------------------------
# Import packages
# -----------------------------------------------------------------------------------------------------------------------
from py_bin.py_class.shap_structure import shap_structure
import os
from py_bin.py_plots.plotstruc3d import plotstruc3d, plotstruc3d_separe

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
exec("from "+folder_def+" import "+tr_data_str+" as tr_data")


# -----------------------------------------------------------------------------------------------------------------------
# Data for the statistics:
#     - index        : index of the field
#     - Hperc        : percolation index
#     - uvw_folder   : folder of the flow field data
#     - uvw_file     : file of the flow field data
#     - umean_file   : file to save the mean velocity
#     - data_folder  : folder to store the calculated data
#     - dx           : downsampling in x
#     - dy           : downsampling in y
#     - dz           : downsampling in z
#     - L_x          : length of the channel in the streamwise direction
#     - L_y          : half-width of the channel in the wall-normal direction
#     - L_z          : length of the channel in the spanwise direction
#     - urms_file    : file to save the rms of the velocity
#     - rey          : Friction Reynolds number
#     - utau         : Friction velocity
#     - padding      : padding of the flow field
#     - sym_quad     : flag for using the symmetry in the direction 2 of the field for the quadrant selection
#     - filvol       : volume for filtering the structures+
#     - shap_folder  : folder of the shap values
#     - shap_folder  : file of the shap values
#     - SHAPq_folder : folder of the shap structures
#     - SHAPq_file   : file of the uv structures
#     - padding      : padding of the field
#     - data_type    : type of data used by the model
#     - nsamples     : number of samples of the shap calculation
#     - SHAPrms_file : file of the rms of the shap
# -----------------------------------------------------------------------------------------------------------------------
index        = 26000
Hperc        = 1.18
uvw_folder   = folders.uvw_folder
uvw_file     = folders.uvw_file
umean_file   = folders.umean_file
data_folder  = folders.data_folder
dx           = chd.dx
dy           = chd.dy
dz           = chd.dz
L_x          = chd.L_x
L_y          = chd.L_y
L_z          = chd.L_z
urms_file    = folders.urms_file
rey          = chd.rey
utau         = chd.utau
padding      = chd.padding
sym_quad     = True
filvol       = chd.filvol
shap_folder  = folders.shap_folder
shap_file    = folders.shap_file
SHAPq_folder = folders.SHAPq_folder+"_real"
SHAPq_file   = folders.SHAPq_file
padding      = chd.padding
data_type    = tr_data.data_type
plot_folder  = folders.plot_folder
nsamples     = sh_data.nsamples
SHAPrms_file = folders.SHAPrms_file

# -----------------------------------------------------------------------------------------------------------------------
# Create the data of the structure
# -----------------------------------------------------------------------------------------------------------------------
shap_struc = shap_structure(data_in={"uvw_folder":uvw_folder,"uvw_file":uvw_file,"Hperc":Hperc,"index":index,"dx":dx,
                                     "dy":dy,"dz":dz,"L_x":L_x,"L_y":L_y,"L_z":L_z,"rey":rey,"utau":utau,
                                     "padding":padding,"data_folder":data_folder,"umean_file":umean_file,
                                     "urms_file":urms_file,"sym_quad":True,"filvol":filvol,"shap_folder":shap_folder,
                                     "shap_file":shap_file,"folder":SHAPq_folder,"file":SHAPq_file,"padding":padding,
                                     "data_type":data_type,"nsamples":nsamples,"SHAPrms_file":SHAPrms_file})
shap_struc.read_struc()

# -----------------------------------------------------------------------------------------------------------------------
# Plot the 3D field
# -----------------------------------------------------------------------------------------------------------------------
plotstruc3d(data_in={"struc":shap_struc,"plot_folder":plot_folder,"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,
                      "fontsize":fontsize,"figsize_x":figsize_x,"figsize_y":figsize_y,"colormap":colormap,
                      "colornum":colornum,"fig_name":fig_name,"dpi":dpi,"dy":dy,"dx":dx,"dz":dz,
                      "uvw_folder":uvw_folder,"uvw_file":uvw_file,"L_x":L_x,"L_y":L_y,"L_z":L_z,
                      "rey":rey,"utau":utau,"cmap_flag":True,"padtext":[padtext_x,padtext_y,padtext_z],"savepdf":False})
plotstruc3d_separe(data_in={"struc":shap_struc,"plot_folder":plot_folder,"xlabel":xlabel,"ylabel":ylabel,
                            "zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,"figsize_y":figsize_y,
                            "colormap":colormap,"colornum":colornum,"fig_name":fig_name,"dpi":dpi,"dy":dy,
                            "dx":dx,"dz":dz,"uvw_folder":uvw_folder,"uvw_file":uvw_file,"L_x":L_x,"L_y":L_y,
                            "L_z":L_z,"rey":rey,"utau":utau,"cmap_flag":True,"padtext":[padtext_x,padtext_y,padtext_z],
                            "savepdf":False})