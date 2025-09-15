# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plot_histuvw_shap_y.py
-------------------------------------------------------------------------------------------------------------------------
Created on Tue Jun 18 11:30:49 2024

@author: Andres Cremades Botella

Function to calculate the coincidence between the uv the shap in the whole domain:
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
tr_data_str = "training_data"

# -----------------------------------------------------------------------------------------------------------------------
# Import packages
# -----------------------------------------------------------------------------------------------------------------------
from py_bin.py_class.shap_structure import shap_structure
import os
from py_bin.py_plots.plot_histuvw_y import plot_histuvw_y
from py_bin.py_class.flow_field import flow_field
from py_bin.py_functions.read_velocity import read_velocity
import numpy as np

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
# Define the variables required for the plot
#     - xlabel                   : label of the x axis
#     - ylabelu                  : label of the y axis for the streamwise velocity
#     - ylabelv                  : label of the y axis for the wall-normal velocity
#     - ylabelw                  : label of the y axis for the spanwise velocity
#     - fontsize                 : size of the text in the figure
#     - figsize_x                : size of the figure in axis x
#     - figsize_y                : size of the figure in axis y
#     - colormap                 : colormap used in the plot
#     - colornum                 : number of levels required in the colormap
#     - fig_name                 : name of the figure after saving
#     - dpi                      : dots per inch of the figure
#     - struc1_lab               : label of the structure 1
#     - struc2_lab               : label of the structure 2
#     - plot_coin_folder_uv_shap : folder to save the coincidence plots between uv and shap structures
#     - plot_coin_file_uv_shap   : folder to save the coincidence plots between uv and shap structures
# -----------------------------------------------------------------------------------------------------------------------
ylabel           = "$y^+$"
xlabelu          = "$u^+$"
xlabelv          = "$v^+$"
xlabelw          = "$w^+$"
fontsize         = 24
figsize_x        = 7
figsize_y        = 6
colormap         = "viridis"
colornum         = 4
dpi              = 400
plot_fileu       = "hist_uy_shap_83pi"
plot_filev       = "hist_vy_shap_83pi"
plot_filew       = "hist_wy_shap_83pi"
bins             = 100
lev_min          = 1e-3
lev_delta        = None
linewidth        = 3
umin             = -8.5
umax             = 8.5
vmin             = -4
vmax             = 4
wmin             = -4.5
wmax             = 4.5

# -----------------------------------------------------------------------------------------------------------------------
# Data for the statistics:
#     - index         : index of the field
#     - Hperc         : percolation index
#     - uvw_folder    : folder of the flow field data
#     - uvw_file      : file of the flow field data
#     - umean_file    : file to save the mean velocity
#     - data_folder   : folder to store the calculated data
#     - dx            : downsampling in x
#     - dy            : downsampling in y
#     - dz            : downsampling in z
#     - L_x           : length of the channel in the streamwise direction
#     - L_y           : half-width of the channel in the wall-normal direction
#     - L_z           : length of the channel in the spanwise direction
#     - urms_file     : file to save the rms of the velocity
#     - rey           : Friction Reynolds number
#     - utau          : Friction velocity
#     - padding       : padding of the flow field
#     - sym_quad      : flag for using the symmetry in the direction 2 of the field for the quadrant selection
#     - filvol        : volume for filtering the structures+
#     - shap_folder   : folder of the shap values
#     - shap_folder   : file of the shap values
#     - padding       : padding of the field
#     - data_type     : type of data used by the model
#     - SHAPq_folder  : folder of the shap structures
#     - SHAPq_file    : file of the uv structures
#     - nsamples      : number of samples of the shap calculation
#     - SHAPrms_file  : file of the rms of the shap
# -----------------------------------------------------------------------------------------------------------------------
index_ini        = st_data.field_ini
index_fin        = st_data.field_fin
index_delta      = st_data.field_delta
Hperc            = 1.41
uvw_folder       = folders.uvw_folder
uvw_file         = folders.uvw_file
umean_file       = folders.umean_file
data_folder      = folders.data_folder
dx               = chd.dx
dy               = chd.dy
dz               = chd.dz
L_x              = chd.L_x
L_y              = chd.L_y
L_z              = chd.L_z
urms_file        = folders.urms_file
rey              = chd.rey
utau             = chd.utau
padding          = chd.padding
sym_quad         = True
filvol           = chd.filvol
shap_folder      = folders.shap_folder
shap_file        = folders.shap_file
padding          = chd.padding
data_type        = tr_data.data_type
plot_folder      = folders.plot_folder
SHAPq_folder     = folders.SHAPq_folder+"_real"
SHAPq_file       = folders.SHAPq_file
nsamples         = sh_data.nsamples
SHAPrms_file     = folders.SHAPrms_file
streak_shap_file = folders.streak_shap_file
umax_file        = folders.umax_file


# -----------------------------------------------------------------------------------------------------------------------
# Read the channel characteristics
# -----------------------------------------------------------------------------------------------------------------------
Data_flow = {"folder":uvw_folder,"file":uvw_file,"down_x":dx,"down_y":dy,
             "down_z":dz,"L_x":L_x,"L_y":L_y,"L_z":L_z,"rey":rey,"utau":utau,"umax_file":umax_file}
flowfield = flow_field(data_in=Data_flow)
flowfield.shape_tensor()
flowfield.flow_grid()

# -----------------------------------------------------------------------------------------------------------------------
# Create the data of the shap structure and read where the structures exist
# -----------------------------------------------------------------------------------------------------------------------
shap_data  = {"uvw_folder":uvw_folder,"uvw_file":uvw_file,"Hperc":Hperc,"index":0,"dx":dx,
              "dy":dy,"dz":dz,"L_x":L_x,"L_y":L_y,"L_z":L_z,"rey":rey,"utau":utau,
              "padding":padding,"data_folder":data_folder,"umean_file":umean_file,
              "urms_file":urms_file,"sym_quad":True,"filvol":filvol,"shap_folder":shap_folder,
              "shap_file":shap_file,"folder":SHAPq_folder,"file":SHAPq_file,"padding":padding,
              "data_type":data_type,"nsamples":nsamples,"SHAPrms_file":SHAPrms_file}
velo_data  = {"folder":uvw_folder,"file":uvw_file,"index":0,"dx":dx,"dy":dy,"dz":dz,
              "shpx":flowfield.shpx,"shpy":flowfield.shpy,"shpz":flowfield.shpz,
              "padding":0,"data_folder":data_folder,"umean_file":umean_file}



# -----------------------------------------------------------------------------------------------------------------------
# For all the fields
# -----------------------------------------------------------------------------------------------------------------------
index_range = range(index_ini,index_fin,index_delta)
for ii in index_range:  
    shap_data["index"] = ii
    velo_data["index"] = ii
    # -----------------------------------------------------------------------------------------------------------------------
    # Read the velocity
    # -----------------------------------------------------------------------------------------------------------------------
    shap_struc    = shap_structure(data_in=shap_data)
    shap_struc.read_struc()
    mat_struc     = shap_struc.mat_struc
    velocity_data = read_velocity(data_in=velo_data)
    uu            = velocity_data["uu"]/utau
    vv            = velocity_data["vv"]/utau
    ww            = velocity_data["ww"]/utau
    
    vv[flowfield.yu_s:,:,:] *= -1 
    ww[flowfield.yu_s:,:,:] *= -1 
    
    # -----------------------------------------------------------------------------------------------------------------------
    # Velocities of the structues
    # -----------------------------------------------------------------------------------------------------------------------
    index_st    = np.where(mat_struc==1)
    if ii == index_range[0]:
        uu_struc    = uu[index_st]
        vv_struc    = vv[index_st]
        ww_struc    = ww[index_st]
        y_h_struc   = flowfield.y_h[index_st[0]]
        yplus_struc = (1-abs(y_h_struc))*rey
    else:
        uu_struc    = np.concatenate((uu_struc,uu[index_st]))
        vv_struc    = np.concatenate((vv_struc,vv[index_st]))
        ww_struc    = np.concatenate((ww_struc,ww[index_st]))
        y_h_struc   = flowfield.y_h[index_st[0]]
        yplus_struc = np.concatenate((yplus_struc,(1-abs(y_h_struc))*rey))

# -----------------------------------------------------------------------------------------------------------------------
# Plot the data
# -----------------------------------------------------------------------------------------------------------------------
plot_format_data = {"plot_folder":plot_folder,"plot_fileu":plot_fileu,"plot_filev":plot_filev,"plot_filew":plot_filew,
                    "ylabel":ylabel,"xlabelu":xlabelu,"xlabelv":xlabelv,"xlabelw":xlabelw,"fontsize":fontsize,
                    "figsize_x":figsize_x,"figsize_y":figsize_y,"colormap":colormap,"colornum":colornum,"dpi":dpi,
                    "uu_struc":uu_struc,"vv_struc":vv_struc,"ww_struc":ww_struc,"yplus_struc":yplus_struc,
                    "yplusmesh":flowfield.yplus,"bins":bins,"lev_min":lev_min,"lev_delta":lev_delta,
                    "linewidth":linewidth,"umin":umin,"umax":umax,"vmin":vmin,"vmax":vmax,"wmin":wmin,"wmax":wmax}
plot_histuvw_y(data_in=plot_format_data)