# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plot_predictions.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 13:38:39 2024

@author: Andres Cremades Botella

Create the plot of the training field. The file requires to set the following paths:
    - folder_def  : (str) name of the folder containing the files for configuring the case of analysis.
    - chd_str     : (str) name of the file containing the data of the channel.
    - folders_str : (str) name of the file containing the folders and files used in the problem.
    - tr_data_str : (str) name of the file containing the information required for the training.
In addition the following variables need to be set:
    - xlabel      : label of the x axis
    - ylabel      : label of the y axis
    - titles      : title of each subfigure
    - fontsize    : size of the text in the figure
    - figsize_x   : size of the figure in axis x
    - figsize_y   : size of the figure in axis y
    - colormap    : colormap used in the plot
    - colornum    : number of levels required in the colormap
    - fig_name    : name of the figure after saving
    - dpi         : dots per inch of the figure
"""
# -----------------------------------------------------------------------------------------------------------------------
# Import packages
# -----------------------------------------------------------------------------------------------------------------------
import numpy as np
from py_bin.py_functions.read_velocity import read_velocity
from py_bin.py_plots.plotshap import plotshap_noise
import py_bin.py_class.shap_config as sc
from py_bin.py_class.flow_field import flow_field
import os
from py_bin.py_functions.normalization import read_norm

# -----------------------------------------------------------------------------------------------------------------------
# Unlock the h5 files for avoiding problems in some clusters
# -----------------------------------------------------------------------------------------------------------------------
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# -----------------------------------------------------------------------------------------------------------------------
# Define the names of the files containing the definitios of the parameters
# - folder_def : folder containing the files with the definitions required in the problem
# - folders    : file containing the folder and file structures
# -----------------------------------------------------------------------------------------------------------------------
folder_def  = "configuration"
chd_str     = "channel_data"
folders_str = "folders"
tr_data_str = "training_data"
sh_data_str = "shap_data"

# -----------------------------------------------------------------------------------------------------------------------
# Define the variables required for the plot
#     - xlabel      : label of the x axis
#     - ylabel      : label of the y axis
#     - titles      : title of each subfigure
#     - fontsize    : size of the text in the figure
#     - figsize_x   : size of the figure in axis x
#     - figsize_y   : size of the figure in axis y
#     - colormap    : colormap used in the plot
#     - colornum    : number of levels required in the colormap
#     - fig_name    : name of the figure after saving
#     - dpi         : dots per inch of the figure
#     - index_ii    : index of the field
#     - index_y     : index of the wall distance
#     - ymin        : minimum y
#     - ymax        : maximum y
#     - xmin        : minimum x
#     - xmax        : maximum x
#     - errmax      : maximum error in the colorbar
#     - errmin      : minimum error in the colorbar
#     - b_velo_sim  : bar text for the simulated velocity
#     - b_velo_pred : bar text for the predicted velocity
#     - b_velo_err  : bar text for the error
#     - scale_shap  : factor to scale the shap values
#     - colormapsh  : colormap for the shaps
# -----------------------------------------------------------------------------------------------------------------------
fields      = ""
nfield      = "11 field for computing SHAP"
xlabel      = "$x^+$"
ylabel      = "$z^+$"
titles      = ["$u^+$","$\phi$"]
fontsize    = 18
figsize_x   = 9
figsize_y   = 8
colormap    = "viridis"
colornum    = 3
fig_name    = "shapfield_noise"+fields
dpi         = 200
index_ii    = 26000
index_y     = 20
ymin        = 0
ymax        = 3*np.pi
xmin        = 0
xmax        = 8*np.pi
errmax      = 0.1
errmin      = 0
b_velo_u    = "$u^+$"
b_shap_u    = "$\phi_u$"
b_velo_v    = "$v^+$"
b_shap_v    = "$\phi_v$"
b_velo_w    = "$w^+$"
b_shap_w    = "$\phi_w$"
b_velo_m    = "$|u|^+$"
b_shap_m    = "$|\phi|$"
scale_shap  = 1e5
colormapsh  = 'PRGn'


   
# -----------------------------------------------------------------------------------------------------------------------
# Read the training data file
# -----------------------------------------------------------------------------------------------------------------------
exec("from "+folder_def+" import "+chd_str+" as chd")
exec("from "+folder_def+" import "+folders_str+" as folders")
exec("from "+folder_def+" import "+tr_data_str+" as tr_data")
exec("from "+folder_def+" import "+sh_data_str+" as sh_data")

# -----------------------------------------------------------------------------------------------------------------------
# Load the channel data to import the information regarding the channel size and the friction Reynolds number 
# and velocity
#     - L_x     : Channel size in the streamwise direction
#     - L_z     : Channel size in the spanwise direction
#     - L_y     : Half of the channel width
#     - rey     : Friction Reynolds number
#     - utau    : Friction velocity
#     - dx      : Downsampling in x
#     - dy      : Downsampling in y
#     - dz      : Downsampling in z
#     - padding : Number of nodes of the padding
# -----------------------------------------------------------------------------------------------------------------------
L_x     = chd.L_x
L_z     = chd.L_z
L_y     = chd.L_y
rey     = chd.rey
utau    = chd.utau
dx      = chd.dx
dy      = chd.dy
dz      = chd.dz
padding = chd.padding

# -----------------------------------------------------------------------------------------------------------------------
# Define the data of the model definition: data, padding, downsampling...
#     - uvw_folder      : Folder of the velocity data
#     - uvw_file        : This file does not contain the file index
#     - data_folder     : Folder for storing the data of the model
#     - umean_file      : File for the mean velocity
#     - unorm_file      : File for the normalization of the velocity
#     - umax_file       : File containing the maximum velocity
# -----------------------------------------------------------------------------------------------------------------------
uvw_folder  = folders.uvw_folder
uvw_file    = folders.uvw_file
shap_folder = folders.shap_folder+fields+"_real"
shap_file   = folders.shap_file
data_folder = folders.data_folder
umean_file  = folders.umean_file
unorm_file  = folders.unorm_file
umax_file   = folders.umax_file
    
# -----------------------------------------------------------------------------------------------------------------------
# Read the channel characteristics
# -----------------------------------------------------------------------------------------------------------------------
Data_flow={"folder":uvw_folder,"file":uvw_file,"down_x":dx,"down_y":dy,
           "down_z":dz,"L_x":L_x,"L_y":L_y,"L_z":L_z,"rey":rey,"utau":utau,"umax_file":umax_file}
flowfield = flow_field(data_in=Data_flow)
flowfield.shape_tensor()
flowfield.flow_grid()

# -----------------------------------------------------------------------------------------------------------------------
# Read the velocity
# -----------------------------------------------------------------------------------------------------------------------
velocity = read_velocity(data_in={"folder":uvw_folder,"file":uvw_file,"index":index_ii,"dx":dx,"dy":dy,"dz":dz,
                                  "shpx":flowfield.shpx,"shpy":flowfield.shpy,"shpz":flowfield.shpz,
                                  "padding":padding,"data_folder":data_folder,"umean_file":umean_file})

# -----------------------------------------------------------------------------------------------------------------------
# Define the data for the shap.
#     - ngpu          : Number of gpus
#     - field_ini     : Initial field of the shaps
#     - field_fin     : Final field of the shaps
#     - field_delta   : Separation between the files
#     - read_model    : Flag to define or read the model (False=define, True=read)
#     - model_folder  : Folder of the trained model files
#     - model_name    : Name of the trained model file
#     - nfil          : Number of filters of the first layer of the Unet
#     - stride        : Stride of the Unet
#     - activation    : Activation function
#     - kernel        : Kernel size of the unet
#     - pooling       : Size of the poolings of the unet
#     - delta_pred    : Number of fields to advance the prediction
#     - data_type     : Format of the data of the training
#     - error_file    : file to store the error
#     - umax_file     : file to store the maximum and minimum velocity
#     - urmspred_file : file to store the rms predicted by the model 
#     - nrep_field      : number of repetitions of each field for calculating the SHAP values
#     - shap_batch      : batch size used for the shap
# -----------------------------------------------------------------------------------------------------------------------
ngpu            = tr_data.ngpu
field_ini       = sh_data.field_ini
field_fin       = sh_data.field_fin
field_delta     = sh_data.field_delta
read_model      = tr_data.read_model
model_folder    = folders.model_folder
model_read      = folders.model_read
nfil            = tr_data.nfil
stride          = tr_data.stride
activation      = tr_data.activation
kernel          = tr_data.kernel
pooling         = tr_data.pooling
delta_pred      = tr_data.delta_pred
nsamples        = sh_data.nsamples
nsamples_max    = sh_data.nsamples_max
data_type       = tr_data.data_type
error_file      = folders.error_file
umax_file       = folders.umax_file
urmspred_file   = folders.urmspred_file
mean_norm       = tr_data.mean_norm
tfrecord_folder = folders.tfrecord_folder
nrep_field      = sh_data.nrep_field
shap_batch      = sh_data.shap_batch
repeat_exist    = sh_data.repeat_exist

# -----------------------------------------------------------------------------------------------------------------------
# Define dict containing the information needed for the shap model
# -----------------------------------------------------------------------------------------------------------------------
data_shap = {"shap_folder":shap_folder,"shap_file":shap_file,"uvw_folder":uvw_folder,"uvw_file":uvw_file,
             "padding":padding,"dx":dx,"dy":dy,"dz":dz,"data_folder":data_folder,"umean_file":umean_file,
             "unorm_file":unorm_file,"L_x":L_x,"L_z":L_z,"L_y":L_y,"rey":rey,"utau":utau,"ngpu":ngpu,
             "field_ini":field_ini,"field_fin":field_fin,"field_delta":field_delta,"model_folder":model_folder,
             "model_read":model_read,"nfil":nfil,"stride":stride,"activation":activation,"kernel":kernel,
             "pooling":pooling,"delta_pred":delta_pred,"nsamples":nsamples,"nsamples_max":nsamples,
             "data_type":data_type,"error_file":error_file,"umax_file":umax_file,"urmspred_file":urmspred_file,
             "mean_norm":mean_norm,"tfrecord_folder":tfrecord_folder,"nrep_field":nrep_field,"shap_batch":shap_batch,
             "repeat_exist":repeat_exist,"flag_model":False}
shap_model = sc.shap_config(data_in=data_shap)
shap_data  = shap_model.read_shap(data_in = {"index":index_ii})

# -----------------------------------------------------------------------------------------------------------------------
# Read the velocity
# -----------------------------------------------------------------------------------------------------------------------
data_velocity = {"folder":uvw_folder,"file":uvw_file,"index":index_ii,"dx":dx,"dy":dy,"dz":dz,"shpx":flowfield.shpx,
                 "shpy":flowfield.shpy,"shpz":flowfield.shpz,"padding":padding,"data_folder":data_folder,
                 "umean_file":umean_file}
vel_data      = read_velocity(data_in=data_velocity)


# -----------------------------------------------------------------------------------------------------------------------
# Define the information of the data read from the files
#     - plot_folder : folder to save the figures
# -----------------------------------------------------------------------------------------------------------------------
plot_folder = folders.plot_folder

# -----------------------------------------------------------------------------------------------------------------------
# Plot the shap values
# -----------------------------------------------------------------------------------------------------------------------
xmin *= flowfield.rey
xmax *= flowfield.rey
ymin *= flowfield.rey
ymax *= flowfield.rey
data_plot={"plot_folder":plot_folder,"xlabel":xlabel,"ylabel":ylabel,"fontsize":fontsize,"figsize_x":figsize_x,
           "figsize_y":figsize_y,"colormap":colormap,"colornum":colornum,"fig_name":fig_name,"dpi":dpi,
           "index_ii":index_ii,"shap_data":shap_data,"vel_data":vel_data,"flowfield":flowfield,"index_y":index_y,
           "xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"b_velo_u":b_velo_u,"b_shap_u":b_shap_u,
           "b_velo_v":b_velo_v,"b_shap_v":b_shap_v,"b_velo_w":b_velo_w,"b_shap_w":b_shap_w,"b_velo_m":b_velo_m,
           "b_shap_m":b_shap_m,"padding":padding,"scale_shap":scale_shap,"colormapsh":colormapsh,"normfield":False,
           "nfield":nfield}
plotshap_noise(data_in=data_plot)