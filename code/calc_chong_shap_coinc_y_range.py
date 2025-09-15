# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
calc_uv_shap_coinc_y.py
-------------------------------------------------------------------------------------------------------------------------
Created on Tue Jun 18 11:30:49 2024

@author: Andres Cremades Botella

Function to calculate the volume occupied by the vortices the shap and the coincidence as a function of y:
    - folder_def  : (str) name of the folder containing the files for configuring the case of analysis.
    - chd_str     : (str) name of the file containing the data of the channel.
    - folders_str : (str) name of the file containing the folders and files used in the problem.
    - st_data_str : (str) name of the file containing the information required for the statistics.
    - sh_data_str : file containing the data of the shap values
    - tr_data_str : file containing the trainin`g data
    - saveh5      : save a file with the data in case the calculation stops
"""
# -----------------------------------------------------------------------------------------------------------------------
# Define the names of the files containing the definitios of the parameters
# - folder_def  : folder containing the files with the definitions required in the problem
# - chd_str     : file containing the data of the channel
# - folders     : file containing the folder and file structures
# - st_data_str : file containing the data of the statistics
# - sh_data_str : file containing the data of the shap values
# - tr_data_str : file containing the trainin`g data
# - saveh5      : save a file with the data in case the calculation stops
# -----------------------------------------------------------------------------------------------------------------------
folder_def  = "configuration"
chd_str     = "channel_data"
folders_str = "folders"
st_data_str = "stats_data_shap"
sh_data_str = "shap_data"
tr_data_str = "training_data"
saveh5      = "saveh5_coin_chong_shap.tmp.h5"

# -----------------------------------------------------------------------------------------------------------------------
# Import packages
# -----------------------------------------------------------------------------------------------------------------------
from py_bin.py_class.chong_structure import chong_structure
from py_bin.py_class.shap_structure import shap_structure
import os
from py_bin.py_functions.calc_coinc import calc_coinc,save_coinc
import h5py
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
# Data for the statistics:
#     - index            : index of the field
#     - Hperc            : percolation index
#     - uvw_folder       : folder of the flow field data
#     - uvw_file         : file of the flow field data
#     - umean_file       : file to save the mean velocity
#     - data_folder      : folder to store the calculated data
#     - dx               : downsampling in x
#     - dy               : downsampling in y
#     - dz               : downsampling in z
#     - L_x              : length of the channel in the streamwise direction
#     - L_y              : half-width of the channel in the wall-normal direction
#     - L_z              : length of the channel in the spanwise direction
#     - urms_file        : file to save the rms of the velocity
#     - rey              : Friction Reynolds number
#     - utau             : Friction velocity
#     - padding          : padding of the flow field
#     - sym_quad         : flag for using the symmetry in the direction 2 of the field for the quadrant selection
#     - filvol           : volume for filtering the structures+
#     - shap_folder      : folder of the shap values
#     - shap_folder      : file of the shap values
#     - chong_folder     : folder of the chong structures
#     - chong_file       : file of the chong structures
#     - padding          : padding of the field
#     - data_type        : type of data used by the model
#     - SHAPq_folder     : folder of the shap structures
#     - SHAPq_file       : file of the uv structures
#     - nsamples         : number of samples of the shap calculation
#     - SHAPrms_file     : file of the rms of the shap
#     - chong_shap_file  : file for saving the coincidence between chongs and shap structures
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
chong_folder     = folders.chong_folder
chong_file       = folders.chong_file
padding          = chd.padding
data_type        = tr_data.data_type
plot_folder      = folders.plot_folder
SHAPq_folder     = folders.SHAPq_folder+"_real"
SHAPq_file       = folders.SHAPq_file
nsamples         = sh_data.nsamples
SHAPrms_file     = folders.SHAPrms_file
chong_shap_file = folders.chong_shap_file

# -----------------------------------------------------------------------------------------------------------------------
# Create the data of the uv structure
# -----------------------------------------------------------------------------------------------------------------------
data_chong  = {"uvw_folder":uvw_folder,"uvw_file":uvw_file,"Hperc":Hperc,"index":0,"dx":dx,
               "dy":dy,"dz":dz,"L_x":L_x,"L_y":L_y,"L_z":L_z,"rey":rey,"utau":utau,
               "padding":padding,"data_folder":data_folder,"umean_file":umean_file,
               "urms_file":urms_file,"sym_quad":True,"filvol":filvol,"shap_folder":shap_folder,
               "shap_file":shap_file,"folder":chong_folder,"file":chong_file,"padding":padding,
               "data_type":data_type}

# -----------------------------------------------------------------------------------------------------------------------
# Create the data of the shap structure
# -----------------------------------------------------------------------------------------------------------------------
data_shap = {"uvw_folder":uvw_folder,"uvw_file":uvw_file,"Hperc":Hperc,"index":0,"dx":dx,
             "dy":dy,"dz":dz,"L_x":L_x,"L_y":L_y,"L_z":L_z,"rey":rey,"utau":utau,
             "padding":padding,"data_folder":data_folder,"umean_file":umean_file,
             "urms_file":urms_file,"sym_quad":True,"filvol":filvol,"shap_folder":shap_folder,
             "shap_file":shap_file,"folder":SHAPq_folder,"file":SHAPq_file,"padding":padding,
             "data_type":data_type,"nsamples":nsamples,"SHAPrms_file":SHAPrms_file}

# -----------------------------------------------------------------------------------------------------------------------
# calculate the coincidence between the uv and the shap structures as a function of y
# -----------------------------------------------------------------------------------------------------------------------
index_range = range(index_ini,index_fin,index_delta)
for ii in index_range:
    print(ii,flush=True)
    data_chong["index"]  = ii
    data_shap["index"]   = ii
    chong_struc          = chong_structure(data_in=data_chong)
    chong_struc.read_struc()
    shap_struc           = shap_structure(data_in=data_shap)
    shap_struc.read_struc()
    data_out             = calc_coinc(data_in={"data_struc1":shap_struc,"data_struc2":chong_struc,"save_data":False,
                                               "calc_coin_file":chong_shap_file,"folder":data_folder,"dy":dy,"dx":dx,
                                               "dz":dz,"uvw_folder":uvw_folder,"uvw_file":uvw_file,"L_x":L_x,
                                               "L_y":L_y,"L_z":L_z,"rey":rey,"utau":utau})
    if ii == index_range[0]:
        frac_struc1  = data_out["frac_struc1"]
        frac_struc2  = data_out["frac_struc2"]
        frac_coinc   = data_out["frac_coinc"]
        yplus        = data_out["yplus"]
    else:
        frac_struc1 += data_out["frac_struc1"]
        frac_struc2 += data_out["frac_struc2"]
        frac_coinc  += data_out["frac_coinc"]
        
    # -----------------------------------------------------------------------------------------------------------------
    # Save the data
    # -----------------------------------------------------------------------------------------------------------------
    nfiles     = ii+1
    if np.mod(nfiles,5)==0:
        fileh5save = h5py.File(data_folder+'/'+saveh5,'w')
        fileh5save.create_dataset('frac_struc1',data=frac_struc1/nfiles)
        fileh5save.create_dataset('frac_struc2',data=frac_struc2/nfiles)
        fileh5save.create_dataset('frac_coinc',data=frac_coinc/nfiles)
        fileh5save.create_dataset('yplus',data=yplus)
        fileh5save.create_dataset('nfields',data=nfiles)
        fileh5save.close()
    print("Iteration "+str(ii),flush=True)
frac_struc1 /= len(index_range)
frac_struc2 /= len(index_range)
frac_coinc  /= len(index_range)
save_coinc(data_in={"frac_struc1":frac_struc1,"frac_struc2":frac_struc2,"frac_coinc":frac_coinc,
                    "yplus":yplus,"calc_coin_file":chong_shap_file,"folder":data_folder})