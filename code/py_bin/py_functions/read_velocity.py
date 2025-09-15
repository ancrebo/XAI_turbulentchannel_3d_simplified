# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
read_velocity.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 21 15:18:38 2024

@author: Andres Cremades Botella

File to read the data of the velocity fields. The file contains the following functions:
    Functions:
        - read_velocity : file to read the velocity
"""
# -----------------------------------------------------------------------------------------------------------------------
# Read the packages for all the functions
# -----------------------------------------------------------------------------------------------------------------------
import sys
import numpy as np

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

def read_velocity(data_in={"folder":"../../P125_21pi_vu","file":"P125_21pi_vu.$INDEX$.h5.uvw","index":1000,
                           "dx":1,"dy":1,"dz":1,"shpx":196,"shpy":201,"shpz":96,"padding":15,
                           "data_folder":"Data","umean_file":"Umean.txt"}):
    """
    .....................................................................................................................
    # read_velocity: Function to read the velocity, the function generates three arrays containing the velocities in
                     the streamwise (uu), wall-normal (vv) and spanwise (ww) directions
    .....................................................................................................................
    Parameters
    ----------
    data_in : TYPE, optional
        DESCRIPTION. The default is {"folder":"../P125_21pi_vu","file":"P125_21pi_vu.$INDEX$.h5.uvw","index":1000,
                                     "dx":1,"dy":1,"dz":1,"shpx":196,"shpy":201,"shpz":96,"padding":15
                                     "data_folder":"Data","umean_file":"Umean.txt"}.
        Data:
            - folder      : folder of the velocity data
            - file        : file of the velocity data without the index
            - index       : index of the velocity data file
            - dx          : downsampling in x
            - dy          : downsampling in y
            - dz          : downsampling in z
            - shpx        : shape in x of the tensors
            - shpy        : shape in y of the tensors
            - shpz        : shape in z of the tensors
            - padding     : padding of the fields
            - data_folder : folder to store generated data
            - umean_file  : file of the mean velocity
    Returns
    -------
    dict
        Velocity fluctuation in the streamwise, wall-normal and spanwise directions.
        Data:
            - uu : velocity fluctuation in the streamwise direction
            - vv : velocity fluctuation in the wall-normal direction
            - ww : velocity fluctuation in the spanwise direction

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    import h5py
    from py_bin.py_functions.umean import read_Umean
    from py_bin.py_functions.padding_field import padding_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    folder        = str(data_in["folder"])      # folder of the data
    file          = str(data_in["file"])        # file of the data
    index         = str(int(data_in["index"]))  # index of the file
    dx            = int(data_in["dx"])          # downsampling in x
    dy            = int(data_in["dy"])          # downsampling in y
    dz            = int(data_in["dz"])          # downsampling in z
    shpx          = int(data_in["shpx"])        # shape of the tensors in the x direction
    shpy          = int(data_in["shpy"])        # shape of the tensors in the y direction
    shpz          = int(data_in["shpz"])        # shape of the tensors in the z direction
    padding       = int(data_in["padding"])     # padding of the fields
    data_folder   = str(data_in["data_folder"]) # folder to store generated data
    umean_file    = str(data_in["umean_file"])  # file of the mean velocity
    file_complete = folder+'/'+file
    file_ii        = file_complete.replace("$INDEX$",index) 
    try:
        dataUmean  = {"folder":data_folder,"file":umean_file,"dy":dy}
        meanU_data = read_Umean(dataUmean)
    except:
        print("Mean velocity file needs to be provided. Breaking calculation...",flush=True)
        sys.exit()
        
    # -------------------------------------------------------------------------------------------------------------------
    # Read the mean velocity in the streamwise direction, other directions have null mean velocity
    # -------------------------------------------------------------------------------------------------------------------
    UUmean  = meanU_data["UUmean"]
    print('Reading field:' + str(file_ii),flush=True)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the information from the files. The information requires the mean values in the wall-normal directions
    # This values should be stored in the data folder, in the case of missing the file, the software will calculate it.
    # -------------------------------------------------------------------------------------------------------------------
    file = h5py.File(file_ii,'r')    
    UU   = np.array(file['u'])[::dy,::dz,::dx]
    uu   = UU-UUmean.reshape(-1,1,1)
    vv   = np.array(file['v'])[::dy,::dz,::dx]
    ww   = np.array(file['w'])[::dy,::dz,::dx]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Apply the padding if necessary. The padding takes the variable padding to add that number of nodes in both sizes
    # of the channel in the streamwise and the spanwise directions. The idea is to preserve the periodicity of the 
    # channel
    # -------------------------------------------------------------------------------------------------------------------
    if padding > 0:
        uu = padding_field(data_in={"field":uu,"shpx":shpx,"shpy":shpy,"shpz":shpz,"padding":padding})["field"]
        vv = padding_field(data_in={"field":vv,"shpx":shpx,"shpy":shpy,"shpz":shpz,"padding":padding})["field"]
        ww = padding_field(data_in={"field":ww,"shpx":shpx,"shpy":shpy,"shpz":shpz,"padding":padding})["field"]
    data_output = {"uu":uu,"vv":vv,"ww":ww}
    return data_output