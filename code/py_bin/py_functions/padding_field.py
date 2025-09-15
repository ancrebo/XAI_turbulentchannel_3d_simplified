# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
padding_field.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 21 15:18:38 2024

@author: Andres Cremades Botella

File to create the padding of a field. The file contains the following functions:
    Functions:
        - padding_field : file to create the padding of the field
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

def padding_field(data_in={"field":[],"shpx":196,"shpy":201,"shpz":96,"padding":15}):
    """
    .....................................................................................................................
    # read_velocity: Function to read the velocity, the function generates three arrays containing the velocities in
                     the streamwise (uu), wall-normal (vv) and spanwise (ww) directions
    .....................................................................................................................
    Parameters
    ----------
    data_in : TYPE, optional
        DESCRIPTION. The default is {"field":[],"shpx":196,"shpy":201,"shpz":96,"padding":15}.
        Data:
            - field       : field without padding
            - shpx        : shape in x of the tensors
            - shpy        : shape in y of the tensors
            - shpz        : shape in z of the tensors
            - padding     : padding of the fields
    Returns
    -------
    dict
        Field after applying padding
        Data:
            - field : field with padding

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    import h5py
    from py_bin.py_functions.umean import read_Umean
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    field         = np.array(data_in["field"])
    shpx          = int(data_in["shpx"])        # shape of the tensors in the x direction
    shpy          = int(data_in["shpy"])        # shape of the tensors in the y direction
    shpz          = int(data_in["shpz"])        # shape of the tensors in the z direction
    padding       = int(data_in["padding"])     # padding of the fields
        
    # -------------------------------------------------------------------------------------------------------------------
    # Apply the padding if necessary. The padding takes the variable padding to add that number of nodes in both sizes
    # of the channel in the streamwise and the spanwise directions. The idea is to preserve the periodicity of the 
    # channel
    # -------------------------------------------------------------------------------------------------------------------
    field_pad = np.zeros((shpy,shpz+2*padding,shpx+2*padding))
    field_pad[:,padding:-padding,padding:-padding] = field.copy()
    field_pad[:,:padding,padding:-padding]         = field[:,-padding:,:]
    field_pad[:,-padding:,padding:-padding]        = field[:,:padding,:]
    field_pad[:,:,:padding]                        = field_pad[:,:,-2*padding:-padding]
    field_pad[:,:,-padding:]                       = field_pad[:,:,padding:2*padding]
    field                                          = field_pad.copy()
    data_output = {"field":field}
    return data_output