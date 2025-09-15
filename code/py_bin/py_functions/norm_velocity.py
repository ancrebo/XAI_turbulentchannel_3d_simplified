# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
norm_velocity.py
-------------------------------------------------------------------------------------------------------------------------
Created on Fri Mar 22 11:59:50 2024

@author:  Andres Cremades Botella

File to normalize the velocity fields. The normalization generates values between 0 and 1 using the minimum and the 
maximum of the velocity values. The file contains the following functions:
    Functions:
        - norm_velocity : function for normalize the velocity
        - dim_velocity  : function for dimensionalize the velocity
"""

# -----------------------------------------------------------------------------------------------------------------------
# Import packages for all the functions
# -----------------------------------------------------------------------------------------------------------------------
import sys
import numpy as np

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

def norm_velocity(data_in={"uu":[],"vv":[],"ww":[],"folder_data":"Data","unorm_file":"norm.txt","data_type":"float32",
                           "mean_norm":False}):
    """
    .....................................................................................................................
    # norm_velocity: function for normalize the velocity
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data for the normalization of the velocity.
        The default is {"uu":[],"vv":[],"ww":[],"folder_data":"Data","unorm_file":"norm.txt","data_type":"float32"}.
        Data:
            - uu          : streamwise velocity
            - vv          : wall-normal velocity
            - ww          : spanwise velocity
            - folder_data : path to the folder containing the normalization
            - unorm_file  : file with the normalization values
            - data_type   : type of the data (float32,float16...)
            - mean_norm   : choose normalizing with the mean

    Returns
    -------
    data_out : dict
        Normalized velocity. The velocity is configured in float16 to save memory during the training
        Data:
            - unorm : streamwise normalized velocity
            - vnorm : wall-normal normalized velocity
            - wnorm : spanwise normalized velocity

    """    
    # -------------------------------------------------------------------------------------------------------------------
    # The normalization of the velocity fluctuations is generated in float format
    # -------------------------------------------------------------------------------------------------------------------
    uu        = np.array(data_in["uu"],dtype="float")               # velocity fluctuation in the streamwise direction.
    vv        = np.array(data_in["vv"],dtype="float")               # velocity fluctuation in the wall-normal direction.
    ww        = np.array(data_in["ww"],dtype="float")               # velocity fluctuation in the spanwise direction.
    mean_norm = bool(data_in["mean_norm"])
    
    # -------------------------------------------------------------------------------------------------------------------
    # import packages
    # -------------------------------------------------------------------------------------------------------------------
    if mean_norm:
        from py_bin.py_functions.normalization_normaldist import read_norm
    else:
        from py_bin.py_functions.normalization import read_norm
        
    # -------------------------------------------------------------------------------------------------------------------
    # Check datatype
    # -------------------------------------------------------------------------------------------------------------------
    if "data_type" in data_in.keys():
        data_type = str(data_in["data_type"])                       # definition of the data type.
        if not (data_type=="float32" or data_type=="float16"):
            data_type = "float32"
    else:
        print("[trainvali_data.py:data_traintest_tf] Data type needs to be selected.")
        sys.exit()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the normalization parameters
    # -------------------------------------------------------------------------------------------------------------------
    folder_data    = str(data_in["folder_data"])               # folder of the generated data
    unorm_file     = str(data_in["unorm_file"])                # file of the normalization data
    unorm_data     = {"folder":folder_data,"file":unorm_file}
    try:
        norm_param = read_norm(unorm_data)
    except:
        print('Normalization file could not be located. Calculation is stopped...',flush=True)
        sys.exit()
    
    if mean_norm:
        uumean = float(norm_param["uumean"])
        vvmean = float(norm_param["vvmean"])
        wwmean = float(norm_param["wwmean"])
        uustd  = float(norm_param["uustd"])
        vvstd  = float(norm_param["vvstd"])
        wwstd  = float(norm_param["wwstd"])
        
        # ---------------------------------------------------------------------------------------------------------------
        # Define the normalized fields using float 16 format
        # ---------------------------------------------------------------------------------------------------------------
        unorm    = np.array((uu-uumean)/(uustd),dtype=data_type)
        vnorm    = np.array((vv-vvmean)/(vvstd),dtype=data_type)
        wnorm    = np.array((ww-wwmean)/(wwstd),dtype=data_type)
    else:
        uumax = float(norm_param["uumax"])
        vvmax = float(norm_param["vvmax"])
        wwmax = float(norm_param["wwmax"])
        uumin = float(norm_param["uumin"])
        vvmin = float(norm_param["vvmin"])
        wwmin = float(norm_param["wwmin"])
        print(uumax)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Define the normalized fields using float 16 format
        # ---------------------------------------------------------------------------------------------------------------
        unorm    = np.array((uu-uumin)/(uumax-uumin),dtype=data_type)
        vnorm    = np.array((vv-vvmin)/(vvmax-vvmin),dtype=data_type)
        wnorm    = np.array((ww-wwmin)/(wwmax-wwmin),dtype=data_type)
    data_out = {"unorm":unorm,"vnorm":vnorm,"wnorm":wnorm}
    return data_out

def dim_velocity(data_in={"unorm":[],"vnorm":[],"wnorm":[],"folder_data":"Data","unorm_file":"norm.txt",
                          "data_type":"float32","mean_norm":False}):
    """
    .....................................................................................................................
    # dim_velocity: function for dimensionalize the velocity
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data for the non-normalization of the velocity.
        The default is {"unorm":[],"vnorm":[],"wnorm":[],"folder_data":"Data","unorm_file":"norm.txt",
                        "data_type":"float32"}.
        Data:
            - unorm       : streamwise velocity
            - vnorm       : wall-normal velocity
            - wnorm       : spanwise velocity
            - folder_data : path to the folder containing the normalization
            - unorm_file  : file with the normalization values
            - data_type   : type of the data (float32,float16...)
            - mean_norm   : choose normalizing with the mean

    Returns
    -------
    data_out : dict
        Normalized velocity. The velocity is configured in float16 to save memory during the training
        Data:
            uu : streamwise normalized velocity
            vv : wall-normal normalized velocity
            ww : spanwise normalized velocity

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate the dimensional velocity
    # -------------------------------------------------------------------------------------------------------------------
    if "data_type" in data_in.keys():
        data_type = str(data_in["data_type"])                   # definition of the data type.
        if not (data_type=="float32" or data_type=="float16"):
            data_type = "float32"
    else:
        print("[trainvali_data.py:data_traintest_tf] Data type needs to be selected.")
        sys.exit()
    uu_norm   = np.array(data_in["unorm"],dtype=data_type)      # velocity fluctuation in the streamwise direction.
    vv_norm   = np.array(data_in["vnorm"],dtype=data_type)      # velocity fluctuation in the wall-normal direction.
    ww_norm   = np.array(data_in["wnorm"],dtype=data_type)      # velocity fluctuation in the spanwise direction.
    mean_norm = bool(data_in["mean_norm"])
    
    # -------------------------------------------------------------------------------------------------------------------
    # import packages
    # -------------------------------------------------------------------------------------------------------------------
    if mean_norm:
        from py_bin.py_functions.normalization_normaldist import read_norm
    else:
        from py_bin.py_functions.normalization import read_norm
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the normalization parameters, use the float 16 format
    # -------------------------------------------------------------------------------------------------------------------
    folder_data    = str(data_in["folder_data"])                # folder of the generated data
    unorm_file     = str(data_in["unorm_file"])                 # file of the normalization data
    unorm_data     = {"folder":folder_data,"file":unorm_file}
    try:
        norm_param = read_norm(unorm_data)
    except:
        print('Normalization file could not be located. Calculation is stopped...',flush=True)
        sys.exit()
    
    if mean_norm:
        uumean = norm_param["uumean"]
        vvmean = norm_param["vvmean"]
        wwmean = norm_param["wwmean"]
        uustd  = norm_param["uustd"]
        vvstd  = norm_param["vvstd"]
        wwstd  = norm_param["wwstd"]
        
        # ---------------------------------------------------------------------------------------------------------------
        # Define the normalized fields using float 16 format
        # ---------------------------------------------------------------------------------------------------------------
        uu = np.array(uu_norm*(uustd)+uumean,dtype=data_type)
        vv = np.array(vv_norm*(vvstd)+vvmean,dtype=data_type)
        ww = np.array(ww_norm*(wwstd)+wwmean,dtype=data_type)
    else:
        uumax = norm_param["uumax"]
        vvmax = norm_param["vvmax"]
        wwmax = norm_param["wwmax"]
        uumin = norm_param["uumin"]
        vvmin = norm_param["vvmin"]
        wwmin = norm_param["wwmin"]
        
        # ---------------------------------------------------------------------------------------------------------------
        # Define the normalized fields using float 16 format
        # ---------------------------------------------------------------------------------------------------------------
        uu = np.array(uu_norm*(uumax-uumin)+uumin,dtype=data_type)
        vv = np.array(vv_norm*(vvmax-vvmin)+vvmin,dtype=data_type)
        ww = np.array(ww_norm*(wwmax-wwmin)+wwmin,dtype=data_type)
    data_out = {"uu":uu,"vv":vv,"ww":ww}
    return data_out
