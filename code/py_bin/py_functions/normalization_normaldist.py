# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
normalization_normaldist.py
-------------------------------------------------------------------------------------------------------------------------
Created on Fri Mar 22 12:33:24 2024

@author: Andres Cremades Botella

File to create the normalization values for the velocity fields. The normalization generates 
values following a normal distribution centered in 0 and with a standard deviation of 1. The file contains
the following functions:
    Functions:
        - save_norm : function for saving the normalization to a file
        - read_norm : function for reading the normalization file
        - calc_norm : function for calculating the normalization
"""

# ---------------------------------------------------------------------------------------------------------------------
# Import packages for all the functions
# ---------------------------------------------------------------------------------------------------------------------
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# Define the functions
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
    
def save_norm(data_in={"folder":"Data","file":"norm.txt","uumean":0,"vvmean":0,"wwmean":0,"uustd":0,"vvstd":0,
                       "wwstd":0,"uvmean":0,"vwmean":0,"uwmean":0,"uvstd":0,"vwstd":0,"uwstd":0}):
    """
    .....................................................................................................................
    # save_norm: function for saving the normalization to a file. The function saves the mean and standar deviation
                 values of the velocity components and stress components
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data for saving the normalization values. 
        The default is {"folder":data,"file":"norm.txt","uumean":0,"vvmean":0,"wwmean":0,"uustd":0,"vvstd":0,
                         "wwstd":0,"uvmean":0,"vwmean":0,"uwmean":0,"uvstd":0,"vwstd":0,"uwstd":0}.
        Data:
            - folder : folder of the generated data
            - file   : file of the normalization data
            - uumean : mean streamwise velocity
            - vvmean : mean wall-normal velocity
            - wwmean : mean spanwise velocity
            - uustd  : standard deviation streamwise velocity
            - vvstd  : standard deviation wall-normal velocity
            - wwstd  : standard deviation spanwise veloctiy
            - uvmean : mean uv stress
            - vwmean : mean vw stress
            - uwmean : mean uw stress
            - uvstd  : standard deviation uv stress
            - vwstd  : standard deviation vw stress
            - uwstd  : standard deviation uw stress
    Returns
    -------
    None.
    """
    
    # -----------------------------------------------------------------------------------------------------------------
    # Read the data
    # -----------------------------------------------------------------------------------------------------------------
    folder    = str(data_in["folder"])    # folder of the normalization data
    file      = str(data_in["file"])      # file of the normalization data
    uumean    = float(data_in["uumean"])  # mean streamwise velocity
    vvmean    = float(data_in["vvmean"])  # mean of the wall-normal velocity 
    wwmean    = float(data_in["wwmean"])  # mean of the spanwise velocity
    uustd     = float(data_in["uustd"])   # standard deviation of the streamwise veloctity
    vvstd     = float(data_in["vvstd"])   # standard deviation of the wall-normal veloctiy
    wwstd     = float(data_in["wwstd"])   # standard deviation of the spanwise velocity
    uvmean    = float(data_in["uvmean"])  # mean of the uv stress
    vwmean    = float(data_in["vwmean"])  # mean of the vw stress
    uwmean    = float(data_in["uwmean"])  # mean of the uw stress
    uvstd     = float(data_in["uvstd"])   # standard deviation of the uv stress
    vwstd     = float(data_in["vwstd"])   # standard deviation of the vw stress
    uwstd     = float(data_in["uwstd"])   # standard deviation of the uw stress
    
    # -----------------------------------------------------------------------------------------------------------------
    # Save the data to a file
    # -----------------------------------------------------------------------------------------------------------------
    file_norm = folder+'/'+file
    file_save = open(file_norm, "w+")           
    content = str(uumean)+'\n'
    file_save.write(content)    
    content = str(vvmean)+'\n'
    file_save.write(content)    
    content = str(wwmean)+'\n'
    file_save.write(content)          
    content = str(uustd)+'\n'
    file_save.write(content)    
    content = str(vvstd)+'\n'
    file_save.write(content)    
    content = str(wwstd)+'\n'
    file_save.write(content)         
    content = str(uvmean)+'\n'
    file_save.write(content)    
    content = str(vwmean)+'\n'
    file_save.write(content)    
    content = str(uwmean)+'\n'
    file_save.write(content)          
    content = str(uvstd)+'\n'
    file_save.write(content)    
    content = str(vwstd)+'\n'
    file_save.write(content)    
    content = str(uwstd)+'\n'
    file_save.write(content) 
    

def read_norm(data_in={"folder":"Data","file":"norm.txt"}):
    """
    .....................................................................................................................
    # read_norm: function for reading the normalization file. The function reads the mean and standard deviation values
                 of the velocity components and stress components
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data for the normalization of the velocity data. 
        The default is {folder:"Data",file:"norm.txt"}.
        Data:
            - folder : folder to read the data
            - file   : file to read the data

    Returns
    -------
    dict
        Data of the mean and standard deviation of values for the normalization.
        Data:
            - uumean : mean streamwise velocity
            - vvmean : mean wall-normal velocity
            - wwmean : mean spanwise velocity
            - uustd  : standard deviation of streamwise velocity
            - vvstd  : standard deviation of wall-normal velocity
            - wwstd  : standard deviation of spanwise velocity
            - uvmean : mean uv stress
            - vwmean : mean vw stress
            - uwmean : mean uw stress
            - uvstd  : standard deviation of uv stress
            - vwstd  : standard deviation of vw stress
            - uwstd  : standard deviation of uw stress

    """
    # -----------------------------------------------------------------------------------------------------------------
    # Read the data
    # -----------------------------------------------------------------------------------------------------------------
    folder = str(data_in["folder"]) # folder to read the normalization data
    file   = str(data_in["file"])   # file to read the normalization data
    
    # -----------------------------------------------------------------------------------------------------------------
    # Read the normalization file
    # -----------------------------------------------------------------------------------------------------------------
    file_norm = folder+'/'+file
    file_read = open(file_norm,"r")
    uumean    = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    vvmean    = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    wwmean    = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    uustd     = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    vvstd     = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    wwstd     = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    uvmean    = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    vwmean    = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    uwmean    = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    uvstd     = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    vwstd     = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    uwstd     = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    data_out  = {"uumean":uumean,"vvmean":vvmean,"wwmean":wwmean,"uustd":uustd,"vvstd":vvstd,"wwstd":wwstd,\
                 "uvmean":uvmean,"vwmean":vwmean,"uwmean":uwmean,"uvstd":uvstd,"vwstd":vwstd,"uwstd":uwstd}
    return data_out

               
def calc_norm(data_in={"field_ini":1000,"field_fin":9999,"data_folder":"Data","umean_file":"Umean.txt",
                       "dx":1,"dy":1,"dz":1,"folder":"../../P125_21pi_vu","file":"P125_21pi_vu.$INDEX$.h5.uvw",
                       "shpx":192,"shpy":201,"shpz":96,"save_file":True,"unorm_file":"norm.txt"}):
    """
    .....................................................................................................................
    # calc_norm: function to calculate the normalization of the velocity. The function calculates the mean and 
                 the standard deviation values of the velocity components and stress components
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data for normalizing the velocity.
        The default is {"field_ini":1000,"field_fin":9999,"data_folder":"Data","umean_file"="Umean.txt",
                        "dx":1,"dy":1,"dz":1,"folder":"../P125_21pi_vu","file":"P125_21pi_vu.$INDEX$.h5.uvw",
                        "shpx":192,"shpy":201,"shpz":96,"padding":15,"save_file":True,"unorm_file":"norm.txt"}.
        Data:
            - field_ini   : initial field of the data to calculate the normalization
            - field_fin   : final field of the data to calculate the normalization
            - data_folder : folder to store the data calculated by the code
            - umean_file  : file of the mean velocity
            - dx          : downsampling of x direction
            - dy          : downsampling of y direction
            - dz          : downsampling of z direction
            - folder      : folder of the velocity data
            - file        : file of the velocity data without index
            - shpx        : shape of the tensor in x
            - shpy        : shape of the tensor in y
            - shpz        : shape of the tensor in z
            - save_file   : flag to save the normalization in a file
            - unorm_file  : file of the normalization data
            

    Returns
    -------
    dict
        Data for the normalization. Only returns it in case of not saving a file
        Data:
            - uumean : mean streamwise velocity
            - vvmean : mean wall-normal velocity
            - wwmean : mean spanwise velocity
            - uustd  : standard deviation of streamwise velocity
            - vvstd  : standard deviation of wall-normal velocity
            - wwstd  : standard deviation of spanwise velocity
            - uvmean : mean uv stress
            - vwmean : mean vw stress
            - uwmean : mean uw stress
            - uvstd  : standard deviation of uv stress
            - vwstd  : standard deviation of vw stress
            - uwstd  : standard deviation of uw stress
    """
    # -----------------------------------------------------------------------------------------------------------------
    # Import packages
    # -----------------------------------------------------------------------------------------------------------------
    from py_bin.py_functions.read_velocity import read_velocity
    
    # -----------------------------------------------------------------------------------------------------------------
    # Read the data
    # -----------------------------------------------------------------------------------------------------------------
    field_ini   = int(data_in["field_ini"])   # initial field for calculating the normalization
    field_fin   = int(data_in["field_fin"])   # final field for calculating the normalization
    data_folder = str(data_in["data_folder"]) # folder for reading the data
    umean_file  = str(data_in["umean_file"])  # file for reading the mean velocity
    folder      = str(data_in["folder"])      # folder to read the velocity fields
    file        = str(data_in["file"])        # file to read the veloctity fields
    dx          = int(data_in["dx"])          # downsampling in x
    dy          = int(data_in["dy"])          # downsampling in y
    dz          = int(data_in["dz"])          # downsampling in z
    shpx        = int(data_in["shpx"])        # shape in the x direction
    shpy        = int(data_in["shpy"])        # shape in the y direction
    shpz        = int(data_in["shpz"])        # shape in the z direction
    save_file   = bool(data_in["save_file"])  # flag to decide if the normalization must be save in a file
    unorm_file  = str(data_in["unorm_file"])  # file to save the normalization
    
    # -----------------------------------------------------------------------------------------------------------------
    # In the loop
    #   - ii : index of the file that we are reading
    # -----------------------------------------------------------------------------------------------------------------
    for ii in range(field_ini,field_fin):
        data_velocity      = {"folder":folder,"file":file,"index":ii,"dx":dx,"dy":dy,"dz":dz,"shpx":shpx,
                              "shpy":shpy,"shpz":shpz,"padding":0,"data_folder":data_folder,
                              "umean_file":umean_file}            
        data_read_velocity = read_velocity(data_velocity)
        uu_i0 = np.array(data_read_velocity['uu'],dtype='float')
        vv_i0 = np.array(data_read_velocity['vv'],dtype='float')
        ww_i0 = np.array(data_read_velocity['ww'],dtype='float')
        uv_i0 = np.multiply(uu_i0,vv_i0)
        vw_i0 = np.multiply(vv_i0,ww_i0)
        uw_i0 = np.multiply(uu_i0,ww_i0)
        if ii == field_ini:
            uu_cum  = np.sum(uu_i0)
            vv_cum  = np.sum(vv_i0)
            ww_cum  = np.sum(ww_i0)
            uu2_cum = np.sum(np.multiply(uu_i0,uu_i0))
            vv2_cum = np.sum(np.multiply(vv_i0,vv_i0))
            ww2_cum = np.sum(np.multiply(ww_i0,ww_i0))
            uv_cum  = np.sum(uv_i0)
            vw_cum  = np.sum(vw_i0)
            uw_cum  = np.sum(uw_i0)
            uv2_cum = np.sum(np.multiply(uv_i0,uv_i0))
            vw2_cum = np.sum(np.multiply(vw_i0,vw_i0))
            uw2_cum = np.sum(np.multiply(uw_i0,uw_i0))
            nn_cum  = shpx*shpy*shpz
        else:
            uu_cum  += np.sum(uu_i0)
            vv_cum  += np.sum(vv_i0)
            ww_cum  += np.sum(ww_i0)
            uu2_cum += np.sum(np.multiply(uu_i0,uu_i0))
            vv2_cum += np.sum(np.multiply(vv_i0,vv_i0))
            ww2_cum += np.sum(np.multiply(ww_i0,ww_i0))
            uv_cum  += np.sum(uv_i0)
            vw_cum  += np.sum(vw_i0)
            uw_cum  += np.sum(uw_i0)
            uv2_cum += np.sum(np.multiply(uv_i0,uv_i0))
            vw2_cum += np.sum(np.multiply(vw_i0,vw_i0))
            uw2_cum += np.sum(np.multiply(uw_i0,uw_i0))
            nn_cum  += shpx*shpy*shpz
    
    uumean  = uu_cum/nn_cum
    vvmean  = vv_cum/nn_cum
    wwmean  = ww_cum/nn_cum
    uu2mean = uu2_cum/nn_cum
    vv2mean = vv2_cum/nn_cum
    ww2mean = ww2_cum/nn_cum
    uustd   = np.sqrt(uu2mean-uumean**2) 
    vvstd   = np.sqrt(vv2mean-vvmean**2) 
    wwstd   = np.sqrt(ww2mean-wwmean**2) 
    uvmean  = uv_cum/nn_cum
    vwmean  = vw_cum/nn_cum
    uwmean  = uw_cum/nn_cum
    uv2mean = uv2_cum/nn_cum
    vw2mean = vw2_cum/nn_cum
    uw2mean = uw2_cum/nn_cum
    uvstd   = np.sqrt(uv2mean-uvmean**2) 
    vwstd   = np.sqrt(vw2mean-vwmean**2) 
    uwstd   = np.sqrt(uw2mean-uwmean**2) 
            
    # -----------------------------------------------------------------------------------------------------------------
    # Save the normalization in a file or return the values of the normalization
    # -----------------------------------------------------------------------------------------------------------------
    if save_file:
        data_norm_save = {"folder":data_folder,"file":unorm_file,"uumean":uumean,"vvmean":vvmean,"wwmean":wwmean,
                          "uustd":uustd,"vvstd":vvstd,"wwstd":wwstd,"uvmean":uvmean,"vwmean":vwmean,"uwmean":uwmean,
                          "uvstd":uvstd,"vwstd":vwstd,"uwstd":uwstd}
        save_norm(data_in=data_norm_save)
    else:
        data_out = {"uumean":uumean,"vvmean":vvmean,"wwmean":wwmean,"uustd":uustd,"vvstd":vvstd,"wwstd":wwstd,
                    "uvmean":uvmean,"vwmean":vwmean,"uwmean":uwmean,"uvstd":uvstd,"vwstd":vwstd,"uwstd":uwstd}
        return data_out
    
        