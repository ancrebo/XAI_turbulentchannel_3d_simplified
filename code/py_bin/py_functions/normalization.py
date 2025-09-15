# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
normalization.py
-------------------------------------------------------------------------------------------------------------------------
Created on Fri Mar 22 12:33:24 2024

@author: Andres Cremades Botella

File to create the normalization values for the velocity fields. The normalization generates 
values between 0 and 1 using the minimum and the maximum of the velocity values. The file contains
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
    
def save_norm(data_in={"folder":"Data","file":"norm.txt","uumax":0,"vvmax":0,"wwmax":0,"uumin":0,"vvmin":0,
                       "wwmin":0,"uvmax":0,"vwmax":0,"uwmax":0,"uvmin":0,"vwmin":0,"uwmin":0}):
    """
    .....................................................................................................................
    # save_norm: function for saving the normalization to a file. The function saves the maximum and minimum values
                 of the velocity components and stress components
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data for saving the normalization values. 
        The default is {"folder":data,"file":"norm.txt","uumax":0,"vvmax":0,"wwmax":0,"uumin":0,"vvmin":0,
                         "wwmin":0,"uvmax":0,"vwmax":0,"uwmax":0,"uvmin":0,"vwmin":0,"uwmin":0}.
        Data:
            - folder : folder of the generated data
            - file   : file of the normalization data
            - uumax  : maximum streamwise velocity
            - vvmax  : maximum wall-normal velocity
            - wwmax  : maximum spanwise velocity
            - uumin  : minimum streamwise velocity
            - vvmin  : minimum wall-normal velocity
            - wwmin  : minimum spanwise veloctiy
            - uvmax  : maximum uv stress
            - vwmax  : maximum vw stress
            - uwmax  : maximum uw stress
            - uvmin  : minimum uv stress
            - vwmin  : minimum vw stress
            - uwmin  : minimum uw stress
    Returns
    -------
    None.
    """
    
    # -----------------------------------------------------------------------------------------------------------------
    # Read the data
    # -----------------------------------------------------------------------------------------------------------------
    folder    = str(data_in["folder"])    # folder of the normalization data
    file      = str(data_in["file"])      # file of the normalization data
    uumax     = float(data_in["uumax"])   # maximum streamwise velocity
    vvmax     = float(data_in["vvmax"])   # maximum of the wall-normal velocity 
    wwmax     = float(data_in["wwmax"])   # maximum of the spanwise velocity
    uumin     = float(data_in["uumin"])   # minimum of the streamwise veloctity
    vvmin     = float(data_in["vvmin"])   # minimum of the wall-normal veloctiy
    wwmin     = float(data_in["wwmin"])   # minimum of the spanwise velocity
    uvmax     = float(data_in["uvmax"])   # maximum of the uv stress
    vwmax     = float(data_in["vwmax"])   # maximum of the vw stress
    uwmax     = float(data_in["uwmax"])   # maximum of the uw stress
    uvmin     = float(data_in["uvmin"])   # minimum of the uv stress
    vwmin     = float(data_in["vwmin"])   # minimum of the vw stress
    uwmin     = float(data_in["uwmin"])   # minimum of the uw stress
    
    # -----------------------------------------------------------------------------------------------------------------
    # Save the data to a file
    # -----------------------------------------------------------------------------------------------------------------
    file_norm = folder+'/'+file
    file_save = open(file_norm, "w+")           
    content = str(uumax)+'\n'
    file_save.write(content)    
    content = str(vvmax)+'\n'
    file_save.write(content)    
    content = str(wwmax)+'\n'
    file_save.write(content)          
    content = str(uumin)+'\n'
    file_save.write(content)    
    content = str(vvmin)+'\n'
    file_save.write(content)    
    content = str(wwmin)+'\n'
    file_save.write(content)         
    content = str(uvmax)+'\n'
    file_save.write(content)    
    content = str(vwmax)+'\n'
    file_save.write(content)    
    content = str(uwmax)+'\n'
    file_save.write(content)          
    content = str(uvmin)+'\n'
    file_save.write(content)    
    content = str(vwmin)+'\n'
    file_save.write(content)    
    content = str(uwmin)+'\n'
    file_save.write(content) 
    

def read_norm(data_in={"folder":"Data","file":"norm.txt"}):
    """
    .....................................................................................................................
    # read_norm: function for reading the normalization file. The function reads the maximum and minimum values
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
        Data of the maximum and minimum values for the normalization.
        Data:
            - uumax : maximum streamwise velocity
            - vvmax : maximum wall-normal velocity
            - wwmax : maximum spanwise velocity
            - uumin : minimum streamwise velocity
            - vvmin : minimum wall-normal velocity
            - wwmin : minimum spanwise velocity
            - uvmax : maximum uv stress
            - vwmax : maximum vw stress
            - uwmax : maximum uw stress
            - uvmin : minimum uv stress
            - vwmin : minimum vw stress
            - uwmin : minimum uw stress

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
    uumax = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    vvmax = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    wwmax = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    uumin = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    vvmin = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    wwmin = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    uvmax = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    vwmax = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    uwmax = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    uvmin = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    vwmin = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    uwmin = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    data_out = {"uumax":uumax,"vvmax":vvmax,"wwmax":wwmax,"uumin":uumin,"vvmin":vvmin,"wwmin":wwmin,\
                "uvmax":uvmax,"vwmax":vwmax,"uwmax":uwmax,"uvmin":uvmin,"vwmin":vwmin,"uwmin":uwmin}
    return data_out

               
def calc_norm(data_in={"field_ini":1000,"field_fin":9999,"data_folder":"Data","umean_file":"Umean.txt",
                       "dx":1,"dy":1,"dz":1,"folder":"../../P125_21pi_vu","file":"P125_21pi_vu.$INDEX$.h5.uvw",
                       "shpx":192,"shpy":201,"shpz":96,"save_file":True,"unorm_file":"norm.txt"}):
    """
    .....................................................................................................................
    # calc_norm: function to calculate the normalization of the velocity. The function calculates the maximum and 
                 minimum values of the velocity components and stress components
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
            - uumax : maximum streamwise velocity
            - vvmax : maximum wall-normal velocity
            - wwmax : maximum spanwise velocity
            - uumin : minimum streamwise velocity
            - vvmin : minimum wall-normal velocity
            - wwmin : minimum spanwise velocity
            - uvmax : maximum uv stress
            - vwmax : maximum vw stress
            - uwmax : maximum uw stress
            - uvmin : minimum uv stress
            - vwmin : minimum vw stress
            - uwmin : minimum uw stress
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
            uumax = np.max(uu_i0)
            vvmax = np.max(vv_i0)
            wwmax = np.max(ww_i0)
            uumin = np.min(uu_i0)
            vvmin = np.min(vv_i0)
            wwmin = np.min(ww_i0)
            uvmax = np.max(uv_i0)
            vwmax = np.max(vw_i0)
            uwmax = np.max(uw_i0)
            uvmin = np.min(uv_i0)
            vwmin = np.min(vw_i0)
            uwmin = np.min(uw_i0)
        else:
            uumax = np.max([uumax,np.max(uu_i0)])
            vvmax = np.max([vvmax,np.max(vv_i0)])
            wwmax = np.max([wwmax,np.max(ww_i0)])
            uumin = np.min([uumin,np.min(uu_i0)])
            vvmin = np.min([vvmin,np.min(vv_i0)])
            wwmin = np.min([wwmin,np.min(ww_i0)])
            uvmax = np.max([uvmax,np.max(uv_i0)])
            vwmax = np.max([vwmax,np.max(vw_i0)])
            uwmax = np.max([uwmax,np.max(uw_i0)])
            uvmin = np.min([uvmin,np.min(uv_i0)])
            vwmin = np.min([vwmin,np.min(vw_i0)])
            uwmin = np.min([uwmin,np.min(uw_i0)])
            
    # -----------------------------------------------------------------------------------------------------------------
    # Save the normalization in a file or return the values of the normalization
    # -----------------------------------------------------------------------------------------------------------------
    if save_file:
        data_norm_save = {"folder":data_folder,"file":unorm_file,"uumax":uumax,"vvmax":vvmax,"wwmax":wwmax,
                          "uumin":uumin,"vvmin":vvmin,"wwmin":wwmin,"uvmax":uvmax,"vwmax":vwmax,"uwmax":uwmax,
                          "uvmin":uvmin,"vwmin":vwmin,"uwmin":uwmin}
        save_norm(data_in=data_norm_save)
    else:
        data_out = {"uumax":uumax,"vvmax":vvmax,"wwmax":wwmax,"uumin":uumin,"vvmin":vvmin,"wwmin":wwmin,
                    "uvmax":uvmax,"vwmax":vwmax,"uwmax":uwmax,"uvmin":uvmin,"vwmin":vwmin,"uwmin":uwmin}
        return data_out
    
        