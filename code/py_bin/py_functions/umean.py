# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
umean.py
-------------------------------------------------------------------------------------------------------------------------
Created on Fri Mar 22 09:23:59 2024

@author: Andres Cremades Botella

File for calculating and reading the mean velocity. The file contains the following functions:
    Functions:
        - save_Umean : function for saving the mean velocity
        - read_Umean : function for reading the mean velocity
        - calc_Umean : function for calculating the mean velocity
"""

# -----------------------------------------------------------------------------------------------------------------------
# Import packages for all the functions
# ----------------------------------------------------------------------------------------------------------------------- 
import numpy as np

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

def save_Umean(data_in={"folder":"Data","file":"Umean.txt","UUmean":[],"VVmean":[],"WWmean":[]}):
    """
    .....................................................................................................................
    # save_Umean: Function for saving the mean velocity
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Information for saving the mean velocity.
        The default is {"file":"Umean.txt","UUmean":[],"VVmean":[],"WWmean":[]}.
        Data:
            - folder : folder to save the information
            - file   : file to save the information
            - UUmean : mean streamwise velocity
            - VVmean : mean wall-normal velocity
            - WWmean : mean spanwise velocity

    Returns
    -------
    None.

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    folder     = str(data_in["folder"])                    # Folder to save teh mean velocity
    file       = str(data_in["file"])                      # File to save the mean velocity
    UUmean     = np.array(data_in["UUmean"],dtype='float') # Mean velocity in the streamwise direction
    VVmean     = np.array(data_in["VVmean"],dtype='float') # Mean velocity in the wall-normal direction
    WWmean     = np.array(data_in["WWmean"],dtype='float') # Mean velocity in the spanwise direction 

    # -------------------------------------------------------------------------------------------------------------------
    # Save the information in a file
    # -------------------------------------------------------------------------------------------------------------------    
    file_umean = folder+'/'+file                     
    file_save  = open(file_umean, "w+")           
    content    = str(UUmean.tolist())+'\n'
    file_save.write(content)          
    content    = str(VVmean.tolist())+'\n'
    file_save.write(content)          
    content    = str(WWmean.tolist())+'\n'
    file_save.write(content)
    file_save.close()
    
    
def read_Umean(data_in={"folder":"Data","file":"Umean.txt","dy":1}):
    """ 
    .....................................................................................................................   
    # read_Umean: Function for read the mean velocity
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data to read the mean velocity.
        The default is {"folder":"Data","file":"Umean.txt","dy":1}.
        Data:
            - folder : folder to read the calculated data
            - file   : file of the mean velocity
            - dy     : downsampling in the y direction

    Returns
    -------
    dict
        Mean velocity in the streamwise, wall-normal and spanwise directions is returned.
        Data:
            - UUmean : streamwise mean velocity as a function of the wall-normal distance
            - VVmean : wall-normal mean velocity as a function of the wall-normal distance
            - WWmean : spanwise mean velocity as a function of the wall-normal distance

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    folder    = str(data_in["folder"]) # folder for reading the mean velocity data
    file      = str(data_in["file"])   # file for reading the mean velocity data
    dy        = int(data_in["dy"])     # downsampling in the wall-normal direction 
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the velocity from file
    # -------------------------------------------------------------------------------------------------------------------
    fileUmean = folder+'/'+file
    file_read = open(fileUmean,"r")
    UUmean    = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')[::dy]
    VVmean    = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')[::dy]
    WWmean    = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')[::dy]
    data_out  = {"UUmean":UUmean,"VVmean":VVmean,"WWmean":WWmean}
    return data_out
    
    
def calc_Umean(data_in={"field_ini":1000,"field_fin":9999,"folder":"../../P125_21pi_vu",\
                           "file":"P125_21pi_vu.$INDEX$.h5.uvw","save_file":True,"umean_file":"Umean.txt",
                           "data_folder":"Data","shpx":192,"shpy":201,"shpz":96}):
    """
    .....................................................................................................................
    # calc_Umean: Function for calculating the mean velocity
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data for the calculation of the mean velocity. 
        The default is {"field_ini":1000,"field_fin":9999,"folder":"../P125_21pi_vu",
                        "file":"P125_21pi_vu.$INDEX$.h5.uvw","save_file":True,"fileUmean":"Umean.txt",
                        "data_folder":"Data"}.
        Data:
            - field_ini   : index of the initial field used for calculating the mean velocity
            - field_fin   : index of the final field used for calculating the mean velocity
            - folder      : path of the folder to read the velocity data base
            - file        : name of the file to read the velocity
            - save_file    : flag to save the mean velocity in a file
            - umean_file  : file for saving the mean velocity
            - data_folder : path of the folder of the data calculated by the code
            - shpx        : shape of the tensors in the streamwise direction
            - shpy        : shape of the tensors in the wall-normal direction
            - shpz        : shape of the tensors in the spanwise direction

    Returns
    -------
    dict
        Mean velocity in the streamwise, wall-normal and spanwise directions is returned. Only used when the saving
        option is not active.
        Data:
            - UUmean : streamwise mean velocity as a function of the wall-normal distance
            - VVmean : wall-normal mean velocity as a function of the wall-normal distance
            - WWmean : spanwise mean velocity as a function of the wall-normal distance

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Load packages
    # -------------------------------------------------------------------------------------------------------------------
    import h5py
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    field_ini   = int(data_in["field_ini"])    # index of the initial field used for calculating the mean velocity
    field_fin   = int(data_in["field_fin"])    # index of the final field used for calculating the mean velocity
    folder      = str(data_in["folder"])       # path to the folder for reading the velocity data base
    file        = str(data_in["file"])         # name of the file containing the velocity data
    file_comp   = folder+'/'+file
    save_file   = bool(data_in["save_file"])   # flag for saving the file
    umean_file  = str(data_in["umean_file"])   # file for the mean velocity
    data_folder = str(data_in["data_folder"])  # folder of the data calculated by the code
    shpx        = int(data_in["shpx"])         # shape of the tensors in the streamwise direction
    shpy        = int(data_in["shpy"])         # shape of the tensors in the wall-normal direction
    shpz        = int(data_in["shpz"])         # shape of the tensors in the spanwise direction
     
    # -------------------------------------------------------------------------------------------------------------------
    # In the following lines:
    #     - ii : index related to the field
    #     - file_ii : file of the velocity field including the index
    #     - UU      : streamwise velocity
    #     - VV      : wall-normal velocity
    #     - WW      : spanwise velocity
    #     - UU_cum  : cumulative streamwise velocity
    #     - VV_cum  : cumulative wall-normal velocity
    #     - WW_cum  : cumulative spanwise velocity
    #     - nn_cum  : number of gridpoints used for calculate the mean velocity
    # -------------------------------------------------------------------------------------------------------------------
    for ii in range(field_ini,field_fin):            
        file_ii = file_comp.replace("$INDEX$",str(ii))
        print('Mean velocity calculation:' + str(file_ii),flush=True)
        file_r = h5py.File(file_ii,'r')
        UU     = np.array(file_r['u'])
        VV     = np.array(file_r['v'])
        WW     = np.array(file_r['w'])
        if ii == field_ini:
            UU_cum = np.sum(UU,axis=(1,2))
            VV_cum = np.sum(VV,axis=(1,2))
            WW_cum = np.sum(WW,axis=(1,2))
            nn_cum = np.ones((shpy,))*shpx*shpz
        else:
            UU_cum += np.sum(UU,axis=(1,2))
            VV_cum += np.sum(VV,axis=(1,2))
            WW_cum += np.sum(WW,axis=(1,2))
            nn_cum += np.ones((shpy,))*shpx*shpz
            
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate the mean velocity
    #     - UUmean : mean velocity in the streamwise velocity
    #     - VVmean : mean velocity in the wall-normal velocity
    #     - WWmean : mean velocity in the spanwise veloctiy
    # -------------------------------------------------------------------------------------------------------------------
    UUmean = np.divide(UU_cum,nn_cum)
    VVmean = np.divide(VV_cum,nn_cum)
    WWmean = np.divide(WW_cum,nn_cum)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Save the mean in a file or return the mean velocity
    # -------------------------------------------------------------------------------------------------------------------
    if save_file:
        data_Umean = {"folder":data_folder,"file":umean_file,"UUmean":UUmean,"VVmean":VVmean,"WWmean":WWmean}
        save_Umean(data_in=data_Umean)
    else:
        data_out  = {"UUmean":UUmean,"VVmean":VVmean,"WWmean":WWmean}
        return data_out
            
    