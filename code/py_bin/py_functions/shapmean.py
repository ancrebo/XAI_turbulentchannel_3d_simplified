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
import glob

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

def save_SHAPmean(data_in={"folder":"Data","file":"SHAPmean.txt","SHAP_umean":[],"SHAP_vmean":[],"SHAP_wmean":[]}):
    """
    .....................................................................................................................
    # save_SHAPmean: Function for saving the mean SHAP values
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Information for saving the mean SHAP values.
        The default is {"folder":"Data","file":"SHAPmean.txt","SHAP_umean":[],"SHAP_vmean":[],"SHAP_wmean":[]}.
        Data:
            - folder     : folder to save the information
            - file       : file to save the information
            - SHAP_umean : mean streamwise shap
            - SHAP_vmean : mean wall-normal shap
            - SHAP_wmean : mean spanwise shap
            - SHAP_mmean : mean of the absolute value of the SHAP

    Returns
    -------
    None.

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    folder     = str(data_in["folder"])                        # Folder to save the mean velocity
    file       = str(data_in["file"])                          # File to save the mean velocity
    SHAP_umean = np.array(data_in["SHAP_umean"],dtype='float') # Mean velocity in the streamwise direction
    SHAP_vmean = np.array(data_in["SHAP_vmean"],dtype='float') # Mean velocity in the wall-normal direction
    SHAP_wmean = np.array(data_in["SHAP_wmean"],dtype='float') # Mean velocity in the spanwise direction
    SHAP_mmean = np.array(data_in["SHAP_mmean"],dtype='float')

    # -------------------------------------------------------------------------------------------------------------------
    # Save the information in a file
    # -------------------------------------------------------------------------------------------------------------------    
    file_umean = folder+'/'+file                     
    file_save  = open(file_umean, "w+")           
    content    = str(SHAP_umean.tolist())+'\n'
    file_save.write(content)          
    content    = str(SHAP_vmean.tolist())+'\n'
    file_save.write(content)          
    content    = str(SHAP_wmean.tolist())+'\n'
    file_save.write(content)        
    content    = str(SHAP_mmean.tolist())+'\n'
    file_save.write(content)
    file_save.close()
    
    
def read_SHAPmean(data_in={"folder":"Data","file":"SHAPmean.txt","dy":1}):
    """ 
    .....................................................................................................................   
    # read_SHAPmean: Function for read the mean velocity
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data to read the mean SHAP.
        The default is {"folder":"Data","file":"SHAPmean.txt","dy":1}.
        Data:
            - folder : folder to read the calculated data
            - file   : file of the mean velocity
            - dy     : downsampling in the y direction

    Returns
    -------
    dict
        Mean velocity in the streamwise, wall-normal and spanwise directions is returned.
        Data:
            - SHAP_umean : streamwise mean shap as a function of the wall-normal distance
            - SHAP_vmean : wall-normal mean shap as a function of the wall-normal distance
            - SHAP_wmean : spanwise mean shap as a function of the wall-normal distance
            - SHAP_mmean : mean of the absolute value of the SHAP

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
    fileSHAPmean = folder+'/'+file
    file_read    = open(fileSHAPmean,"r")
    SHAP_umean   = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')[::dy]
    SHAP_vmean   = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')[::dy]
    SHAP_wmean   = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')[::dy]
    SHAP_mmean   = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')[::dy]
    data_out     = {"SHAP_umean":SHAP_umean,"SHAP_vmean":SHAP_vmean,"SHAP_wmean":SHAP_wmean,"SHAP_mmean":SHAP_mmean}
    return data_out
    
    
def calc_SHAPmean(data_in={"field_ini":1000,"field_fin":9999,"field_delta":1,"folder":"../../P125_21pi_vu",
                           "file":"P125_21pi_vu.$INDEX$.h5.uvw","save_file":True,"SHAPmean_file":"SHAPmean.txt",
                           "data_folder":"Data","shpx":192,"shpy":201,"shpz":96}):
    """
    .....................................................................................................................
    # calc_SHAPmean: Function for calculating the mean velocity
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data for the calculation of the mean velocity. 
        The default is {"field_ini":1000,"field_fin":9999,"folder":"../../P125_21pi_vu",
                        "file":"P125_21pi_vu.$INDEX$.h5.uvw","save_file":True,"SHAPmean_file":"SHAPmean.txt",
                        "data_folder":"Data","shpx":192,"shpy":201,"shpz":96}.
        Data:
            - field_ini     : index of the initial field used for calculating the mean velocity
            - field_fin     : index of the final field used for calculating the mean velocity
            - field_delta   : separation between fields
            - folder        : path of the folder to read the velocity data base
            - file          : name of the file to read the velocity
            - save_file     : flag to save the mean velocity in a file
            - SHAPmean_file : file for saving the mean velocity
            - data_folder   : path of the folder of the data calculated by the code
            - shpx          : shape of the tensors in the streamwise direction
            - shpy          : shape of the tensors in the wall-normal direction
            - shpz          : shape of the tensors in the spanwise direction

    Returns
    -------
    dict
        Mean velocity in the streamwise, wall-normal and spanwise directions is returned. Only used when the saving
        option is not active.
        Data:
            - SHAP_umean : streamwise mean SHAP as a function of the wall-normal distance
            - SHAP_vmean : wall-normal mean SHAP as a function of the wall-normal distance
            - SHAP_wmean : spanwise mean SHAP as a function of the wall-normal distance
            - SHAP_mmean : mean of the absolute value of the SHAP

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Load packages
    # -------------------------------------------------------------------------------------------------------------------
    import h5py
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    field_ini     = int(data_in["field_ini"])     # index of the initial field used for calculating the mean velocity
    field_fin     = int(data_in["field_fin"])     # index of the final field used for calculating the mean velocity
    field_delta   = int(data_in["field_delta"])
    folder        = str(data_in["folder"])        # path to the folder for reading the velocity data base
    file          = str(data_in["file"])          # name of the file containing the velocity data
    file_comp     = folder+'/'+file
    save_file     = bool(data_in["save_file"])    # flag for saving the file
    SHAPmean_file = str(data_in["SHAPmean_file"]) # file for the mean velocity
    data_folder   = str(data_in["data_folder"])   # folder of the data calculated by the code
    shpx          = int(data_in["shpx"])          # shape of the tensors in the streamwise direction
    shpy          = int(data_in["shpy"])          # shape of the tensors in the wall-normal direction
    shpz          = int(data_in["shpz"])          # shape of the tensors in the spanwise direction
     
    # -------------------------------------------------------------------------------------------------------------------
    # In the following lines:
    #     - ii : index related to the field
    #     - file_ii   : file of the shap field including the index
    #     - SHAP_u    : streamwise shap
    #     - SHAP_v    : wall-normal shap
    #     - SHAP_w    : spanwise shap
    #     - SHAP_m    : absolute value of the SHAP
    #     - SHAP_ucum : cumulative streamwise shap
    #     - SHAP_vcum : cumulative wall-normal shap
    #     - SHAP_wcum : cumulative spanwise shap
    #     - SHAP_mcum : cumulative value of the SHAP
    #     - nn_cum vv : number of gridpoints used for calculate the mean shap
    # -------------------------------------------------------------------------------------------------------------------
    for ii in range(field_ini,field_fin,field_delta):            
        file_ii = file_comp.replace("$INDEX$",str(ii))
        print('Mean velocity calculation:' + str(file_ii),flush=True)
        if glob.glob(file_ii):
            file_r = h5py.File(file_ii,'r+')
            SHAP_u = np.array(file_r['SHAP_u'])
            SHAP_v = np.array(file_r['SHAP_v'])
            SHAP_w = np.array(file_r['SHAP_w'])
            SHAP_m = np.sqrt(SHAP_u**2+SHAP_v**2+SHAP_w**2)
            if ii == field_ini:
                SHAP_ucum = np.sum(SHAP_u,axis=(1,2))
                SHAP_vcum = np.sum(SHAP_v,axis=(1,2))
                SHAP_wcum = np.sum(SHAP_w,axis=(1,2))
                SHAP_mcum = np.sum(SHAP_m,axis=(1,2))
                nn_cum    = np.ones((shpy,))*shpx*shpz
            else:
                SHAP_ucum += np.sum(SHAP_u,axis=(1,2))
                SHAP_vcum += np.sum(SHAP_v,axis=(1,2))
                SHAP_wcum += np.sum(SHAP_w,axis=(1,2))
                SHAP_mcum += np.sum(SHAP_m,axis=(1,2))
                nn_cum    += np.ones((shpy,))*shpx*shpz
        else:
            print('Skiping field '+str(ii)+' as file was not found',flush=True)
            
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate the mean velocity
    #     - SHAP_umean : mean shap in the streamwise velocity
    #     - SHAP_vmean : mean shap in the wall-normal velocity
    #     - SHAP_wmean : mean shap in the spanwise veloctiy
    #     - SHAP_mmean : mean value of the absolute SHAP
    # -------------------------------------------------------------------------------------------------------------------
    SHAP_umean = np.divide(SHAP_ucum,nn_cum)
    SHAP_vmean = np.divide(SHAP_vcum,nn_cum)
    SHAP_wmean = np.divide(SHAP_wcum,nn_cum)
    SHAP_mmean = np.divide(SHAP_mcum,nn_cum)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Save the mean in a file or return the mean velocity
    # -------------------------------------------------------------------------------------------------------------------
    if save_file:
        data_SHAPmean = {"folder":data_folder,"file":SHAPmean_file,"SHAP_umean":SHAP_umean,"SHAP_vmean":SHAP_vmean,
                         "SHAP_wmean":SHAP_wmean,"SHAP_mmean":SHAP_mmean}
        save_SHAPmean(data_in=data_SHAPmean)
    else:
        data_out  = {"SHAP_umean":SHAP_umean,"SHAP_vmean":SHAP_vmean,"SHAP_wmean":SHAP_wmean,"SHAP_mmean":SHAP_mmean}
        return data_out
            
    