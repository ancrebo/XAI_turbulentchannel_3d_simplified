# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
shaprms.py
-------------------------------------------------------------------------------------------------------------------------
Created on Wed Mar 27 08:50:04 2024

@author: Andres Cremades Botella

File to create the RMS values for the SHAP fields. Functions contained in the file:
    Functions:
        - read_rms : function to read the rms of the velocity
        - save_rms : function to save the rms of the velocity
        - calc_rms : function to calculate the rms of the velocity
"""

# -----------------------------------------------------------------------------------------------------------------------
# Import packages for all the functions
# -----------------------------------------------------------------------------------------------------------------------
import sys
import numpy as np
import glob

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
def read_rms(data_in={"file":"SHAPrms.txt","folder":"Data"}):
    """
    .....................................................................................................................
    # read_rms: Function for reading the RMS
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data for reading the RMS file. The default is {"file":"SHAPrms.txt","folder":"Data"}.
        Data:
            - file   : file of the RMS of the velocity
            - folder : folder of the generated data

    Returns
    -------
    data_out : dict
        Data of the RMS of the velocity.
        Data:
            - SHAP_urms : RMS of the streamwise SHAP
            - SHAP_vrms : RMS of the wall-normal SHAP
            - SHAP_wrms : RMS of the spanwise SHAP
            - SHAP_uv   : Mean uv SHAP along the wall-normal distance
            - SHAP_vw   : Mean vw SHAP along the wall-normal distance
            - SHAP_uw   : Mean uw SHAP along the wall-normal distance
            - SHAP_mrms : RMS of the absolute value of the SHAP

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    file   = str(data_in["file"])   # File of the RMS of the SHAP
    folder = str(data_in["folder"]) # Folder of the generated data
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data of the rms
    # -------------------------------------------------------------------------------------------------------------------
    file_rms  = folder+'/'+file
    file_read = open(file_rms,"r")
    SHAP_urms = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    SHAP_vrms = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    SHAP_wrms = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    SHAP_uv   = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    SHAP_vw   = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    SHAP_uw   = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    SHAP_mrms = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    data_out  = {"SHAP_urms":SHAP_urms,"SHAP_vrms":SHAP_vrms,"SHAP_wrms":SHAP_wrms,"SHAP_uv":SHAP_uv,
                 "SHAP_vw":SHAP_vw,"SHAP_uw":SHAP_uw,"SHAP_mrms":SHAP_mrms}
    return data_out


      
def save_rms(data_in={"file":"SHAPrms.txt","folder":"Data","SHAP_urms":[],"SHAP_vrms":[],"SHAP_wrms":[],
                      "SHAP_uv":[],"SHAP_vw":[],"SHAP_uw":[],"SHAP_mrms":[]}):
    """
    .....................................................................................................................
    # save_rms: Function for saving the RMS of the SHAP
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data for saving the RMS of the velocity.
        The default is {"file":"SHAPrms.txt","folder":"Data","SHAP_urms":[],"SHAP_vrms":[],"SHAP_wrms":[],
                        "SHAP_uv":[],"SHAP_vw":[],"SHAP_uw":[]}.
        Data:
            - file      : file of the RMS of the SHAP
            - folder    : folder of the generated data
            - SHAP_urms : RMS of the streamwise SHAP
            - SHAP_vrms : RMS of the wall-normal SHAP
            - SHAP_wrms : RMS of the spanwise SHAP
            - SHAP_uv   : Mean uv SHAP along the wall-normal distance
            - SHAP_vw   : Mean vw SHAP along the wall-normal distance
            - SHAP_uw   : Mean uw SHAP along the wall-normal distance
            - SHAP_mrms : RMS of the mean value of the SHAP

    Returns
    -------
    None.

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    file      = str(data_in["file"])                         # file of the RMS
    folder    = str(data_in["folder"])                       # folder of the RMS
    SHAP_urms = np.array(data_in["SHAP_urms"],dtype='float') # RMS of the streamwise SHAP
    SHAP_vrms = np.array(data_in["SHAP_vrms"],dtype='float') # RMS of the wall-normal SHAP
    SHAP_wrms = np.array(data_in["SHAP_wrms"],dtype='float') # RMS of the spanwise SHAP
    SHAP_uv   = np.array(data_in["SHAP_uv"],dtype='float')    # Mean uv stress
    SHAP_vw   = np.array(data_in["SHAP_vw"],dtype='float')    # Mean vw stress
    SHAP_uw   = np.array(data_in["SHAP_uw"],dtype='float')    # Mean uw stress
    SHAP_mrms = np.array(data_in["SHAP_mrms"],dtype='float')
    file_rms  = folder+'/'+file
    
    # -------------------------------------------------------------------------------------------------------------------
    # Save in the file
    # -------------------------------------------------------------------------------------------------------------------
    file_save = open(file_rms, "w+")           
    content   = str(SHAP_urms.tolist())+'\n'
    file_save.write(content)    
    content   = str(SHAP_vrms.tolist())+'\n'
    file_save.write(content)    
    content   = str(SHAP_wrms.tolist())+'\n'
    file_save.write(content)          
    content   = str(SHAP_uv.tolist())+'\n'
    file_save.write(content)    
    content   = str(SHAP_vw.tolist())+'\n'
    file_save.write(content)    
    content   = str(SHAP_uw.tolist())+'\n'
    file_save.write(content)
    content   = str(SHAP_mrms.tolist())+'\n'
    file_save.write(content)


def calc_rms(data_in={"field_ini":1000,"field_fin":9999,"field_delta":1,"SHAPmean_file":"SHAPmean.txt",
                      "data_folder":"Data","file":"../../P125_21pi_vu","folder":"P125_21pi_vu.$INDEX$.h5.uvw",
                      "dx":1,"dy":1,"dz":1,"shpx":192,"shpy":201,"shpz":96,"save_file":True,
                      "SHAPrms_file":"SHAPrms.txt"}):
    """
    .....................................................................................................................
    # calc_rms: Function to calculate the RMS of the SHAP data along the wall-normal direction
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        The data required for the calculation of the RMS.
        The default is {"field_ini":1000,"field_fin":9999,"SHAPmean_file":"SHAPmean.txt","data_folder":"Data", 
                        "file":"../P125_21pi_vu","folder":"P125_21pi_vu.$INDEX$.h5.uvw","dx":1,"dy":1,"dz":1,
                        "shpx":192,"shpy":201,"shpz":96,"save_file":True,"SHAPrms_file":"SHAPrms.txt"}.
        Data:
            - field_ini     : index of the initial field
            - field_fin     : index of the final field
            - field_delta   : separation between fields
            - SHAPmean_file : file of the mean velocity
            - data_folder   : folder of the generated data
            - file          : file of the velocity flow
            - folder        : folder of the velocity flow
            - dx            : downsampling in the streamwise direction
            - dy            : downsampling in the wall-normal direction
            - dz            : downsampling in the spanwise direction
            - shpx          : shape of the tensors in the streamwise direction
            - shpy          : shape of the tensors in the wall-normal direction
            - shpz          : shape of the tensors in the spanwise direction
            - save_file     : flag for saving the information in a file (True: the information is saved in a file,
                                                                         False: the information is stored in a variable)
            - SHAPrms_file  : file containing the information of the RMS of the SHAP

    Returns
    -------
    data_out : dict
        Data of the RMS. Only used in the case of not saving the information in a file.
        Data:
            - SHAP_urms : RMS of the streamwise SHAP
            - SHAP_vrms : RMS of the wall-normal SHAP
            - SHAP_wrms : RMS of the spanwise SHAP
            - SHAP_uv   : Mean uv SHAP along the wall-normal distance
            - SHAP_vw   : Mean vw SHAP along the wall-normal distance
            - SHAP_uw   : Mean uw SHAP along the wall-normal distance
            - SHAP_mrms : RMS of the absolute value of the SHAP

    """
    
    # -------------------------------------------------------------------------------------------------------------------
    # Load packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_functions.shapmean import read_SHAPmean
    import h5py
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    field_ini      = int(data_in["field_ini"])     # initial index of the fields
    field_fin      = int(data_in["field_fin"])     # final index of the fields
    field_delta    = int(data_in["field_delta"])
    SHAPmean_file  = str(data_in["SHAPmean_file"]) # file for the mean velocity
    data_folder    = str(data_in["data_folder"])   # folder of the generated data
    file           = str(data_in["file"])          # file name of the flow fields
    folder         = str(data_in["folder"])        # folder name of the flow fields
    dy             = int(data_in["dy"])            # downsampling in the y direction
    dx             = int(data_in["dx"])            # downsampling in the x direction
    dz             = int(data_in["dz"])            # downsampling in the z direction
    shpx           = int(data_in["shpx"])          # shape of the tensor in the x direction
    shpy           = int(data_in["shpy"])          # shape of the tensor in the y direction
    shpz           = int(data_in["shpz"])          # shape of the tensor in the z direction
    save_file      = bool(data_in["save_file"])    # flag to choose if the RMS is saved in a file
    SHAPrms_file   = str(data_in["SHAPrms_file"])  # file to store the RMS information
    file_comp      = folder+'/'+file
    for ii in range(field_ini,field_fin,field_delta):
        file_ii = file_comp.replace("$INDEX$",str(ii))
        print('RMS velocity calculation:'+str(file_ii),flush=True)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the velocity fields from the files and then calculate their maximum and minimum values
        # ---------------------------------------------------------------------------------------------------------------
        if glob.glob(file_ii):
            file = h5py.File(file_ii,'r+')
            SHAP_u   = np.array(file['SHAP_u'])[::dy,::dz,::dx]
            SHAP_v   = np.array(file['SHAP_v'])[::dy,::dz,::dx]
            SHAP_w   = np.array(file['SHAP_w'])[::dy,::dz,::dx]
            SHAP_m2  = SHAP_u**2+SHAP_v**2+SHAP_w**2
            SHAP_u2  = np.multiply(SHAP_u,SHAP_u)
            SHAP_v2  = np.multiply(SHAP_v,SHAP_v)
            SHAP_w2  = np.multiply(SHAP_w,SHAP_w)
            SHAP_uv  = np.multiply(SHAP_u,SHAP_v)
            SHAP_vw  = np.multiply(SHAP_v,SHAP_w)
            SHAP_uw  = np.multiply(SHAP_u,SHAP_w)
            if ii == field_ini:
                SHAP_u2_cum = np.sum(SHAP_u2,axis=(1,2))
                SHAP_v2_cum = np.sum(SHAP_v2,axis=(1,2))
                SHAP_w2_cum = np.sum(SHAP_w2,axis=(1,2))
                SHAP_uv_cum = np.sum(SHAP_uv,axis=(1,2))
                SHAP_vw_cum = np.sum(SHAP_vw,axis=(1,2))
                SHAP_uw_cum = np.sum(SHAP_uw,axis=(1,2))
                SHAP_m2_cum = np.sum(SHAP_m2,axis=(1,2))
                nn_cum      = np.ones((shpy,))*shpx*shpz
            else:
                SHAP_u2_cum += np.sum(SHAP_u2,axis=(1,2))
                SHAP_v2_cum += np.sum(SHAP_v2,axis=(1,2))
                SHAP_w2_cum += np.sum(SHAP_w2,axis=(1,2))
                SHAP_uv_cum += np.sum(SHAP_uv,axis=(1,2))
                SHAP_vw_cum += np.sum(SHAP_vw,axis=(1,2))
                SHAP_uw_cum += np.sum(SHAP_uw,axis=(1,2))
                SHAP_m2_cum += np.sum(SHAP_m2,axis=(1,2))
                nn_cum      += np.ones((shpy,))*shpx*shpz
        else:
            print('Skiping field '+str(ii)+' as file was not found',flush=True)
    SHAP_urms = np.sqrt(np.divide(SHAP_u2_cum,nn_cum))    
    SHAP_vrms = np.sqrt(np.divide(SHAP_v2_cum,nn_cum))   
    SHAP_wrms = np.sqrt(np.divide(SHAP_w2_cum,nn_cum)) 
    SHAP_uv   = np.divide(SHAP_uv_cum,nn_cum)
    SHAP_vw   = np.divide(SHAP_vw_cum,nn_cum)
    SHAP_uw   = np.divide(SHAP_uw_cum,nn_cum)   
    SHAP_mrms = np.sqrt(np.divide(SHAP_m2_cum,nn_cum))
    
    # -------------------------------------------------------------------------------------------------------------------
    # Save the RMS in a file or return the values of the RMS
    # -------------------------------------------------------------------------------------------------------------------
    if save_file:
        data_rms_save = {"folder":data_folder,"file":SHAPrms_file,"SHAP_urms":SHAP_urms,"SHAP_vrms":SHAP_vrms,
                         "SHAP_wrms":SHAP_wrms,"SHAP_uv":SHAP_uv,"SHAP_vw":SHAP_vw,"SHAP_uw":SHAP_uw,
                         "SHAP_mrms":SHAP_mrms}
        save_rms(data_in=data_rms_save)
    else:
        data_out = {"SHAP_urms":SHAP_urms,"SHAP_vrms":SHAP_vrms,"SHAP_wrms":SHAP_wrms,"SHAP_uv":SHAP_uv,
                    "SHAP_vw":SHAP_vw,"SHAP_uw":SHAP_uw,"SHAP_mrms":SHAP_mrms}
        return data_out
    
    
def calc_rms_nomean(data_in={"field_ini":1000,"field_fin":9999,"field_delta":1,"SHAPmean_file":"SHAPmean.txt",
                             "data_folder":"Data","file":"../../P125_21pi_vu","folder":"P125_21pi_vu.$INDEX$.h5.uvw",
                             "dx":1,"dy":1,"dz":1,"shpx":192,"shpy":201,"shpz":96,"save_file":True,
                             "SHAPrms_file":"SHAPrms.txt"}):
    """
    .....................................................................................................................
    # calc_rms_nomean: Function to calculate the RMS of the SHAP data along the wall-normal direction without the
                       mean value
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        The data required for the calculation of the RMS.
        The default is {"field_ini":1000,"field_fin":9999,"SHAPmean_file":"SHAPmean.txt","data_folder":"Data", 
                        "file":"../P125_21pi_vu","folder":"P125_21pi_vu.$INDEX$.h5.uvw","dx":1,"dy":1,"dz":1,
                        "shpx":192,"shpy":201,"shpz":96,"save_file":True,"SHAPrms_file":"SHAPrms.txt"}.
        Data:
            - field_ini     : index of the initial field
            - field_fin     : index of the final field
            - field_delta   : separation between fields
            - SHAPmean_file : file of the mean velocity
            - data_folder   : folder of the generated data
            - file          : file of the velocity flow
            - folder        : folder of the velocity flow
            - dx            : downsampling in the streamwise direction
            - dy            : downsampling in the wall-normal direction
            - dz            : downsampling in the spanwise direction
            - shpx          : shape of the tensors in the streamwise direction
            - shpy          : shape of the tensors in the wall-normal direction
            - shpz          : shape of the tensors in the spanwise direction
            - save_file     : flag for saving the information in a file (True: the information is saved in a file,
                                                                         False: the information is stored in a variable)
            - SHAPrms_file  : file containing the information of the RMS of the SHAP

    Returns
    -------
    data_out : dict
        Data of the RMS. Only used in the case of not saving the information in a file.
        Data:
            - SHAP_urms : RMS of the streamwise SHAP
            - SHAP_vrms : RMS of the wall-normal SHAP
            - SHAP_wrms : RMS of the spanwise SHAP
            - SHAP_uv   : Mean uv SHAP along the wall-normal distance
            - SHAP_vw   : Mean vw SHAP along the wall-normal distance
            - SHAP_uw   : Mean uw SHAP along the wall-normal distance
            - SHAP_mrms : RMS of the absolute value of the SHAP

    """
    
    # -------------------------------------------------------------------------------------------------------------------
    # Load packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_functions.shapmean import read_SHAPmean
    import h5py
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    field_ini      = int(data_in["field_ini"])     # initial index of the fields
    field_fin      = int(data_in["field_fin"])     # final index of the fields
    field_delta    = int(data_in["field_delta"])
    SHAPmean_file  = str(data_in["SHAPmean_file"]) # file for the mean velocity
    data_folder    = str(data_in["data_folder"])   # folder of the generated data
    file           = str(data_in["file"])          # file name of the flow fields
    folder         = str(data_in["folder"])        # folder name of the flow fields
    dy             = int(data_in["dy"])            # downsampling in the y direction
    dx             = int(data_in["dx"])            # downsampling in the x direction
    dz             = int(data_in["dz"])            # downsampling in the z direction
    shpx           = int(data_in["shpx"])          # shape of the tensor in the x direction
    shpy           = int(data_in["shpy"])          # shape of the tensor in the y direction
    shpz           = int(data_in["shpz"])          # shape of the tensor in the z direction
    save_file      = bool(data_in["save_file"])    # flag to choose if the RMS is saved in a file
    SHAPrms_file   = str(data_in["SHAPrms_file"])  # file to store the RMS information
    SHAPrms_file   = SHAPrms_file.replace(".txt","_nomean.txt")
    file_comp      = folder+'/'+file
    try:
        data_SHAPmean  = read_SHAPmean(data_in={"folder":data_folder,"file":SHAPmean_file,"dy":dy})
        SHAP_umean     = data_SHAPmean["SHAP_umean"]
        SHAP_vmean     = data_SHAPmean["SHAP_vmean"]
        SHAP_wmean     = data_SHAPmean["SHAP_wmean"]
    except:
        print("RMS calculations require mean velocity file. Breaking calculation...",flush=True)
        sys.exit()
    for ii in range(field_ini,field_fin,field_delta):
        file_ii = file_comp.replace("$INDEX$",str(ii))
        print('RMS velocity calculation:'+str(file_ii),flush=True)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the velocity fields from the files and then calculate their maximum and minimum values
        # ---------------------------------------------------------------------------------------------------------------
        if glob.glob(file_ii):
            file = h5py.File(file_ii,'r+')
            SHAP_u   = np.array(file['SHAP_u'])[::dy,::dz,::dx]-SHAP_umean.reshape(-1,1,1)
            SHAP_v   = np.array(file['SHAP_v'])[::dy,::dz,::dx]-SHAP_vmean.reshape(-1,1,1)
            SHAP_w   = np.array(file['SHAP_w'])[::dy,::dz,::dx]-SHAP_wmean.reshape(-1,1,1)
            SHAP_m2  = SHAP_u**2+SHAP_v**2+SHAP_w**2
            SHAP_u2  = np.multiply(SHAP_u,SHAP_u)
            SHAP_v2  = np.multiply(SHAP_v,SHAP_v)
            SHAP_w2  = np.multiply(SHAP_w,SHAP_w)
            SHAP_uv  = np.multiply(SHAP_u,SHAP_v)
            SHAP_vw  = np.multiply(SHAP_v,SHAP_w)
            SHAP_uw  = np.multiply(SHAP_u,SHAP_w)
            if ii == field_ini:
                SHAP_u2_cum = np.sum(SHAP_u2,axis=(1,2))
                SHAP_v2_cum = np.sum(SHAP_v2,axis=(1,2))
                SHAP_w2_cum = np.sum(SHAP_w2,axis=(1,2))
                SHAP_uv_cum = np.sum(SHAP_uv,axis=(1,2))
                SHAP_vw_cum = np.sum(SHAP_vw,axis=(1,2))
                SHAP_uw_cum = np.sum(SHAP_uw,axis=(1,2))
                SHAP_m2_cum = np.sum(SHAP_m2,axis=(1,2))
                nn_cum      = np.ones((shpy,))*shpx*shpz
            else:
                SHAP_u2_cum += np.sum(SHAP_u2,axis=(1,2))
                SHAP_v2_cum += np.sum(SHAP_v2,axis=(1,2))
                SHAP_w2_cum += np.sum(SHAP_w2,axis=(1,2))
                SHAP_uv_cum += np.sum(SHAP_uv,axis=(1,2))
                SHAP_vw_cum += np.sum(SHAP_vw,axis=(1,2))
                SHAP_uw_cum += np.sum(SHAP_uw,axis=(1,2))
                SHAP_m2_cum += np.sum(SHAP_m2,axis=(1,2))
                nn_cum      += np.ones((shpy,))*shpx*shpz
        else:
            print('Skiping field '+str(ii)+' as file was not found',flush=True)
    SHAP_urms = np.sqrt(np.divide(SHAP_u2_cum,nn_cum))    
    SHAP_vrms = np.sqrt(np.divide(SHAP_v2_cum,nn_cum))   
    SHAP_wrms = np.sqrt(np.divide(SHAP_w2_cum,nn_cum)) 
    SHAP_uv   = np.divide(SHAP_uv_cum,nn_cum)
    SHAP_vw   = np.divide(SHAP_vw_cum,nn_cum)
    SHAP_uw   = np.divide(SHAP_uw_cum,nn_cum)   
    SHAP_mrms = np.sqrt(np.divide(SHAP_m2_cum,nn_cum))
    
    # -------------------------------------------------------------------------------------------------------------------
    # Save the RMS in a file or return the values of the RMS
    # -------------------------------------------------------------------------------------------------------------------
    if save_file:
        data_rms_save = {"folder":data_folder,"file":SHAPrms_file,"SHAP_urms":SHAP_urms,"SHAP_vrms":SHAP_vrms,
                         "SHAP_wrms":SHAP_wrms,"SHAP_uv":SHAP_uv,"SHAP_vw":SHAP_vw,"SHAP_uw":SHAP_uw,
                         "SHAP_mrms":SHAP_mrms}
        save_rms(data_in=data_rms_save)
    else:
        data_out = {"SHAP_urms":SHAP_urms,"SHAP_vrms":SHAP_vrms,"SHAP_wrms":SHAP_wrms,"SHAP_uv":SHAP_uv,
                    "SHAP_vw":SHAP_vw,"SHAP_uw":SHAP_uw,"SHAP_mrms":SHAP_mrms}
        return data_out