# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
urms.py
-------------------------------------------------------------------------------------------------------------------------
Created on Wed Mar 27 08:50:04 2024

@author: Andres Cremades Botella

File to create the RMS values for the velocity fields. Functions contained in the file:
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

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
def read_rms(data_in={"file":"Urms.txt","folder":"Data"}):
    """
    .....................................................................................................................
    # read_rms: Function for reading the RMS
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data for reading the RMS file. The default is {"file":"Urms.txt","folder":"Data"}.
        Data:
            - file   : file of the RMS of the velocity
            - folder : folder of the generated data

    Returns
    -------
    data_out : dict
        Data of the RMS of the velocity.
        Data:
            - uurms : RMS of the streamwise velocity
            - vvrms : RMS of the wall-normal velocity
            - wwrms : RMS of the spanwise velocity
            - uv    : Mean uv stress along the wall-normal distance
            - vw    : Mean vw stress along the wall-normal distance
            - uw    : Mean uw stress along the wall-normal distance

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    file   = str(data_in["file"])   # File of the RMS of the velocity
    folder = str(data_in["folder"]) # Folder of the generated data
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data of the rms
    # -------------------------------------------------------------------------------------------------------------------
    file_rms  = folder+'/'+file
    file_read = open(file_rms,"r")
    uurms     = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    vvrms     = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    wwrms     = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    uv        = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    vw        = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    uw        = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    data_out  = {"uurms":uurms,"vvrms":vvrms,"wwrms":wwrms,"uv":uv,"vw":vw,"uw":uw}
    return data_out


      
def save_rms(data_in={"file":"Urms.txt","folder":"Data","uurms":[],"vvrms":[],"wwrms":[],"uv":[],"vw":[],"uw":[]}):
    """
    .....................................................................................................................
    # save_rms: Function for saving the RMS of the velocity
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data for saving the RMS of the velocity.
        The default is {"file":"Urms.txt","folder":"Data","uurms":[],"vvrms":[],"wwrms":[],"uv":[],"vw":[],"uw":[]}.
        Data:
            - file : file of the RMS of the velocity
            - folder : folder of the generated data
            - uurms : RMS of the streamwise velocity
            - vvrms : RMS of the wall-normal velocity
            - wwrms : RMS of the spanwise velocity
            - uv    : Mean uv stress along the wall-normal distance
            - vw    : Mean vw stress along the wall-normal distance
            - uw    : Mean uw stress along the wall-normal distance

    Returns
    -------
    None.

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    file      = str(data_in["file"])                     # file of the RMS
    folder    = str(data_in["folder"])                   # folder of the RMS
    uurms     = np.array(data_in["uurms"],dtype='float') # RMS of the streamwise velocity
    vvrms     = np.array(data_in["vvrms"],dtype='float') # RMS of the wall-normal velocity
    wwrms     = np.array(data_in["wwrms"],dtype='float') # RMS of the spanwise velocity
    uv        = np.array(data_in["uv"],dtype='float')    # Mean uv stress
    vw        = np.array(data_in["vw"],dtype='float')    # Mean vw stress
    uw        = np.array(data_in["uw"],dtype='float')    # Mean uw stress
    file_rms  = folder+'/'+file
    
    # -------------------------------------------------------------------------------------------------------------------
    # Save in the file
    # -------------------------------------------------------------------------------------------------------------------
    file_save = open(file_rms, "w+")           
    content   = str(uurms.tolist())+'\n'
    file_save.write(content)    
    content   = str(vvrms.tolist())+'\n'
    file_save.write(content)    
    content   = str(wwrms.tolist())+'\n'
    file_save.write(content)          
    content   = str(uv.tolist())+'\n'
    file_save.write(content)    
    content   = str(vw.tolist())+'\n'
    file_save.write(content)    
    content   = str(uw.tolist())+'\n'
    file_save.write(content)


def calc_rms(data_in={"field_ini":1000,"field_fin":9999,"umean_file":"Umean.txt","data_folder":"Data",\
                       "file":"../../P125_21pi_vu","folder":"P125_21pi_vu.$INDEX$.h5.uvw","dx":1,"dy":1,"dz":1,\
                           "shpx":192,"shpy":201,"shpz":96,"save_file":True,"urms_file":"Urms.txt"}):
    """
    .....................................................................................................................
    # calc_rms: Function to calculate the RMS of the velocity data along the wall-normal direction
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        The data required for the calculation of the RMS.
        The default is {"field_ini":1000,"field_fin":9999,"umean_file":"Umean.txt","data_folder":"Data", 
                        "file":"../P125_21pi_vu","folder":"P125_21pi_vu.$INDEX$.h5.uvw","dx":1,"dy":1,"dz":1,
                           "shpx":192,"shpy":201,"shpz":96,"save_file":True,"urms_file":"Urms.txt"}.
        Data:
            - field_ini   : index of the initial field
            - field_fin   : index of the final field
            - umean_file  : file of the mean velocity
            - data_folder : folder of the generated data
            - file        : file of the velocity flow
            - folder      : folder of the velocity flow
            - dx          : downsampling in the streamwise direction
            - dy          : downsampling in the wall-normal direction
            - dz          : downsampling in the spanwise direction
            - shpx        : shape of the tensors in the streamwise direction
            - shpy        : shape of the tensors in the wall-normal direction
            - shpz        : shape of the tensors in the spanwise direction
            - save_file   : flag for saving the information in a file (True: the information is saved in a file,
                                                                       False: the information is stored in a variable)
            - urms_file   : file containing the information of the RMS of the velocity

    Returns
    -------
    data_out : dict
        Data of the RMS. Only used in the case of not saving the information in a file.
        Data:
            - uurms : RMS of the streamwise velocity
            - vvrms : RMS of the wall-normal velocity
            - wwrms : RMS of the spanwise velocity
            - uv    : Mean uv stress along the wall-normal distance
            - vw    : Mean vw stress along the wall-normal distance
            - uw    : Mean uw stress along the wall-normal distance

    """
    
    # -------------------------------------------------------------------------------------------------------------------
    # Load packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_functions.umean import read_Umean
    import h5py
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    field_ini   = int(data_in["field_ini"])    # initial index of the fields
    field_fin   = int(data_in["field_fin"])    # final index of the fields
    umean_file  = str(data_in["umean_file"])   # file for the mean velocity
    data_folder = str(data_in["data_folder"])  # folder of the generated data
    file        = str(data_in["file"])         # file name of the flow fields
    folder      = str(data_in["folder"])       # folder name of the flow fields
    dy          = int(data_in["dy"])           # downsampling in the y direction
    dx          = int(data_in["dx"])           # downsampling in the x direction
    dz          = int(data_in["dz"])           # downsampling in the z direction
    shpx        = int(data_in["shpx"])         # shape of the tensor in the x direction
    shpy        = int(data_in["shpy"])         # shape of the tensor in the y direction
    shpz        = int(data_in["shpz"])         # shape of the tensor in the z direction
    save_file   = bool(data_in["save_file"])   # flag to choose if the RMS is saved in a file
    urms_file   = str(data_in["urms_file"])    # file to store the RMS information
    file_comp   = folder+'/'+file
    try:
        data_umean  = read_Umean(data_in={"folder":data_folder,"file":umean_file,"dy":dy})
        UUmean      = data_umean["UUmean"]
    except:
        print("RMS calculations require mean velocity file. Breaking calculation...",flush=True)
        sys.exit()
    for ii in range(field_ini,field_fin):
        file_ii = file_comp.replace("$INDEX$",str(ii))
        print('RMS velocity calculation:'+str(file_ii),flush=True)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the velocity fields from the files and then calculate their maximum and minimum values
        # ---------------------------------------------------------------------------------------------------------------
        file = h5py.File(file_ii,'r')
        UU   = np.array(file['u'])[::dy,::dz,::dx]
        uu   = UU-UUmean.reshape(-1,1,1)
        vv   = np.array(file['v'])[::dy,::dz,::dx]
        ww   = np.array(file['w'])[::dy,::dz,::dx]
        uu2  = np.multiply(uu,uu)
        vv2  = np.multiply(vv,vv)
        ww2  = np.multiply(ww,ww)
        uv   = np.multiply(uu,vv)
        vw   = np.multiply(vv,ww)
        uw   = np.multiply(uu,ww)
        if ii == field_ini:
            uu2_cum = np.sum(uu2,axis=(1,2))
            vv2_cum = np.sum(vv2,axis=(1,2))
            ww2_cum = np.sum(ww2,axis=(1,2))
            uv_cum  = np.sum(uv,axis=(1,2))
            vw_cum  = np.sum(vw,axis=(1,2))
            uw_cum  = np.sum(uw,axis=(1,2))
            nn_cum  = np.ones((shpy,))*shpx*shpz
        else:
            uu2_cum += np.sum(uu2,axis=(1,2))
            vv2_cum += np.sum(vv2,axis=(1,2))
            ww2_cum += np.sum(ww2,axis=(1,2))
            uv_cum  += np.sum(uv,axis=(1,2))
            vw_cum  += np.sum(vw,axis=(1,2))
            uw_cum  += np.sum(uw,axis=(1,2))
            nn_cum  += np.ones((shpy,))*shpx*shpz
    uurms = np.sqrt(np.divide(uu2_cum,nn_cum))    
    vvrms = np.sqrt(np.divide(vv2_cum,nn_cum))   
    wwrms = np.sqrt(np.divide(ww2_cum,nn_cum)) 
    uv    = np.divide(uv_cum,nn_cum)
    vw    = np.divide(vw_cum,nn_cum)
    uw    = np.divide(uw_cum,nn_cum)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Save the RMS in a file or return the values of the RMS
    # -------------------------------------------------------------------------------------------------------------------
    if save_file:
        data_rms_save = {"folder":data_folder,"file":urms_file,"uurms":uurms,"vvrms":vvrms,"wwrms":wwrms,\
                          "uv":uv,"vw":vw,"uw":uw}
        save_rms(data_in=data_rms_save)
    else:
        data_out = {"uurms":uurms,"vvrms":vvrms,"wwrms":wwrms,"uv":uv,"vw":vw,"uw":uw}
        return data_out