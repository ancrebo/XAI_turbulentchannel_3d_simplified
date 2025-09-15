# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
calc_coinc_shap_uvw.py
-------------------------------------------------------------------------------------------------------------------------
Created on Wed May 15 11:58:23 2024

@author: Andres Cremades Botella

File containing the function to calculate the coincidence between the quadrant structures of the SHAP uvw structures 
The file contains the following functions:
    Functions:
        - calc_coinc : function for calculating the coincidence
        - save_coinc : function for saving the coincidence
        - read_coinc : function to read the coincidence
"""
# -----------------------------------------------------------------------------------------------------------------------
# Import packages
# -----------------------------------------------------------------------------------------------------------------------
import numpy as np
import sys
    

def save_coinc(data_in={"frac_strucQ100":[],"frac_strucQ001":[],"frac_strucQ010":[],"frac_strucQ020":[],
                        "frac_strucQ101":[],"frac_strucQ110":[],"frac_strucQ120":[],"frac_strucQ011":[],
                        "frac_strucQ021":[],"frac_strucQ111":[],"frac_strucQ121":[],"yplus":[],
                        "calc_coin_file":"calc_coin.txt","folder":"data"}):
    """
    .....................................................................................................................
    # save_coinc: Function to save the coincidence in a file
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"frac_strucQ100":[],"frac_strucQ001":[],"frac_strucQ010":[],"frac_strucQ020":[],
                                     "frac_strucQ101":[],"frac_strucQ110":[],"frac_strucQ120":[],"frac_strucQ011":[],
                                     "frac_strucQ021":[],"frac_strucQ111":[],"frac_strucQ121":[],"yplus":[],
                                     "calc_coin_file":"calc_coin.txt","folder":"data"}.
        Data:
            - frac_strucQ100  : fraction of volume of the structures Q100
            - frac_strucQ001  : fraction of volume of the structures Q001
            - frac_strucQ010  : fraction of volume of the structures Q010
            - frac_strucQ020  : fraction of volume of the structures Q020
            - frac_strucQ101  : fraction of volume of the structures Q101
            - frac_strucQ110  : fraction of volume of the structures Q110
            - frac_strucQ120  : fraction of volume of the structures Q120
            - frac_strucQ011  : fraction of volume of the structures Q011
            - frac_strucQ021  : fraction of volume of the structures Q021
            - frac_strucQ111  : fraction of volume of the structures Q111
            - frac_strucQ121  : fraction of volume of the structures Q121
            - yplus           : wall-normal distance
    
    Returns
    -------
    None
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    frac_strucQ100    = np.array(data_in["frac_strucQ100"],dtype="float32")
    frac_strucQ001    = np.array(data_in["frac_strucQ001"],dtype="float32")
    frac_strucQ010    = np.array(data_in["frac_strucQ010"],dtype="float32")
    frac_strucQ020    = np.array(data_in["frac_strucQ020"],dtype="float32")
    frac_strucQ101    = np.array(data_in["frac_strucQ101"],dtype="float32")
    frac_strucQ110    = np.array(data_in["frac_strucQ110"],dtype="float32")
    frac_strucQ120    = np.array(data_in["frac_strucQ120"],dtype="float32")
    frac_strucQ011    = np.array(data_in["frac_strucQ011"],dtype="float32")
    frac_strucQ021    = np.array(data_in["frac_strucQ021"],dtype="float32")
    frac_strucQ111    = np.array(data_in["frac_strucQ111"],dtype="float32")
    frac_strucQ121    = np.array(data_in["frac_strucQ121"],dtype="float32")
    yplus             = np.array(data_in["yplus"],dtype="float32")
    calc_coin_file    = str(data_in["calc_coin_file"])
    folder            = str(data_in["folder"])
    
    # -------------------------------------------------------------------------------------------------------------------
    # Save the information in a file
    # -------------------------------------------------------------------------------------------------------------------    
    file_coin = folder+'/'+calc_coin_file                     
    file_save = open(file_coin, "w")           
    content   = str(yplus.tolist())+'\n'
    file_save.write(content)         
    content   = str(frac_strucQ100.tolist())+'\n'
    file_save.write(content)         
    content   = str(frac_strucQ001.tolist())+'\n'
    file_save.write(content)         
    content   = str(frac_strucQ010.tolist())+'\n'
    file_save.write(content)         
    content   = str(frac_strucQ020.tolist())+'\n'
    file_save.write(content)         
    content   = str(frac_strucQ101.tolist())+'\n'
    file_save.write(content)         
    content   = str(frac_strucQ110.tolist())+'\n'
    file_save.write(content)         
    content   = str(frac_strucQ120.tolist())+'\n'
    file_save.write(content)         
    content   = str(frac_strucQ011.tolist())+'\n'
    file_save.write(content)         
    content   = str(frac_strucQ021.tolist())+'\n'
    file_save.write(content)         
    content   = str(frac_strucQ111.tolist())+'\n'
    file_save.write(content)         
    content   = str(frac_strucQ121.tolist())+'\n'
    file_save.write(content)  
    file_save.close()


def read_coinc(data_in={"calc_coin_file":"calc_coin.txt","folder":"data"}):
    """
    .....................................................................................................................
    # read_coinc: Function to read the coincidence between structures in a file
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"calc_coin_file":"calc_coin.txt","folder":"data"}.
        Data:
            - calc_coin_file : file of the coincidence analysis
            - folder         : folder to save the analysis
    
    Returns
    -------
    dict
        Structure containing the information of the percolations
        Data:
            - frac_strucQ100  : fraction of volume of the structures Q100
            - frac_strucQ001  : fraction of volume of the structures Q001
            - frac_strucQ010  : fraction of volume of the structures Q010
            - frac_strucQ020  : fraction of volume of the structures Q020
            - frac_strucQ101  : fraction of volume of the structures Q101
            - frac_strucQ110  : fraction of volume of the structures Q110
            - frac_strucQ120  : fraction of volume of the structures Q120
            - frac_strucQ011  : fraction of volume of the structures Q011
            - frac_strucQ021  : fraction of volume of the structures Q021
            - frac_strucQ111  : fraction of volume of the structures Q111
            - frac_strucQ121  : fraction of volume of the structures Q121
            - yplus           : wall-normal distance
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    folder    = str(data_in["folder"]) # folder for reading the mean velocity data
    file      = str(data_in["calc_coin_file"])   # file for reading the mean velocity data
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the velocity from file
    # -------------------------------------------------------------------------------------------------------------------
    file_perc      = folder+'/'+file
    file_read      = open(file_perc,"r")
    yplus          = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_strucQ100 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_strucQ001 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_strucQ010 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_strucQ020 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_strucQ101 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_strucQ110 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_strucQ120 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_strucQ011 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_strucQ021 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_strucQ111 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_strucQ121 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    data_out       = {"frac_strucQ100":frac_strucQ100,"frac_strucQ001":frac_strucQ001,
                      "frac_strucQ010":frac_strucQ010,"frac_strucQ020":frac_strucQ020,
                      "frac_strucQ101":frac_strucQ101,"frac_strucQ110":frac_strucQ110,
                      "frac_strucQ120":frac_strucQ120,"frac_strucQ011":frac_strucQ011,
                      "frac_strucQ021":frac_strucQ021,"frac_strucQ111":frac_strucQ111,
                      "frac_strucQ121":frac_strucQ121,"yplus":yplus}
    return data_out 

def calc_coinc(data_in={"data_struc":[],"save_data":True,"calc_coin_file":"calc_coin.txt",
                        "folder":"data","dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                        "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                        "rey":125,"utau":0.060523258443963}):
    """
    .....................................................................................................................
    # calc_coinc: Function for calculating the coincidence between structures
    .....................................................................................................................

    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"data_struc":[],"save_data":True,
                                     "calc_coin_file":"calc_coin.txt","folder":"data","dy":1,"dx":1,"dz":1,
                                     "uvw_folder":"../../P125_21pi_vu/",
                                     "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                                     "rey":125,"utau":0.060523258443963}.
        Data:
            - data_struc     : data required for the shap uvw structures
            - save_data      : flag for saving the data (True: saves in a file, False: not save in a file)
            - calc_coin_file : percolation file
            - folder         : folder to store the data
            - dy             : downsampling in the wall-normal direction
            - dx             : downsampling in the streamwise direction
            - dz             : downsampling in the spanwise direction
            - uvw_folder     : folder of the flow fields
            - uvw_file       : file of the flow fields
            - L_x            : streamwise dimension of the channel
            - L_y            : wall-normal dimension of the channel
            - L_z            : spanwise dimension of the channel
            - rey            : friction Reynolds number
            - utau           : friction velocity

    Returns
    -------
    dict
        Structure containing the information of the percolations
        Data:
            - frac_strucQ100  : fraction of volume of the structures Q100
            - frac_strucQ001  : fraction of volume of the structures Q001
            - frac_strucQ010  : fraction of volume of the structures Q010
            - frac_strucQ020  : fraction of volume of the structures Q020
            - frac_strucQ101  : fraction of volume of the structures Q101
            - frac_strucQ110  : fraction of volume of the structures Q110
            - frac_strucQ120  : fraction of volume of the structures Q120
            - frac_strucQ011  : fraction of volume of the structures Q011
            - frac_strucQ021  : fraction of volume of the structures Q021
            - frac_strucQ111  : fraction of volume of the structures Q111
            - frac_strucQ121  : fraction of volume of the structures Q121
            - yplus           : wall-normal distance

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    struc          = data_in["data_struc"]
    save_data      = bool(data_in["save_data"])
    calc_coin_file = str(data_in["calc_coin_file"])
    folder         = str(data_in["folder"])
    dy             = int(data_in["dy"])
    dx             = int(data_in["dx"])
    dz             = int(data_in["dz"])
    uvw_folder     = str(data_in["uvw_folder"])
    uvw_file       = str(data_in["uvw_file"])
    L_x            = float(data_in["L_x"])
    L_y            = float(data_in["L_y"])
    L_z            = float(data_in["L_z"])
    rey            = float(data_in["rey"])
    utau           = float(data_in["utau"])
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the file mean velocity
    # -------------------------------------------------------------------------------------------------------------------
    Data_flow = {"folder":uvw_folder,"file":uvw_file,"down_x":dx,"down_y":dy,"down_z":dz,
                 "L_x":L_x,"L_y":L_y,"L_z":L_z,"rey":rey,"utau":utau}
    flow_data = flow_field(data_in=Data_flow)
    flow_data.shape_tensor()
    flow_data.flow_grid()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data of the structures
    # -------------------------------------------------------------------------------------------------------------------
    mat_Q100   = np.array(struc.structures_Q100.mat_struc,dtype='float')
    mat_Q001   = np.array(struc.structures_Q001.mat_struc,dtype='float')
    mat_check  = mat_Q100+mat_Q001
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q010   = np.array(struc.structures_Q010.mat_struc,dtype='float')
    mat_check += mat_Q010
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q020   = np.array(struc.structures_Q020.mat_struc,dtype='float')
    mat_check += mat_Q020
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q101   = np.array(struc.structures_Q101.mat_struc,dtype='float')
    mat_check += mat_Q101
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q110   = np.array(struc.structures_Q110.mat_struc,dtype='float')
    mat_check += mat_Q110
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q120   = np.array(struc.structures_Q120.mat_struc,dtype='float')
    mat_check += mat_Q120
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q011   = np.array(struc.structures_Q011.mat_struc,dtype='float')
    mat_check += mat_Q011
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q021   = np.array(struc.structures_Q021.mat_struc,dtype='float')
    mat_check += mat_Q021
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q111   = np.array(struc.structures_Q111.mat_struc,dtype='float')
    mat_check += mat_Q111
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q121   = np.array(struc.structures_Q121.mat_struc,dtype='float')
    mat_check += mat_Q121
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_struc1       = mat_Q100
    shp              = mat_struc1.shape
    shp_slice        = shp[1]*shp[2]
    frac_strucQ100_h = np.sum(mat_Q100,axis=(1,2))/shp_slice
    frac_strucQ001_h = np.sum(mat_Q001,axis=(1,2))/shp_slice
    frac_strucQ010_h = np.sum(mat_Q010,axis=(1,2))/shp_slice
    frac_strucQ020_h = np.sum(mat_Q020,axis=(1,2))/shp_slice
    frac_strucQ101_h = np.sum(mat_Q101,axis=(1,2))/shp_slice
    frac_strucQ110_h = np.sum(mat_Q110,axis=(1,2))/shp_slice
    frac_strucQ120_h = np.sum(mat_Q120,axis=(1,2))/shp_slice
    frac_strucQ011_h = np.sum(mat_Q011,axis=(1,2))/shp_slice
    frac_strucQ021_h = np.sum(mat_Q021,axis=(1,2))/shp_slice
    frac_strucQ111_h = np.sum(mat_Q111,axis=(1,2))/shp_slice
    frac_strucQ121_h = np.sum(mat_Q121,axis=(1,2))/shp_slice
    frac_strucQ100   = (frac_strucQ100_h[:flow_data.yl_s]+np.flip(frac_strucQ100_h[flow_data.yu_s:]))/2
    frac_strucQ001   = (frac_strucQ001_h[:flow_data.yl_s]+np.flip(frac_strucQ001_h[flow_data.yu_s:]))/2
    frac_strucQ010   = (frac_strucQ010_h[:flow_data.yl_s]+np.flip(frac_strucQ010_h[flow_data.yu_s:]))/2
    frac_strucQ020   = (frac_strucQ020_h[:flow_data.yl_s]+np.flip(frac_strucQ020_h[flow_data.yu_s:]))/2
    frac_strucQ101   = (frac_strucQ101_h[:flow_data.yl_s]+np.flip(frac_strucQ101_h[flow_data.yu_s:]))/2
    frac_strucQ110   = (frac_strucQ110_h[:flow_data.yl_s]+np.flip(frac_strucQ110_h[flow_data.yu_s:]))/2
    frac_strucQ120   = (frac_strucQ120_h[:flow_data.yl_s]+np.flip(frac_strucQ120_h[flow_data.yu_s:]))/2
    frac_strucQ011   = (frac_strucQ011_h[:flow_data.yl_s]+np.flip(frac_strucQ011_h[flow_data.yu_s:]))/2
    frac_strucQ021   = (frac_strucQ021_h[:flow_data.yl_s]+np.flip(frac_strucQ021_h[flow_data.yu_s:]))/2
    frac_strucQ111   = (frac_strucQ111_h[:flow_data.yl_s]+np.flip(frac_strucQ111_h[flow_data.yu_s:]))/2
    frac_strucQ121   = (frac_strucQ121_h[:flow_data.yl_s]+np.flip(frac_strucQ121_h[flow_data.yu_s:]))/2
    
        
    # -------------------------------------------------------------------------------------------------------------------
    # Store the output data in a file
    # -------------------------------------------------------------------------------------------------------------------
    if save_data:
        save_coinc(data_in={"frac_strucQ100":frac_strucQ100,"frac_strucQ001":frac_strucQ001,
                            "frac_strucQ010":frac_strucQ010,"frac_strucQ020":frac_strucQ020,
                            "frac_strucQ101":frac_strucQ101,"frac_strucQ110":frac_strucQ110,
                            "frac_strucQ120":frac_strucQ120,"frac_strucQ011":frac_strucQ011,
                            "frac_strucQ021":frac_strucQ021,"frac_strucQ111":frac_strucQ111,
                            "frac_strucQ121":frac_strucQ121,"calc_coin_file":calc_coin_file,
                            "folder":folder,"yplus":flow_data.yplus})
    else:
        data_out = {"frac_strucQ100":frac_strucQ100,"frac_strucQ001":frac_strucQ001,
                    "frac_strucQ010":frac_strucQ010,"frac_strucQ020":frac_strucQ020,
                    "frac_strucQ101":frac_strucQ101,"frac_strucQ110":frac_strucQ110,
                    "frac_strucQ120":frac_strucQ120,"frac_strucQ011":frac_strucQ011,
                    "frac_strucQ021":frac_strucQ021,"frac_strucQ111":frac_strucQ111,
                    "frac_strucQ121":frac_strucQ121,"yplus":flow_data.yplus}
        return data_out

