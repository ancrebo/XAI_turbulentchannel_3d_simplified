# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
calc_coinc_shap_uw_vsign.py
-------------------------------------------------------------------------------------------------------------------------
Created on Wed May 15 11:58:23 2024

@author: Andres Cremades Botella

File containing the function to calculate the coincidence between the quadrant structures of the SHAP streaks with the
sign of the shap v. The file contains the following functions:
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
    

def save_coinc(data_in={"frac_struc1":[],"frac_struc2":[],"yplus":[],
                        "calc_coin_file":"calc_coin.txt","folder":"data"}):
    """
    .....................................................................................................................
    # save_coinc: Function to save the coincidence in a file
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"frac_struc1":[],"frac_struc2":[],"yplus":[],
                                     "calc_coin_file":"calc_coin.txt","folder":"data"}.
        Data:
            - frac_struc1    : fraction of volume of the structures Q1
            - frac_struc2    : fraction of volume of the structures Q2
            - yplus          : wall-normal distance
            - calc_coin_file : file to save the coincidence
            - folder         : folder to save the coincidence file
    
    Returns
    -------
    None
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    frac_struc1    = np.array(data_in["frac_struc1"],dtype="float32")
    frac_struc2    = np.array(data_in["frac_struc2"],dtype="float32")
    yplus          = np.array(data_in["yplus"],dtype="float32")
    calc_coin_file = str(data_in["calc_coin_file"])
    folder         = str(data_in["folder"])
    
    # -------------------------------------------------------------------------------------------------------------------
    # Save the information in a file
    # -------------------------------------------------------------------------------------------------------------------    
    file_coin = folder+'/'+calc_coin_file                     
    file_save = open(file_coin, "w")           
    content   = str(yplus.tolist())+'\n'
    file_save.write(content)         
    content   = str(frac_struc1.tolist())+'\n'
    file_save.write(content)         
    content   = str(frac_struc2.tolist())+'\n'
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
            - frac_struc1  : fraction of volume of the structures Q100
            - frac_struc2  : fraction of volume of the structures Q001
            - yplus        : wall-normal distance
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    folder    = str(data_in["folder"]) # folder for reading the mean velocity data
    file      = str(data_in["calc_coin_file"])   # file for reading the mean velocity data
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the velocity from file
    # -------------------------------------------------------------------------------------------------------------------
    file_perc   = folder+'/'+file
    file_read   = open(file_perc,"r")
    yplus       = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_struc1 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_struc2 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    data_out    = {"frac_struc1":frac_struc1,"frac_struc2":frac_struc2,"yplus":yplus}
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
            - frac_struc1  : fraction of volume of the structures Q1
            - frac_struc2  : fraction of volume of the structures Q2
            - yplus        : wall-normal distance

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
    mat_1         = np.array(struc.structures_1.mat_struc,dtype='float')
    mat_2         = np.array(struc.structures_2.mat_struc,dtype='float')
    shp           = mat_1.shape
    shp_slice     = shp[1]*shp[2]
    frac_struc1_h = np.sum(mat_1,axis=(1,2))/shp_slice
    frac_struc2_h = np.sum(mat_2,axis=(1,2))/shp_slice
    frac_struc1   = (frac_struc1_h[:flow_data.yl_s]+np.flip(frac_struc1_h[flow_data.yu_s:]))/2
    frac_struc2   = (frac_struc2_h[:flow_data.yl_s]+np.flip(frac_struc2_h[flow_data.yu_s:]))/2
        
    # -------------------------------------------------------------------------------------------------------------------
    # Store the output data in a file
    # -------------------------------------------------------------------------------------------------------------------
    if save_data:
        save_coinc(data_in={"frac_struc1":frac_struc1,"frac_struc2":frac_struc2,"calc_coin_file":calc_coin_file,
                            "folder":folder,"yplus":flow_data.yplus})
    else:
        data_out = {"frac_struc1":frac_struc1,"frac_struc2":frac_struc2,"yplus":flow_data.yplus}
        return data_out

