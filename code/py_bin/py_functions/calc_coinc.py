# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
calc_coinc.py
-------------------------------------------------------------------------------------------------------------------------
Created on Wed May 15 11:58:23 2024

@author: Andres Cremades Botella

File containing the function to calculate the coincidence between two types of structure. 
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
    

def save_coinc(data_in={"frac_struc1":[],"frac_struc2":[],"frac_coinc":[],
                        "yplus":[],"calc_coin_file":"calc_coin.txt","folder":"data"}):
    """
    .....................................................................................................................
    # save_coinc: Function to save the coincidence in a file
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"frac_struc1":[],"frac_struc2":[],"frac_coinc":[],
                                     "yplus":[],"calc_coin_file":"calc_coin.txt","folder":"data"}.
        Data:
            - frac_struc1    : fraction of volume of the structure 1
            - frac_struc2    : fraction of volume of the structure 2
            - frac_coinc     : fraction of volume of the coincidence of 1 and 2
            - yplus          : wall-normal distance
            - calc_coin_file : percolation file
            - folder         : folder to store the data
    
    Returns
    -------
    None
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    frac_struc1    = np.array(data_in["frac_struc1"],dtype="float32")
    frac_struc2    = np.array(data_in["frac_struc2"],dtype="float32")
    frac_coinc     = np.array(data_in["frac_coinc"],dtype="float32")
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
    content   = str(frac_coinc.tolist())+'\n'
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
            - frac_struc1 : fraction of volume of the structure 1
            - frac_struc2 : fraction of volume of the structure 2
            - frac_coinc  : fraction of volume of the coincidence of 1 and 2
            - yplus       : wall-normal distance
    
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
    frac_coinc  = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    data_out    = {"frac_struc1":frac_struc1,"frac_struc2":frac_struc2,"frac_coinc":frac_coinc,"yplus":yplus}
    return data_out

def calc_coinc_mat(data_in={"mat_struc1":[],"mat_struc2":[]}):
    """
    .....................................................................................................................
    # calc_coinc_mat: Function for calculate the matrix of the structures
    .....................................................................................................................

    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"mat_struc1":[],"mat_struc2":[]}.
        Data:
            - mat_struc1 : data required for the structure 1
            - mat_struc2 : data required for the structure 2

    Returns
    -------
    dict
        Structure containing the information of the percolations
        Data:
            - mat_comb : matrix of the combined structures

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    mat_struc1 = data_in["mat_struc1"]
    mat_struc2 = data_in["mat_struc2"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate the volume occupied by structure 1 and 2
    # -------------------------------------------------------------------------------------------------------------------
    mat_comb    = mat_struc1+mat_struc2*2
    data_out    = {"mat_comb":mat_comb}
    
    return data_out    

def calc_coinc(data_in={"data_struc1":[],"data_struc2":[],"save_data":True,"calc_coin_file":"calc_coin.txt",
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
        DESCRIPTION. The default is {"data_struc1":[],"data_struc2":[],"save_data":True,
                                     "calc_coin_file":"calc_coin.txt","folder":"data","dy":1,"dx":1,"dz":1,
                                     "uvw_folder":"../../P125_21pi_vu/",
                                     "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                                     "rey":125,"utau":0.060523258443963}.
        Data:
            - data_struc1    : data required for the structure 1
            - data_struc2    : data required for the structure 2
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
            - frac_struc1 : fraction of volume of the structure 1
            - frac_struc2 : fraction of volume of the structure 2
            - frac_coinc  : fraction of volume of the coincidence of 1 and 2
            - yplus       : wall-normal distance

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    data_struc1    = data_in["data_struc1"]
    data_struc2    = data_in["data_struc2"]
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
    # Calculate the volume occupied by structure 1 and 2
    # -------------------------------------------------------------------------------------------------------------------
    mat_struc1    = data_struc1.mat_struc
    shp           = mat_struc1.shape
    shp_slice     = shp[1]*shp[2]
    mat_struc2    = data_struc2.mat_struc
    mat_comb      = calc_coinc_mat(data_in={"mat_struc1":mat_struc1,
                                            "mat_struc2":mat_struc2})["mat_comb"]
    frac_struc1_h = np.sum(mat_struc1,axis=(1,2))/shp_slice
    frac_struc2_h = np.sum(mat_struc2,axis=(1,2))/shp_slice
    frac_struc1   = (frac_struc1_h[:flow_data.yl_s]+np.flip(frac_struc1_h[flow_data.yu_s:]))/2
    frac_struc2   = (frac_struc2_h[:flow_data.yl_s]+np.flip(frac_struc2_h[flow_data.yu_s:]))/2
    
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate the volume occupied by both structures
    # -------------------------------------------------------------------------------------------------------------------
    mat_coinc              = mat_comb.copy()
    mat_coinc[mat_comb==3] = 1
    mat_coinc[mat_comb<3]  = 0
    frac_coinc_h           = np.sum(mat_coinc,axis=(1,2))/shp_slice
    frac_coinc             = (frac_coinc_h[:flow_data.yl_s]+np.flip(frac_coinc_h[flow_data.yu_s:]))/2
        
    # -------------------------------------------------------------------------------------------------------------------
    # Store the output data in a file
    # -------------------------------------------------------------------------------------------------------------------
    if save_data:
        save_coinc(data_in={"frac_struc1":frac_struc1,"frac_struc2":frac_struc2,"frac_coinc":frac_coinc,
                            "calc_coin_file":calc_coin_file,"folder":folder,"yplus":flow_data.yplus})
    else:
        data_out = {"frac_struc1":frac_struc1,"frac_struc2":frac_struc2,"frac_coinc":frac_coinc,
                    "yplus":flow_data.yplus}
        return data_out



def calc_coinc_mat_all(data_in={"mat_struc1":[],"mat_struc2":[],"mat_struc3":[],"mat_struc4":[],"mat_struc5":[]}):
    """
    .....................................................................................................................
    # calc_coinc_mat: Function for calculate the matrix of the structures
    .....................................................................................................................

    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"mat_struc1":[],"mat_struc2":[],"mat_struc3":[],"mat_struc4":[],"mat_struc5":[]}.
        Data:
            - mat_struc1 : data required for the structure 1
            - mat_struc2 : data required for the structure 2
            - mat_struc3 : data required for the structure 3
            - mat_struc4 : data required for the structure 4
            - mat_struc5 : data required for the structure 5

    Returns
    -------
    dict
        Structure containing the information of the percolations
        Data:
            - mat_comb : matrix of the combined structures

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    mat_struc1 = data_in["mat_struc1"]
    mat_struc2 = data_in["mat_struc2"]
    mat_struc3 = data_in["mat_struc3"]
    mat_struc4 = data_in["mat_struc4"]
    mat_struc5 = data_in["mat_struc5"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate the volume occupied by structure 1 and 2
    # -------------------------------------------------------------------------------------------------------------------
    mat_comb    = mat_struc1+mat_struc2*2+mat_struc3*4+mat_struc4*8+mat_struc5*16
    data_out    = {"mat_comb":mat_comb}
    
    return data_out    



def calc_coinc_all(data_in={"data_struc1":[],"data_struc2":[],"data_struc3":[],"data_struc4":[],"data_struc5":[],
                            "save_data":True,"calc_coin_file":"calc_coin.txt","folder":"data","dy":1,"dx":1,
                            "dz":1,"uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                            "L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963}):
    """
    .....................................................................................................................
    # calc_coinc_all: Function for calculating the coincidence between structures
    .....................................................................................................................

    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"data_struc1":[],"data_struc2":[],"data_struc3":[],"data_struc4":[],
                                     "data_struc5":[],"save_data":True,"calc_coin_file":"calc_coin.txt",
                                     "folder":"data","dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                                     "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                                     "rey":125,"utau":0.060523258443963}.
        Data:
            - data_struc1    : data required for the structure 1
            - data_struc2    : data required for the structure 2
            - data_struc3    : data required for the structure 3
            - data_struc4    : data required for the structure 4
            - data_struc4    : data required for the structure 5
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
            - frac_struc1 : fraction of volume of the structure 1
            - frac_struc2 : fraction of volume of the structure 2
            - frac_coinc  : fraction of volume of the coincidence of 1 and 2
            - yplus       : wall-normal distance

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    data_struc1    = data_in["data_struc1"]
    data_struc2    = data_in["data_struc2"]
    data_struc3    = data_in["data_struc3"]
    data_struc4    = data_in["data_struc4"]
    data_struc5    = data_in["data_struc5"]
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
    # Calculate the volume occupied by structure 1 and 2
    # -------------------------------------------------------------------------------------------------------------------
    mat_struc1    = data_struc1.mat_struc
    shp           = mat_struc1.shape
    shp_slice     = shp[1]*shp[2]
    mat_struc2    = data_struc2.mat_struc
    mat_struc3    = data_struc3.mat_struc
    mat_struc4    = data_struc4.mat_struc
    mat_struc5    = data_struc5.mat_struc
    mat_comb      = calc_coinc_mat_all(data_in={"mat_struc1":mat_struc1,
                                                "mat_struc2":mat_struc2,
                                                "mat_struc3":mat_struc3,
                                                "mat_struc4":mat_struc4,
                                                "mat_struc5":mat_struc5})["mat_comb"]
    len_comb      = 31
    for index_comb in np.arange(len_comb):
        mat_coinc                         = np.zeros_like(mat_comb)
        mat_coinc[mat_comb==index_comb+1] = 1
        frac_coinc_h                      = np.sum(mat_coinc,axis=(1,2))/shp_slice
        frac_coinc                        = (frac_coinc_h[:flow_data.yl_s]+np.flip(frac_coinc_h[flow_data.yu_s:]))/2
        if index_comb == 0:
            frac_coinc_tot = [frac_coinc]
        else:
            frac_coinc_tot.append(frac_coinc)
        
    # -------------------------------------------------------------------------------------------------------------------
    # Store the output data in a file
    # -------------------------------------------------------------------------------------------------------------------
    if save_data:
        save_coinc_all(data_in={"frac_coinc_tot":frac_coinc_tot,"calc_coin_file":calc_coin_file,
                                "folder":folder,"yplus":flow_data.yplus})
    else:
        data_out = {"frac_coinc_tot":frac_coinc_tot,"yplus":flow_data.yplus}
        return data_out


def save_coinc_all(data_in={"frac_coinc_tot":[],"yplus":[],"calc_coin_file":"calc_coin.txt","folder":"data"}):
    """
    .....................................................................................................................
    # save_coinc_all: Function to save the coincidence in a file
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"frac_coinc_tot":[],"yplus":[],"calc_coin_file":"calc_coin.txt","folder":"data"}.
        Data:
            - frac_coinc_tot : fraction of volume of the different combinations
            - yplus          : wall-normal distance
            - calc_coin_file : percolation file
            - folder         : folder to store the data
    
    Returns
    -------
    None
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    frac_coinc_tot = np.array(data_in["frac_coinc_tot"],dtype="float32")
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
    for ii in np.arange(31):         
        content   = str(frac_coinc_tot[ii].tolist())+'\n'
        file_save.write(content)  
    file_save.close()


def read_coinc_all(data_in={"calc_coin_file":"calc_coin.txt","folder":"data"}):
    """
    .....................................................................................................................
    # read_coinc_all: Function to read the coincidence between structures in a file with all coincidences
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
            - frac_coinc_tot : fraction of volume of the structure 1
            - yplus          : wall-normal distance
    
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
    for ii in np.arange(31):
        if ii==0:
            frac_coinc_tot = [np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')]
        else:
            frac_coinc_tot.append(np.array(file_read.readline().replace('[','').replace(']','').split(','),
                                           dtype='float'))
    data_out    = {"frac_coinc_tot":frac_coinc_tot,"yplus":yplus}
    return data_out


def calc_coinc_type(data_in={"data_struc1":[],"data_struc2":[],"save_data":True,"calc_coin_file":"calc_coin.txt",
                             "folder":"data","dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                             "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                             "rey":125,"utau":0.060523258443963}):
    """
    .....................................................................................................................
    # calc_coinc_type: Function for calculating the coincidence between structures using the type of structure
    .....................................................................................................................

    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"data_struc1":[],"data_struc2":[],"save_data":True,
                                     "calc_coin_file":"calc_coin.txt","folder":"data","dy":1,"dx":1,"dz":1,
                                     "uvw_folder":"../../P125_21pi_vu/",
                                     "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                                     "rey":125,"utau":0.060523258443963}.
        Data:
            - data_struc1    : data required for the structure 1
            - data_struc2    : data required for the structure 2
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
            - frac_struc1  : fraction of volume of the structure 1
            - frac_struc2a : fraction of volume of the structure 2 type a
            - frac_struc2b : fraction of volume of the structure 2 type b
            - frac_struc2c : fraction of volume of the structure 2 type c
            - frac_struc2d : fraction of volume of the structure 2 type d
            - frac_coinc_a : fraction of volume of the coincidence of 1 and 2a
            - frac_coinc_b : fraction of volume of the coincidence of 1 and 2b
            - frac_coinc_c : fraction of volume of the coincidence of 1 and 2c
            - frac_coinc_d : fraction of volume of the coincidence of 1 and 2d
            - yplus        : wall-normal distance

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    data_struc1    = data_in["data_struc1"]
    data_struc2    = data_in["data_struc2"]
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
    # Calculate the volume occupied by structure 1 and 2
    # -------------------------------------------------------------------------------------------------------------------
    mat_struc1                 = data_struc1.mat_struc
    shp                        = mat_struc1.shape
    shp_slice                  = shp[1]*shp[2]
    mat_struc2                 = data_struc2.structures.mat_event
    mat_struc2a                = np.zeros_like(mat_struc2)
    mat_struc2a[mat_struc2==1] = 1
    mat_struc2b                = np.zeros_like(mat_struc2)
    mat_struc2b[mat_struc2==2] = 1
    mat_struc2c                = np.zeros_like(mat_struc2)
    mat_struc2c[mat_struc2==3] = 1
    mat_struc2d                = np.zeros_like(mat_struc2)
    mat_struc2d[mat_struc2==4] = 1
    mat_comb_a                 = calc_coinc_mat(data_in={"mat_struc1":mat_struc1,
                                                         "mat_struc2":mat_struc2a})["mat_comb"]
    mat_comb_b                 = calc_coinc_mat(data_in={"mat_struc1":mat_struc1,
                                                         "mat_struc2":mat_struc2b})["mat_comb"]
    mat_comb_c                 = calc_coinc_mat(data_in={"mat_struc1":mat_struc1,
                                                         "mat_struc2":mat_struc2c})["mat_comb"]
    mat_comb_d                 = calc_coinc_mat(data_in={"mat_struc1":mat_struc1,
                                                         "mat_struc2":mat_struc2d})["mat_comb"]
    frac_struc1_h              = np.sum(mat_struc1,axis=(1,2))/shp_slice
    frac_struc2a_h             = np.sum(mat_struc2a,axis=(1,2))/shp_slice
    frac_struc2b_h             = np.sum(mat_struc2b,axis=(1,2))/shp_slice
    frac_struc2c_h             = np.sum(mat_struc2c,axis=(1,2))/shp_slice
    frac_struc2d_h             = np.sum(mat_struc2d,axis=(1,2))/shp_slice
    frac_struc1                = (frac_struc1_h[:flow_data.yl_s]+np.flip(frac_struc1_h[flow_data.yu_s:]))/2
    frac_struc2a               = (frac_struc2a_h[:flow_data.yl_s]+np.flip(frac_struc2a_h[flow_data.yu_s:]))/2
    frac_struc2b               = (frac_struc2b_h[:flow_data.yl_s]+np.flip(frac_struc2b_h[flow_data.yu_s:]))/2
    frac_struc2c               = (frac_struc2c_h[:flow_data.yl_s]+np.flip(frac_struc2c_h[flow_data.yu_s:]))/2
    frac_struc2d               = (frac_struc2d_h[:flow_data.yl_s]+np.flip(frac_struc2d_h[flow_data.yu_s:]))/2
    
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate the volume occupied by both structures
    # -------------------------------------------------------------------------------------------------------------------
    mat_coinc_a                = np.zeros_like(mat_comb_a)
    mat_coinc_a[mat_comb_a==3] = 1
    mat_coinc_b                = np.zeros_like(mat_comb_b)
    mat_coinc_b[mat_comb_b==3] = 1
    mat_coinc_c                = np.zeros_like(mat_comb_c)
    mat_coinc_c[mat_comb_c==3] = 1
    mat_coinc_d                = np.zeros_like(mat_comb_d)
    mat_coinc_d[mat_comb_d==3] = 1
    frac_coinc_a_h             = np.sum(mat_coinc_a,axis=(1,2))/shp_slice
    frac_coinc_a               = (frac_coinc_a_h[:flow_data.yl_s]+np.flip(frac_coinc_a_h[flow_data.yu_s:]))/2
    frac_coinc_b_h             = np.sum(mat_coinc_b,axis=(1,2))/shp_slice
    frac_coinc_b               = (frac_coinc_b_h[:flow_data.yl_s]+np.flip(frac_coinc_b_h[flow_data.yu_s:]))/2
    frac_coinc_c_h             = np.sum(mat_coinc_c,axis=(1,2))/shp_slice
    frac_coinc_c               = (frac_coinc_c_h[:flow_data.yl_s]+np.flip(frac_coinc_c_h[flow_data.yu_s:]))/2
    frac_coinc_d_h             = np.sum(mat_coinc_d,axis=(1,2))/shp_slice
    frac_coinc_d               = (frac_coinc_d_h[:flow_data.yl_s]+np.flip(frac_coinc_d_h[flow_data.yu_s:]))/2
        
    # -------------------------------------------------------------------------------------------------------------------
    # Store the output data in a file
    # -------------------------------------------------------------------------------------------------------------------
    if save_data:
        save_coinc_type(data_in={"frac_struc1":frac_struc1,"frac_struc2a":frac_struc2a,"frac_coinc_a":frac_coinc_a,
                                 "frac_struc2b":frac_struc2b,"frac_coinc_b":frac_coinc_b,
                                 "frac_struc2c":frac_struc2c,"frac_coinc_c":frac_coinc_c,
                                 "frac_struc2d":frac_struc2d,"frac_coinc_d":frac_coinc_d,
                                 "calc_coin_file":calc_coin_file,"folder":folder,"yplus":flow_data.yplus})
    else:
        data_out = {"frac_struc1":frac_struc1,"frac_struc2a":frac_struc2a,"frac_coinc_a":frac_coinc_a,
                    "frac_struc2b":frac_struc2b,"frac_coinc_b":frac_coinc_b,"frac_struc2c":frac_struc2c,
                    "frac_coinc_c":frac_coinc_c,"frac_struc2d":frac_struc2d,"frac_coinc_d":frac_coinc_d,
                    "yplus":flow_data.yplus}
        return data_out
    
    
def save_coinc_type(data_in={"frac_struc1":[],"frac_struc2a":[],"frac_coinc_a":[],"frac_struc2b":[],
                             "frac_coinc_b":[],"frac_struc2c":[],"frac_coinc_c":[],"frac_struc2d":[],
                             "frac_coinc_d":[],"yplus":[],"calc_coin_file":"calc_coin.txt","folder":"data"}):
    """
    .....................................................................................................................
    # save_coinc_type: Function to save the coincidence of the type of structure
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"frac_coinc_tot":[],"yplus":[],"calc_coin_file":"calc_coin.txt","folder":"data"}.
        Data:
            - frac_struc1    : fraction of volume of the structure 1
            - frac_struc2a   : fraction of volume of the structure 2 type a
            - frac_coinc_a   : fraction of volume of the coincidence between structure 1 and 2a
            - frac_struc2b   : fraction of volume of the structure 2 type b
            - frac_coinc_b   : fraction of volume of the coincidence between structure 1 and 2b
            - frac_struc2c   : fraction of volume of the structure 2 type c
            - frac_coinc_c   : fraction of volume of the coincidence between structure 1 and 2c
            - frac_struc2d   : fraction of volume of the structure 2 type d
            - frac_coinc_d   : fraction of volume of the coincidence between structure 1 and 2d
            - yplus          : wall-normal distance
            - calc_coin_file : percolation file
            - folder         : folder to store the data
    
    Returns
    -------
    None
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    frac_struc1    = np.array(data_in["frac_struc1"],dtype="float32")
    frac_struc2a   = np.array(data_in["frac_struc2a"],dtype="float32")
    frac_coinc_a   = np.array(data_in["frac_coinc_a"],dtype="float32")
    frac_struc2b   = np.array(data_in["frac_struc2b"],dtype="float32")
    frac_coinc_b   = np.array(data_in["frac_coinc_b"],dtype="float32")
    frac_struc2c   = np.array(data_in["frac_struc2c"],dtype="float32")
    frac_coinc_c   = np.array(data_in["frac_coinc_c"],dtype="float32")
    frac_struc2d   = np.array(data_in["frac_struc2d"],dtype="float32")
    frac_coinc_d   = np.array(data_in["frac_coinc_d"],dtype="float32")
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
    content   = str(frac_struc2a.tolist())+'\n'
    file_save.write(content)       
    content   = str(frac_struc2b.tolist())+'\n'
    file_save.write(content)        
    content   = str(frac_struc2c.tolist())+'\n'
    file_save.write(content)       
    content   = str(frac_struc2d.tolist())+'\n'
    file_save.write(content)       
    content   = str(frac_coinc_a.tolist())+'\n'
    file_save.write(content) 
    content   = str(frac_coinc_b.tolist())+'\n'
    file_save.write(content) 
    content   = str(frac_coinc_c.tolist())+'\n'
    file_save.write(content) 
    content   = str(frac_coinc_d.tolist())+'\n'
    file_save.write(content) 
    file_save.close()


def read_coinc_type(data_in={"calc_coin_file":"calc_coin.txt","folder":"data"}):
    """
    .....................................................................................................................
    # read_coinc_type: Function to read the coincidence between types of structures in a file
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
            - frac_struc1  : fraction of volume of the structure 1
            - frac_struc2a : fraction of volume of the structure 2 type a
            - frac_struc2b : fraction of volume of the structure 2 type b
            - frac_struc2c : fraction of volume of the structure 2 type c
            - frac_struc2d : fraction of volume of the structure 2 type d
            - frac_coinc_a : fraction of volume of the coincidence of 1 and 2a
            - frac_coinc_b : fraction of volume of the coincidence of 1 and 2b
            - frac_coinc_c : fraction of volume of the coincidence of 1 and 2c
            - frac_coinc_d : fraction of volume of the coincidence of 1 and 2d
            - yplus        : wall-normal distance
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
    file_perc    = folder+'/'+file
    file_read    = open(file_perc,"r")
    yplus        = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_struc1  = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_struc2a = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_struc2b = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_struc2c = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_struc2d = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_coinc_a = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_coinc_b = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_coinc_c = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_coinc_d = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    data_out     = {"frac_struc1":frac_struc1,"frac_struc2a":frac_struc2a,"frac_coinc_a":frac_coinc_a,
                    "frac_struc2b":frac_struc2b,"frac_coinc_b":frac_coinc_b,"frac_struc2c":frac_struc2c,
                    "frac_coinc_c":frac_coinc_c,"frac_struc2d":frac_struc2d,"frac_coinc_d":frac_coinc_d,
                    "yplus":yplus}
    return data_out

   

def calc_coinc_4struc(data_in={"data_struc1":[],"data_struc2":[],"data_struc3":[],"data_struc4":[],
                               "save_data":True,"calc_coin_file":"calc_coin.txt",
                               "folder":"data","dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                               "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                               "rey":125,"utau":0.060523258443963}):
    """
    .....................................................................................................................
    # calc_coinc_4struc: Function for calculating the coincidence between 4 structures
    .....................................................................................................................

    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"data_struc1":[],"data_struc2":[],"data_struc3":[],"data_struc4":[],
                                     "save_data":True,"calc_coin_file":"calc_coin.txt",
                                     "folder":"data","dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                                     "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                                     "rey":125,"utau":0.060523258443963}.
        Data:
            - data_struc1    : data required for the structure 1
            - data_struc2    : data required for the structure 2
            - data_struc3    : data required for the structure 3
            - data_struc4    : data required for the structure 4
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
            - frac_struc1 : fraction of volume of the structure 1
            - frac_struc2 : fraction of volume of the structure 2
            - frac_struc3 : fraction of volume of the structure 3
            - frac_struc4 : fraction of volume of the structure 4
            - frac_coinc  : fraction of volume of the coincidence of 1, 2, 3 and 4
            - yplus       : wall-normal distance

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    data_struc1    = data_in["data_struc1"]
    data_struc2    = data_in["data_struc2"]
    data_struc3    = data_in["data_struc3"]
    data_struc4    = data_in["data_struc4"]
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
    # Calculate the volume occupied by structure 1 and 2
    # -------------------------------------------------------------------------------------------------------------------
    mat_struc1    = data_struc1.mat_struc
    shp           = mat_struc1.shape
    shp_slice     = shp[1]*shp[2]
    mat_struc2    = data_struc2.mat_struc
    mat_struc3    = data_struc3.mat_struc
    mat_struc4    = data_struc4.mat_struc
    mat_comb      = calc_coinc_mat_4struc(data_in={"mat_struc1":mat_struc1,
                                                   "mat_struc2":mat_struc2,
                                                   "mat_struc3":mat_struc3,
                                                   "mat_struc4":mat_struc4})["mat_comb"]
    frac_struc1_h = np.sum(mat_struc1,axis=(1,2))/shp_slice
    frac_struc2_h = np.sum(mat_struc2,axis=(1,2))/shp_slice
    frac_struc3_h = np.sum(mat_struc3,axis=(1,2))/shp_slice
    frac_struc4_h = np.sum(mat_struc4,axis=(1,2))/shp_slice
    frac_struc1   = (frac_struc1_h[:flow_data.yl_s]+np.flip(frac_struc1_h[flow_data.yu_s:]))/2
    frac_struc2   = (frac_struc2_h[:flow_data.yl_s]+np.flip(frac_struc2_h[flow_data.yu_s:]))/2
    frac_struc3   = (frac_struc3_h[:flow_data.yl_s]+np.flip(frac_struc3_h[flow_data.yu_s:]))/2
    frac_struc4   = (frac_struc4_h[:flow_data.yl_s]+np.flip(frac_struc4_h[flow_data.yu_s:]))/2
    
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate the volume occupied by both structures
    # -------------------------------------------------------------------------------------------------------------------
    mat_coinc               = mat_comb.copy()
    mat_coinc[mat_comb==10] = 1
    mat_coinc[mat_comb<10]  = 0
    frac_coinc_h            = np.sum(mat_coinc,axis=(1,2))/shp_slice
    frac_coinc              = (frac_coinc_h[:flow_data.yl_s]+np.flip(frac_coinc_h[flow_data.yu_s:]))/2
        
    # -------------------------------------------------------------------------------------------------------------------
    # Store the output data in a file
    # -------------------------------------------------------------------------------------------------------------------
    if save_data:
        save_coinc_4struc(data_in={"frac_struc1":frac_struc1,"frac_struc2":frac_struc2,"frac_struc3":frac_struc3,
                                   "frac_struc4":frac_struc4,"frac_coinc":frac_coinc,"calc_coin_file":calc_coin_file,
                                   "folder":folder,"yplus":flow_data.yplus})
    else:
        data_out = {"frac_struc1":frac_struc1,"frac_struc2":frac_struc2,"frac_struc3":frac_struc3,
                    "frac_struc4":frac_struc4,"frac_coinc":frac_coinc,"yplus":flow_data.yplus}
        return data_out


def save_coinc_4struc(data_in={"frac_struc1":[],"frac_struc2":[],"frac_struc3":[],"frac_struc4":[],"frac_coinc":[],
                               "yplus":[],"calc_coin_file":"calc_coin.txt","folder":"data"}):
    """
    .....................................................................................................................
    # save_coinc_4struc: Function to save the coincidence between 4 structures in a file
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"frac_struc1":[],"frac_struc2":[],"frac_struc3":[],"frac_struc4":[],
                                     "frac_coinc":[],"yplus":[],"calc_coin_file":"calc_coin.txt","folder":"data"}.
        Data:
            - frac_struc1    : fraction of volume of the structure 1
            - frac_struc2    : fraction of volume of the structure 2
            - frac_struc3    : fraction of volume of the structure 3
            - frac_struc4    : fraction of volume of the structure 4
            - frac_coinc     : fraction of volume of the coincidence of 1, 2, 3 and 4
            - yplus          : wall-normal distance
            - calc_coin_file : percolation file
            - folder         : folder to store the data
    
    Returns
    -------
    None
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    frac_struc1    = np.array(data_in["frac_struc1"],dtype="float32")
    frac_struc2    = np.array(data_in["frac_struc2"],dtype="float32")
    frac_struc3    = np.array(data_in["frac_struc3"],dtype="float32")
    frac_struc4    = np.array(data_in["frac_struc4"],dtype="float32")
    frac_coinc     = np.array(data_in["frac_coinc"],dtype="float32")
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
    content   = str(frac_struc3.tolist())+'\n'
    file_save.write(content)          
    content   = str(frac_struc4.tolist())+'\n'
    file_save.write(content)          
    content   = str(frac_coinc.tolist())+'\n'
    file_save.write(content)
    file_save.close()


def read_coinc_4struc(data_in={"calc_coin_file":"calc_coin.txt","folder":"data"}):
    """
    .....................................................................................................................
    # read_coinc_4struc: Function to read the coincidence between 4 structures in a file
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
            - frac_struc1 : fraction of volume of the structure 1
            - frac_struc2 : fraction of volume of the structure 2
            - frac_struc3 : fraction of volume of the structure 3
            - frac_struc4 : fraction of volume of the structure 4
            - frac_coinc  : fraction of volume of the coincidence of 1 and 2
            - yplus       : wall-normal distance
    
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
    frac_struc3 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_struc4 = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    frac_coinc  = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    data_out    = {"frac_struc1":frac_struc1,"frac_struc2":frac_struc2,"frac_struc3":frac_struc3,
                   "frac_struc4":frac_struc4,"frac_coinc":frac_coinc,"yplus":yplus}
    return data_out

def calc_coinc_mat_4struc(data_in={"mat_struc1":[],"mat_struc2":[],"mat_struc3":[],"mat_struc4":[]}):
    """
    .....................................................................................................................
    # calc_coinc_mat_4struc: Function for calculate the matrix of the coincidence between 4 structures
    .....................................................................................................................

    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"mat_struc1":[],"mat_struc2":[],"mat_struc3":[],"mat_struc4":[]}.
        Data:
            - mat_struc1 : data required for the structure 1
            - mat_struc2 : data required for the structure 2
            - mat_struc3 : data required for the structure 3
            - mat_struc4 : data required for the structure 4

    Returns
    -------
    dict
        Structure containing the information of the percolations
        Data:
            - mat_comb : matrix of the combined structures

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    mat_struc1 = data_in["mat_struc1"]
    mat_struc2 = data_in["mat_struc2"]
    mat_struc3 = data_in["mat_struc3"]
    mat_struc4 = data_in["mat_struc4"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate the volume occupied by structure 1 and 2
    # -------------------------------------------------------------------------------------------------------------------
    mat_comb    = mat_struc1+mat_struc2*2+mat_struc3*3+mat_struc4*4
    data_out    = {"mat_comb":mat_comb}
    
    return data_out    



def calc_coinc_mat_all_4types(data_in={"mat_struc1":[],"mat_struc2":[],"mat_struc3":[],"mat_struc4":[]}):
    """
    .....................................................................................................................
    # calc_coinc_mat_all_4types: Function for calculate the matrix of the structures for all the types
    .....................................................................................................................

    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"mat_struc1":[],"mat_struc2":[],"mat_struc3":[],"mat_struc4":[],"mat_struc5":[]}.
        Data:
            - mat_struc1 : data required for the structure 1
            - mat_struc2 : data required for the structure 2
            - mat_struc3 : data required for the structure 3
            - mat_struc4 : data required for the structure 4

    Returns
    -------
    dict
        Structure containing the information of the percolations
        Data:
            - mat_comb : matrix of the combined structures

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    mat_struc1 = data_in["mat_struc1"]
    mat_struc2 = data_in["mat_struc2"]
    mat_struc3 = data_in["mat_struc3"]
    mat_struc4 = data_in["mat_struc4"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate the volume occupied by structure 1 and 2
    # -------------------------------------------------------------------------------------------------------------------
    mat_comb    = mat_struc1+mat_struc2*2+mat_struc3*4+mat_struc4*8
    data_out    = {"mat_comb":mat_comb}
    
    return data_out    

def calc_coinc_all_4types(data_in={"data_struc1":[],"data_struc2":[],"data_struc3":[],"data_struc4":[],
                                   "save_data":True,"calc_coin_file":"calc_coin.txt","folder":"data","dy":1,"dx":1,
                                   "dz":1,"uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                                   "L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963}):
    """
    .....................................................................................................................
    # calc_coinc_all_4types: Function for calculating the coincidence between structures with 4 types of structures
    .....................................................................................................................

    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"data_struc1":[],"data_struc2":[],"data_struc3":[],"data_struc4":[],
                                     "save_data":True,"calc_coin_file":"calc_coin.txt",
                                     "folder":"data","dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                                     "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                                     "rey":125,"utau":0.060523258443963}.
        Data:
            - data_struc1    : data required for the structure 1
            - data_struc2    : data required for the structure 2
            - data_struc3    : data required for the structure 3
            - data_struc4    : data required for the structure 4
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
            - frac_struc1 : fraction of volume of the structure 1
            - frac_struc2 : fraction of volume of the structure 2
            - frac_coinc  : fraction of volume of the coincidence of 1 and 2
            - yplus       : wall-normal distance

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    data_struc1    = data_in["data_struc1"]
    data_struc2    = data_in["data_struc2"]
    data_struc3    = data_in["data_struc3"]
    data_struc4    = data_in["data_struc4"]
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
    # Calculate the volume occupied by structure 1 and 2
    # -------------------------------------------------------------------------------------------------------------------
    mat_struc1    = data_struc1.mat_struc
    shp           = mat_struc1.shape
    shp_slice     = shp[1]*shp[2]
    mat_struc2    = data_struc2.mat_struc
    mat_struc3    = data_struc3.mat_struc
    mat_struc4    = data_struc4.mat_struc
    mat_comb      = calc_coinc_mat_all_4types(data_in={"mat_struc1":mat_struc1,
                                                       "mat_struc2":mat_struc2,
                                                       "mat_struc3":mat_struc3,
                                                       "mat_struc4":mat_struc4})["mat_comb"]
    len_comb      = 15
    for index_comb in np.arange(len_comb):
        mat_coinc                         = np.zeros_like(mat_comb)
        mat_coinc[mat_comb==index_comb+1] = 1
        frac_coinc_h                      = np.sum(mat_coinc,axis=(1,2))/shp_slice
        frac_coinc                        = (frac_coinc_h[:flow_data.yl_s]+np.flip(frac_coinc_h[flow_data.yu_s:]))/2
        if index_comb == 0:
            frac_coinc_tot = [frac_coinc]
        else:
            frac_coinc_tot.append(frac_coinc)
        
    # -------------------------------------------------------------------------------------------------------------------
    # Store the output data in a file
    # -------------------------------------------------------------------------------------------------------------------
    if save_data:
        save_coinc_all_4types(data_in={"frac_coinc_tot":frac_coinc_tot,"calc_coin_file":calc_coin_file,
                                       "folder":folder,"yplus":flow_data.yplus})
    else:
        data_out = {"frac_coinc_tot":frac_coinc_tot,"yplus":flow_data.yplus}
        return data_out


def save_coinc_all_4types(data_in={"frac_coinc_tot":[],"yplus":[],"calc_coin_file":"calc_coin.txt","folder":"data"}):
    """
    .....................................................................................................................
    # save_coinc_all_4types: Function to save the coincidence in a file
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"frac_coinc_tot":[],"yplus":[],"calc_coin_file":"calc_coin.txt","folder":"data"}.
        Data:
            - frac_coinc_tot : fraction of volume of the different combinations
            - yplus          : wall-normal distance
            - calc_coin_file : percolation file
            - folder         : folder to store the data
    
    Returns
    -------
    None
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    frac_coinc_tot = np.array(data_in["frac_coinc_tot"],dtype="float32")
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
    for ii in np.arange(15):         
        content   = str(frac_coinc_tot[ii].tolist())+'\n'
        file_save.write(content)  
    file_save.close()


def read_coinc_all_4types(data_in={"calc_coin_file":"calc_coin.txt","folder":"data"}):
    """
    .....................................................................................................................
    # read_coinc_all_4types: Function to read the coincidence between structures in a file with all coincidences for 4
                             structures
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
            - frac_coinc_tot : fraction of volume of the structure 1
            - yplus          : wall-normal distance
    
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
    for ii in np.arange(15):
        if ii==0:
            frac_coinc_tot = [np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')]
        else:
            frac_coinc_tot.append(np.array(file_read.readline().replace('[','').replace(']','').split(','),
                                           dtype='float'))
    data_out    = {"frac_coinc_tot":frac_coinc_tot,"yplus":yplus}
    return data_out


def calc_coinc_mat_all_3types(data_in={"mat_struc1":[],"mat_struc2":[],"mat_struc3":[]}):
    """
    .....................................................................................................................
    # calc_coinc_mat_all_3types: Function for calculate the matrix of the structures for all the types
    .....................................................................................................................

    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"mat_struc1":[],"mat_struc2":[],"mat_struc3":[],"mat_struc5":[]}.
        Data:
            - mat_struc1 : data required for the structure 1
            - mat_struc2 : data required for the structure 2
            - mat_struc3 : data required for the structure 3

    Returns
    -------
    dict
        Structure containing the information of the percolations
        Data:
            - mat_comb : matrix of the combined structures

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    mat_struc1 = data_in["mat_struc1"]
    mat_struc2 = data_in["mat_struc2"]
    mat_struc3 = data_in["mat_struc3"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate the volume occupied by structure 1 and 2
    # -------------------------------------------------------------------------------------------------------------------
    mat_comb    = mat_struc1+mat_struc2*2+mat_struc3*4
    data_out    = {"mat_comb":mat_comb}
    
    return data_out    


def calc_coinc_all_3types(data_in={"data_struc1":[],"data_struc2":[],"data_struc3":[],
                                   "save_data":True,"calc_coin_file":"calc_coin.txt","folder":"data","dy":1,"dx":1,
                                   "dz":1,"uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                                   "L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963}):
    """
    .....................................................................................................................
    # calc_coinc_all_3types: Function for calculating the coincidence between structures with 4 types of structures
    .....................................................................................................................

    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"data_struc1":[],"data_struc2":[],"data_struc3":[],
                                     "save_data":True,"calc_coin_file":"calc_coin.txt",
                                     "folder":"data","dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                                     "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                                     "rey":125,"utau":0.060523258443963}.
        Data:
            - data_struc1    : data required for the structure 1
            - data_struc2    : data required for the structure 2
            - data_struc3    : data required for the structure 3
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
            - frac_struc1 : fraction of volume of the structure 1
            - frac_struc2 : fraction of volume of the structure 2
            - frac_coinc  : fraction of volume of the coincidence of 1 and 2
            - yplus       : wall-normal distance

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    data_struc1    = data_in["data_struc1"]
    data_struc2    = data_in["data_struc2"]
    data_struc3    = data_in["data_struc3"]
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
    # Calculate the volume occupied by structure 1 and 2
    # -------------------------------------------------------------------------------------------------------------------
    mat_struc1    = data_struc1.mat_struc
    shp           = mat_struc1.shape
    shp_slice     = shp[1]*shp[2]
    mat_struc2    = data_struc2.mat_struc
    mat_struc3    = data_struc3.mat_struc
    mat_comb      = calc_coinc_mat_all_3types(data_in={"mat_struc1":mat_struc1,
                                                       "mat_struc2":mat_struc2,
                                                       "mat_struc3":mat_struc3})["mat_comb"]
    len_comb      = 7
    for index_comb in np.arange(len_comb):
        mat_coinc                         = np.zeros_like(mat_comb)
        mat_coinc[mat_comb==index_comb+1] = 1
        frac_coinc_h                      = np.sum(mat_coinc,axis=(1,2))/shp_slice
        frac_coinc                        = (frac_coinc_h[:flow_data.yl_s]+np.flip(frac_coinc_h[flow_data.yu_s:]))/2
        if index_comb == 0:
            frac_coinc_tot = [frac_coinc]
        else:
            frac_coinc_tot.append(frac_coinc)
        
    # -------------------------------------------------------------------------------------------------------------------
    # Store the output data in a file
    # -------------------------------------------------------------------------------------------------------------------
    if save_data:
        save_coinc_all_3types(data_in={"frac_coinc_tot":frac_coinc_tot,"calc_coin_file":calc_coin_file,
                                       "folder":folder,"yplus":flow_data.yplus})
    else:
        data_out = {"frac_coinc_tot":frac_coinc_tot,"yplus":flow_data.yplus}
        return data_out


def save_coinc_all_3types(data_in={"frac_coinc_tot":[],"yplus":[],"calc_coin_file":"calc_coin.txt","folder":"data"}):
    """
    .....................................................................................................................
    # save_coinc_all_3types: Function to save the coincidence in a file
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"frac_coinc_tot":[],"yplus":[],"calc_coin_file":"calc_coin.txt","folder":"data"}.
        Data:
            - frac_coinc_tot : fraction of volume of the different combinations
            - yplus          : wall-normal distance
            - calc_coin_file : percolation file
            - folder         : folder to store the data
    
    Returns
    -------
    None
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    frac_coinc_tot = np.array(data_in["frac_coinc_tot"],dtype="float32")
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
    for ii in np.arange(7):         
        content   = str(frac_coinc_tot[ii].tolist())+'\n'
        file_save.write(content)  
    file_save.close()


def read_coinc_all_3types(data_in={"calc_coin_file":"calc_coin.txt","folder":"data"}):
    """
    .....................................................................................................................
    # read_coinc_all_3types: Function to read the coincidence between structures in a file with all coincidences for 3
                             structures
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
            - frac_coinc_tot : fraction of volume of the structure 1
            - yplus          : wall-normal distance
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    folder    = str(data_in["folder"])           # folder for reading the mean velocity data
    file      = str(data_in["calc_coin_file"])   # file for reading the mean velocity data
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the velocity from file
    # -------------------------------------------------------------------------------------------------------------------
    file_perc   = folder+'/'+file
    file_read   = open(file_perc,"r")
    yplus       = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    for ii in np.arange(7):
        if ii==0:
            frac_coinc_tot = [np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')]
        else:
            frac_coinc_tot.append(np.array(file_read.readline().replace('[','').replace(']','').split(','),
                                           dtype='float'))
    data_out    = {"frac_coinc_tot":frac_coinc_tot,"yplus":yplus}
    return data_out

