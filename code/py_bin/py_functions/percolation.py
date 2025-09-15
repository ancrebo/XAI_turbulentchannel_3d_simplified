# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
percolation.py
-------------------------------------------------------------------------------------------------------------------------
Created on Wed May 15 11:58:23 2024

@author: Andres Cremades Botella

File containing the function to percolate the structures. The file contains the following functions:
    Functions:
        - percolation      : function for calculating the percolation
        - save_percolation : function to save the percolation
"""
# -----------------------------------------------------------------------------------------------------------------------
# Import packages
# -----------------------------------------------------------------------------------------------------------------------
import numpy as np


def save_percolation(data_in={"nstruc":[],"Vstruc":[],"H_perc":[],"perc_file":"perc_uv.txt","folder":"data"}):
    """
    .....................................................................................................................
    # save_percolation: Function to save the percolation in a file
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"nstruc":[],"Vstruc":[],"H_perc":[],"perc_file":"perc_uv.txt","folder":"data"}.
        Data:
            - nstruc    : number of structures of the percolation index divided by the maximum number of structures
            - Vstruc    : volume of the largest structures respect to the volume of the channel
            - H_perc    : percolation index
            - perc_file : file of the percolation analysis
            - folder    : folder to save the analysis
    
    Returns
    -------
    None
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    nstruc    = np.array(data_in["nstruc"],dtype="float32")
    Vstruc    = np.array(data_in["Vstruc"],dtype="float32")
    H_perc    = np.array(data_in["H_perc"],dtype="float32")
    perc_file = str(data_in["perc_file"])
    folder    = str(data_in["folder"])
    
    # -------------------------------------------------------------------------------------------------------------------
    # Save the information in a file
    # -------------------------------------------------------------------------------------------------------------------    
    file_perc = folder+'/'+perc_file                     
    file_save = open(file_perc, "w")           
    content   = str(H_perc.tolist())+'\n'
    file_save.write(content)          
    content   = str(nstruc.tolist())+'\n'
    file_save.write(content)          
    content   = str(Vstruc.tolist())+'\n'
    file_save.write(content)
    file_save.close()


def read_percolation(data_in={"perc_file":"perc_uv.txt","folder":"data"}):
    """
    .....................................................................................................................
    # read_percolation: Function to read the percolation in a file
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"perc_file":"perc_uv.txt","folder":"data"}.
        Data:
            - perc_file : file of the percolation analysis
            - folder    : folder to save the analysis
    
    Returns
    -------
    dict
        Structure containing the information of the percolations
        Data:
            - nstruc : number of structures of the percolation index divided by the maximum number of structures
            - Vstruc : volume of the largest structures respect to the volume of the channel
            - H_perc : percolation index
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    folder    = str(data_in["folder"]) # folder for reading the mean velocity data
    file      = str(data_in["perc_file"])   # file for reading the mean velocity data
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the velocity from file
    # -------------------------------------------------------------------------------------------------------------------
    file_perc = folder+'/'+file
    file_read = open(file_perc,"r")
    H_perc    = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    nstruc    = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    Vstruc    = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    data_out  = {"H_perc":H_perc,"nstruc":nstruc,"Vstruc":Vstruc}
    return data_out
    
    
    
def percolation(data_in={"Hmin":0.1,"Hmax":10,"Hnum":100,"data_struc":[],"save_data":True,
                         "perc_file":"perc_uv.txt","coherent_structure":"uv_structure"}):
    """
    .....................................................................................................................
    # percolation: Function for percolate the data
    .....................................................................................................................

    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"Hmin":0.1,"Hmax":10,"Hnum":100,"data_struc":[],"save_data":True,
                                     "perc_file":"perc_uv.txt","coherent_structure":"uv_structure"}.
        Data:
            - Hmin               : minimum percolation index
            - Hmax               : maximum percolation index
            - Hnum               : number of percolation indices
            - data_struc         : data required for the structure
            - save_data          : flag for saving the data (True: saves in a file, False: not save in a file)
            - perc_file          : percolation file
            - coherent_structure : coherent structure used for the percolation (package)

    Returns
    -------
    dict
        Structure containing the information of the percolations
        Data:
            - nstruc : number of structures of the percolation index divided by the maximum number of structures
            - Vstruc : volume of the largest structures respect to the volume of the channel
            - H_perc : percolation index

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import Packages
    # -------------------------------------------------------------------------------------------------------------------
    coherent_structure = data_in["coherent_structure"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    Hmin        = float(data_in["Hmin"])
    Hmax        = float(data_in["Hmax"])
    Hnum        = int(data_in["Hnum"])
    data_struc  = data_in["data_struc"]
    save_data   = bool(data_in["save_data"])
    perc_file   = str(data_in["perc_file"])
    filvol      = data_struc["filvol"]
    data_folder = data_struc["data_folder"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Distribute logarithmically
    # -------------------------------------------------------------------------------------------------------------------
    Hexp_min = np.log10(Hmin)
    Hexp_max = np.log10(Hmax)
    Hexp_vec = np.linspace(Hexp_min,Hexp_max,Hnum)
    Hvec     = 10**Hexp_vec
    
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate the percolation of the field
    # -------------------------------------------------------------------------------------------------------------------
    nstruc   = np.zeros((Hnum,))
    Vstruc   = np.zeros((Hnum,))
    for index in np.arange(Hnum):
        # ---------------------------------------------------------------------------------------------------------------
        # Create the class for the structure
        # ---------------------------------------------------------------------------------------------------------------
        data_struc["Hperc"] = Hvec[index]
        uv_struc            = coherent_structure(data_in=data_struc)
        print("-"*100,flush=True)
        print("Calculating the percolation for H="+str(Hvec[index])+"...",flush=True)
        uv_struc.calculate_matstruc()
        uv_struc.segment_struc()
        nstruc_val    = len(uv_struc.structures.vol[uv_struc.structures.vol>filvol])
        if nstruc_val > 0:
            vol_val   = np.max(uv_struc.structures.vol[uv_struc.structures.vol>filvol])/np.sum(uv_struc.structures.vol)
        else:
            vol_val   = 0
        nstruc[index] = nstruc_val
        Vstruc[index] = vol_val
        print("Number of structures:"+str(nstruc_val),flush=True)
        print("-"*100,flush=True)
    nstruc  /= np.max(nstruc)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Store the output data in a file
    # -------------------------------------------------------------------------------------------------------------------
    if save_data:
        save_percolation(data_in={"nstruc":nstruc,"Vstruc":Vstruc,"H_perc":Hvec,
                                  "perc_file":perc_file,"folder":data_folder})
    else:
        data_out = {"nstruc":nstruc,"Vstruc":Vstruc,"H_perc":Hvec}
        return data_out



def save_percolation_uvw(data_in={"nstruc_u":[],"Vstruc_u":[],"nstruc_v":[],"Vstruc_v":[],"nstruc_w":[],"Vstruc_w":[],
                                  "H_perc":[],"perc_file":"perc_uv.txt","folder":"data"}):
    """
    .....................................................................................................................
    # save_percolation_uvw: Function to save the percolation of the different components in a file
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"nstruc":[],"Vstruc":[],"H_perc":[],"perc_file":"perc_uv.txt","folder":"data"}.
        Data:
            - nstruc_u  : number of structures of the percolation index divided by the maximum number of structures in u
            - Vstruc_u  : volume of the largest structures respect to the volume of the channel in u
            - nstruc_v  : number of structures of the percolation index divided by the maximum number of structures in v
            - Vstruc_v  : volume of the largest structures respect to the volume of the channel in v
            - nstruc_w  : number of structures of the percolation index divided by the maximum number of structures in w
            - Vstruc_w  : volume of the largest structures respect to the volume of the channel in w
            - H_perc    : percolation index
            - perc_file : file of the percolation analysis
            - folder    : folder to save the analysis
    
    Returns
    -------
    None
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    nstruc_u  = np.array(data_in["nstruc_u"],dtype="float32")
    Vstruc_u  = np.array(data_in["Vstruc_u"],dtype="float32")
    nstruc_v  = np.array(data_in["nstruc_v"],dtype="float32")
    Vstruc_v  = np.array(data_in["Vstruc_v"],dtype="float32")
    nstruc_w  = np.array(data_in["nstruc_w"],dtype="float32")
    Vstruc_w  = np.array(data_in["Vstruc_w"],dtype="float32")
    H_perc    = np.array(data_in["H_perc"],dtype="float32")
    perc_file = str(data_in["perc_file"])
    folder    = str(data_in["folder"])
    
    # -------------------------------------------------------------------------------------------------------------------
    # Save the information in a file
    # -------------------------------------------------------------------------------------------------------------------    
    file_perc = folder+'/'+perc_file                     
    file_save = open(file_perc, "w")           
    content   = str(H_perc.tolist())+'\n'
    file_save.write(content)          
    content   = str(nstruc_u.tolist())+'\n'
    file_save.write(content)          
    content   = str(Vstruc_u.tolist())+'\n'
    file_save.write(content)        
    content   = str(nstruc_v.tolist())+'\n'
    file_save.write(content)          
    content   = str(Vstruc_v.tolist())+'\n'
    file_save.write(content)        
    content   = str(nstruc_w.tolist())+'\n'
    file_save.write(content)          
    content   = str(Vstruc_w.tolist())+'\n'
    file_save.write(content)
    file_save.close()


def read_percolation_uvw(data_in={"perc_file":"perc_uv.txt","folder":"data"}):
    """
    .....................................................................................................................
    # read_percolation_uvw: Function to read the percolation in a file in the different components
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"perc_file":"perc_uv.txt","folder":"data"}.
        Data:
            - perc_file : file of the percolation analysis
            - folder    : folder to save the analysis
    
    Returns
    -------
    dict
        Structure containing the information of the percolations
        Data:
            - nstruc_u : number of structures of the percolation index divided by the maximum number of structures in u
            - Vstruc_u : volume of the largest structures respect to the volume of the channel in u
            - nstruc_v : number of structures of the percolation index divided by the maximum number of structures in v
            - Vstruc_v : volume of the largest structures respect to the volume of the channel in v
            - nstruc_w : number of structures of the percolation index divided by the maximum number of structures in w
            - Vstruc_w : volume of the largest structures respect to the volume of the channel in w
            - H_perc   : percolation index
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    folder    = str(data_in["folder"]) # folder for reading the mean velocity data
    file      = str(data_in["perc_file"])   # file for reading the mean velocity data
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the velocity from file
    # -------------------------------------------------------------------------------------------------------------------
    file_perc = folder+'/'+file
    file_read = open(file_perc,"r")
    H_perc    = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    nstruc_u  = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    Vstruc_u  = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    nstruc_v  = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    Vstruc_v  = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    nstruc_w  = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    Vstruc_w  = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    data_out  = {"H_perc":H_perc,"nstruc_u":nstruc_u,"Vstruc_u":Vstruc_u,"nstruc_v":nstruc_v,"Vstruc_v":Vstruc_v,
                 "nstruc_w":nstruc_w,"Vstruc_w":Vstruc_w}
    return data_out
    
   
    
def percolation_uvw(data_in={"Hmin":0.1,"Hmax":10,"Hnum":100,"data_struc":[],"save_data":True,
                             "perc_file":"perc_uv.txt","coherent_structure":"uv_structure"}):
    """
    .....................................................................................................................
    # percolation_uvw: Function for percolate the data in the different directions
    .....................................................................................................................

    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"Hmin":0.1,"Hmax":10,"Hnum":100,"data_struc":[],"save_data":True,
                                     "perc_file":"perc_uv.txt","coherent_structure":"uv_structure"}.
        Data:
            - Hmin               : minimum percolation index
            - Hmax               : maximum percolation index
            - Hnum               : number of percolation indices
            - data_struc         : data required for the structure
            - save_data          : flag for saving the data (True: saves in a file, False: not save in a file)
            - perc_file          : percolation file
            - coherent_structure : coherent structure used for the percolation (package)

    Returns
    -------
    dict
        Structure containing the information of the percolations
        Data:
            - nstruc : number of structures of the percolation index divided by the maximum number of structures
            - Vstruc : volume of the largest structures respect to the volume of the channel
            - H_perc : percolation index

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import Packages
    # -------------------------------------------------------------------------------------------------------------------
    coherent_structure = data_in["coherent_structure"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    Hmin        = float(data_in["Hmin"])
    Hmax        = float(data_in["Hmax"])
    Hnum        = int(data_in["Hnum"])
    data_struc  = data_in["data_struc"]
    save_data   = bool(data_in["save_data"])
    perc_file   = str(data_in["perc_file"])
    filvol      = data_struc["filvol"]
    data_folder = data_struc["data_folder"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Distribute logarithmically
    # -------------------------------------------------------------------------------------------------------------------
    Hexp_min = np.log10(Hmin)
    Hexp_max = np.log10(Hmax)
    Hexp_vec = np.linspace(Hexp_min,Hexp_max,Hnum)
    Hvec     = 10**Hexp_vec
    
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate the percolation of the field
    # -------------------------------------------------------------------------------------------------------------------
    nstruc_u   = np.zeros((Hnum,))
    Vstruc_u   = np.zeros((Hnum,))
    nstruc_v   = np.zeros((Hnum,))
    Vstruc_v   = np.zeros((Hnum,))
    nstruc_w   = np.zeros((Hnum,))
    Vstruc_w   = np.zeros((Hnum,))
    for index in np.arange(Hnum):
        # ---------------------------------------------------------------------------------------------------------------
        # Create the class for the structure
        # ---------------------------------------------------------------------------------------------------------------
        data_struc["Hperc"] = Hvec[index]
        uvw_struc           = coherent_structure(data_in=data_struc)
        print("-"*100,flush=True)
        print("Calculating the percolation for H="+str(Hvec[index])+"...",flush=True)
        uvw_struc.calculate_matstruc()
        uvw_struc.segment_struc()
        nstruc_u_val    = len(uvw_struc.structures_u.vol[uvw_struc.structures_u.vol>filvol])
        nstruc_v_val    = len(uvw_struc.structures_v.vol[uvw_struc.structures_v.vol>filvol])
        nstruc_w_val    = len(uvw_struc.structures_w.vol[uvw_struc.structures_w.vol>filvol])
        if nstruc_u_val > 0:
            vol_u_val   = np.max(uvw_struc.structures_u.vol[uvw_struc.structures_u.vol
                                                            >filvol])/np.sum(uvw_struc.structures_u.vol)
        else:
            vol_u_val   = 0
        if nstruc_v_val > 0:
            vol_v_val   = np.max(uvw_struc.structures_v.vol[uvw_struc.structures_v.vol
                                                            >filvol])/np.sum(uvw_struc.structures_v.vol)
        else:
            vol_v_val   = 0
        if nstruc_w_val > 0:
            vol_w_val   = np.max(uvw_struc.structures_w.vol[uvw_struc.structures_w.vol
                                                            >filvol])/np.sum(uvw_struc.structures_w.vol)
        else:
            vol_w_val   = 0
        nstruc_u[index] = nstruc_u_val
        Vstruc_u[index] = vol_u_val
        nstruc_v[index] = nstruc_v_val
        Vstruc_v[index] = vol_v_val
        nstruc_w[index] = nstruc_w_val
        Vstruc_w[index] = vol_w_val
        print("Number of structures u:"+str(nstruc_u_val),flush=True)
        print("Number of structures v:"+str(nstruc_v_val),flush=True)
        print("Number of structures w:"+str(nstruc_w_val),flush=True)
        print("-"*100,flush=True)
    nstruc_u  /= np.max(nstruc_u)
    nstruc_v  /= np.max(nstruc_v)
    nstruc_w  /= np.max(nstruc_w)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Store the output data in a file
    # -------------------------------------------------------------------------------------------------------------------
    if save_data:
        save_percolation_uvw(data_in={"nstruc_u":nstruc_u,"nstruc_v":nstruc_v,"nstruc_w":nstruc_w,"Vstruc_u":Vstruc_u,
                                      "Vstruc_v":Vstruc_v,"Vstruc_w":Vstruc_w,"H_perc":Hvec,"perc_file":perc_file,
                                      "folder":data_folder})
    else:
        data_out = {"nstruc_u":nstruc_u,"nstruc_v":nstruc_v,"nstruc_w":nstruc_w,"Vstruc_u":Vstruc_u,
                    "Vstruc_v":Vstruc_v,"Vstruc_w":Vstruc_w,"H_perc":Hvec}
        return data_out


def save_percolation_uw_vsign(data_in={"nstruc_1":[],"Vstruc_1":[],"nstruc_2":[],"Vstruc_2":[],"H_perc":[],
                                       "perc_file":"perc_uv.txt","folder":"data"}):
    """
    .....................................................................................................................
    # save_percolation_uw_vsign: Function to save the percolation of the different components in a file
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"nstruc_1":[],"Vstruc_1":[],"nstruc_2":[],"Vstruc_2":[],"H_perc":[],
                                     "perc_file":"perc_uv.txt","folder":"data"}.
        Data:
            - nstruc_1  : number of structures of the percolation index divided by the maximum number of structures in 1
            - Vstruc_1  : volume of the largest structures respect to the volume of the channel in 1
            - nstruc_2  : number of structures of the percolation index divided by the maximum number of structures in 2
            - Vstruc_2  : volume of the largest structures respect to the volume of the channel in 2
            - H_perc    : percolation index
            - perc_file : file of the percolation analysis
            - folder    : folder to save the analysis
    
    Returns
    -------
    None
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    nstruc_1  = np.array(data_in["nstruc_1"],dtype="float32")
    Vstruc_1  = np.array(data_in["Vstruc_1"],dtype="float32")
    nstruc_2  = np.array(data_in["nstruc_2"],dtype="float32")
    Vstruc_2  = np.array(data_in["Vstruc_2"],dtype="float32")
    H_perc    = np.array(data_in["H_perc"],dtype="float32")
    perc_file = str(data_in["perc_file"])
    folder    = str(data_in["folder"])
    
    # -------------------------------------------------------------------------------------------------------------------
    # Save the information in a file
    # -------------------------------------------------------------------------------------------------------------------    
    file_perc = folder+'/'+perc_file                     
    file_save = open(file_perc, "w")           
    content   = str(H_perc.tolist())+'\n'
    file_save.write(content)          
    content   = str(nstruc_1.tolist())+'\n'
    file_save.write(content)          
    content   = str(Vstruc_1.tolist())+'\n'
    file_save.write(content)        
    content   = str(nstruc_2.tolist())+'\n'
    file_save.write(content)          
    content   = str(Vstruc_2.tolist())+'\n'
    file_save.write(content)
    file_save.close()


def read_percolation_uw_vsign(data_in={"perc_file":"perc_uv.txt","folder":"data"}):
    """
    .....................................................................................................................
    # read_percolation_uw_vsign: Function to read the percolation in a file in the different components
    .....................................................................................................................
    
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"perc_file":"perc_uv.txt","folder":"data"}.
        Data:
            - perc_file : file of the percolation analysis
            - folder    : folder to save the analysis
    
    Returns
    -------
    dict
        Structure containing the information of the percolations
        Data:
            - nstruc_1 : number of structures of the percolation index divided by the maximum number of structures in u
            - Vstruc_1 : volume of the largest structures respect to the volume of the channel in 1
            - nstruc_2 : number of structures of the percolation index divided by the maximum number of structures in v
            - Vstruc_2 : volume of the largest structures respect to the volume of the channel in 2
            - H_perc   : percolation index
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    folder    = str(data_in["folder"]) # folder for reading the mean velocity data
    file      = str(data_in["perc_file"])   # file for reading the mean velocity data
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the velocity from file
    # -------------------------------------------------------------------------------------------------------------------
    file_perc = folder+'/'+file
    file_read = open(file_perc,"r")
    H_perc    = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    nstruc_1  = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    Vstruc_1  = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    nstruc_2  = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    Vstruc_2  = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='float')
    data_out  = {"H_perc":H_perc,"nstruc_1":nstruc_1,"Vstruc_1":Vstruc_1,"nstruc_2":nstruc_2,"Vstruc_2":Vstruc_2}
    return data_out
    
   
    
def percolation_uw_vsign(data_in={"Hmin":0.1,"Hmax":10,"Hnum":100,"data_struc":[],"save_data":True,
                                  "perc_file":"perc_uv.txt","coherent_structure":"uv_structure"}):
    """
    .....................................................................................................................
    # percolation_uw_vsign: Function for percolate the data in the different directions
    .....................................................................................................................

    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"Hmin":0.1,"Hmax":10,"Hnum":100,"data_struc":[],"save_data":True,
                                     "perc_file":"perc_uv.txt","coherent_structure":"uv_structure"}.
        Data:
            - Hmin               : minimum percolation index
            - Hmax               : maximum percolation index
            - Hnum               : number of percolation indices
            - data_struc         : data required for the structure
            - save_data          : flag for saving the data (True: saves in a file, False: not save in a file)
            - perc_file          : percolation file
            - coherent_structure : coherent structure used for the percolation (package)

    Returns
    -------
    dict
        Structure containing the information of the percolations
        Data:
            - nstruc_1 : number of structures of the percolation index divided by the maximum number of structures in 1
            - Vstruc_1 : volume of the largest structures respect to the volume of the channel of the structures 1
            - nstruc_2 : number of structures of the percolation index divided by the maximum number of structures in 2
            - Vstruc_2 : volume of the largest structures respect to the volume of the channel of the structures 2
            - H_perc   : percolation index

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import Packages
    # -------------------------------------------------------------------------------------------------------------------
    coherent_structure = data_in["coherent_structure"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    Hmin        = float(data_in["Hmin"])
    Hmax        = float(data_in["Hmax"])
    Hnum        = int(data_in["Hnum"])
    data_struc  = data_in["data_struc"]
    save_data   = bool(data_in["save_data"])
    perc_file   = str(data_in["perc_file"])
    filvol      = data_struc["filvol"]
    data_folder = data_struc["data_folder"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Distribute logarithmically
    # -------------------------------------------------------------------------------------------------------------------
    Hexp_min = np.log10(Hmin)
    Hexp_max = np.log10(Hmax)
    Hexp_vec = np.linspace(Hexp_min,Hexp_max,Hnum)
    Hvec     = 10**Hexp_vec
    
    # -------------------------------------------------------------------------------------------------------------------
    # Calculate the percolation of the field
    # -------------------------------------------------------------------------------------------------------------------
    nstruc_1   = np.zeros((Hnum,))
    Vstruc_1   = np.zeros((Hnum,))
    nstruc_2   = np.zeros((Hnum,))
    Vstruc_2   = np.zeros((Hnum,))
    for index in np.arange(Hnum):
        # ---------------------------------------------------------------------------------------------------------------
        # Create the class for the structure
        # ---------------------------------------------------------------------------------------------------------------
        data_struc["Hperc"] = Hvec[index]
        uvw_struc           = coherent_structure(data_in=data_struc)
        print("-"*100,flush=True)
        print("Calculating the percolation for H="+str(Hvec[index])+"...",flush=True)
        uvw_struc.calculate_matstruc()
        uvw_struc.segment_struc()
        nstruc_1_val    = len(uvw_struc.structures_1.vol[uvw_struc.structures_1.vol>filvol])
        nstruc_2_val    = len(uvw_struc.structures_2.vol[uvw_struc.structures_2.vol>filvol])
        if nstruc_1_val > 0:
            vol_1_val   = np.max(uvw_struc.structures_1.vol[uvw_struc.structures_1.vol
                                                            >filvol])/np.sum(uvw_struc.structures_1.vol)
        else:
            vol_1_val   = 0
        if nstruc_2_val > 0:
            vol_2_val   = np.max(uvw_struc.structures_2.vol[uvw_struc.structures_2.vol
                                                            >filvol])/np.sum(uvw_struc.structures_2.vol)
        else:
            vol_2_val   = 0
        nstruc_1[index] = nstruc_1_val
        Vstruc_1[index] = vol_1_val
        nstruc_2[index] = nstruc_2_val
        Vstruc_2[index] = vol_2_val
        print("Number of structures 1:"+str(nstruc_1_val),flush=True)
        print("Number of structures 2:"+str(nstruc_2_val),flush=True)
        print("-"*100,flush=True)
    nstruc_1  /= np.max(nstruc_1)
    nstruc_2  /= np.max(nstruc_2)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Store the output data in a file
    # -------------------------------------------------------------------------------------------------------------------
    if save_data:
        save_percolation_uw_vsign(data_in={"nstruc_1":nstruc_1,"nstruc_2":nstruc_2,"Vstruc_1":Vstruc_1,
                                           "Vstruc_2":Vstruc_2,"H_perc":Hvec,"perc_file":perc_file,
                                           "folder":data_folder})
    else:
        data_out = {"nstruc_1":nstruc_1,"nstruc_2":nstruc_2,"Vstruc_1":Vstruc_1,
                    "Vstruc_2":Vstruc_2,"H_perc":Hvec}
        return data_out

