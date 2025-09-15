# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plotstruc3d_4struc.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella

File to plot a structure of each type:
    - Functions:
        - plotstruc3d        : function to plot the whole channel
        - plotstruc3d_separe : function to plot half of the channel
"""

# -----------------------------------------------------------------------------------------------------------------------
# Import packages for all functions
# -----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

def plotstruc3d(data_in={"struc1":[],"struc2":[],"struc3":[],"struc4":[],"plot_folder":"plots","xlabel":"$x^+$",
                         "ylabel":"$y^+$","zlabel":"z^+","fontsize":18,"figsize_x":10,"figsize_y":8,
                         "colormap1":[],"colormap2":[],"colormap3":[],"colormap4":[],"fig_name":"struc3d",
                         "dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                         "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,
                         "L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963,"cmap_flag":False}):
    """
    .....................................................................................................................
    # plotstruc3d: Function to generate the plot of the 3d structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc1":[],"struc2":[],"struc3":[],"struc4":[],"plot_folder":"plots","xlabel":"$x^+$",
                        "ylabel":"$y^+$","zlabel":"z^+","fontsize":18,"figsize_x":10,"figsize_y":8,
                        "colormap1":[],"colormap2":[],"colormap3":[],
                        "colormap4":[],"colornum":2,"fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,
                        "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,
                        "L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963,"cmap_flag":False}.
        Data:
            - struc1      : data of the structure 1
            - struc2      : data of the structure 2
            - struc3      : data of the structure 3
            - struc4      : data of the structure 4
            - plot_folder : folder to store the figures
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - zlabel      : label of the z axis
            - fontsize    : font size used for the figure
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colormap1   : colormap 1 used for the figure
            - colormap2   : colormap 2 used for the figure
            - colormap3   : colormap 3 used for the figure
            - colormap4   : colormap 4 used for the figure
            - fig_name    : name of the saved figure
            - dpi         : dots per inch of the saved figure
            - dy          : downsampling in the wall-normal direction
            - dx          : downsampling in the streamwise direction
            - dz          : downsampling in the spanwise direction
            - uvw_folder  : folder of the flow fields
            - uvw_file    : file of the flow fields
            - L_x         : streamwise dimension of the channel
            - L_y         : wall-normal dimension of the channel
            - L_z         : spanwise dimension of the channel
            - rey         : friction Reynolds number
            - utau        : friction velocity
            - cmap_flag   : flag to use a color map (True) or solid color (False)

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    struc1      = data_in["struc1"]
    struc2      = data_in["struc2"]
    struc3      = data_in["struc3"]
    struc4      = data_in["struc4"]
    plot_folder = str(data_in["plot_folder"]) 
    xlabel      = str(data_in["xlabel"]) 
    ylabel      = str(data_in["ylabel"])
    zlabel      = str(data_in["zlabel"])
    fontsize    = int(data_in["fontsize"])
    figsize_x   = int(data_in["figsize_x"])
    figsize_y   = int(data_in["figsize_y"])
    colormap1   = data_in["colormap1"]
    colormap2   = data_in["colormap2"]
    colormap3   = data_in["colormap3"]
    colormap4   = data_in["colormap4"]
    fig_name    = str(data_in["fig_name"])
    dpi         = float(data_in["dpi"])
    dy          = int(data_in["dy"])
    dx          = int(data_in["dx"])
    dz          = int(data_in["dz"])
    uvw_folder  = str(data_in["uvw_folder"])
    uvw_file    = str(data_in["uvw_file"])
    L_x         = float(data_in["L_x"])
    L_y         = float(data_in["L_y"])
    L_z         = float(data_in["L_z"])
    rey         = float(data_in["rey"])
    utau        = float(data_in["utau"])
    cmapflag    = bool(data_in["cmap_flag"])
    
    if isinstance(colormap1, str):
        colormap1 = str(colormap1)
    else:
        colormap1 = np.array(colormap1,dtype="float")
    if isinstance(colormap2, str):
        colormap2 = str(colormap2)
    else:
        colormap2 = np.array(colormap2,dtype="float")
    if isinstance(colormap3, str):
        colormap3 = str(colormap3)
    else:
        colormap3 = np.array(colormap3,dtype="float")
    if isinstance(colormap4, str):
        colormap4 = str(colormap4)
    else:
        colormap4 = np.array(colormap4,dtype="float")
    
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
    mat_segment1                                              = np.zeros_like(struc1.structures.mat_segment)
    maxvol1                                                   = np.flip(struc1.structures.vol.argsort(),0)
    mat_segment1[struc1.structures.mat_segment==maxvol1[0]+1] = 1
    mat_segment1[struc1.structures.mat_segment==maxvol1[1]+1] = 2
    mat_segment1[struc1.structures.mat_segment==maxvol1[2]+1] = 3
    mat_segment1[struc1.structures.mat_segment==maxvol1[3]+1] = 4
    mat_segment1[struc1.structures.mat_segment==maxvol1[4]+1] = 5
    mat_segment2                                              = np.zeros_like(struc2.structures.mat_segment)
    maxvol2                                                   = np.flip(struc2.structures.vol.argsort(),0)
    mat_segment2[struc2.structures.mat_segment==maxvol2[0]+1] = 1
    mat_segment2[struc2.structures.mat_segment==maxvol2[1]+1] = 2
    mat_segment2[struc2.structures.mat_segment==maxvol2[2]+1] = 3
    mat_segment2[struc2.structures.mat_segment==maxvol2[3]+1] = 4
    mat_segment2[struc2.structures.mat_segment==maxvol2[4]+1] = 5
    mat_segment3                                              = np.zeros_like(struc3.structures.mat_segment)
    maxvol3                                                   = np.flip(struc3.structures.vol.argsort(),0)
    mat_segment3[struc3.structures.mat_segment==maxvol3[0]+1] = 1
    mat_segment3[struc3.structures.mat_segment==maxvol3[1]+1] = 2
    mat_segment3[struc3.structures.mat_segment==maxvol3[2]+1] = 3
    mat_segment3[struc3.structures.mat_segment==maxvol3[3]+1] = 4
    mat_segment3[struc3.structures.mat_segment==maxvol3[4]+1] = 5
    mat_segment4                                              = np.zeros_like(struc4.structures.mat_segment)
    maxvol4                                                   = np.flip(struc4.structures.vol.argsort(),0)
    mat_segment4[struc4.structures.mat_segment==maxvol4[0]+1] = 1
    mat_segment4[struc4.structures.mat_segment==maxvol4[1]+1] = 2
    mat_segment4[struc4.structures.mat_segment==maxvol4[2]+1] = 3
    mat_segment4[struc4.structures.mat_segment==maxvol4[3]+1] = 4
    mat_segment4[struc4.structures.mat_segment==maxvol4[4]+1] = 5
    index                                                     = str(struc1.index)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap1,
                  "colornum":0,"legend":False,"fig_name":fig_name+"_field"+index,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1 = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_segment1,
                  "color":colormap1,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_info2 = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_segment2,
                  "color":colormap2,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info2)
    plot_info3 = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_segment3,
                  "color":colormap3,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info3)
    plot_info4 = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_segment4,
                  "color":colormap4,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info4)
    plot_train.plot_layout_3d(data_in={"xticks":[0,
                                                 L_x/3*rey,
                                                 2*L_x/3*rey,
                                                 L_x*rey],
                                       "yticks":[0,
                                                 L_z*rey],
                                       "zticks":[-L_y*rey,
                                                 0,
                                                 L_y*rey],
                                       "xticklabels":["0",
                                                      str(int(np.round(L_x/np.pi/3*rey)))+"$\pi$",
                                                      str(int(np.round(2*L_x/np.pi/3*rey)))+"$\pi$",
                                                      str(int(np.round(L_x/np.pi*rey)))+"$\pi$"],
                                       "yticklabels":["0",
                                                      str(int(np.round(L_z/np.pi*rey)))+"$\pi$"],
                                       "zticklabels":["0",
                                                      str(int(np.round(L_y*rey))),
                                                      "0"],
                                       "L_x":L_x,"L_y":L_y*2,"L_z":L_z,"xpad":30,"ypad":10,"zpad":7})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    

    
       

def plotstruc3d_separe(data_in={"struc1":[],"struc2":[],"struc3":[],"struc4":[],"plot_folder":"plots","xlabel":"$x^+$",
                                "ylabel":"$y^+$","zlabel":"z^+","fontsize":18,"figsize_x":10,"figsize_y":8,
                                "colormap1":[],"colormap2":[],"colormap3":[],"colormap4":[],"fig_name":"struc3d",
                                "dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                                "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,
                                "utau":0.060523258443963,"cmap_flag":False}):
    """
    .....................................................................................................................
    # plotstruc3d_separe: Function to generate the plot of the 3d structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc1":[],"struc2":[],"struc3":[],"struc4":[],"plot_folder":"plots","xlabel":"$x^+$",
                        "ylabel":"$y^+$","zlabel":"z^+","fontsize":18,"figsize_x":10,"figsize_y":8,
                        "colormap1":[],"colormap2":[],"colormap3":[],"colormap4":[],"fig_name":"struc3d",
                        "dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                        "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,
                        "utau":0.060523258443963,"cmap_flag":False}.
        Data:
            - struc1      : data of the structure 1
            - struc2      : data of the structure 2
            - struc3      : data of the structure 3
            - struc4      : data of the structure 4
            - plot_folder : folder to store the figures
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - zlabel      : label of the z axis
            - fontsize    : font size used for the figure
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colormap1   : colormap 1 used for the figure
            - colormap2   : colormap 2 used for the figure
            - colormap3   : colormap 3 used for the figure
            - colormap4   : colormap 4 used for the figure
            - fig_name    : name of the saved figure
            - dpi         : dots per inch of the saved figure
            - dy          : downsampling in the wall-normal direction
            - dx          : downsampling in the streamwise direction
            - dz          : downsampling in the spanwise direction
            - uvw_folder  : folder of the flow fields
            - uvw_file    : file of the flow fields
            - L_x         : streamwise dimension of the channel
            - L_y         : wall-normal dimension of the channel
            - L_z         : spanwise dimension of the channel
            - rey         : friction Reynolds number
            - utau        : friction velocity
            - cmap_flag   : flag to use a color map (True) or solid color (False)

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    struc1      = data_in["struc1"]
    struc2      = data_in["struc2"]
    struc3      = data_in["struc3"]
    struc4      = data_in["struc4"]
    plot_folder = str(data_in["plot_folder"]) 
    xlabel      = str(data_in["xlabel"]) 
    ylabel      = str(data_in["ylabel"])
    zlabel      = str(data_in["zlabel"])
    fontsize    = int(data_in["fontsize"])
    figsize_x   = int(data_in["figsize_x"])
    figsize_y   = int(data_in["figsize_y"])
    colormap1   = data_in["colormap1"]
    colormap2   = data_in["colormap2"]
    colormap3   = data_in["colormap3"]
    colormap4   = data_in["colormap4"]
    fig_name    = str(data_in["fig_name"])
    dpi         = float(data_in["dpi"])
    dy          = int(data_in["dy"])
    dx          = int(data_in["dx"])
    dz          = int(data_in["dz"])
    uvw_folder  = str(data_in["uvw_folder"])
    uvw_file    = str(data_in["uvw_file"])
    L_x         = float(data_in["L_x"])
    L_y         = float(data_in["L_y"])
    L_z         = float(data_in["L_z"])
    rey         = float(data_in["rey"])
    utau        = float(data_in["utau"])
    cmapflag    = bool(data_in["cmap_flag"])
    
    if isinstance(colormap1, str):
        colormap1 = str(colormap1)
    else:
        colormap1 = np.array(colormap1,dtype="float")
    if isinstance(colormap2, str):
        colormap2 = str(colormap2)
    else:
        colormap2 = np.array(colormap2,dtype="float")
    if isinstance(colormap3, str):
        colormap3 = str(colormap3)
    else:
        colormap3 = np.array(colormap3,dtype="float")
    if isinstance(colormap4, str):
        colormap4 = str(colormap4)
    else:
        colormap4 = np.array(colormap4,dtype="float")
    
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
    mat_segment_tot1                                         = struc1.structures.mat_segment.copy()
    mat_segment_tot_low1                                     = mat_segment_tot1[:flow_data.yl_s,:,:]
    mat_segment_tot_upp1                                     = np.flip(mat_segment_tot1[flow_data.yu_s:,:,:],0)
    uniqueval_low1,countval_low1                             = np.unique(mat_segment_tot_low1,return_counts=True)
    uniqueval_upp1,countval_upp1                             = np.unique(mat_segment_tot_upp1,return_counts=True)
    maxvol_low1                                              = uniqueval_low1[np.flip(np.argsort(countval_low1[1:])+1,0)]
    maxvol_upp1                                              = uniqueval_upp1[np.flip(np.argsort(countval_upp1[1:])+1,0)]
    mat_segment_low1                                         = np.zeros_like(mat_segment_tot_low1)
    mat_segment_upp1                                         = np.zeros_like(mat_segment_tot_upp1)
    mat_segment_low1[mat_segment_tot_low1 == maxvol_low1[0]] = 1
    mat_segment_low1[mat_segment_tot_low1 == maxvol_low1[1]] = 2
    mat_segment_low1[mat_segment_tot_low1 == maxvol_low1[2]] = 3
    mat_segment_low1[mat_segment_tot_low1 == maxvol_low1[3]] = 4
    mat_segment_low1[mat_segment_tot_low1 == maxvol_low1[4]] = 5
    mat_segment_upp1[mat_segment_tot_upp1 == maxvol_upp1[0]] = 1
    mat_segment_upp1[mat_segment_tot_upp1 == maxvol_upp1[1]] = 2
    mat_segment_upp1[mat_segment_tot_upp1 == maxvol_upp1[2]] = 3
    mat_segment_upp1[mat_segment_tot_upp1 == maxvol_upp1[3]] = 4
    mat_segment_upp1[mat_segment_tot_upp1 == maxvol_upp1[4]] = 5
    mat_segment_tot2                                         = struc2.structures.mat_segment.copy()
    mat_segment_tot_low2                                     = mat_segment_tot2[:flow_data.yl_s,:,:]
    mat_segment_tot_upp2                                     = np.flip(mat_segment_tot2[flow_data.yu_s:,:,:],0)
    uniqueval_low2,countval_low2                             = np.unique(mat_segment_tot_low2,return_counts=True)
    uniqueval_upp2,countval_upp2                             = np.unique(mat_segment_tot_upp2,return_counts=True)
    maxvol_low2                                              = uniqueval_low2[np.flip(np.argsort(countval_low2[1:])+1,0)]
    maxvol_upp2                                              = uniqueval_upp2[np.flip(np.argsort(countval_upp2[1:])+1,0)]
    mat_segment_low2                                         = np.zeros_like(mat_segment_tot_low2)
    mat_segment_upp2                                         = np.zeros_like(mat_segment_tot_upp2)
    mat_segment_low2[mat_segment_tot_low2 == maxvol_low2[0]] = 1
    mat_segment_low2[mat_segment_tot_low2 == maxvol_low2[1]] = 2
    mat_segment_low2[mat_segment_tot_low2 == maxvol_low2[2]] = 3
    mat_segment_low2[mat_segment_tot_low2 == maxvol_low2[3]] = 4
    mat_segment_low2[mat_segment_tot_low2 == maxvol_low2[4]] = 5
    mat_segment_upp2[mat_segment_tot_upp2 == maxvol_upp2[0]] = 1
    mat_segment_upp2[mat_segment_tot_upp2 == maxvol_upp2[1]] = 2
    mat_segment_upp2[mat_segment_tot_upp2 == maxvol_upp2[2]] = 3
    mat_segment_upp2[mat_segment_tot_upp2 == maxvol_upp2[3]] = 4
    mat_segment_upp2[mat_segment_tot_upp2 == maxvol_upp2[4]] = 5
    mat_segment_tot3                                         = struc3.structures.mat_segment.copy()
    mat_segment_tot_low3                                     = mat_segment_tot3[:flow_data.yl_s,:,:]
    mat_segment_tot_upp3                                     = np.flip(mat_segment_tot3[flow_data.yu_s:,:,:],0)
    uniqueval_low3,countval_low3                             = np.unique(mat_segment_tot_low3,return_counts=True)
    uniqueval_upp3,countval_upp3                             = np.unique(mat_segment_tot_upp3,return_counts=True)
    maxvol_low3                                              = uniqueval_low3[np.flip(np.argsort(countval_low3[1:])+1,0)]
    maxvol_upp3                                              = uniqueval_upp3[np.flip(np.argsort(countval_upp3[1:])+1,0)]
    mat_segment_low3                                         = np.zeros_like(mat_segment_tot_low3)
    mat_segment_upp3                                         = np.zeros_like(mat_segment_tot_upp3)
    mat_segment_low3[mat_segment_tot_low3 == maxvol_low3[0]] = 1
    mat_segment_low3[mat_segment_tot_low3 == maxvol_low3[1]] = 2
    mat_segment_low3[mat_segment_tot_low3 == maxvol_low3[2]] = 3
    mat_segment_low3[mat_segment_tot_low3 == maxvol_low3[3]] = 4
    mat_segment_low3[mat_segment_tot_low3 == maxvol_low3[4]] = 5
    mat_segment_upp3[mat_segment_tot_upp3 == maxvol_upp3[0]] = 1
    mat_segment_upp3[mat_segment_tot_upp3 == maxvol_upp3[1]] = 2
    mat_segment_upp3[mat_segment_tot_upp3 == maxvol_upp3[2]] = 3
    mat_segment_upp3[mat_segment_tot_upp3 == maxvol_upp3[3]] = 4
    mat_segment_upp3[mat_segment_tot_upp3 == maxvol_upp3[4]] = 5
    mat_segment_tot4                                         = struc4.structures.mat_segment.copy()
    mat_segment_tot_low4                                     = mat_segment_tot4[:flow_data.yl_s,:,:]
    mat_segment_tot_upp4                                     = np.flip(mat_segment_tot4[flow_data.yu_s:,:,:],0)
    uniqueval_low4,countval_low4                             = np.unique(mat_segment_tot_low4,return_counts=True)
    uniqueval_upp4,countval_upp4                             = np.unique(mat_segment_tot_upp4,return_counts=True)
    maxvol_low4                                              = uniqueval_low4[np.flip(np.argsort(countval_low4[1:])+1,0)]
    maxvol_upp4                                              = uniqueval_upp4[np.flip(np.argsort(countval_upp4[1:])+1,0)]
    mat_segment_low4                                         = np.zeros_like(mat_segment_tot_low4)
    mat_segment_upp4                                         = np.zeros_like(mat_segment_tot_upp4)
    mat_segment_low4[mat_segment_tot_low4 == maxvol_low4[0]] = 1
    mat_segment_low4[mat_segment_tot_low4 == maxvol_low4[1]] = 2
    mat_segment_low4[mat_segment_tot_low4 == maxvol_low4[2]] = 3
    mat_segment_low4[mat_segment_tot_low4 == maxvol_low4[3]] = 4
    mat_segment_low4[mat_segment_tot_low4 == maxvol_low4[4]] = 5
    mat_segment_upp4[mat_segment_tot_upp4 == maxvol_upp4[0]] = 1
    mat_segment_upp4[mat_segment_tot_upp4 == maxvol_upp4[1]] = 2
    mat_segment_upp4[mat_segment_tot_upp4 == maxvol_upp4[2]] = 3
    mat_segment_upp4[mat_segment_tot_upp4 == maxvol_upp4[3]] = 4
    mat_segment_upp4[mat_segment_tot_upp4 == maxvol_upp4[4]] = 5
    index                                                    = str(struc1.index)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap1,
                  "colornum":0,"legend":False,"fig_name":fig_name+"_field"+index+"_low","dpi":dpi,"plot_folder":plot_folder,
                  "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_segment_low1,
                  "color":colormap1,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_info2 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_segment_low2,
                  "color":colormap2,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info2)
    plot_info3 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_segment_low3,
                  "color":colormap3,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info3)
    plot_info4 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_segment_low4,
                  "color":colormap4,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info4)
    plot_train.plot_layout_3d(data_in={"xticks":[0,
                                                 L_x/3*rey,
                                                 2*L_x/3*rey,
                                                 L_x*rey],
                                       "yticks":[0,
                                                 L_z*rey],
                                       "zticks":[0,
                                                 L_y*rey],
                                       "xticklabels":["0",
                                                      str(int(np.round(L_x/np.pi/3*rey)))+"$\pi$",
                                                      str(int(np.round(2*L_x/np.pi/3*rey)))+"$\pi$",
                                                      str(int(np.round(L_x/np.pi*rey)))+"$\pi$"],
                                       "yticklabels":["0",
                                                      str(int(np.round(L_z/np.pi*rey)))+"$\pi$"],
                                       "zticklabels":["0",
                                                      str(int(np.round(L_y*rey)))],
                                       "L_x":L_x,"L_y":L_y,"L_z":L_z,"xpad":30,"ypad":10,"zpad":7})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()   
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap1,
                  "colornum":0,"legend":False,"fig_name":fig_name+"_field"+index+"_upp","dpi":dpi,
                  "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_segment_upp1,
                  "color":colormap1,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_info2 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_segment_upp2,
                  "color":colormap2,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info2)
    plot_info3 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_segment_upp3,
                  "color":colormap3,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info3)
    plot_info4 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_segment_upp4,
                  "color":colormap4,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info4)
    plot_train.plot_layout_3d(data_in={"xticks":[0,
                                                 L_x/3*rey,
                                                 2*L_x/3*rey,
                                                 L_x*rey],
                                       "yticks":[0,
                                                 L_z*rey],
                                       "zticks":[0,
                                                 L_y*rey],
                                       "xticklabels":["0",
                                                      str(int(np.round(L_x/np.pi/3*rey)))+"$\pi$",
                                                      str(int(np.round(2*L_x/np.pi/3*rey)))+"$\pi$",
                                                      str(int(np.round(L_x/np.pi*rey)))+"$\pi$"],
                                       "yticklabels":["0",
                                                      str(int(np.round(L_z/np.pi*rey)))+"$\pi$"],
                                       "zticklabels":["0",
                                                      str(int(np.round(L_y*rey)))],
                                       "L_x":L_x,"L_y":L_y,"L_z":L_z,"xpad":30,"ypad":10,"zpad":7})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()  