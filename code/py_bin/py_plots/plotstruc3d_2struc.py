# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plotstruc3d_2struc.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella

File to plot training information:
    - Functions:
        - plotstruc3d_2struc : function to plot the 3d structures of 2 types
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

def plotstruc3d_2struc(data_in={"struc1":[],"struc2":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$",
                                "zlabel":"z^+","fontsize":18,"figsize_x":10,"figsize_y":8,"colormap1":"viridis",
                                "colormap2":"viridis","colornum":2,"fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,
                                "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                                "L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963,"cmap_flag":False}):
    """
    .....................................................................................................................
    # plotstruc3d_2struc: Function to generate the plot of the 3d structures of 2 types
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc1":[],"struc2":[],"plot_folder":"plots","xlabel":"$x^+$",
                        "ylabel":"$y^+$","zlabel":"z^+","fontsize":18,"figsize_x":10,
                        "figsize_y":8,"colormap1":"viridis","colormap2":"viridis","colornum":2,
                        "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,
                        "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                        "L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963}.
        Data:
            - struc1      : data of the structures of type 1
            - struc2      : data of the structures of type 2
            - plot_folder : folder to store the figures
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - zlabel      : label of the z axis
            - fontsize    : font size used for the figure
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colormap1   : colormap used for the figure
            - colormap2   : colormap used for the figure
            - colornum    : number of colors of the colormap, two curves are used. The number of levels of the 
                            colormap needs to be higher than 2 
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
    plot_folder = str(data_in["plot_folder"]) 
    xlabel      = str(data_in["xlabel"]) 
    ylabel      = str(data_in["ylabel"])
    zlabel      = str(data_in["zlabel"])
    fontsize    = int(data_in["fontsize"])
    figsize_x   = int(data_in["figsize_x"])
    figsize_y   = int(data_in["figsize_y"])
    colormap1   = data_in["colormap1"]
    colormap2   = data_in["colormap2"]
    colornum    = int(data_in["colornum"])
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
    mat_segment1 = struc1.structures.mat_segment_filtered
    mat_segment2 = struc2.structures.mat_segment_filtered
    index        = str(struc1.index)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap1,
                  "colornum":colornum,"legend":False,"fig_name":fig_name+"_field"+index,"dpi":dpi,
                  "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1 = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_segment1,
                  "color":colormap1,"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_info2 = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_segment2,
                  "color":colormap2,"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_train.add_plot_3d_structure(data_in=plot_info2)
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
    
    
def plotstruc3d_2struc_separate(data_in={"struc1":[],"struc2":[],"plot_folder":"plots","xlabel":"$x^+$",
                                         "ylabel":"$y^+$","zlabel":"z^+","fontsize":18,"figsize_x":10,
                                         "figsize_y":8,"colormap1":"viridis","colormap2":"viridis","colornum":2,
                                         "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,
                                         "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                                         "L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963,
                                         "cmap_flag":False}):
    """
    .....................................................................................................................
    # plotstruc3d_2struc_separate: Function to generate the plot of the 3d structures of 2 types in separate 
                                   semichannel
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc1":[],"struc2":[],"plot_folder":"plots","xlabel":"$x^+$",
                        "ylabel":"$y^+$","zlabel":"z^+","fontsize":18,"figsize_x":10,
                        "figsize_y":8,"colormap1":"viridis","colormap2":"viridis","colornum":2,
                        "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,
                        "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                        "L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963,"cmap_flag":False}.
        Data:
            - struc1      : data of the structures of type 1
            - struc2      : data of the structures of type 2
            - plot_folder : folder to store the figures
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - zlabel      : label of the z axis
            - fontsize    : font size used for the figure
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colormap1   : colormap used for the figure
            - colormap2   : colormap used for the figure
            - colornum    : number of colors of the colormap, two curves are used. The number of levels of the 
                            colormap needs to be higher than 2 
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
    plot_folder = str(data_in["plot_folder"]) 
    xlabel      = str(data_in["xlabel"]) 
    ylabel      = str(data_in["ylabel"])
    zlabel      = str(data_in["zlabel"])
    fontsize    = int(data_in["fontsize"])
    figsize_x   = int(data_in["figsize_x"])
    figsize_y   = int(data_in["figsize_y"])
    colormap1   = data_in["colormap1"]
    colormap2   = data_in["colormap2"]
    colornum    = int(data_in["colornum"])
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
    mat_segment1     = struc1.structures.mat_segment_filtered
    mat_segment1_low = mat_segment1[:flow_data.yl_s,:,:]
    mat_segment1_upp = np.flip(mat_segment1[flow_data.yu_s:,:,:],0)
    mat_segment2     = struc2.structures.mat_segment_filtered
    mat_segment2_low = mat_segment2[:flow_data.yl_s,:,:]
    mat_segment2_upp = np.flip(mat_segment2[flow_data.yu_s:,:,:],0)
    index            = str(struc1.index)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap1,
                  "colornum":colornum,"legend":False,"fig_name":fig_name+"_low"+"_field"+index,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_segment1_low,
                  "color":colormap1,"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_info2 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_segment2_low,
                  "color":colormap2,"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_train.add_plot_3d_structure(data_in=plot_info2)
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
                  "colornum":colornum,"legend":False,"fig_name":fig_name+"_upp"+"_field"+index,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_segment1_upp,
                  "color":colormap1,"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_info2 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_segment2_upp,
                  "color":colormap2,"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_train.add_plot_3d_structure(data_in=plot_info2)
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