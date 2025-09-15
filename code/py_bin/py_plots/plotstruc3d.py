# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plotmean.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella

File to plot training information:
    - Functions:
        - plotumean : function to plot the mean velocity
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

def plotstruc3d(data_in={"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$","zlabel":"z^+",
                         "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                         "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                         "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                         "rey":125,"utau":0.060523258443963,"cmap_flag":False,"padtext":[0,0,0],"savepdf":False}):
    """
    .....................................................................................................................
    # plotstruc3d: Function to generate the plot of the 3d structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$","zlabel":"z^+",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                        "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                        "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                        "rey":125,"utau":0.060523258443963,"cmap_flag":False,"padtext":[0,0,0],"savepdf":False}.
        Data:
            - struc       : data of the structures
            - plot_folder : folder to store the figures
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - zlabel      : label of the z axis
            - fontsize    : font size used for the figure
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colormap    : colormap used for the figure
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
            - padtext     : padding used for the labels text
            - savepdf     : activate flag to save figure in pdf

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
    struc       = data_in["struc"]
    plot_folder = str(data_in["plot_folder"]) 
    xlabel      = str(data_in["xlabel"]) 
    ylabel      = str(data_in["ylabel"])
    zlabel      = str(data_in["zlabel"])
    fontsize    = int(data_in["fontsize"])
    figsize_x   = int(data_in["figsize_x"])
    figsize_y   = int(data_in["figsize_y"])
    colormap    = data_in["colormap"]
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
    savepdf     = bool(data_in["savepdf"])
    padtext     = np.array(data_in["padtext"],dtype="int")
    padtext_x   = padtext[0]
    padtext_y   = padtext[1]
    padtext_z   = padtext[2]
    
    if isinstance(colormap, str):
        colormap = str(colormap)
    else:
        colormap = np.array(colormap,dtype="float")
    
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
    mat_segment = struc.structures.mat_segment_filtered
    index       = str(struc.index)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":False,"fig_name":fig_name+"_field"+index,"dpi":dpi,
                  "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1 = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_segment,
                  "color":None,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_train.plot_layout_3d(data_in={"xticks":[0,
                                                 L_x/4*rey,
                                                 2*L_x/4*rey,
                                                 3*L_x/4*rey,
                                                 L_x*rey],
                                       "yticks":[0,
                                                 L_z/2*rey,
                                                 L_z*rey],
                                       "zticks":[-L_y*rey,
                                                 0,
                                                 L_y*rey],
                                       "xticklabels":["0",
                                                      str(int(np.round(L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(2*L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(3*L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(L_x/np.pi*rey)))+"$\pi$"],
                                       "yticklabels":["0",
                                                      str(int(np.round(L_z/np.pi/2*rey)))+"$\pi$",
                                                      str(int(np.round(L_z/np.pi*rey)))+"$\pi$"],
                                       "zticklabels":["0",
                                                      str(int(np.round(L_y*rey))),
                                                      "0"],
                                       "L_x":L_x,"L_y":L_y*2,"L_z":L_z,"xpad":padtext_x,"ypad":padtext_y,
                                       "zpad":padtext_z})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    if savepdf == True:
        plot_train.plot_save_pdf()
    
    

def plotstruc3d_separe(data_in={"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$","zlabel":"z^+",
                                "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                                "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                                "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                                "rey":125,"utau":0.060523258443963,"cmap_flag":False,"padtext":[0,0,0],"savepdf":False}):
    """
    .....................................................................................................................
    # plotstruc3d_separe: Function to generate the plot of the 3d structures separated in 2 semichannels
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$","zlabel":"z^+",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                        "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                        "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                        "rey":125,"utau":0.060523258443963,"cmap_flag":False,"padtext":[0,0,0],"savepdf":False}.
        Data:
            - struc       : data of the structures
            - plot_folder : folder to store the figures
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - zlabel      : label of the z axis
            - fontsize    : font size used for the figure
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colormap    : colormap used for the figure
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
            - padtext     : padding used for the labels text
            - savepdf     : activate flag to save figure in pdf

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
    struc       = data_in["struc"]
    plot_folder = str(data_in["plot_folder"]) 
    xlabel      = str(data_in["xlabel"]) 
    ylabel      = str(data_in["ylabel"])
    zlabel      = str(data_in["zlabel"])
    fontsize    = int(data_in["fontsize"])
    figsize_x   = int(data_in["figsize_x"])
    figsize_y   = int(data_in["figsize_y"])
    colormap    = data_in["colormap"]
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
    savepdf     = bool(data_in["savepdf"])
    padtext     = np.array(data_in["padtext"],dtype="int")
    padtext_x   = padtext[0]
    padtext_y   = padtext[1]
    padtext_z   = padtext[2]
    
    if isinstance(colormap, str):
        colormap = str(colormap)
    else:
        colormap = np.array(colormap,dtype="float")
    
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
    mat_segment     = struc.structures.mat_segment_filtered
    mat_segment_low = mat_segment[:flow_data.yl_s,:,:]
    mat_segment_upp = np.flip(mat_segment[flow_data.yu_s:,:,:],0)
    index           = str(struc.index)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":False,"fig_name":fig_name+'_low'+"_field"+index,"dpi":dpi,
                  "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_segment_low,
                  "color":None,"cmap_flag":True,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_train.plot_layout_3d(data_in={"xticks":[0,
                                                 L_x/4*rey,
                                                 2*L_x/4*rey,
                                                 3*L_x/4*rey,
                                                 L_x*rey],
                                       "yticks":[0,
                                                 L_z/2*rey,
                                                 L_z*rey],
                                       "zticks":[0,
                                                 L_y*rey],
                                       "xticklabels":["0",
                                                      str(int(np.round(L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(2*L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(3*L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(L_x/np.pi*rey)))+"$\pi$"],
                                       "yticklabels":["0",
                                                      str(int(np.round(L_z/np.pi/2*rey)))+"$\pi$",
                                                      str(int(np.round(L_z/np.pi*rey)))+"$\pi$"],
                                       "zticklabels":["0",
                                                      str(int(np.round(L_y*rey)))],
                                       "L_x":L_x,"L_y":L_y,"L_z":L_z,"xpad":padtext_x,"ypad":padtext_y,
                                       "zpad":padtext_z})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    if savepdf == True:
        plot_train.plot_save_pdf()
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":False,"fig_name":fig_name+'_upp'+"_field"+index,"dpi":dpi,
                  "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_segment_upp,
                  "color":None,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
    plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_train.plot_layout_3d(data_in={"xticks":[0,
                                                 L_x/4*rey,
                                                 2*L_x/4*rey,
                                                 3*L_x/4*rey,
                                                 L_x*rey],
                                       "yticks":[0,
                                                 L_z/2*rey,
                                                 L_z*rey],
                                       "zticks":[0,
                                                 L_y*rey],
                                       "xticklabels":["0",
                                                      str(int(np.round(L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(2*L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(3*L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(L_x/np.pi*rey)))+"$\pi$"],
                                       "yticklabels":["0",
                                                      str(int(np.round(L_z/np.pi/2*rey)))+"$\pi$",
                                                      str(int(np.round(L_z/np.pi*rey)))+"$\pi$"],
                                       "zticklabels":["0",
                                                      str(int(np.round(L_y*rey)))],
                                       "L_x":L_x,"L_y":L_y,"L_z":L_z,"xpad":padtext_x,"ypad":padtext_y,
                                       "zpad":padtext_z})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    if savepdf == True:
        plot_train.plot_save_pdf()
    
    
def plotstruc3d_2Q(data_in={"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$","zlabel":"z^+",
                            "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap1":"viridis","colormap2":"viridis",
                            "colornum":2,"fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,
                            "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                            "L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963,"cmap_flag":False}):
    """
    .....................................................................................................................
    # plotstruc3d_2Q: Function to generate the plot of 2 Quadrants of the 3d structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$","zlabel":"z^+",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap1":"viridis","colormap2":"viridis",
                        "colornum":2,"fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,
                        "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                        "L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963,"cmap_flag":False}.
        Data:
            - struc       : data of the structures
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
    struc       = data_in["struc"]
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
    mat_segment = struc.structures.mat_event
    mat_q2      = np.zeros_like(mat_segment)
    mat_q4      = np.zeros_like(mat_segment)
    mat_q2[mat_segment==2] = 1
    mat_q4[mat_segment==4] = 1
    index       = str(struc.index)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap1,
                  "colornum":colornum,"legend":False,"fig_name":fig_name+"_field"+index,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1 = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_q2,
                  "color":colormap1,"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_info2 = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_q4,
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
    
    
    
def plotstruc3d_separe_2Q(data_in={"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$","zlabel":"z^+",
                                   "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap1":"viridis",
                                   "colormap2":"viridis","colornum":2,"fig_name":"struc3d","dpi":60,"dy":1,"dx":1,
                                   "dz":1,"uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                                   "L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963,"cmap_flag":False}):
    """
    .....................................................................................................................
    # plotstruc3d_separa_2Q: Function to generate the plot of 2 Quadrants of the 3d structures separe semichannels
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$","zlabel":"z^+",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap1":"viridis","colormap2":"viridis",
                        "colornum":2,"fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,
                        "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                        "L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963,"cmap_flag":False}.
        Data:
            - struc       : data of the structures
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
    struc       = data_in["struc"]
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
    mat_segment     = struc.structures.mat_event
    mat_segment_low = mat_segment[:flow_data.yl_s,:,:]
    mat_segment_upp = np.flip(mat_segment[flow_data.yu_s:,:,:],0)
    mat_q2_low      = np.zeros_like(mat_segment_low)
    mat_q4_low      = np.zeros_like(mat_segment_low)
    mat_q2_upp      = np.zeros_like(mat_segment_upp)
    mat_q4_upp      = np.zeros_like(mat_segment_upp)
    mat_q2_low[mat_segment_low==2] = 1
    mat_q4_low[mat_segment_low==4] = 1
    mat_q2_upp[mat_segment_upp==2] = 1
    mat_q4_upp[mat_segment_upp==4] = 1
    index       = str(struc.index)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap1,
                  "colornum":colornum,"legend":False,"fig_name":fig_name+"_field"+index+"_low","dpi":dpi,
                  "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_q2_low,
                  "color":colormap1,"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_info2 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_q4_low,
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
    
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap1,
                  "colornum":colornum,"legend":False,"fig_name":fig_name+"_field"+index+"_upp","dpi":dpi,
                  "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_q2_upp,
                  "color":colormap1,"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_info2 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_q4_upp,
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
    

def plotstruc3d_shapcol(data_in={"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$","zlabel":"z^+",
                                 "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                                 "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                                 "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                                 "rey":125,"utau":0.060523258443963,"padtext":[0,0,0],"shap":[],
                                 "shap_ind":[],"shap_max":1,"shap_min":0}):
    """
    .....................................................................................................................
    # plotstruc3d_shapcol: Function to generate the plot of the 3d structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$","zlabel":"z^+",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                        "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                        "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                        "rey":125,"utau":0.060523258443963,padtext":[0,0,0],"shapdata":[],
                        "shap_ind":[],"shap_max":1,"shap_min":0}.
        Data:
            - struc       : data of the structures
            - plot_folder : folder to store the figures
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - zlabel      : label of the z axis
            - fontsize    : font size used for the figure
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colormap    : colormap used for the figure
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
            - padtext     : padding used for the labels text
            - shap        : array of the shap values
            - shap_ind    : array of the index of structures used for the shap values
            - shap_max    : maximum value of the SHAP, if not input, default
            - shap_min    : minimum value of the SHAP, if not input, default

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from py_bin.py_class.flow_field import flow_field
    import matplotlib
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    struc        = data_in["struc"]
    plot_folder  = str(data_in["plot_folder"]) 
    xlabel       = str(data_in["xlabel"]) 
    ylabel       = str(data_in["ylabel"])
    zlabel       = str(data_in["zlabel"])
    fontsize     = int(data_in["fontsize"])
    figsize_x    = int(data_in["figsize_x"])
    figsize_y    = int(data_in["figsize_y"])
    colormap     = data_in["colormap"]
    colornum     = int(data_in["colornum"])
    fig_name     = str(data_in["fig_name"])
    dpi          = float(data_in["dpi"])
    dy           = int(data_in["dy"])
    dx           = int(data_in["dx"])
    dz           = int(data_in["dz"])
    uvw_folder   = str(data_in["uvw_folder"])
    uvw_file     = str(data_in["uvw_file"])
    L_x          = float(data_in["L_x"])
    L_y          = float(data_in["L_y"])
    L_z          = float(data_in["L_z"])
    rey          = float(data_in["rey"])
    utau         = float(data_in["utau"])
    cmapflag     = False
    padtext      = np.array(data_in["padtext"],dtype="int")
    padtext_x    = padtext[0]
    padtext_y    = padtext[1]
    padtext_z    = padtext[2]
    shap         = np.array(data_in["shap"],dtype="float")
    shap_ind     = np.array(data_in["shap_ind"],dtype="int")
    if "shap_max" in data_in:
        shap_max = float(data_in["shap_max"])
    else:
        shap_max = np.max(shap)
    if "shap_min" in data_in:
        shap_min = float(data_in["shap_min"])
    else:
        shap_min = np.min(shap)
    
    if isinstance(colormap, str):
        colormap = str(colormap)
    else:
        colormap = np.array(colormap,dtype="float")
    normshap     = (shap-shap_min)/(shap_max-shap_min)
        
    
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
    mat_segment = struc.structures.mat_segment_filtered
    index       = str(struc.index)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":False,"fig_name":fig_name+"_field"+index,"dpi":dpi,
                  "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    
    for ind_str in np.arange(len(shap_ind)):
        mat_struc                      = np.zeros_like(mat_segment)
        mat_struc[mat_segment==
                  shap_ind[ind_str]+1] = 1
        cmap                           = matplotlib.cm.get_cmap(colormap)
        color                          = cmap(normshap[ind_str])
        plot_info1                     = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,
                                          "struc":mat_struc,"color":color,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
        plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_train.plot_layout_3d(data_in={"xticks":[0,
                                                 L_x/4*rey,
                                                 2*L_x/4*rey,
                                                 3*L_x/4*rey,
                                                 L_x*rey],
                                       "yticks":[0,
                                                 L_z/2*rey,
                                                 L_z*rey],
                                       "zticks":[-L_y*rey,
                                                 0,
                                                 L_y*rey],
                                       "xticklabels":["0",
                                                      str(int(np.round(L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(2*L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(3*L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(L_x/np.pi*rey)))+"$\pi$"],
                                       "yticklabels":["0",
                                                      str(int(np.round(L_z/np.pi/2*rey)))+"$\pi$",
                                                      str(int(np.round(L_z/np.pi*rey)))+"$\pi$"],
                                       "zticklabels":["0",
                                                      str(int(np.round(L_y*rey))),
                                                      "0"],
                                       "L_x":L_x,"L_y":L_y*2,"L_z":L_z,"xpad":padtext_x,"ypad":padtext_y,
                                       "zpad":padtext_z})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    
    
def plotstruc3d_separe_shapcol(data_in={"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$","zlabel":"z^+",
                                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                                        "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                                        "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                                        "rey":125,"utau":0.060523258443963,"padtext":[0,0,0],"shap":[],
                                        "shap_ind":[],"shap_max":1,"shap_min":0}):
    """
    .....................................................................................................................
    # plotstruc3d_shapcol: Function to generate the plot of the 3d structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$","zlabel":"z^+",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                        "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                        "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                        "rey":125,"utau":0.060523258443963,padtext":[0,0,0],"shapdata":[],
                        "shap_ind":[],"shap_max":1,"shap_min":0}.
        Data:
            - struc       : data of the structures
            - plot_folder : folder to store the figures
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - zlabel      : label of the z axis
            - fontsize    : font size used for the figure
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colormap    : colormap used for the figure
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
            - padtext     : padding used for the labels text
            - shap        : array of the shap values
            - shap_ind    : array of the index of structures used for the shap values
            - shap_max    : maximum value of the SHAP, if not input, default
            - shap_min    : minimum value of the SHAP, if not input, default

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from py_bin.py_class.flow_field import flow_field
    import matplotlib
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    struc        = data_in["struc"]
    plot_folder  = str(data_in["plot_folder"]) 
    xlabel       = str(data_in["xlabel"]) 
    ylabel       = str(data_in["ylabel"])
    zlabel       = str(data_in["zlabel"])
    fontsize     = int(data_in["fontsize"])
    figsize_x    = int(data_in["figsize_x"])
    figsize_y    = int(data_in["figsize_y"])
    colormap     = data_in["colormap"]
    colornum     = int(data_in["colornum"])
    fig_name     = str(data_in["fig_name"])
    dpi          = float(data_in["dpi"])
    dy           = int(data_in["dy"])
    dx           = int(data_in["dx"])
    dz           = int(data_in["dz"])
    uvw_folder   = str(data_in["uvw_folder"])
    uvw_file     = str(data_in["uvw_file"])
    L_x          = float(data_in["L_x"])
    L_y          = float(data_in["L_y"])
    L_z          = float(data_in["L_z"])
    rey          = float(data_in["rey"])
    utau         = float(data_in["utau"])
    cmapflag     = False
    padtext      = np.array(data_in["padtext"],dtype="int")
    padtext_x    = padtext[0]
    padtext_y    = padtext[1]
    padtext_z    = padtext[2]
    shap         = np.array(data_in["shap"],dtype="float")
    shap_ind     = np.array(data_in["shap_ind"],dtype="int")
    if "shap_max" in data_in:
        shap_max = float(data_in["shap_max"])
    else:
        shap_max = np.max(shap)
    if "shap_min" in data_in:
        shap_min = float(data_in["shap_min"])
    else:
        shap_min = np.min(shap)
    
    if isinstance(colormap, str):
        colormap = str(colormap)
    else:
        colormap = np.array(colormap,dtype="float")
    normshap     = (shap-shap_min)/(shap_max-shap_min)
        
    
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
    mat_segment     = struc.structures.mat_segment_filtered
    mat_segment_low = mat_segment[:flow_data.yl_s,:,:]
    mat_segment_upp = np.flip(mat_segment[flow_data.yu_s:,:,:],0)
    index           = str(struc.index)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":False,"fig_name":fig_name+'_low'+"_field"+index,"dpi":dpi,
                  "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    
    for ind_str in np.arange(len(shap_ind)):
        mat_struc                      = np.zeros_like(mat_segment_low)
        mat_struc[mat_segment_low==
                  shap_ind[ind_str]+1] = 1
        cmap                           = matplotlib.cm.get_cmap(colormap)
        color                          = cmap(normshap[ind_str])
        plot_info1                     = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,
                                          "struc":mat_struc,"color":color,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
        plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_train.plot_layout_3d(data_in={"xticks":[0,
                                                 L_x/4*rey,
                                                 2*L_x/4*rey,
                                                 3*L_x/4*rey,
                                                 L_x*rey],
                                       "yticks":[0,
                                                 L_z/2*rey,
                                                 L_z*rey],
                                       "zticks":[0,
                                                 L_y*rey],
                                       "xticklabels":["0",
                                                      str(int(np.round(L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(2*L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(3*L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(L_x/np.pi*rey)))+"$\pi$"],
                                       "yticklabels":["0",
                                                      str(int(np.round(L_z/np.pi/2*rey)))+"$\pi$",
                                                      str(int(np.round(L_z/np.pi*rey)))+"$\pi$"],
                                       "zticklabels":["0",
                                                      str(int(np.round(L_y*rey)))],
                                       "L_x":L_x,"L_y":L_y,"L_z":L_z,"xpad":padtext_x,"ypad":padtext_y,
                                       "zpad":padtext_z})
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
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":False,"fig_name":fig_name+'_upp'+"_field"+index,"dpi":dpi,
                  "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    
    for ind_str in np.arange(len(shap_ind)):
        mat_struc                      = np.zeros_like(mat_segment_upp)
        mat_struc[mat_segment_upp==
                  shap_ind[ind_str]+1] = 1
        cmap                           = matplotlib.cm.get_cmap(colormap)
        color                          = cmap(normshap[ind_str])
        plot_info1                     = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,
                                          "struc":mat_struc,"color":color,"cmap_flag":cmapflag,"vmax":None,"vmin":None}
        plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_train.plot_layout_3d(data_in={"xticks":[0,
                                                 L_x/4*rey,
                                                 2*L_x/4*rey,
                                                 3*L_x/4*rey,
                                                 L_x*rey],
                                       "yticks":[0,
                                                 L_z/2*rey,
                                                 L_z*rey],
                                       "zticks":[0,
                                                 L_y*rey],
                                       "xticklabels":["0",
                                                      str(int(np.round(L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(2*L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(3*L_x/np.pi/4*rey)))+"$\pi$",
                                                      str(int(np.round(L_x/np.pi*rey)))+"$\pi$"],
                                       "yticklabels":["0",
                                                      str(int(np.round(L_z/np.pi/2*rey)))+"$\pi$",
                                                      str(int(np.round(L_z/np.pi*rey)))+"$\pi$"],
                                       "zticklabels":["0",
                                                      str(int(np.round(L_y*rey)))],
                                       "L_x":L_x,"L_y":L_y,"L_z":L_z,"xpad":padtext_x,"ypad":padtext_y,
                                       "zpad":padtext_z})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()