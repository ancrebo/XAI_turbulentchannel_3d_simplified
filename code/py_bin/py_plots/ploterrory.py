# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
ploterrory.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella

File to plot training information:
    - Functions:
        - ploterrory : function to plot the error of the prediction along the y direction
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

def ploterrory(data_in={"file":"hist.txt","folder":"Data","plot_folder":"plots","xlabel":"$y^+$",
                       "ylabel":"$\overline{U}$","fontsize":18,"figsize_x":10,"figsize_y":8,
                       "colormap":"viridis","colornum":2,"fig_name":"training_info","dpi":60,
                       "dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                       "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                       "rey":125,"utau":0.060523258443963}):
    """
    .....................................................................................................................
    # ploterrory: Function to generate the plot of the error along the y direction
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"file":"Umean.txt","folder":"Data","plot_folder":"plots","xlabel":"$y^+$",
                        "ylabel":"$\overline{U}$","fontsize":18,"figsize_x":10,
                        "figsize_y":8,"xscale":"linear","yscale":"log","colormap":"viridis","colornum":2,
                        "fig_name":"training_info","dpi":60,"dy":1,"L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                        "rey":125,"utau":0.060523258443963}.
        Data:
            - file : file of the training information
            - folder : folder of the data generated during the training
            - plot_folder : folder to store the figures
            - fontsize    : font size used for the figure
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
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

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from py_bin.py_functions.umean import read_Umean
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    folder      = str(data_in["folder"])      # Folder to read the epoch data
    file        = str(data_in["file"])        # File to read for the epochs data
    plot_folder = str(data_in["plot_folder"]) # folder to save the plots
    xlabel      = str(data_in["xlabel"])      # label of the x axis
    ylabel      = str(data_in["ylabel"])      # label of the y axis
    fontsize    = int(data_in["fontsize"])    # size of the text in the plot
    figsize_x   = int(data_in["figsize_x"])   # size of the figure in direction x
    figsize_y   = int(data_in["figsize_y"])   # size of the figure in direction y
    colormap    = str(data_in["colormap"])    # colormap of the figure
    colornum    = int(data_in["colornum"])    # number of colors of the colormap
    fig_name    = str(data_in["fig_name"])    # name of the figure to be saved
    dpi         = float(data_in["dpi"])       # dots per inch to save the figure
    dy          = int(data_in["dy"])          # downsampling in the wall-normal direction
    dx          = int(data_in["dx"])          # downsampling in the streamwise direction
    dz          = int(data_in["dz"])          # downsampling in the spanwise direction
    uvw_folder  = str(data_in["uvw_folder"])  # folder of the flow fields
    uvw_file    = str(data_in["uvw_file"])    # file of the flow fields
    L_x         = float(data_in["L_x"])       # streamwise dimension of the channel
    L_y         = float(data_in["L_y"])       # wall-normal dimension of the channel
    L_z         = float(data_in["L_z"])       # spanwise dimension of the channel
    rey         = float(data_in["rey"])       # Friction reynolds number
    utau        = float(data_in["utau"])      # Friction velocity
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the file of the error along the y direction. For reading this file, the function of reading the mean 
    # velocity is used as they share the same format
    # -------------------------------------------------------------------------------------------------------------------
    Data_flow   = {"folder":uvw_folder,"file":uvw_file,"down_x":dx,"down_y":dy,"down_z":dz,
                   "L_x":L_x,"L_y":L_y,"L_z":L_z,"rey":rey,"utau":utau}
    flow_data   = flow_field(data_in=Data_flow)
    flow_data.shape_tensor()
    flow_data.flow_grid()
    data_errory = read_Umean(data_in={"folder":folder,"file":file,"dy":dy})
    error_u     = (data_errory["UUmean"][:flow_data.yl_s]+np.flip(data_errory["UUmean"][flow_data.yu_s:]))/2*100
    error_v     = (data_errory["VVmean"][:flow_data.yl_s]+np.flip(data_errory["VVmean"][flow_data.yu_s:]))/2*100
    error_w     = (data_errory["WWmean"][:flow_data.yl_s]+np.flip(data_errory["WWmean"][flow_data.yu_s:]))/2*100
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":flow_data.yplus,"data_y":error_u,"label":"$\epsilon_u$",\
                  "color":None,"linewidth":2,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":flow_data.yplus,"data_y":error_v,"label":"$\epsilon_v$",\
                  "color":None,"linewidth":2,"plot_number":1,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info2)
    plot_info3 = {"data_x":flow_data.yplus,"data_y":error_w,"label":"$\epsilon_w$",\
                  "color":None,"linewidth":2,"plot_number":2,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info3)
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    