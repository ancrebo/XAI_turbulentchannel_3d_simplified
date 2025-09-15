# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plottrain.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella

File to plot training information. The file contains the following functions:
    - Functions:
        - plottrain : function to plot the training evolution
"""
# -----------------------------------------------------------------------------------------------------------------------
# Import packages for all the functions
# -----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define functions
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

def plottrain(plot_format_data={"file":"hist.txt","folder":"Data","plot_folder":"plots","xlabel":"Epoch",
                                "ylabel":"Loss function (-)","fontsize":18,"figsize_x":10,"figsize_y":8,
                                "colormap":"viridis","colornum":2,"fig_name":"training_info","dpi":60}):
    """
    .....................................................................................................................
    # plottrain: Function to generate the plot of the loss function during the training
    .....................................................................................................................
    Parameters
    ----------
    plot_format_data : dict, optional
        Data required for generating the plot. 
        The default is {"file":"hist.txt","folder":"Data","plot_folder":"plots","xlabel":"Epoch",
                        "ylabel":"Loss function (-)","fontsize":18,"figsize_x":10,
                        "figsize_y":8,"xscale":"linear","yscale":"log","colormap":"viridis","colornum":2,
                        "fig_name":"training_info","dpi":60}.
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

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    folder      = str(plot_format_data["folder"])      # Folder to read the epoch data
    file        = str(plot_format_data["file"])        # File to read for the epochs data
    xlabel      = str(plot_format_data["xlabel"])      # label of the x axis
    ylabel      = str(plot_format_data["ylabel"])      # label of the y axis
    plot_folder = str(plot_format_data["plot_folder"]) # folder to save the plots
    fontsize    = int(plot_format_data["fontsize"])    # size of the text in the plot
    figsize_x   = int(plot_format_data["figsize_x"])   # size of the figure in direction x
    figsize_y   = int(plot_format_data["figsize_y"])   # size of the figure in direction y
    colormap    = str(plot_format_data["colormap"])    # colormap of the figure
    colornum    = int(plot_format_data["colornum"])    # number of colors of the colormap
    fig_name    = str(plot_format_data["fig_name"])    # name of the figure to be saved
    dpi         = float(plot_format_data["dpi"])       # dots per inch to save the figure
    file_com    = folder+'/'+file
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the file information
    # -------------------------------------------------------------------------------------------------------------------
    with open(file_com, 'r') as fread:
        data_train = np.array([[float(ii) for ii in line.split(',')] for line in fread])
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"log","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":np.arange(len(data_train[:,0])),"data_y":data_train[:,1],"label":"Training Loss",
                  "color":None,"linewidth":2,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":np.arange(len(data_train[:,0])),"data_y":data_train[:,2],"label":"Validation Loss",
                  "color":None,"linewidth":2,"plot_number":1,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info2)
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    