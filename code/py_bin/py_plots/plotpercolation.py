# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plotpercolation.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella

File to plot training information:
    - Functions:
        - plotpercolation : function to plot the percolation
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

def plotpercolation(data_in={"file":"hist.txt","folder":"Data","plot_folder":"plots","xlabel":"$H$",
                             "ylabel":"$N/N_{max}$","ylabel2":"$V_{large}/V_{tot}$","fontsize":18,
                             "figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                             "fig_name":"training_info","dpi":60}):
    """
    .....................................................................................................................
    # plotpercolation: Function to generate the plot of the percolation
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"file":"Umean.txt","folder":"Data","plot_folder":"plots","xlabel":"$H$",
                        "ylabel":"$N/N_{max}$","ylabel2":"$V_{large}/V_{channel}$","fontsize":18,"figsize_x":10,
                        "figsize_y":8,"xscale":"linear","yscale":"log","colormap":"viridis","colornum":2,
                        "fig_name":"training_info","dpi":60,"dy":1,"L_x":2*np.pi,"L_y":1,"L_z":np.pi}.
        Data:
            - file : file of the training information
            - folder : folder of the data generated during the training
            - plot_folder : folder to store the figures
            - fontsize    : font size used for the figure
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - ylabel2     : second label of the y axis
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
    from py_bin.py_functions.percolation import read_percolation
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    folder      = str(data_in["folder"])      # Folder to read the epoch data
    file        = str(data_in["file"])        # File to read for the epochs data
    plot_folder = str(data_in["plot_folder"]) # folder to save the plots
    xlabel      = str(data_in["xlabel"])      # label of the x axis
    ylabel      = str(data_in["ylabel"])      # label of the y axis
    ylabel2     = str(data_in["ylabel2"])     # second label of the y axis
    fontsize    = int(data_in["fontsize"])    # size of the text in the plot
    figsize_x   = int(data_in["figsize_x"])   # size of the figure in direction x
    figsize_y   = int(data_in["figsize_y"])   # size of the figure in direction y
    colormap    = str(data_in["colormap"])    # colormap of the figure
    colornum    = int(data_in["colornum"])    # number of colors of the colormap
    fig_name    = str(data_in["fig_name"])    # name of the figure to be saved
    dpi         = float(data_in["dpi"])       # dots per inch to save the figure
    linewidth   = 5
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the file mean velocity
    # -------------------------------------------------------------------------------------------------------------------
    data_perc  = read_percolation(data_in={"perc_file":file,"folder":folder})
    H_perc     = data_perc["H_perc"]
    nstruc     = data_perc["nstruc"]
    Vstruc     = data_perc["Vstruc"]
    maxperc    = H_perc[np.argmax(nstruc)]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel+', '+ylabel2,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":False,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info  = {"data_x":H_perc,"data_y":nstruc,"label":ylabel,\
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info)
    plot_info  = {"data_x":H_perc,"data_y":Vstruc,"label":ylabel2,\
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info)
    plot_train.plot_layout()
    plot_info  = {"data_x":[maxperc,maxperc],"data_y":[0,1],"label":ylabel2,\
                  "color":'k',"linewidth":linewidth,"plot_number":1,"style":"-."}
    plot_train.add_plot_2d(data_in=plot_info)
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    
def plotpercolation_uvw(data_in={"file":"hist.txt","folder":"Data","plot_folder":"plots","xlabel":"$H$",
                                 "ylabel":"$N/N_{max}$","ylabel2":"$V_{large}/V_{tot}$","fontsize":18,
                                 "figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                                 "fig_name":"training_info","dpi":60}):
    """
    .....................................................................................................................
    # plotpercolation_uvw: Function to generate the plot of the uvw percolation
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"file":"Umean.txt","folder":"Data","plot_folder":"plots","xlabel":"$H$",
                        "ylabel":"$N/N_{max}$","ylabel2":"$V_{large}/V_{channel}$","fontsize":18,"figsize_x":10,
                        "figsize_y":8,"xscale":"linear","yscale":"log","colormap":"viridis","colornum":2,
                        "fig_name":"training_info","dpi":60,"dy":1,"L_x":2*np.pi,"L_y":1,"L_z":np.pi}.
        Data:
            - file : file of the training information
            - folder : folder of the data generated during the training
            - plot_folder : folder to store the figures
            - fontsize    : font size used for the figure
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - ylabel2     : second label of the y axis
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
    from py_bin.py_functions.percolation import read_percolation_uvw
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    folder      = str(data_in["folder"])      # Folder to read the epoch data
    file        = str(data_in["file"])        # File to read for the epochs data
    plot_folder = str(data_in["plot_folder"]) # folder to save the plots
    xlabel      = str(data_in["xlabel"])      # label of the x axis
    ylabel      = str(data_in["ylabel"])      # label of the y axis
    ylabel2     = str(data_in["ylabel2"])     # second label of the y axis
    fontsize    = int(data_in["fontsize"])    # size of the text in the plot
    figsize_x   = int(data_in["figsize_x"])   # size of the figure in direction x
    figsize_y   = int(data_in["figsize_y"])   # size of the figure in direction y
    colormap    = str(data_in["colormap"])    # colormap of the figure
    colornum    = int(data_in["colornum"])    # number of colors of the colormap
    fig_name    = str(data_in["fig_name"])    # name of the figure to be saved
    dpi         = float(data_in["dpi"])       # dots per inch to save the figure
    linewidth   = 2
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the file mean velocity
    # -------------------------------------------------------------------------------------------------------------------
    data_perc  = read_percolation_uvw(data_in={"perc_file":file,"folder":folder})
    H_perc     = data_perc["H_perc"]
    nstruc_u   = data_perc["nstruc_u"]
    Vstruc_u   = data_perc["Vstruc_u"]
    nstruc_v   = data_perc["nstruc_v"]
    Vstruc_v   = data_perc["Vstruc_v"]
    nstruc_w   = data_perc["nstruc_w"]
    Vstruc_w   = data_perc["Vstruc_w"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel+', '+ylabel2,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":False,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info  = {"data_x":H_perc,"data_y":nstruc_u,"label":"$($"+ylabel+"$)_u$",\
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info)
    plot_info  = {"data_x":H_perc,"data_y":nstruc_v,"label":"$($"+ylabel+"$)_v$",\
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info)
    plot_info  = {"data_x":H_perc,"data_y":nstruc_w,"label":"$($"+ylabel+"$)_w$",\
                  "color":None,"linewidth":linewidth,"plot_number":2,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info)
    plot_info  = {"data_x":H_perc,"data_y":Vstruc_u,"label":"$($"+ylabel2+"$)_u$",\
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"--"}
    plot_train.add_plot_2d(data_in=plot_info)
    plot_info  = {"data_x":H_perc,"data_y":Vstruc_v,"label":"$($"+ylabel2+"$)_v$",\
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"--"}
    plot_train.add_plot_2d(data_in=plot_info)
    plot_info  = {"data_x":H_perc,"data_y":Vstruc_w,"label":"$($"+ylabel2+"$)_w$",\
                  "color":None,"linewidth":linewidth,"plot_number":2,"style":"--"}
    plot_train.add_plot_2d(data_in=plot_info)
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
        
def plotpercolation_uw_vsign(data_in={"file":"hist.txt","folder":"Data","plot_folder":"plots","xlabel":"$H$",
                                      "ylabel":"$N/N_{max}$","ylabel2":"$V_{large}/V_{tot}$","fontsize":18,
                                      "figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                                      "fig_name":"training_info","dpi":60}):
    """
    .....................................................................................................................
    # plotpercolation_uw_vsign: Function to generate the plot of the uvw percolation
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"file":"Umean.txt","folder":"Data","plot_folder":"plots","xlabel":"$H$",
                        "ylabel":"$N/N_{max}$","ylabel2":"$V_{large}/V_{channel}$","fontsize":18,"figsize_x":10,
                        "figsize_y":8,"xscale":"linear","yscale":"log","colormap":"viridis","colornum":2,
                        "fig_name":"training_info","dpi":60,"dy":1,"L_x":2*np.pi,"L_y":1,"L_z":np.pi}.
        Data:
            - file : file of the training information
            - folder : folder of the data generated during the training
            - plot_folder : folder to store the figures
            - fontsize    : font size used for the figure
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - ylabel2     : second label of the y axis
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
    from py_bin.py_functions.percolation import read_percolation_uw_vsign
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    folder      = str(data_in["folder"])      # Folder to read the epoch data
    file        = str(data_in["file"])        # File to read for the epochs data
    plot_folder = str(data_in["plot_folder"]) # folder to save the plots
    xlabel      = str(data_in["xlabel"])      # label of the x axis
    ylabel      = str(data_in["ylabel"])      # label of the y axis
    ylabel2     = str(data_in["ylabel2"])     # second label of the y axis
    fontsize    = int(data_in["fontsize"])    # size of the text in the plot
    figsize_x   = int(data_in["figsize_x"])   # size of the figure in direction x
    figsize_y   = int(data_in["figsize_y"])   # size of the figure in direction y
    colormap    = str(data_in["colormap"])    # colormap of the figure
    colornum    = int(data_in["colornum"])    # number of colors of the colormap
    fig_name    = str(data_in["fig_name"])    # name of the figure to be saved
    dpi         = float(data_in["dpi"])       # dots per inch to save the figure
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the file mean velocity
    # -------------------------------------------------------------------------------------------------------------------
    data_perc  = read_percolation_uw_vsign(data_in={"perc_file":file,"folder":folder})
    H_perc     = data_perc["H_perc"]
    nstruc_1   = data_perc["nstruc_1"]
    Vstruc_1   = data_perc["Vstruc_1"]
    nstruc_2   = data_perc["nstruc_2"]
    Vstruc_2   = data_perc["Vstruc_2"]
    linewidth  = 2
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel+', '+ylabel2,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":False,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info  = {"data_x":H_perc,"data_y":nstruc_1,"label":"$($"+ylabel+"$)_1$",\
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info)
    plot_info  = {"data_x":H_perc,"data_y":nstruc_2,"label":"$($"+ylabel+"$)_2$",\
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info)
    plot_info  = {"data_x":H_perc,"data_y":Vstruc_1,"label":"$($"+ylabel2+"$)_1$",\
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"--"}
    plot_train.add_plot_2d(data_in=plot_info)
    plot_info  = {"data_x":H_perc,"data_y":Vstruc_2,"label":"$($"+ylabel2+"$)_2$",\
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"--"}
    plot_train.add_plot_2d(data_in=plot_info)
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    