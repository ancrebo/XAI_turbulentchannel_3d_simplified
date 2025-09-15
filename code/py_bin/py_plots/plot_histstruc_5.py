# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plot_histuvw_y.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella

File to plot the velocity pdf in a type of structure. The file contains the following functions:
    - Functions:
        - plot_histuvw_y : function to plot the pdf of the velocities in the structures
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

    
def plot_histstruc_5_lowmem(data_in={"plot_folder":"plots","plot_file":"file",
                                     "ylabel":"-","xlabel":"-","fontsize":18,"figsize_x":10,
                                     "figsize_y":8,"colormap":"viridis","colornum":2,"dpi":60,"grid_1":[],
                                     "grid_2":[],"grid_3":[],"grid_4":[],"grid_5":[],"grid_x":[],"grid_y":[],
                                     "lev_min":1e-2,"lev_delta":None,"linewidth":2,"xmin":None,
                                     "xmax":None,"ymin":None,"ymax":None}):
    """""
    .....................................................................................................................
    # plot_histstruc_5_lowmem: Function to plot the pdf of the structures 1 to 5 for low memory consumption
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"plot_folder":"plots","plot_file":"file",
                        "ylabel":"-","xlabel":"-","fontsize":18,"figsize_x":10,
                        "figsize_y":8,"colormap1":"viridis","colormap2":"viridis","colormap3":"viridis",
                        "colormap4":"viridis","colormap5":"viridis","colornum":2,"dpi":60,"grid_1":[],
                        "grid_2":[],"grid_3":[],"grid_4":[],"grid_5":[],"grid_x":[],"grid_y":[],
                        "lev_min":1e-2,"linewidth":2,"xmin":None,
                        "xmax":None,"ymin":None,"ymax":None}.
        Data:
            - plot_folder      : folder to store the plots
            - plot_file        : file to save the pdf 
            - ylabel           : label of the y axis
            - xlabel           : label of the x axis 
            - fontsize         : font size used for the figure
            - figsize_x        : size of the figure in x
            - figsize_y        : size of the figure in y
            - colormap1        : colormap 1 used for the figure
            - colormap2        : colormap 2 used for the figure
            - colormap3        : colormap 3 used for the figure
            - colormap4        : colormap 4 used for the figure
            - colormap5        : colormap 5 used for the figure
            - colornum         : number of colors of the colormap, two curves are used. The number of levels of the 
                                 colormap needs to be higher than 2 
            - dpi              : dots per inch of the saved figure
            - grid_1           : grid of the repetitions of structure 1
            - grid_2           : grid of the repetitions of structure 2
            - grid_3           : grid of the repetitions of structure 3
            - grid_4           : grid of the repetitions of structure 4
            - grid_5           : grid of the repetitions of structure 5
            - grid_x           : grid of the repetitions in x
            - grid_y           : grid of the repetitions in y
            - lev_min          : minimum value of the levels of the pdf
            - linewidth        : width of the line
            - xmin             : minimum value of the x in the histogram
            - xmax             : maximum value of the x in the histogram
            - ymin             : minimum value of the y in the histogram
            - ymax             : maximum value of the y in the histogram

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from matplotlib import ticker
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    plot_folder           = str(data_in["plot_folder"])
    plot_file             = str(data_in["plot_file"])
    ylabel                = str(data_in["ylabel"])
    xlabel                = str(data_in["xlabel"])
    fontsize              = int(data_in["fontsize"])
    figsize_x             = int(data_in["figsize_x"])
    figsize_y             = int(data_in["figsize_y"])
    colormap1             = str(data_in["colormap1"])
    colormap2             = str(data_in["colormap2"])
    colormap3             = str(data_in["colormap3"])
    colormap4             = str(data_in["colormap4"])
    colormap5             = str(data_in["colormap5"])
    colornum              = int(data_in["colornum"])
    dpi                   = float(data_in["dpi"])
    grid_1                = np.array(data_in["grid_1"],dtype="float")
    grid_2                = np.array(data_in["grid_2"],dtype="float")
    grid_3                = np.array(data_in["grid_3"],dtype="float")
    grid_4                = np.array(data_in["grid_4"],dtype="float")
    grid_5                = np.array(data_in["grid_5"],dtype="float")
    grid_x                = np.array(data_in["grid_x"],dtype="float")
    grid_y                = np.array(data_in["grid_y"],dtype="float")
    lev_min               = float(data_in["lev_min"])
    linewidth             = int(data_in["linewidth"])
    xmin_in               = data_in["xmin"]
    xmax_in               = data_in["xmax"]
    ymin_in               = data_in["ymin"]
    ymax_in               = data_in["ymax"]
    
      

    # ----------------------------------------------------------------------------------------------------------------
    # Calculate the pdf for 1
    # ----------------------------------------------------------------------------------------------------------------
    if len(grid_1) > 0:
        grid_1               /= np.max(grid_1)
        content1              = grid_1[np.where(grid_1>=lev_min)]
        if xmin_in is None:
            xmin              = np.min(content1)
        else:
            xmin              = float(xmin_in)
        if xmax_in is None:
            xmax              = np.max(content1) 
        else:
            xmax              = float(xmax_in)
        if ymin_in is None:
            ymin              = np.min(content1)
        else:
            ymin              = float(ymin_in)
        if ymax_in is None:
            ymax              = np.max(content1) 
        else:
            ymax              = float(ymax_in)
    if len(grid_2) > 0:
        grid_2               /= np.max(grid_2)
        content2              = grid_2[np.where(grid_2>=lev_min)]
    if len(grid_3) > 0:
        grid_3               /= np.max(grid_3)
        content3              = grid_3[np.where(grid_3>=lev_min)]
    if len(grid_4) > 0:
        grid_4               /= np.max(grid_4)
        content4              = grid_4[np.where(grid_4>=lev_min)]
    if len(grid_5) > 0:
        grid_5               /= np.max(grid_5)
        content5              = grid_5[np.where(grid_5>=lev_min)]
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot for u
    # -------------------------------------------------------------------------------------------------------------------
    titlefig   = " "
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap1,
                  "colornum":colornum,"legend":True,"fig_name":plot_file,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure()
    try:
        if len(grid_1) > 0:
            plot_pred.add_hist2d_cont(data_in={"xx":grid_x,"yy":grid_y,"xxyy":grid_1,"levels":lev_min,
                                               "colormap":colormap1,"alp":0.65,"linewidth":linewidth,
                                               "cmap_flag":False})
        if len(grid_2) > 0:
            plot_pred.add_hist2d_cont(data_in={"xx":grid_x,"yy":grid_y,"xxyy":grid_2,"levels":lev_min,
                                               "colormap":colormap2,"alp":0.65,"linewidth":linewidth,
                                               "cmap_flag":False})
        if len(grid_3) > 0:
            plot_pred.add_hist2d_cont(data_in={"xx":grid_x,"yy":grid_y,"xxyy":grid_3,"levels":lev_min,
                                               "colormap":colormap3,"alp":0.65,"linewidth":linewidth,
                                               "cmap_flag":False})
        if len(grid_4) > 0:
            plot_pred.add_hist2d_cont(data_in={"xx":grid_x,"yy":grid_y,"xxyy":grid_4,"levels":lev_min,
                                               "colormap":colormap4,"alp":0.65,"linewidth":linewidth,
                                               "cmap_flag":False})
        if len(grid_5) > 0:
            plot_pred.add_hist2d_cont(data_in={"xx":grid_x,"yy":grid_y,"xxyy":grid_5,"levels":lev_min,
                                               "colormap":colormap5,"alp":0.65,"linewidth":linewidth,
                                               "cmap_flag":False})
        plot_pred.plot_layout_pcolor(data_in={"title":None,"colorbar":None,"b_text":None,
                                              "colorticks":[],"colorlabels":[],"equal":False,
                                              "xticks":None,"yticks":None,"xticklabels":None,"yticklabels":None})
    except:
        print("Error in contours",flush=True)
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    plot_pred.close()
   