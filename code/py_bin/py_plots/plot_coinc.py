# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plot_coinc.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella

File to plot the coincidence between the structures.
    - Functions:
        - plot_coinc : function to plot the coincidence
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

def plot_coinc(data_in={"file":"coinc.txt","folder":"Data","plot_folder":"plots","xlabel":"$y^+$",
                        "ylabel":"$\overline{U}$","ylabel2":"$\overline{U}$","fontsize":18,
                        "figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,"fig_name":"training_info",
                        "dpi":60,"struc1_lab":"uv","struc2_lab":"SHAP","linewidth":2}):
    """
    .....................................................................................................................
    # plot_coinc: Function to generate the plot of the coincidence between structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"file":"coinc.txt","folder":"Data","plot_folder":"plots","xlabel":"$y^+$",
                        "ylabel":"$\overline{U}$","fontsize":18,"figsize_x":10,"figsize_y":8,
                        "colormap":"viridis","colornum":2,"fig_name":"training_info","dpi":60,"linewidth":2}.
        Data:
            - file        : file of the training information
            - folder      : folder of the data generated during the training
            - plot_folder : folder to store the figures
            - fontsize    : font size used for the figure
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - ylabel2     : label of the y axis of a second plot
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colormap    : colormap used for the figure
            - colornum    : number of colors of the colormap, two curves are used. The number of levels of the 
                            colormap needs to be higher than 2 
            - fig_name    : name of the saved figure
            - dpi         : dots per inch of the saved figure
            - struc1_lab  : label of the structure 1
            - struc2_lab  : label of the structure 2
            - linewidth   : width of the line

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from py_bin.py_functions.calc_coinc import read_coinc
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    folder      = str(data_in["folder"])      # Folder to read the epoch data
    file        = str(data_in["file"])        # File to read for the epochs data
    plot_folder = str(data_in["plot_folder"]) # folder to save the plots
    xlabel      = str(data_in["xlabel"])      # label of the x axis
    ylabel      = str(data_in["ylabel"])      # label of the y axis
    ylabel2     = str(data_in["ylabel2"])      # label of the y axis
    fontsize    = int(data_in["fontsize"])    # size of the text in the plot
    figsize_x   = int(data_in["figsize_x"])   # size of the figure in direction x
    figsize_y   = int(data_in["figsize_y"])   # size of the figure in direction y
    colormap    = str(data_in["colormap"])    # colormap of the figure
    colornum    = int(data_in["colornum"])    # number of colors of the colormap
    fig_name    = str(data_in["fig_name"])    # name of the figure to be saved
    dpi         = float(data_in["dpi"])       # dots per inch to save the figure
    struc1_lab  = str(data_in["struc1_lab"])
    struc2_lab  = str(data_in["struc2_lab"])
    linewidth   = int(data_in["linewidth"])
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the file of coincidence
    # -------------------------------------------------------------------------------------------------------------------
    data_coinc = read_coinc(data_in={"folder":folder,"calc_coin_file":file})
    frac_struc1 = data_coinc["frac_struc1"]
    frac_struc2 = data_coinc["frac_struc2"]
    frac_coinc  = data_coinc["frac_coinc"]
    yplus       = data_coinc["yplus"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":1,"xmax":125,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":yplus,"data_y":frac_struc1,"label":struc1_lab,
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":yplus,"data_y":frac_struc2,"label":struc2_lab,
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info2) 
    plot_info3 = {"data_x":yplus,"data_y":frac_coinc,"label":struc1_lab+"+"+struc2_lab,
                  "color":None,"linewidth":linewidth,"plot_number":2,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info3)
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel2,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_name+"percentage","dpi":dpi,
                  "plot_folder":plot_folder,"xmin":1,"xmax":125,"ymin":0,"ymax":100,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":yplus,"data_y":frac_coinc/frac_struc1*100,
                  "label":"("+struc1_lab+"+"+struc2_lab+")/"+struc1_lab,
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":yplus,"data_y":frac_coinc/frac_struc2*100,
                  "label":"("+struc1_lab+"+"+struc2_lab+")/"+struc2_lab,
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info2) 
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    

def plot_coinc_type(data_in={"file":"coinc.txt","folder":"Data","plot_folder":"plots","xlabel":"$y^+$",
                             "ylabel":"$\overline{U}$","ylabel2":"$\overline{U}$","fontsize":18,
                             "figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,"fig_namea":"Q1",
                             "fig_nameb":"Q2","fig_namec":"Q3","fig_named":"Q4","dpi":60,"struc1_lab":"shap",
                             "struc2a_lab":"oi","struc2b_lab":"ejection","struc2c_lab":"ii","struc2d_lab":"sweeps",
                             "linewidth":2}):
    """
    .....................................................................................................................
    # plot_coinc_type: Function to generate the plot of the coincidence between structures of different type
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"file":"coinc.txt","folder":"Data","plot_folder":"plots","xlabel":"$y^+$",
                        "ylabel":"$\overline{U}$","ylabel2":"$\overline{U}$","fontsize":18,
                        "figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,"fig_name":"training_info",
                        "dpi":60,"struc1_lab":"shap","struc2a_lab":"oi","struc2b_lab":"ejection",
                        "struc2c_lab":"ii","struc2d_lab":"sweeps","linewidth":2}.
        Data:
            - file        : file of the training information
            - folder      : folder of the data generated during the training
            - plot_folder : folder to store the figures
            - fontsize    : font size used for the figure
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - ylabel2     : label of the y axis of a second plot
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colormap    : colormap used for the figure
            - colornum    : number of colors of the colormap, two curves are used. The number of levels of the 
                            colormap needs to be higher than 2 
            - fig_namea   : name of the saved figure a
            - fig_nameb   : name of the saved figure b
            - fig_namec   : name of the saved figure c
            - fig_named   : name of the saved figure d
            - dpi         : dots per inch of the saved figure
            - struc1_lab  : label of the structure 1
            - struc2a_lab : label of the structure 2 type a
            - struc2b_lab : label of the structure 2 type b
            - struc2c_lab : label of the structure 2 type c
            - struc2d_lab : label of the structure 2 type d
            - linewidth   : linewidth of the line

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from py_bin.py_functions.calc_coinc import read_coinc_type
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    folder      = str(data_in["folder"])      # Folder to read the epoch data
    file        = str(data_in["file"])        # File to read for the epochs data
    plot_folder = str(data_in["plot_folder"]) # folder to save the plots
    xlabel      = str(data_in["xlabel"])      # label of the x axis
    ylabel      = str(data_in["ylabel"])      # label of the y axis
    ylabel2     = str(data_in["ylabel2"])      # label of the y axis
    fontsize    = int(data_in["fontsize"])    # size of the text in the plot
    figsize_x   = int(data_in["figsize_x"])   # size of the figure in direction x
    figsize_y   = int(data_in["figsize_y"])   # size of the figure in direction y
    colormap    = str(data_in["colormap"])    # colormap of the figure
    colornum    = int(data_in["colornum"])    # number of colors of the colormap
    fig_namea   = str(data_in["fig_namea"])    # name of the figure to be saved
    fig_nameb   = str(data_in["fig_nameb"])    # name of the figure to be saved
    fig_namec   = str(data_in["fig_namec"])    # name of the figure to be saved
    fig_named   = str(data_in["fig_named"])    # name of the figure to be saved
    dpi         = float(data_in["dpi"])       # dots per inch to save the figure
    struc1_lab  = str(data_in["struc1_lab"])
    struc2a_lab = str(data_in["struc2a_lab"])
    struc2b_lab = str(data_in["struc2b_lab"])
    struc2c_lab = str(data_in["struc2c_lab"])
    struc2d_lab = str(data_in["struc2d_lab"])
    linewidth   = int(data_in["linewidth"])
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the file of coincidence
    # -------------------------------------------------------------------------------------------------------------------
    data_coinc   = read_coinc_type(data_in={"folder":folder,"calc_coin_file":file})
    frac_struc1  = data_coinc["frac_struc1"]
    frac_struc2a = data_coinc["frac_struc2a"]
    frac_coinc_a = data_coinc["frac_coinc_a"]
    frac_struc2b = data_coinc["frac_struc2b"]
    frac_coinc_b = data_coinc["frac_coinc_b"]
    frac_struc2c = data_coinc["frac_struc2c"]
    frac_coinc_c = data_coinc["frac_coinc_c"]
    frac_struc2d = data_coinc["frac_struc2d"]
    frac_coinc_d = data_coinc["frac_coinc_d"]
    yplus        = data_coinc["yplus"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot Q1
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_namea,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":1,"xmax":125,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":yplus,"data_y":frac_struc1,"label":struc1_lab,
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":yplus,"data_y":frac_struc2a,"label":struc2a_lab,
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info2) 
    plot_info3 = {"data_x":yplus,"data_y":frac_coinc_a,"label":struc1_lab+"+"+struc2a_lab,
                  "color":None,"linewidth":linewidth,"plot_number":2,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info3)
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot Q1 perc
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel2,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_namea+"percentage","dpi":dpi,
                  "plot_folder":plot_folder,"xmin":1,"xmax":125,"ymin":0,"ymax":100,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":yplus,"data_y":frac_coinc_a/frac_struc1*100,
                  "label":"("+struc1_lab+"+"+struc2a_lab+")/"+struc1_lab,
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":yplus,"data_y":frac_coinc_a/frac_struc2a*100,
                  "label":"("+struc1_lab+"+"+struc2a_lab+")/"+struc2a_lab,
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info2) 
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_nameb,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":1,"xmax":125,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":yplus,"data_y":frac_struc1,"label":struc1_lab,
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":yplus,"data_y":frac_struc2b,"label":struc2b_lab,
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info2) 
    plot_info3 = {"data_x":yplus,"data_y":frac_coinc_b,"label":struc1_lab+"+"+struc2b_lab,
                  "color":None,"linewidth":linewidth,"plot_number":2,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info3)
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot Q2 perc
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel2,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_nameb+"percentage","dpi":dpi,
                  "plot_folder":plot_folder,"xmin":1,"xmax":125,"ymin":0,"ymax":100,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":yplus,"data_y":frac_coinc_b/frac_struc1*100,
                  "label":"("+struc1_lab+"+"+struc2b_lab+")/"+struc1_lab,
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":yplus,"data_y":frac_coinc_b/frac_struc2b*100,
                  "label":"("+struc1_lab+"+"+struc2b_lab+")/"+struc2b_lab,
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info2) 
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_namec,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":1,"xmax":125,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":yplus,"data_y":frac_struc1,"label":struc1_lab,
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":yplus,"data_y":frac_struc2c,"label":struc2c_lab,
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info2) 
    plot_info3 = {"data_x":yplus,"data_y":frac_coinc_c,"label":struc1_lab+"+"+struc2c_lab,
                  "color":None,"linewidth":linewidth,"plot_number":2,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info3)
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot Q3 perc
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel2,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_namec+"percentage","dpi":dpi,
                  "plot_folder":plot_folder,"xmin":1,"xmax":125,"ymin":0,"ymax":100,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":yplus,"data_y":frac_coinc_c/frac_struc1*100,
                  "label":"("+struc1_lab+"+"+struc2c_lab+")/"+struc1_lab,
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":yplus,"data_y":frac_coinc_c/frac_struc2c*100,
                  "label":"("+struc1_lab+"+"+struc2c_lab+")/"+struc2c_lab,
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info2) 
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_named,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":1,"xmax":125,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":yplus,"data_y":frac_struc1,"label":struc1_lab,
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":yplus,"data_y":frac_struc2d,"label":struc2d_lab,
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info2) 
    plot_info3 = {"data_x":yplus,"data_y":frac_coinc_d,"label":struc1_lab+"+"+struc2d_lab,
                  "color":None,"linewidth":linewidth,"plot_number":2,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info3)
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot Q4 perc
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel2,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_named+"percentage","dpi":dpi,
                  "plot_folder":plot_folder,"xmin":1,"xmax":125,"ymin":0,"ymax":100,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":yplus,"data_y":frac_coinc_d/frac_struc1*100,
                  "label":"("+struc1_lab+"+"+struc2d_lab+")/"+struc1_lab,
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":yplus,"data_y":frac_coinc_d/frac_struc2d*100,
                  "label":"("+struc1_lab+"+"+struc2d_lab+")/"+struc2d_lab,
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info2) 
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    
    
def plot_coinc_4struc(data_in={"file":"coinc.txt","folder":"Data","plot_folder":"plots","xlabel":"$y^+$",
                               "ylabel":"$\overline{U}$","ylabel2":"$\overline{U}$","fontsize":18,
                               "figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                               "fig_name":"training_info","dpi":60,"struc1_lab":"uv","struc2_lab":"SHAP",
                               "struc3_lab":"SHAP","struc4_lab":"SHAP","linewidth":2}):
    """
    .....................................................................................................................
    # plot_coinc_4struc: Function to generate the plot of the coincidence between 4 structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"file":"coinc.txt","folder":"Data","plot_folder":"plots","xlabel":"$y^+$",
                        "ylabel":"$\overline{U}$","ylabel2":"$\overline{U}$","fontsize":18,
                        "figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                        "fig_name":"training_info","dpi":60,"struc1_lab":"uv","struc2_lab":"SHAP",
                        "struc3_lab":"SHAP","struc4_lab":"SHAP","linewidth":2}.
        Data:
            - file        : file of the training information
            - folder      : folder of the data generated during the training
            - plot_folder : folder to store the figures
            - fontsize    : font size used for the figure
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - ylabel2     : label of the y axis of a second plot
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colormap    : colormap used for the figure
            - colornum    : number of colors of the colormap, two curves are used. The number of levels of the 
                            colormap needs to be higher than 2 
            - fig_name    : name of the saved figure
            - dpi         : dots per inch of the saved figure
            - struc1_lab  : label of the structure 1
            - struc2_lab  : label of the structure 2
            - struc3_lab  : label of the structure 3
            - struc4_lab  : label of the structure 4
            - linewidth   : width of the line

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from py_bin.py_functions.calc_coinc import read_coinc_4struc
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    folder      = str(data_in["folder"])      # Folder to read the epoch data
    file        = str(data_in["file"])        # File to read for the epochs data
    plot_folder = str(data_in["plot_folder"]) # folder to save the plots
    xlabel      = str(data_in["xlabel"])      # label of the x axis
    ylabel      = str(data_in["ylabel"])      # label of the y axis
    ylabel2     = str(data_in["ylabel2"])      # label of the y axis
    fontsize    = int(data_in["fontsize"])    # size of the text in the plot
    figsize_x   = int(data_in["figsize_x"])   # size of the figure in direction x
    figsize_y   = int(data_in["figsize_y"])   # size of the figure in direction y
    colormap    = str(data_in["colormap"])    # colormap of the figure
    colornum    = int(data_in["colornum"])    # number of colors of the colormap
    fig_name    = str(data_in["fig_name"])    # name of the figure to be saved
    dpi         = float(data_in["dpi"])       # dots per inch to save the figure
    struc1_lab  = str(data_in["struc1_lab"])
    struc2_lab  = str(data_in["struc2_lab"])
    struc3_lab  = str(data_in["struc3_lab"])
    struc4_lab  = str(data_in["struc4_lab"])
    linewidth   = int(data_in["linewidth"])
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the file of coincidence
    # -------------------------------------------------------------------------------------------------------------------
    data_coinc = read_coinc_4struc(data_in={"folder":folder,"calc_coin_file":file})
    frac_struc1 = data_coinc["frac_struc1"]
    frac_struc2 = data_coinc["frac_struc2"]
    frac_struc3 = data_coinc["frac_struc3"]
    frac_struc4 = data_coinc["frac_struc4"]
    frac_coinc  = data_coinc["frac_coinc"]
    yplus       = data_coinc["yplus"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":1,"xmax":125,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":yplus,"data_y":frac_struc1,"label":struc1_lab,
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":yplus,"data_y":frac_struc2,"label":struc2_lab,
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info2) 
    plot_info3 = {"data_x":yplus,"data_y":frac_struc3,"label":struc3_lab,
                  "color":None,"linewidth":linewidth,"plot_number":2,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info3) 
    plot_info4 = {"data_x":yplus,"data_y":frac_struc4,"label":struc4_lab,
                  "color":None,"linewidth":linewidth,"plot_number":3,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info4) 
    plot_info5 = {"data_x":yplus,"data_y":frac_coinc,"label":struc1_lab+"+"+struc2_lab+"+"+struc3_lab+"+"+struc4_lab,
                  "color":None,"linewidth":linewidth,"plot_number":4,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info5)
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel2,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_name+"percentage","dpi":dpi,
                  "plot_folder":plot_folder,"xmin":1,"xmax":125,"ymin":0,"ymax":100,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":yplus,"data_y":frac_coinc/frac_struc1*100,
                  "label":"("+struc1_lab+"+"+struc2_lab+"+"+struc3_lab+"+"+struc4_lab+")/"+struc1_lab,
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":yplus,"data_y":frac_coinc/frac_struc2*100,
                  "label":"("+struc1_lab+"+"+struc2_lab+"+"+struc3_lab+"+"+struc4_lab+")/"+struc2_lab,
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info2) 
    plot_info3 = {"data_x":yplus,"data_y":frac_coinc/frac_struc3*100,
                  "label":"("+struc1_lab+"+"+struc2_lab+"+"+struc3_lab+"+"+struc4_lab+")/"+struc3_lab,
                  "color":None,"linewidth":linewidth,"plot_number":2,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info3) 
    plot_info4 = {"data_x":yplus,"data_y":frac_coinc/frac_struc4*100,
                  "label":"("+struc1_lab+"+"+struc2_lab+"+"+struc3_lab+"+"+struc4_lab+")/"+struc4_lab,
                  "color":None,"linewidth":linewidth,"plot_number":3,"style":"-"}  
    plot_train.add_plot_2d(data_in=plot_info4) 
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    
def plot_coinc_3coinc(data_in={"file_1":"coinc.txt","file_2":"coinc.txt","file_3":"coinc.txt","folder":"Data",
                               "plot_folder":"plots","xlabel":"$y^+$","ylabel":"$\overline{U}$",
                               "ylabel2":"$\overline{U}$","fontsize":18,"figsize_x":10,"figsize_y":8,
                               "colormap":"viridis","colornum":2,"fig_name":"training_info","dpi":60,
                               "struc1a_lab":"uv","struc1b_lab":"SHAP","struc2a_lab":"uv","struc2b_lab":"SHAP",
                               "struc3a_lab":"uv","struc3b_lab":"SHAP","linewidth":2}):
    """
    .....................................................................................................................
    # plot_coinc_3coinc: Function to generate the plot of three coincidence plots between structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"file_1":"coinc.txt","file_2":"coinc.txt","file_3":"coinc.txt","folder":"Data",
                        "plot_folder":"plots","xlabel":"$y^+$","ylabel":"$\overline{U}$",
                        "ylabel2":"$\overline{U}$","fontsize":18,"figsize_x":10,"figsize_y":8,
                        "colormap":"viridis","colornum":2,"fig_name":"training_info","dpi":60,
                        "struc1a_lab":"uv","struc1b_lab":"SHAP","struc2a_lab":"uv","struc2b_lab":"SHAP",
                        "struc3a_lab":"uv","struc3b_lab":"SHAP","linewidth":2}.
        Data:
            - file_1      : file of the training information of the coincidence 1
            - file_2      : file of the training information of the coincidence 2
            - file_3      : file of the training information of the coincidence 3
            - folder      : folder of the data generated during the training
            - plot_folder : folder to store the figures
            - fontsize    : font size used for the figure
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - ylabel2     : label of the y axis of a second plot
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colormap    : colormap used for the figure
            - colornum    : number of colors of the colormap, two curves are used. The number of levels of the 
                            colormap needs to be higher than 2 
            - fig_name    : name of the saved figure
            - dpi         : dots per inch of the saved figure
            - struc1a_lab : label of the structure 1a
            - struc1b_lab : label of the structure 1b
            - struc2a_lab : label of the structure 2a
            - struc2b_lab : label of the structure 2b
            - struc3a_lab : label of the structure 3a
            - struc3b_lab : label of the structure 3b
            - linewidth   : width of the line

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from py_bin.py_functions.calc_coinc import read_coinc
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    folder      = str(data_in["folder"])      # Folder to read the epoch data
    file_1      = str(data_in["file_1"])      # File to read for the epochs data
    file_2      = str(data_in["file_2"])      # File to read for the epochs data
    file_3      = str(data_in["file_3"])      # File to read for the epochs data
    plot_folder = str(data_in["plot_folder"]) # folder to save the plots
    xlabel      = str(data_in["xlabel"])      # label of the x axis
    ylabel      = str(data_in["ylabel"])      # label of the y axis
    ylabel2     = str(data_in["ylabel2"])     # label of the y axis
    fontsize    = int(data_in["fontsize"])    # size of the text in the plot
    figsize_x   = int(data_in["figsize_x"])   # size of the figure in direction x
    figsize_y   = int(data_in["figsize_y"])   # size of the figure in direction y
    colormap    = str(data_in["colormap"])    # colormap of the figure
    colornum    = int(data_in["colornum"])    # number of colors of the colormap
    fig_name    = str(data_in["fig_name"])    # name of the figure to be saved
    dpi         = float(data_in["dpi"])       # dots per inch to save the figure
    struc1a_lab = str(data_in["struc1a_lab"])
    struc1b_lab = str(data_in["struc1b_lab"])
    struc2a_lab = str(data_in["struc2a_lab"])
    struc2b_lab = str(data_in["struc2b_lab"])
    struc3a_lab = str(data_in["struc3a_lab"])
    struc3b_lab = str(data_in["struc3b_lab"])
    linewidth   = int(data_in["linewidth"])
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the file of coincidence
    # -------------------------------------------------------------------------------------------------------------------
    data_coinc_1 = read_coinc(data_in={"folder":folder,"calc_coin_file":file_1})
    frac_struc1a = data_coinc_1["frac_struc1"]
    frac_struc1b = data_coinc_1["frac_struc2"]
    frac_coinc_1 = data_coinc_1["frac_coinc"]
    yplus_1      = data_coinc_1["yplus"]
    data_coinc_2 = read_coinc(data_in={"folder":folder,"calc_coin_file":file_2})
    frac_struc2a = data_coinc_2["frac_struc1"]
    frac_struc2b = data_coinc_2["frac_struc2"]
    frac_coinc_2 = data_coinc_2["frac_coinc"]
    yplus_2      = data_coinc_2["yplus"]
    data_coinc_3 = read_coinc(data_in={"folder":folder,"calc_coin_file":file_3})
    frac_struc3a = data_coinc_3["frac_struc1"]
    frac_struc3b = data_coinc_3["frac_struc2"]
    frac_coinc_3 = data_coinc_3["frac_coinc"]
    yplus_3      = data_coinc_3["yplus"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel2,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"log","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_name+"percentage","dpi":dpi,
                  "plot_folder":plot_folder,"xmin":1,"xmax":125,"ymin":0,"ymax":100,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":yplus_1,"data_y":frac_coinc_1/frac_struc1a*100,
                  "label":"("+struc1a_lab+"+"+struc1b_lab+")/"+struc1a_lab,
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":yplus_1,"data_y":frac_coinc_1/frac_struc1b*100,
                  "label":"("+struc1a_lab+"+"+struc1b_lab+")/"+struc1b_lab,
                  "color":None,"linewidth":linewidth,"plot_number":0,"style":"--"}  
    plot_train.add_plot_2d(data_in=plot_info2) 
    plot_info3 = {"data_x":yplus_2,"data_y":frac_coinc_2/frac_struc2a*100,
                  "label":"("+struc2a_lab+"+"+struc2b_lab+")/"+struc2a_lab,
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info3)
    plot_info4 = {"data_x":yplus_2,"data_y":frac_coinc_2/frac_struc2b*100,
                  "label":"("+struc2a_lab+"+"+struc2b_lab+")/"+struc2b_lab,
                  "color":None,"linewidth":linewidth,"plot_number":1,"style":"--"}  
    plot_train.add_plot_2d(data_in=plot_info4)
    plot_info5 = {"data_x":yplus_3,"data_y":frac_coinc_3/frac_struc3a*100,
                  "label":"("+struc3a_lab+"+"+struc3b_lab+")/"+struc3a_lab,
                  "color":None,"linewidth":linewidth,"plot_number":2,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info5)
    plot_info6 = {"data_x":yplus_3,"data_y":frac_coinc_3/frac_struc3b*100,
                  "label":"("+struc3a_lab+"+"+struc3b_lab+")/"+struc3b_lab,
                  "color":None,"linewidth":linewidth,"plot_number":2,"style":"--"}  
    plot_train.add_plot_2d(data_in=plot_info6)
    plot_train.plot_layout()
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    