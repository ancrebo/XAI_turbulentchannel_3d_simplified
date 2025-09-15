# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
ploturms.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella

File to plot training information
"""

# -----------------------------------------------------------------------------------------------------------------------
# Import packages for all functions
# -----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define functions
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

def plotSHAPrms(data_in={"file":"Urms.txt","folder":"Data","plot_folder":"plots","xlabel":"$y^+$",
                         "ylabel":"$u'^+$","ylabel2":"$|u_iu_j|^+$","fontsize":18,"figsize_x":10,"figsize_y":8,
                         "colormap":"viridis","colornum":2,"fig_name":"urms","fig_name2":"SHAP_uvabs","dpi":60,
                         "dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                         "SHAP_uSHAP_vw_file":"P125_21pi_vu.$INDEX$.h5.SHAP_uSHAP_vw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                         "rey":125,"utau":0.060523258443963}):
    """
    .....................................................................................................................
    # plotSHAPrms: Function to generate the plot of the rms of the SHAP
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"file":"Urms.txt","folder":"Data","plot_folder":"plots","xlabel":"$y^+$",
                        "ylabel":"$u'^+$","ylabel2":"$|u_iu_j|^+$","fontsize":18,"figsize_x":10,"figsize_y":8,
                        "colormap":"viridis","colornum":2,"fig_name":"urms","fig_name2":"SHAP_uvabs","dpi":60,
                        "dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                        "SHAP_uSHAP_vw_file":"P125_21pi_vu.$INDEX$.h5.SHAP_uSHAP_vw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                        "rey":125,"utau":0.060523258443963}.
        Data:
            - file        : file of the training information
            - folder      : folder of the data generated during the training
            - plot_folder : folder to store the figures
            - fontsize    : font size used for the figure
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - ylabel2     : label of the y axis of the second figure
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colormap    : colormap used for the figure
            - colornum    : number of colors of the colormap, two curves are used. The number of levels of the 
                            colormap needs to be higher than 2 
            - fig_name    : name of the saved figure
            - fig_name2   : name of the saved figure for the second plot
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
    from py_bin.py_functions.shaprms import read_rms
    from py_bin.py_class.flow_field import flow_field
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    folder      = str(data_in["folder"])      # Folder to read the epoch data
    file        = str(data_in["file"])        # File to read for the epochs data
    plot_folder = str(data_in["plot_folder"]) # folder to save the plots
    xlabel      = str(data_in["xlabel"])      # label of the x axis
    ylabel      = str(data_in["ylabel"])      # label of the y axis
    ylabel2     = str(data_in["ylabel2"])     # label of the y axis of the second plot
    fontsize    = int(data_in["fontsize"])    # size of the text in the plot
    figsize_x   = int(data_in["figsize_x"])   # size of the figure in direction x
    figsize_y   = int(data_in["figsize_y"])   # size of the figure in direction y
    colormap    = str(data_in["colormap"])    # colormap of the figure
    colornum    = int(data_in["colornum"])    # number of colors of the colormap
    fig_name    = str(data_in["fig_name"])    # name of the figure to be saved
    fig_name2   = str(data_in["fig_name2"])   # name of the figure to be saved for the second plot
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
    # Read the file mean velocity
    # -------------------------------------------------------------------------------------------------------------------
    Data_flow    = {"folder":uvw_folder,"file":uvw_file,"down_x":dx,"down_y":dy,"down_z":dz,\
                    "L_x":L_x,"L_y":L_y,"L_z":L_z,"rey":rey,"utau":utau}
    flow_data    = flow_field(data_in=Data_flow)
    flow_data.shape_tensor()
    flow_data.flow_grid()
    data_SHAPrms = read_rms(data_in={"folder":folder,"file":file,"dy":dy})
    SHAP_urms    = (data_SHAPrms["SHAP_urms"][:flow_data.yl_s]+np.flip(data_SHAPrms["SHAP_urms"][flow_data.yu_s:]))/2
    SHAP_vrms    = (data_SHAPrms["SHAP_vrms"][:flow_data.yl_s]+np.flip(data_SHAPrms["SHAP_vrms"][flow_data.yu_s:]))/2
    SHAP_wrms    = (data_SHAPrms["SHAP_wrms"][:flow_data.yl_s]+np.flip(data_SHAPrms["SHAP_wrms"][flow_data.yu_s:]))/2
    SHAP_uv      = (abs(data_SHAPrms["SHAP_uv"][:flow_data.yl_s])+
                    abs(np.flip(data_SHAPrms["SHAP_uv"][flow_data.yu_s:])))/2
    SHAP_uw      = (abs(data_SHAPrms["SHAP_uw"][:flow_data.yl_s])+
                    abs(np.flip(data_SHAPrms["SHAP_uw"][flow_data.yu_s:])))/2
    SHAP_vw      = (abs(data_SHAPrms["SHAP_vw"][:flow_data.yl_s])+
                    abs(np.flip(data_SHAPrms["SHAP_vw"][flow_data.yu_s:])))/2
    
    SHAP_mrms    = (data_SHAPrms["SHAP_mrms"][:flow_data.yl_s]+np.flip(data_SHAPrms["SHAP_mrms"][flow_data.yu_s:]))/2
    SHAP_rms_m   = np.sqrt(SHAP_urms**2+SHAP_vrms**2+SHAP_wrms**2)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum+1,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":flow_data.yplus,"data_y":SHAP_urms,"label":"$\phi'_u$",\
                  "color":None,"linewidth":2,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":flow_data.yplus,"data_y":SHAP_vrms,"label":"$\phi'_v$",\
                  "color":None,"linewidth":2,"plot_number":1,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info2)
    plot_info3 = {"data_x":flow_data.yplus,"data_y":SHAP_wrms,"label":"$\phi'_w$",\
                  "color":None,"linewidth":2,"plot_number":2,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info3)
    plot_info4 = {"data_x":flow_data.yplus,"data_y":SHAP_mrms,"label":"$|\phi|'$",\
                  "color":None,"linewidth":2,"plot_number":3,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info4)
    plot_train.plot_layout()
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel2,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_name2,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":flow_data.yplus,"data_y":SHAP_uv,"label":"$\phi'_{uv}$",\
                  "color":None,"linewidth":2,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":flow_data.yplus,"data_y":SHAP_uw,"label":"$\phi'_{uw}$",\
                  "color":None,"linewidth":2,"plot_number":1,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info2)
    plot_info3 = {"data_x":flow_data.yplus,"data_y":SHAP_vw,"label":"$\phi'_{vw}$",\
                  "color":None,"linewidth":2,"plot_number":2,"style":"-"}
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
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_name+"_compperc","dpi":dpi,"plot_folder":plot_folder,
                  "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure()
    plot_info1 = {"data_x":flow_data.yplus,"data_y":SHAP_rms_m,"label":"$|\phi'|$",\
                  "color":None,"linewidth":2,"plot_number":0,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info1)
    plot_info2 = {"data_x":flow_data.yplus,"data_y":SHAP_mrms,"label":"$|\phi|'$",\
                  "color":None,"linewidth":2,"plot_number":1,"style":"-"}
    plot_train.add_plot_2d(data_in=plot_info2)
    plot_train.plot_layout()
    plot_train.plot_save_png()
    plot_train.plot_save_pdf()