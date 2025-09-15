# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plotstruc3d_quadrants_uvw.py
-------------------------------------------------------------------------------------------------------------------------
Created on Sat Jul  6 11:51:39 2024

@author: Andres Cremades Botella

File to plot 3d structures of the clasification of the shap structures with 11 quadrants:
    - Functions:
        - plotstruc3d_11color        : function to plot the structures in the whole channel
        - plotstruc3d_separe_11color : function to plot the structures in half of the channel
"""

# -----------------------------------------------------------------------------------------------------------------------
# Import packages for all functions
# -----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os
     
def plotstruc3d_11color(data_in={"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$","zlabel":"z^+",
                                 "fontsize":18,"figsize_x":10,"figsize_y":8,"colors_11":[],"fig_name":"struc3d",
                                 "dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                                 "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,
                                 "utau":0.060523258443963,"cmap_flag":False}):
    """
    .....................................................................................................................
    # plotstruc3d_11color: Function to generate the plot of 11 types of structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$","zlabel":"z^+",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colors_11":[],"fig_name":"struc3d",
                        "dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                        "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,
                        "utau":0.060523258443963,"cmap_flag":False}.
        Data:
            - struc       : data of the structures
            - plot_folder : folder to store the figures
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - zlabel      : label of the z axis
            - fontsize    : font size used for the figure
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colors_11   : colormap used for the figure
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
    colors_11   = data_in["colors_11"]
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
    mat_Q100 = struc.structures_Q100.mat_segment_filtered
    mat_Q001 = struc.structures_Q001.mat_segment_filtered
    mat_Q010 = struc.structures_Q010.mat_segment_filtered
    mat_Q020 = struc.structures_Q020.mat_segment_filtered
    mat_Q101 = struc.structures_Q101.mat_segment_filtered
    mat_Q110 = struc.structures_Q110.mat_segment_filtered
    mat_Q120 = struc.structures_Q120.mat_segment_filtered
    mat_Q011 = struc.structures_Q011.mat_segment_filtered
    mat_Q021 = struc.structures_Q021.mat_segment_filtered
    mat_Q111 = struc.structures_Q111.mat_segment_filtered
    mat_Q121 = struc.structures_Q121.mat_segment_filtered
    index    = str(struc.index)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create a legend with a figure
    # -------------------------------------------------------------------------------------------------------------------
    lab_list     = ['Q100',
                    'Q001',
                    'Q010',
                    'Q020',
                    'Q101',
                    'Q110',
                    'Q120',
                    'Q011',
                    'Q021',
                    'Q111',
                    'Q121']
    data_plot    = {"xlabel":[],"ylabel":[],"zlabel":[],"fontsize":fontsize,"figsize_x":12,
                    "figsize_y":6,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":'viridis',
                    "colornum":0,"legend":True,"fig_name":fig_name+"_legend","dpi":dpi,"plot_folder":plot_folder,
                    "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure()
    plot_pred.legend_infig(data_in={"colors":colors_11,"labels":lab_list,"barstype":"horizontal"})
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    plot_pred.close()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colors_11[0],
                  "colornum":0,"legend":False,"fig_name":fig_name+"_field"+index,"dpi":dpi,
                  "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1  = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_Q100,
                   "color":colors_11[0],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_info2  = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_Q001,
                   "color":colors_11[1],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info2)
    plot_info3  = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_Q010,
                   "color":colors_11[2],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info3)
    plot_info4  = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_Q020,
                   "color":colors_11[3],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info4)
    plot_info5  = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_Q101,
                   "color":colors_11[4],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info5)
    plot_info6  = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_Q110,
                   "color":colors_11[5],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info6)
    plot_info7  = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_Q120,
                   "color":colors_11[6],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info7)
    plot_info8  = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_Q011,
                   "color":colors_11[7],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info8)
    plot_info9  = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_Q021,
                   "color":colors_11[8],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info9)
    plot_info10 = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_Q111,
                   "color":colors_11[9],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info10)
    plot_info11 = {"data_x":flow_data.xplus,"data_y":flow_data.y_h*rey,"data_z":flow_data.zplus,"struc":mat_Q121,
                   "color":colors_11[10],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info11)
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
    
    
     
def plotstruc3d_separe_11color(data_in={"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$",
                                        "zlabel":"z^+","fontsize":18,"figsize_x":10,"figsize_y":8,"colors_11":[],
                                        "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,
                                        "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                                        "L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963,
                                        "cmap_flag":False}):
    """
    .....................................................................................................................
    # plotstruc3d_separe_11color: Function to generate the plot of 11 types of structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$",
                        "zlabel":"z^+","fontsize":18,"figsize_x":10,"figsize_y":8,"colors_11":[],
                        "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,
                        "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                        "L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,"utau":0.060523258443963,
                        "cmap_flag":False}.
        Data:
            - struc       : data of the structures
            - plot_folder : folder to store the figures
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - zlabel      : label of the z axis
            - fontsize    : font size used for the figure
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colors_11   : colormap used for the figure
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
    colors_11   = data_in["colors_11"]
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
    mat_Q100     = struc.structures_Q100.mat_segment_filtered
    mat_Q100_low = mat_Q100[:flow_data.yl_s,:,:]
    mat_Q100_upp = np.flip(mat_Q100[flow_data.yu_s:,:,:],0)
    mat_Q001     = struc.structures_Q001.mat_segment_filtered
    mat_Q001_low = mat_Q001[:flow_data.yl_s,:,:]
    mat_Q001_upp = np.flip(mat_Q001[flow_data.yu_s:,:,:],0)
    mat_Q010     = struc.structures_Q010.mat_segment_filtered
    mat_Q010_low = mat_Q010[:flow_data.yl_s,:,:]
    mat_Q010_upp = np.flip(mat_Q010[flow_data.yu_s:,:,:],0)
    mat_Q020     = struc.structures_Q020.mat_segment_filtered
    mat_Q020_low = mat_Q020[:flow_data.yl_s,:,:]
    mat_Q020_upp = np.flip(mat_Q020[flow_data.yu_s:,:,:],0)
    mat_Q101     = struc.structures_Q101.mat_segment_filtered
    mat_Q101_low = mat_Q101[:flow_data.yl_s,:,:]
    mat_Q101_upp = np.flip(mat_Q101[flow_data.yu_s:,:,:],0)
    mat_Q110     = struc.structures_Q110.mat_segment_filtered
    mat_Q110_low = mat_Q110[:flow_data.yl_s,:,:]
    mat_Q110_upp = np.flip(mat_Q110[flow_data.yu_s:,:,:],0)
    mat_Q120     = struc.structures_Q120.mat_segment_filtered
    mat_Q120_low = mat_Q120[:flow_data.yl_s,:,:]
    mat_Q120_upp = np.flip(mat_Q120[flow_data.yu_s:,:,:],0)
    mat_Q011     = struc.structures_Q011.mat_segment_filtered
    mat_Q011_low = mat_Q011[:flow_data.yl_s,:,:]
    mat_Q011_upp = np.flip(mat_Q011[flow_data.yu_s:,:,:],0)
    mat_Q021     = struc.structures_Q021.mat_segment_filtered
    mat_Q021_low = mat_Q021[:flow_data.yl_s,:,:]
    mat_Q021_upp = np.flip(mat_Q021[flow_data.yu_s:,:,:],0)
    mat_Q111     = struc.structures_Q111.mat_segment_filtered
    mat_Q111_low = mat_Q111[:flow_data.yl_s,:,:]
    mat_Q111_upp = np.flip(mat_Q111[flow_data.yu_s:,:,:],0)
    mat_Q121     = struc.structures_Q121.mat_segment_filtered
    mat_Q121_low = mat_Q121[:flow_data.yl_s,:,:]
    mat_Q121_upp = np.flip(mat_Q121[flow_data.yu_s:,:,:],0)
    index    = str(struc.index)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create a legend with a figure
    # -------------------------------------------------------------------------------------------------------------------
    lab_list     = ['Q100',
                    'Q001',
                    'Q010',
                    'Q020',
                    'Q101',
                    'Q110',
                    'Q120',
                    'Q011',
                    'Q021',
                    'Q111',
                    'Q121']
    data_plot    = {"xlabel":[],"ylabel":[],"zlabel":[],"fontsize":fontsize,"figsize_x":12,
                    "figsize_y":6,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":'viridis',
                    "colornum":0,"legend":True,"fig_name":fig_name+"_legend","dpi":dpi,"plot_folder":plot_folder,
                    "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure()
    plot_pred.legend_infig(data_in={"colors":colors_11,"labels":lab_list,"barstype":"horizontal"})
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    plot_pred.close()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colors_11[0],
                  "colornum":0,"legend":False,"fig_name":fig_name+"_field"+index+"_low","dpi":dpi,
                  "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q100_low,
                   "color":colors_11[0],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_info2  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q001_low,
                   "color":colors_11[1],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info2)
    plot_info3  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q010_low,
                   "color":colors_11[2],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info3)
    plot_info4  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q020_low,
                   "color":colors_11[3],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info4)
    plot_info5  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q101_low,
                   "color":colors_11[4],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info5)
    plot_info6  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q110_low,
                   "color":colors_11[5],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info6)
    plot_info7  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q120_low,
                   "color":colors_11[6],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info7)
    plot_info8  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q011_low,
                   "color":colors_11[7],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info8)
    plot_info9  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q021_low,
                   "color":colors_11[8],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info9)
    plot_info10 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q111_low,
                   "color":colors_11[9],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info10)
    plot_info11 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q121_low,
                   "color":colors_11[10],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info11)
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
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colors_11[0],
                  "colornum":0,"legend":False,"fig_name":fig_name+"_field"+index+"_upp","dpi":dpi,
                  "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_train = plot_format(data_in=data_plot)
    plot_train.create_figure3d()
    plot_info1  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q100_upp,
                   "color":colors_11[0],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info1)
    plot_info2  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q001_upp,
                   "color":colors_11[1],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info2)
    plot_info3  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q010_upp,
                   "color":colors_11[2],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info3)
    plot_info4  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q020_upp,
                   "color":colors_11[3],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info4)
    plot_info5  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q101_upp,
                   "color":colors_11[4],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info5)
    plot_info6  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q110_upp,
                   "color":colors_11[5],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info6)
    plot_info7  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q120_upp,
                   "color":colors_11[6],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info7)
    plot_info8  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q011_upp,
                   "color":colors_11[7],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info8)
    plot_info9  = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q021_upp,
                   "color":colors_11[8],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info9)
    plot_info10 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q111_upp,
                   "color":colors_11[9],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info10)
    plot_info11 = {"data_x":flow_data.xplus,"data_y":flow_data.yplus,"data_z":flow_data.zplus,"struc":mat_Q121_upp,
                   "color":colors_11[10],"cmap_flag":cmapflag,"vmax":rey,"vmin":-3*rey}
    plot_train.add_plot_3d_structure(data_in=plot_info11)
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