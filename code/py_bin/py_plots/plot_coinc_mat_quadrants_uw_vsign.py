# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plot_coinc_mat_quadrants_uw_vsign.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella

File to plot the coincidence between the structures of the shap streaks structures.
The file contains the following functions:
    - Functions:
        - plot_coinc_mat_2color : function to plot the coincidence between structures
"""
# -----------------------------------------------------------------------------------------------------------------------
# Import packages for all the functions
# -----------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define functions
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

def plot_coinc_mat_2color(data_in={"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$",
                                   "fontsize":18,"figsize_x":10,"figsize_y":8,"colors_2":[],"fig_name":"struc3d",
                                   "dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                                   "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,
                                   "utau":0.060523258443963,"cmap_flag":False,"index_ii":0,"plot_coin_folder":"coin",
                                   "calc_coin_shap_uvw":"-","data_folder":"-","ylabelbar":"y"}):
    """
    .....................................................................................................................
    # plot_coinc_mat_2color: Function to plot the coincidence between the structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colors_2":[],"fig_name":"struc3d",
                        "dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                        "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,
                        "utau":0.060523258443963,"cmap_flag":False,"index_ii":0,"plot_coin_folder":"coin",
                        "calc_coin_shap_uvw":"-","data_folder":"-","ylabelbar":"y"}.
        Data:
            - struc              : data of the structures
            - plot_folder        : folder to store the figures
            - xlabel             : label of the x axis
            - ylabel             : label of the y axis
            - fontsize           : font size used for the figure
            - figsize_x          : size of the figure in x
            - figsize_y          : size of the figure in y
            - colors_2           : colormap used for the figure
            - fig_name           : name of the saved figure
            - dpi                : dots per inch of the saved figure
            - dy                 : downsampling in the wall-normal direction
            - dx                 : downsampling in the streamwise direction
            - dz                 : downsampling in the spanwise direction
            - uvw_folder         : folder of the flow fields
            - uvw_file           : file of the flow fields
            - L_x                : streamwise dimension of the channel
            - L_y                : wall-normal dimension of the channel
            - L_z                : spanwise dimension of the channel
            - rey                : friction Reynolds number
            - utau               : friction velocity
            - cmap_flag          : flag to use a color map (True) or solid color (False)
            - index_ii           : index of the field
            - plot_coin_folder   : folder of the images
            - calc_coin_shap_uvw : file containing the coincidence of the shap uvw structures
            - data_folder        : folder storing the data
            - ylabelbar          : y label for the bar plot

    Returns
    -------
    None.

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from py_bin.py_class.flow_field import flow_field
    from matplotlib.colors import ListedColormap
    from py_bin.py_functions.calc_coinc_shap_uw_vsign import read_coinc
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    struc              = data_in["struc"]
    plot_folder        = str(data_in["plot_folder"]) 
    xlabel             = str(data_in["xlabel"]) 
    ylabel             = str(data_in["ylabel"])
    fontsize           = int(data_in["fontsize"])
    figsize_x          = int(data_in["figsize_x"])
    figsize_y          = int(data_in["figsize_y"])
    colors_2           = data_in["colors_2"]
    fig_nameread       = str(data_in["fig_name"])
    dpi                = float(data_in["dpi"])
    dy                 = int(data_in["dy"])
    dx                 = int(data_in["dx"])
    dz                 = int(data_in["dz"])
    uvw_folder         = str(data_in["uvw_folder"])
    uvw_file           = str(data_in["uvw_file"])
    L_x                = float(data_in["L_x"])
    L_y                = float(data_in["L_y"])
    L_z                = float(data_in["L_z"])
    rey                = float(data_in["rey"])
    utau               = float(data_in["utau"])
    cmapflag           = bool(data_in["cmap_flag"])
    index_ii           = int(data_in["index_ii"])
    plot_coin_folder   = str(data_in["plot_coin_folder"])
    calc_coin_shap_uvw = str(data_in["calc_coin_shap_uvw"])
    data_folder        = str(data_in["data_folder"])
    ylabelbar          = str(data_in["ylabelbar"])
    fig_name0          = plot_coin_folder+'/'+fig_nameread
    fig_name1          = plot_coin_folder+'/percbar_'+fig_nameread
    
    
    
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
    mat_1   = np.array(struc.structures_1.mat_struc,dtype='float')
    mat_2   = np.array(struc.structures_2.mat_struc,dtype='float')
    index      = str(struc.index)
    lab_list   = ['Q1',
                  'Q2']
    
    colormap   = ListedColormap(colors_2)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the coincidence file
    # -------------------------------------------------------------------------------------------------------------------
    data_coinc  = read_coinc(data_in={"calc_coin_file":calc_coin_shap_uvw,"folder":data_folder})
    frac_struc1 = data_coinc["frac_struc1"]
    frac_struc2 = data_coinc["frac_struc2"]
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create combined matrix
    # -------------------------------------------------------------------------------------------------------------------
    mat_comb              = mat_1+2*mat_2
    mat_comb[mat_comb==0] = np.nan
    
    # -------------------------------------------------------------------------------------------------------------------
    # Coordinates
    # -------------------------------------------------------------------------------------------------------------------
    xx,zz       = np.meshgrid(flow_data.xplus,flow_data.zplus)
    xmin        = 0
    xmax        = flow_data.L_x*flow_data.rey
    ymin        = 0
    ymax        = flow_data.L_z*flow_data.rey
    yplustot    = (1-abs(flow_data.y_h))*flow_data.rey
    
    for index_y in np.arange(len(mat_comb[:,0,0])):
        titlefig    = "y^+ = "+'{0:.2f}'.format(yplustot[index_y])
        fig_name    = fig_name0+"_field"+str(index_ii)+"_y"+str(index_y)
        # -------------------------------------------------------------------------------------------------------------------
        # Create the plot
        # -------------------------------------------------------------------------------------------------------------------
        data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                      "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                      "colornum":0,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                      "xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"zmin":None,"zmax":None}
        plot_pred = plot_format(data_in=data_plot)
        plot_pred.create_figure()
        plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":mat_comb[index_y,:,:],"colormap":colormap,
                                      "plot_number_x":0,"plot_number_y":0,"vmax":2,"vmin":1,"Ncolor":2})
        plot_pred.plot_layout_pcolor(data_in={"title":titlefig,"colorbar":True,"b_text":None,
                                              "colorticks":np.unique(mat_comb)[:-1],
                                              "colorlabels":lab_list,
                                              "equal":True})
        try:
            os.mkdir(plot_folder)
        except:
            print("Existing folder...",flush=True)
        try:
            os.mkdir(plot_folder+'/'+plot_coin_folder)
        except:
            print("Existing folder...",flush=True)
        plot_pred.plot_save_png()
        plot_pred.plot_save_pdf()
        plot_pred.close()
    
        # ---------------------------------------------------------------------------------------------------------------
        # Name of the figure
        # ---------------------------------------------------------------------------------------------------------------
        fig_name = fig_name1+"_field"+str(index_ii)+"_y"+str(index_y)
        # ---------------------------------------------------------------------------------------------------------------
        # Create a colorbar with the percentage of each type of structure
        # ---------------------------------------------------------------------------------------------------------------
        if index_y <= len(frac_struc1)-1:
            index_y_mod = index_y
        else:
            index_y_mod = len(frac_struc1)-index_y-2
        fs1  = frac_struc1[index_y_mod]
        fs2  = frac_struc2[index_y_mod]
        ftot = fs1+fs2
        fsQ  = np.array([fs1,fs2]/ftot,dtype='float')
        
        data_plot  = {"xlabel":xlabel,"ylabel":ylabelbar,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x+6,
                      "figsize_y":figsize_y+6,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colors_2,
                      "colornum":0,"legend":True,"fig_name":fig_name,"dpi":dpi,
                      "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":0,"ymax":1,
                      "zmin":None,"zmax":None}
        plot_pred = plot_format(data_in=data_plot)
        plot_pred.create_figure()
        plot_pred.add_cust_colorbar(data_in={"labels":lab_list,"data":fsQ,"colors":colors_2,"title":titlefig,
                                             "ynum":5,"posplot":[0.15,0.95,0.8,0.6]})
        plot_pred.plot_save_png()
        plot_pred.plot_save_pdf()
        plot_pred.close()
  
    
def plot_coinc_mat_2color_withcontour(data_in={"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$",
                                               "fontsize":18,"figsize_x":10,"figsize_y":8,"colors_2":[],
                                               "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,
                                               "uvw_folder":"../../P125_21pi_vu/",
                                               "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,
                                               "L_z":np.pi,"rey":125,"utau":0.060523258443963,"cmap_flag":False,
                                               "index_ii":0,"plot_coin_folder":"coin","calc_coin_shap_uvw":"-",
                                               "data_folder":"-","cont_struc":[]}):
    """
    .....................................................................................................................
    # plot_coinc_mat_2color_withcontour: Function to plot the coincidence between the structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colors_2":[],"fig_name":"struc3d",
                        "dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                        "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,
                        "utau":0.060523258443963,"cmap_flag":False,"index_ii":0,"plot_coin_folder":"coin",
                        "calc_coin_shap_uvw":"-","data_folder":"-","cont_struc":[]}.
        Data:
            - struc              : data of the structures
            - plot_folder        : folder to store the figures
            - xlabel             : label of the x axis
            - ylabel             : label of the y axis
            - fontsize           : font size used for the figure
            - figsize_x          : size of the figure in x
            - figsize_y          : size of the figure in y
            - colors_11          : colormap used for the figure
            - fig_name           : name of the saved figure
            - dpi                : dots per inch of the saved figure
            - dy                 : downsampling in the wall-normal direction
            - dx                 : downsampling in the streamwise direction
            - dz                 : downsampling in the spanwise direction
            - uvw_folder         : folder of the flow fields
            - uvw_file           : file of the flow fields
            - L_x                : streamwise dimension of the channel
            - L_y                : wall-normal dimension of the channel
            - L_z                : spanwise dimension of the channel
            - rey                : friction Reynolds number
            - utau               : friction velocity
            - cmap_flag          : flag to use a color map (True) or solid color (False)
            - index_ii           : index of the field
            - plot_coin_folder   : folder of the images
            - calc_coin_shap_uvw : file containing the coincidence of the shap uvw structures
            - data_folder        : folder storing the data
            - cont_struc         : structure for the contours

    Returns
    -------
    None.

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from py_bin.py_class.flow_field import flow_field
    from matplotlib.colors import ListedColormap
    from py_bin.py_functions.calc_coinc_shap_uvw import read_coinc
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    struc              = data_in["struc"]
    plot_folder        = str(data_in["plot_folder"]) 
    xlabel             = str(data_in["xlabel"]) 
    ylabel             = str(data_in["ylabel"])
    fontsize           = int(data_in["fontsize"])
    figsize_x          = int(data_in["figsize_x"])
    figsize_y          = int(data_in["figsize_y"])
    colors_2           = data_in["colors_2"]
    fig_nameread       = str(data_in["fig_name"])
    dpi                = float(data_in["dpi"])
    dy                 = int(data_in["dy"])
    dx                 = int(data_in["dx"])
    dz                 = int(data_in["dz"])
    uvw_folder         = str(data_in["uvw_folder"])
    uvw_file           = str(data_in["uvw_file"])
    L_x                = float(data_in["L_x"])
    L_y                = float(data_in["L_y"])
    L_z                = float(data_in["L_z"])
    rey                = float(data_in["rey"])
    utau               = float(data_in["utau"])
    cmapflag           = bool(data_in["cmap_flag"])
    index_ii           = int(data_in["index_ii"])
    plot_coin_folder   = str(data_in["plot_coin_folder"])
    calc_coin_shap_uvw = str(data_in["calc_coin_shap_uvw"])
    data_folder        = str(data_in["data_folder"])
    cont_struc         = data_in["cont_struc"]
    fig_name0          = plot_coin_folder+'/'+fig_nameread
    fig_name1          = plot_coin_folder+'/percbar_'+fig_nameread
    
    
    
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
    mat_cont   = np.array(cont_struc.mat_struc,dtype='int')
    mat_1      = np.array(struc.structures_1.mat_struc,dtype='float')
    mat_2      = np.array(struc.structures_2.mat_struc,dtype='float')
    index      = str(struc.index)
    lab_list   = ['Q1',
                  'Q2']
    
    colormap   = ListedColormap(colors_2)

    
    # -------------------------------------------------------------------------------------------------------------------
    # Create combined matrix
    # -------------------------------------------------------------------------------------------------------------------
    mat_comb              = mat_1+2*mat_2
    mat_comb[mat_comb==0] = np.nan
    
    # -------------------------------------------------------------------------------------------------------------------
    # Coordinates
    # -------------------------------------------------------------------------------------------------------------------
    xx,zz       = np.meshgrid(flow_data.xplus,flow_data.zplus)
    xmin        = 0
    xmax        = flow_data.L_x*flow_data.rey
    ymin        = 0
    ymax        = flow_data.L_z*flow_data.rey
    yplustot    = (1-abs(flow_data.y_h))*flow_data.rey
    
    for index_y in np.arange(len(mat_comb[:,0,0])):
        titlefig    = "y^+ = "+'{0:.2f}'.format(yplustot[index_y])
        fig_name    = fig_name0+"_field"+str(index_ii)+"_y"+str(index_y)
        # -------------------------------------------------------------------------------------------------------------------
        # Create the plot
        # -------------------------------------------------------------------------------------------------------------------
        data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                      "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                      "colornum":0,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                      "xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"zmin":None,"zmax":None}
        plot_pred = plot_format(data_in=data_plot)
        plot_pred.create_figure()
        plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":mat_comb[index_y,:,:],"colormap":colormap,
                                      "plot_number_x":0,"plot_number_y":0,"vmax":2,"vmin":1,"Ncolor":2})
        plot_pred.add_contline(data_in={"data_x":xx,"data_y":zz,"data_cont":mat_cont[index_y,:,:],"val":0.5,
                                        "colors":"k","linewidth":2})
        plot_pred.plot_layout_pcolor(data_in={"title":titlefig,"colorbar":True,"b_text":None,
                                              "colorticks":np.unique(mat_comb)[:-1],
                                              "colorlabels":lab_list,
                                              "equal":True})
        try:
            os.mkdir(plot_folder)
        except:
            print("Existing folder...",flush=True)
        try:
            os.mkdir(plot_folder+'/'+plot_coin_folder)
        except:
            print("Existing folder...",flush=True)
        plot_pred.plot_save_png()
        plot_pred.plot_save_pdf()
        plot_pred.close()