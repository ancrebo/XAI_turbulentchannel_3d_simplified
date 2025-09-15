# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plot_coinc_mat.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella

File to plot the coincidence between structures. The file contains the following functions:
    - Functions:
        - plot_coinc_mat_11color : function to plot the coincidence between structures
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

def plot_coinc_mat_11color(data_in={"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$",
                                    "fontsize":18,"figsize_x":10,"figsize_y":8,"colors_11":[],"fig_name":"struc3d",
                                    "dpi":60,"dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                                    "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,
                                    "utau":0.060523258443963,"cmap_flag":False,"index_ii":0,"plot_coin_folder":"coin",
                                    "calc_coin_shap_uvw":"-","data_folder":"-","ylabelbar":"y"}):
    """
    .....................................................................................................................
    # plot_coinc_mat_11color: Function to plot the coincidence between the structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colors_11":[],"fig_name":"struc3d",
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
    colors_11          = data_in["colors_11"]
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
    mat_Q100   = np.array(struc.structures_Q100.mat_struc,dtype='float')
    mat_Q001   = np.array(struc.structures_Q001.mat_struc,dtype='float')
    mat_check  = mat_Q100+mat_Q001
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q010   = np.array(struc.structures_Q010.mat_struc,dtype='float')
    mat_check += mat_Q010
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q020   = np.array(struc.structures_Q020.mat_struc,dtype='float')
    mat_check += mat_Q020
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q101   = np.array(struc.structures_Q101.mat_struc,dtype='float')
    mat_check += mat_Q101
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q110   = np.array(struc.structures_Q110.mat_struc,dtype='float')
    mat_check += mat_Q110
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q120   = np.array(struc.structures_Q120.mat_struc,dtype='float')
    mat_check += mat_Q120
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q011   = np.array(struc.structures_Q011.mat_struc,dtype='float')
    mat_check += mat_Q011
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q021   = np.array(struc.structures_Q021.mat_struc,dtype='float')
    mat_check += mat_Q021
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q111   = np.array(struc.structures_Q111.mat_struc,dtype='float')
    mat_check += mat_Q111
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q121   = np.array(struc.structures_Q121.mat_struc,dtype='float')
    mat_check += mat_Q121
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    index      = str(struc.index)
    lab_list   = ['Q100',
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
    
    colormap   = ListedColormap(colors_11)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the coincidence file
    # -------------------------------------------------------------------------------------------------------------------
    data_coinc     = read_coinc(data_in={"calc_coin_file":calc_coin_shap_uvw,"folder":data_folder})
    frac_strucQ100 = data_coinc["frac_strucQ100"]
    frac_strucQ001 = data_coinc["frac_strucQ001"]
    frac_strucQ010 = data_coinc["frac_strucQ010"]
    frac_strucQ020 = data_coinc["frac_strucQ020"]
    frac_strucQ101 = data_coinc["frac_strucQ101"]
    frac_strucQ110 = data_coinc["frac_strucQ110"]
    frac_strucQ120 = data_coinc["frac_strucQ120"]
    frac_strucQ011 = data_coinc["frac_strucQ011"]
    frac_strucQ021 = data_coinc["frac_strucQ021"]
    frac_strucQ111 = data_coinc["frac_strucQ111"]
    frac_strucQ121 = data_coinc["frac_strucQ121"]
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create combined matrix
    # -------------------------------------------------------------------------------------------------------------------
    mat_comb = mat_Q100+2*mat_Q001+3*mat_Q010+4*mat_Q020+5*mat_Q101+6*mat_Q110+7*mat_Q120+8*mat_Q011+9*mat_Q021+\
        10*mat_Q111+11*mat_Q121
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
                                      "plot_number_x":0,"plot_number_y":0,"vmax":11,"vmin":1,"Ncolor":11})
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
        if index_y <= len(frac_strucQ100)-1:
            index_y_mod = index_y
        else:
            index_y_mod = len(frac_strucQ100)-index_y-2
        fsQ100 = frac_strucQ100[index_y_mod]
        fsQ001 = frac_strucQ001[index_y_mod]
        fsQ010 = frac_strucQ010[index_y_mod]
        fsQ020 = frac_strucQ020[index_y_mod]
        fsQ101 = frac_strucQ101[index_y_mod]
        fsQ110 = frac_strucQ110[index_y_mod]
        fsQ120 = frac_strucQ120[index_y_mod]
        fsQ011 = frac_strucQ011[index_y_mod]
        fsQ021 = frac_strucQ021[index_y_mod]
        fsQ111 = frac_strucQ111[index_y_mod]
        fsQ121 = frac_strucQ121[index_y_mod]
        ftot   = fsQ100+fsQ001+fsQ010+fsQ020+fsQ101+fsQ110+fsQ120+fsQ011+fsQ021+fsQ111+fsQ121
        fsQ    = np.array([fsQ100,fsQ001,fsQ010,fsQ020,fsQ101,fsQ110,fsQ120,fsQ011,fsQ021,fsQ111,fsQ121]/ftot,
                          dtype='float')
        
        data_plot  = {"xlabel":xlabel,"ylabel":ylabelbar,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x+6,
                      "figsize_y":figsize_y+6,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colors_11,
                      "colornum":0,"legend":True,"fig_name":fig_name,"dpi":dpi,
                      "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":0,"ymax":0.5,
                      "zmin":None,"zmax":None}
        plot_pred = plot_format(data_in=data_plot)
        plot_pred.create_figure()
        plot_pred.add_cust_colorbar(data_in={"labels":lab_list,"data":fsQ,"colors":colors_11,"title":titlefig,
                                             "ynum":5,"posplot":[0.15,0.95,0.8,0.6]})
        plot_pred.plot_save_png()
        plot_pred.plot_save_pdf()
        plot_pred.close()
  
    
def plot_coinc_mat_11color_withcontour(data_in={"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$",
                                                "fontsize":18,"figsize_x":10,"figsize_y":8,"colors_11":[],
                                                "fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,
                                                "uvw_folder":"../../P125_21pi_vu/",
                                                "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","L_x":2*np.pi,"L_y":1,
                                                "L_z":np.pi,"rey":125,"utau":0.060523258443963,"cmap_flag":False,
                                                "index_ii":0,"plot_coin_folder":"coin","calc_coin_shap_uvw":"-",
                                                "data_folder":"-","cont_struc":[]}):
    """
    .....................................................................................................................
    # plot_coinc_mat_11color_withcontour: Function to plot the coincidence between the structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc":[],"plot_folder":"plots","xlabel":"$x^+$","ylabel":"$y^+$",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colors_11":[],"fig_name":"struc3d",
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
    colors_11          = data_in["colors_11"]
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
    mat_Q100   = np.array(struc.structures_Q100.mat_struc,dtype='float')
    mat_Q001   = np.array(struc.structures_Q001.mat_struc,dtype='float')
    mat_check  = mat_Q100+mat_Q001
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q010   = np.array(struc.structures_Q010.mat_struc,dtype='float')
    mat_check += mat_Q010
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q020   = np.array(struc.structures_Q020.mat_struc,dtype='float')
    mat_check += mat_Q020
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q101   = np.array(struc.structures_Q101.mat_struc,dtype='float')
    mat_check += mat_Q101
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q110   = np.array(struc.structures_Q110.mat_struc,dtype='float')
    mat_check += mat_Q110
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q120   = np.array(struc.structures_Q120.mat_struc,dtype='float')
    mat_check += mat_Q120
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q011   = np.array(struc.structures_Q011.mat_struc,dtype='float')
    mat_check += mat_Q011
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q021   = np.array(struc.structures_Q021.mat_struc,dtype='float')
    mat_check += mat_Q021
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q111   = np.array(struc.structures_Q111.mat_struc,dtype='float')
    mat_check += mat_Q111
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    mat_Q121   = np.array(struc.structures_Q121.mat_struc,dtype='float')
    mat_check += mat_Q121
    if np.max(mat_check) > 1:
        print('Error in matrices',flush=True)
        sys.exit()
    index      = str(struc.index)
    lab_list   = ['Q100',
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
    
    colormap   = ListedColormap(colors_11)
    
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create combined matrix
    # -------------------------------------------------------------------------------------------------------------------
    mat_comb = mat_Q100+2*mat_Q001+3*mat_Q010+4*mat_Q020+5*mat_Q101+6*mat_Q110+7*mat_Q120+8*mat_Q011+9*mat_Q021+\
        10*mat_Q111+11*mat_Q121
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
                                      "plot_number_x":0,"plot_number_y":0,"vmax":11,"vmin":1,"Ncolor":11})
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