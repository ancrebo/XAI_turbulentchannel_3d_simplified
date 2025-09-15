# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plot_coinc_mat.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella

File to plot the coincidence between structures. The file contains the following functions:
    - Functions:
        - plot_coinc_mat : function to plot the coincidence between structures
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

def plot_coinc_mat(data_in={"plot_folder":"plots","plot_coin_folder":"uv_shap_coin_y","plot_coin_file":"uv_shap_coin_y",
                            "xlabel":"x","ylabel":"z","fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis",
                            "colornum":2,"dpi":60,"struc1_lab":"uv","struc2_lab":"SHAP","flowfield":[],"mat_comb":[],
                            "index_ii":0}):
    """
    .....................................................................................................................
    # plot_coinc_mat: Function to plot the coincidence between the structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"plot_folder":"plots","plot_coin_folder":"uv_shap_coin_y","plot_coin_file":"uv_shap_coin_y",
                        "xlabel":"x","ylabel":"z","fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis",
                        "colornum":2,"dpi":60,"struc1_lab":"uv","struc2_lab":"SHAP",
                        "flowfield":[],"mat_comb":[],"index_ii":0}.
        Data:
            - plot_folder      : folder to store the plots
            - plot_coin_folder : folder to save the coincidence plots between structures
            - plot_coin_file   : file to save the coincidence plots between structures
            - xlabel           : label of the x axis
            - ylabel           : label of the y axis
            - fontsize         : font size used for the figure
            - figsize_x        : size of the figure in x
            - figsize_y        : size of the figure in y
            - colormap         : colormap used for the figure
            - colornum         : number of colors of the colormap, two curves are used. The number of levels of the 
                                 colormap needs to be higher than 2 
            - dpi              : dots per inch of the saved figure
            - struc1_lab       : label of the structure 1
            - struc2_lab       : label of the structure 2
            - flowfield        : data of the flow field
            - mat_comb         : matrix with the combination of the structures
            - index_ii         : index of the fields

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
    plot_folder           = str(data_in["plot_folder"])
    plot_coin_folder      = str(data_in["plot_coin_folder"])
    plot_coin_file        = str(data_in["plot_coin_file"])
    xlabel                = str(data_in["xlabel"])
    ylabel                = str(data_in["ylabel"])
    fontsize              = int(data_in["fontsize"])
    figsize_x             = int(data_in["figsize_x"])
    figsize_y             = int(data_in["figsize_y"])
    colormap              = str(data_in["colormap"])
    colornum              = int(data_in["colornum"])
    dpi                   = float(data_in["dpi"])
    struc1_lab            = str(data_in["struc1_lab"])
    struc2_lab            = str(data_in["struc2_lab"])
    flowfield             = data_in["flowfield"]
    mat_comb              = np.array(data_in["mat_comb"],dtype="float")
    index_ii              = data_in["index_ii"]
    mat_comb[mat_comb==0] = np.nan
    fig_name0             = plot_coin_folder+'/'+plot_coin_file
    
     
    xx,zz       = np.meshgrid(flowfield.xplus,flowfield.zplus)
    xmin        = 0
    xmax        = flowfield.L_x*flowfield.rey
    ymin        = 0
    ymax        = flowfield.L_z*flowfield.rey
    yplustot    = (1-abs(flowfield.y_h))*flowfield.rey
    
    for index_y in np.arange(len(mat_comb[:,0,0])):
        titlefig    = "$y^+\simeq$"+'{0:.0f}'.format(yplustot[index_y])
        fig_name    = fig_name0+"_field"+str(index_ii)+"_y"+str(index_y)
        # -------------------------------------------------------------------------------------------------------------------
        # Create the plot
        # -------------------------------------------------------------------------------------------------------------------
        data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                      "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                      "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                      "xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"zmin":None,"zmax":None}
        plot_pred = plot_format(data_in=data_plot)
        plot_pred.create_figure()
        plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":mat_comb[index_y,:,:],"colormap":colormap,
                                      "plot_number_x":0,"plot_number_y":0,"vmax":3,"vmin":1,"Ncolor":3})
        plot_pred.plot_layout_pcolor(data_in={"title":titlefig,"colorbar":True,"b_text":None,
                                              "colorticks":np.unique(mat_comb)[:-1],
                                              "colorlabels":[struc1_lab,struc2_lab,struc1_lab+'+'+struc2_lab],
                                              "equal":True,
                                              "xticks":[0,
                                                        flowfield.L_x/4*flowfield.rey,
                                                        2*flowfield.L_x/4*flowfield.rey,
                                                        3*flowfield.L_x/4*flowfield.rey,
                                                        flowfield.L_x*flowfield.rey],
                                              "yticks":[0,
                                                        flowfield.L_z/2*flowfield.rey,
                                                        flowfield.L_z*flowfield.rey],
                                              "xticklabels":["0",
                                                             str(int(np.round(flowfield.L_x/np.pi/4*flowfield.rey)))+
                                                             "$\pi$",
                                                             str(int(np.round(2*flowfield.L_x/np.pi/4*flowfield.rey)))+
                                                             "$\pi$",
                                                             str(int(np.round(3*flowfield.L_x/np.pi/4*flowfield.rey)))+
                                                             "$\pi$",
                                                             str(int(np.round(flowfield.L_x/np.pi*flowfield.rey)))+
                                                             "$\pi$"],
                                              "yticklabels":["0",
                                                             str(int(np.round(flowfield.L_z/np.pi/2*flowfield.rey)))+
                                                             "$\pi$",
                                                             str(int(np.round(flowfield.L_z/np.pi*flowfield.rey)))+
                                                             "$\pi$"]})
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
    
    
def plot_coinc_mat_all(data_in={"plot_folder":"plots","plot_coin_folder":"uv_shap_coin_y",
                                "plot_coin_file":"uv_shap_coin_y","xlabel":"x","ylabel":"z","fontsize":18,
                                "figsize_x":10,"figsize_y":8,"colornum":2,"dpi":60,
                                "struc1_lab":"SHAP","struc2_lab":"uv","struc3_lab":"streak","struc4_lab":"chong",
                                "struc5_lab":"hunt","flowfield":[],"mat_comb":[],"index_ii":0,
                                "calc_coin_tot":"calc.txt","data_folder":"data","yminbar":0,"ymaxbar":1,
                                "ynumbar":4,"ylabelbar":"-"}):
    """
    .....................................................................................................................
    # plot_coinc_mat_all: Function to plot the coincidence between all the structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"plot_folder":"plots","plot_coin_folder":"uv_shap_coin_y",
                        "plot_coin_file":"uv_shap_coin_y","xlabel":"x","ylabel":"z","fontsize":18,
                        "figsize_x":10,"figsize_y":8,"colornum":2,"dpi":60,
                        "struc1_lab":"SHAP","struc2_lab":"uv","struc3_lab":"streak","struc4_lab":"chong",
                        "struc5_lab":"hunt","flowfield":[],"mat_comb":[],"index_ii":0,"calc_coin_tot":"calc.txt",
                        "data_folder":"data","ylabelbar":"-"}.
        Data:
            - plot_folder      : folder to store the plots
            - plot_coin_folder : folder to save the coincidence plots between structures
            - plot_coin_file   : file to save the coincidence plots between structures
            - xlabel           : label of the x axis
            - ylabel           : label of the y axis
            - fontsize         : font size used for the figure
            - figsize_x        : size of the figure in x
            - figsize_y        : size of the figure in y
            - colornum         : number of colors of the colormap, two curves are used. The number of levels of the 
                                 colormap needs to be higher than 2 
            - dpi              : dots per inch of the saved figure
            - struc1_lab       : label of the structure 1
            - struc2_lab       : label of the structure 2
            - struc3_lab       : label of the structure 3
            - struc4_lab       : label of the structure 4
            - struc5_lab       : label of the structure 5
            - flowfield        : data of the flow field
            - mat_comb         : matrix with the combination of the structures
            - index_ii         : index of the fields
            - calc_coin_tot    : file to read the coincidence between structures
            - data_folder      : folder to read data files
            - yminbar          : minimum value of the y axis of the bar plot
            - ymaxbar          : maximum value of the y axis of the bar plot
            - ynumbar          : number of ticks of the bar plot
            - ylabelbar        : label for the y axis of the bar plot

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from matplotlib.colors import ListedColormap
    from py_bin.py_functions.calc_coinc import read_coinc_all
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    plot_folder           = str(data_in["plot_folder"])
    plot_coin_folder      = str(data_in["plot_coin_folder"])
    plot_coin_file        = str(data_in["plot_coin_file"])
    xlabel                = str(data_in["xlabel"])
    ylabel                = str(data_in["ylabel"])
    fontsize              = int(data_in["fontsize"])
    figsize_x             = int(data_in["figsize_x"])
    figsize_y             = int(data_in["figsize_y"])
    colornum              = int(data_in["colornum"])
    dpi                   = float(data_in["dpi"])
    struc1_lab            = str(data_in["struc1_lab"])
    struc2_lab            = str(data_in["struc2_lab"])
    struc3_lab            = str(data_in["struc3_lab"])
    struc4_lab            = str(data_in["struc4_lab"])
    struc5_lab            = str(data_in["struc5_lab"])
    flowfield             = data_in["flowfield"]
    mat_comb              = np.array(data_in["mat_comb"],dtype="float")
    index_ii              = data_in["index_ii"]
    calc_coin_tot         = str(data_in["calc_coin_tot"])
    data_folder           = str(data_in["data_folder"])
    yminbar               = float(data_in["yminbar"])
    ymaxbar               = float(data_in["ymaxbar"])
    ynumbar               = int(data_in["ynumbar"])
    ylabelbar             = str(data_in["ylabelbar"])
    mat_comb[mat_comb==0] = np.nan
    fig_name0             = plot_coin_folder+'/'+plot_coin_file
    fig_name1             = plot_coin_folder+'/percbar_'+plot_coin_file
    
     
    xx,zz       = np.meshgrid(flowfield.xplus,flowfield.zplus)
    xmin        = 0
    xmax        = flowfield.L_x*flowfield.rey
    ymin        = 0
    ymax        = flowfield.L_z*flowfield.rey
    yplustot    = (1-abs(flowfield.y_h))*flowfield.rey
    
    # -------------------------------------------------------------------------------------------------------------------
    # Define the colors
    # -------------------------------------------------------------------------------------------------------------------
    colors   = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22',
                '#17becf','#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5','#c49c94','#f7b6d2','#c7c7c7',
                '#dbdb8d','#9edae5','#393b79','#637939','#8c6d31','#843c39','#7b4173','#5254a3','#8ca252',
                '#bd9e39','#ad494a','#a55194','#6b6ecf']
    lab_list = [struc1_lab,struc2_lab,struc2_lab+"+"+struc1_lab,struc3_lab,struc3_lab+"+"+struc1_lab,
                struc3_lab+"+"+struc2_lab,struc3_lab+"+"+struc2_lab+"+"+struc1_lab,struc4_lab,
                struc4_lab+"+"+struc1_lab,struc4_lab+"+"+struc2_lab,struc4_lab+"+"+struc2_lab+"+"+struc1_lab,
                struc4_lab+"+"+struc3_lab,struc4_lab+"+"+struc3_lab+"+"+struc1_lab,
                struc4_lab+"+"+struc3_lab+"+"+struc2_lab,struc4_lab+"+"+struc3_lab+"+"+struc2_lab+"+"+struc1_lab,
                struc5_lab,struc5_lab+"+"+struc1_lab,struc5_lab+"+"+struc2_lab,struc5_lab+"+"+struc2_lab+"+"+struc1_lab,
                struc5_lab+"+"+struc3_lab,struc5_lab+"+"+struc3_lab+"+"+struc1_lab,
                struc5_lab+"+"+struc3_lab+"+"+struc2_lab,struc5_lab+"+"+struc3_lab+"+"+struc2_lab+"+"+struc1_lab,
                struc5_lab+"+"+struc4_lab,struc5_lab+"+"+struc4_lab+"+"+struc1_lab,
                struc5_lab+"+"+struc4_lab+"+"+struc2_lab,struc5_lab+"+"+struc4_lab+"+"+struc2_lab+"+"+struc1_lab,
                struc5_lab+"+"+struc4_lab+"+"+struc3_lab,struc5_lab+"+"+struc4_lab+"+"+struc3_lab+"+"+struc1_lab,
                struc5_lab+"+"+struc4_lab+"+"+struc3_lab+"+"+struc2_lab,
                struc5_lab+"+"+struc4_lab+"+"+struc3_lab+"+"+struc2_lab+"+"+struc1_lab]
    colormap = ListedColormap(colors)
    
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    try:
        os.mkdir(plot_folder+'/'+plot_coin_folder)
    except:
        print("Existing folder...",flush=True)
            
    # -------------------------------------------------------------------------------------------------------------------
    # Read the coincidence file
    # -------------------------------------------------------------------------------------------------------------------
    data_coinc     = read_coinc_all(data_in={"calc_coin_file":calc_coin_tot,"folder":data_folder})
    frac_coinc_tot = data_coinc["frac_coinc_tot"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create a legend with a figure
    # -------------------------------------------------------------------------------------------------------------------
    fig_name_leg = plot_coin_folder+'/legend_'+plot_coin_file
    data_plot    = {"xlabel":[],"ylabel":[],"zlabel":[],"fontsize":fontsize,"figsize_x":12,
                    "figsize_y":6,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                    "colornum":colornum,"legend":True,"fig_name":fig_name_leg,"dpi":dpi,"plot_folder":plot_folder,
                    "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure()
    plot_pred.legend_infig(data_in={"colors":colors,"labels":lab_list,"barstype":"vertical"})
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    plot_pred.close()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Do it for all the wall distances
    # -------------------------------------------------------------------------------------------------------------------
    for index_y in np.arange(len(mat_comb[:,0,0])):
        titlefig    = "$y^+\simeq$"+'{0:.0f}'.format(yplustot[index_y])
        fig_name    = fig_name0+"_field"+str(index_ii)+"_y"+str(index_y)
        # ---------------------------------------------------------------------------------------------------------------
        # Create the plot
        # ---------------------------------------------------------------------------------------------------------------
        data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                      "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                      "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                      "xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"zmin":None,"zmax":None}
        plot_pred = plot_format(data_in=data_plot)
        plot_pred.create_figure()
        plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":mat_comb[index_y,:,:],"colormap":colormap,
                                      "plot_number_x":0,"plot_number_y":0,"vmax":31,"vmin":1,"Ncolor":None})
        plot_pred.plot_layout_pcolor(data_in={"title":titlefig,"colorbar":False,"b_text":None,
                                              "colorticks":[],
                                              "colorlabels":[],
                                              "equal":True,
                                              "xticks":[0,
                                                        flowfield.L_x/4*flowfield.rey,
                                                        2*flowfield.L_x/4*flowfield.rey,
                                                        3*flowfield.L_x/4*flowfield.rey,
                                                        flowfield.L_x*flowfield.rey],
                                              "yticks":[0,
                                                        flowfield.L_z/2*flowfield.rey,
                                                        flowfield.L_z*flowfield.rey],
                                              "xticklabels":["0",
                                                             str(int(np.round(flowfield.L_x/np.pi/4*flowfield.rey)))+
                                                             "$\pi$",
                                                             str(int(np.round(2*flowfield.L_x/np.pi/4*flowfield.rey)))+
                                                             "$\pi$",
                                                             str(int(np.round(3*flowfield.L_x/np.pi/4*flowfield.rey)))+
                                                             "$\pi$",
                                                             str(int(np.round(flowfield.L_x/np.pi*flowfield.rey)))+
                                                             "$\pi$"],
                                              "yticklabels":["0",
                                                             str(int(np.round(flowfield.L_z/np.pi/2*flowfield.rey)))+
                                                             "$\pi$",
                                                             str(int(np.round(flowfield.L_z/np.pi*flowfield.rey)))+
                                                             "$\pi$"]})
        plot_pred.plot_save_png()
        plot_pred.plot_save_pdf()
        plot_pred.close()
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create a colorbar with the percentage of each type of structure
        # ---------------------------------------------------------------------------------------------------------------
        fig_name       = fig_name1+"_field"+str(index_ii)+"_y"+str(index_y)
        frac_coinc_yii = np.zeros((len(frac_coinc_tot)))
        for ii_coinc in np.arange(len(frac_coinc_tot)):
            if index_y <= len(frac_coinc_tot[ii_coinc])-1:
                index_y_mod          = index_y
            else:
                index_y_mod          = len(frac_coinc_tot[ii_coinc])-index_y-2
            frac_coinc_yii[ii_coinc] = frac_coinc_tot[ii_coinc][index_y_mod]
        frac_coinc_shap = np.sum(frac_coinc_yii[::2])
        frac_coinc_yii /= frac_coinc_shap
        data_plot  = {"xlabel":xlabel,"ylabel":ylabelbar,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x+6,
                      "figsize_y":figsize_y+6,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                      "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,
                      "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":yminbar,"ymax":ymaxbar,
                      "zmin":None,"zmax":None}
        plot_pred = plot_format(data_in=data_plot)
        plot_pred.create_figure()
        plot_pred.add_cust_colorbar(data_in={"labels":lab_list,"data":frac_coinc_yii,"colors":colors,"title":titlefig,
                                             "ynum":ynumbar,"posplot":[0.15,0.95,0.8,0.6]})
        plot_pred.plot_save_png()
        plot_pred.plot_save_pdf()
        plot_pred.close()
      
def plot_coinc_mat_all_4types(data_in={"plot_folder":"plots","plot_coin_folder":"uv_shap_coin_y",
                                       "plot_coin_file":"uv_shap_coin_y","xlabel":"x","ylabel":"z","fontsize":18,
                                       "figsize_x":10,"figsize_y":8,"colornum":2,"dpi":60,
                                       "struc1_lab":"SHAP","struc2_lab":"uv","struc3_lab":"streak","struc4_lab":"chong",
                                       "flowfield":[],"mat_comb":[],"index_ii":0,
                                       "calc_coin_tot":"calc.txt","data_folder":"data","yminbar":0,"ymaxbar":1,
                                       "ynumbar":4,"ylabelbar":"-"}):
    """
    .....................................................................................................................
    # plot_coinc_mat_all: Function to plot the coincidence between all the structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"plot_folder":"plots","plot_coin_folder":"uv_shap_coin_y",
                        "plot_coin_file":"uv_shap_coin_y","xlabel":"x","ylabel":"z","fontsize":18,
                        "figsize_x":10,"figsize_y":8,"colornum":2,"dpi":60,
                        "struc1_lab":"SHAP","struc2_lab":"uv","struc3_lab":"streak","struc4_lab":"chong",
                        "flowfield":[],"mat_comb":[],"index_ii":0,"calc_coin_tot":"calc.txt",
                        "data_folder":"data","ylabelbar":"-"}.
        Data:
            - plot_folder      : folder to store the plots
            - plot_coin_folder : folder to save the coincidence plots between structures
            - plot_coin_file   : file to save the coincidence plots between structures
            - xlabel           : label of the x axis
            - ylabel           : label of the y axis
            - fontsize         : font size used for the figure
            - figsize_x        : size of the figure in x
            - figsize_y        : size of the figure in y
            - colornum         : number of colors of the colormap, two curves are used. The number of levels of the 
                                 colormap needs to be higher than 2 
            - dpi              : dots per inch of the saved figure
            - struc1_lab       : label of the structure 1
            - struc2_lab       : label of the structure 2
            - struc3_lab       : label of the structure 3
            - struc4_lab       : label of the structure 4
            - flowfield        : data of the flow field
            - mat_comb         : matrix with the combination of the structures
            - index_ii         : index of the fields
            - calc_coin_tot    : file to read the coincidence between structures
            - data_folder      : folder to read data files
            - yminbar          : minimum value of the y axis of the bar plot
            - ymaxbar          : maximum value of the y axis of the bar plot
            - ynumbar          : number of ticks of the bar plot
            - ylabelbar        : label for the y axis of the bar plot

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from matplotlib.colors import ListedColormap
    from py_bin.py_functions.calc_coinc import read_coinc_all_4types
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    plot_folder           = str(data_in["plot_folder"])
    plot_coin_folder      = str(data_in["plot_coin_folder"])
    plot_coin_file        = str(data_in["plot_coin_file"])
    xlabel                = str(data_in["xlabel"])
    ylabel                = str(data_in["ylabel"])
    fontsize              = int(data_in["fontsize"])
    figsize_x             = int(data_in["figsize_x"])
    figsize_y             = int(data_in["figsize_y"])
    colornum              = int(data_in["colornum"])
    dpi                   = float(data_in["dpi"])
    struc1_lab            = str(data_in["struc1_lab"])
    struc2_lab            = str(data_in["struc2_lab"])
    struc3_lab            = str(data_in["struc3_lab"])
    struc4_lab            = str(data_in["struc4_lab"])
    flowfield             = data_in["flowfield"]
    mat_comb              = np.array(data_in["mat_comb"],dtype="float")
    index_ii              = data_in["index_ii"]
    calc_coin_tot         = str(data_in["calc_coin_tot"])
    data_folder           = str(data_in["data_folder"])
    yminbar               = float(data_in["yminbar"])
    ymaxbar               = float(data_in["ymaxbar"])
    ynumbar               = int(data_in["ynumbar"])
    ylabelbar             = str(data_in["ylabelbar"])
    mat_comb[mat_comb==0] = np.nan
    fig_name0             = plot_coin_folder+'/'+plot_coin_file
    fig_name1             = plot_coin_folder+'/percbar_'+plot_coin_file
    
     
    xx,zz       = np.meshgrid(flowfield.xplus,flowfield.zplus)
    xmin        = 0
    xmax        = flowfield.L_x*flowfield.rey
    ymin        = 0
    ymax        = flowfield.L_z*flowfield.rey
    yplustot    = (1-abs(flowfield.y_h))*flowfield.rey
    
    # -------------------------------------------------------------------------------------------------------------------
    # Define the colors
    # -------------------------------------------------------------------------------------------------------------------
    colors   = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8',
                '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                '#bcf60c', '#fabebe', '#008080', '#e6beff',
                '#9a6324', '#fffac8', '#800000']
    lab_list = [struc1_lab+"\\("+struc2_lab+" $\cup$ "+struc3_lab+" $\cup$ "+struc4_lab+")",
                struc2_lab+"\\("+struc1_lab+" $\cup$ "+struc3_lab+" $\cup$ "+struc4_lab+")",
                "("+struc2_lab+" $\cup$ "+struc1_lab+")\\("+struc3_lab+" $\cup$ "+struc4_lab+")",
                struc3_lab+"\\("+struc1_lab+" $\cup$ "+struc2_lab+" $\cup$ "+struc4_lab+")",
                "("+struc3_lab+" $\cup$ "+struc1_lab+")\\("+struc2_lab+" $\cup$ "+struc4_lab+")",
                "("+struc3_lab+" $\cup$ "+struc2_lab+")\\("+struc1_lab+" $\cup$ "+struc4_lab+")",
                "("+struc3_lab+" $\cup$ "+struc2_lab+" $\cup$ "+struc1_lab+")\\("+struc4_lab+")",
                struc4_lab+"\\("+struc1_lab+" $\cup$ "+struc2_lab+" $\cup$ "+struc3_lab+")",
                "("+struc4_lab+" $\cup$ "+struc1_lab+")\\("+struc2_lab+" $\cup$ "+struc3_lab+")",
                "("+struc4_lab+" $\cup$ "+struc2_lab+")\\("+struc1_lab+" $\cup$ "+struc3_lab+")",
                "("+struc4_lab+" $\cup$ "+struc2_lab+" $\cup$ "+struc1_lab+")\\("+struc3_lab+")",
                "("+struc4_lab+" $\cup$ "+struc3_lab+")\\("+struc1_lab+" $\cup$ "+struc2_lab+")",
                "("+struc4_lab+" $\cup$ "+struc3_lab+" $\cup$ "+struc1_lab+")\\("+struc2_lab+")",
                "("+struc4_lab+" $\cup$ "+struc3_lab+" $\cup$ "+struc2_lab+")\\("+struc1_lab+")",
                struc4_lab+" $\cup$ "+struc3_lab+" $\cup$ "+struc2_lab+" $\cup$ "+struc1_lab]
    colormap = ListedColormap(colors)
    
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    try:
        os.mkdir(plot_folder+'/'+plot_coin_folder)
    except:
        print("Existing folder...",flush=True)
    # -------------------------------------------------------------------------------------------------------------------
    # Read the coincidence file
    # -------------------------------------------------------------------------------------------------------------------
    data_coinc     = read_coinc_all_4types(data_in={"calc_coin_file":calc_coin_tot,"folder":data_folder})
    frac_coinc_tot = data_coinc["frac_coinc_tot"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create a legend with a figure
    # -------------------------------------------------------------------------------------------------------------------
    fig_name_leg = plot_coin_folder+'/legend_'+plot_coin_file
    data_plot    = {"xlabel":[],"ylabel":[],"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x-2,
                    "figsize_y":figsize_y+2,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                    "colornum":colornum,"legend":True,"fig_name":fig_name_leg,"dpi":dpi,"plot_folder":plot_folder,
                    "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure()
    plot_pred.legend_infig(data_in={"colors":colors,"labels":lab_list,"barstype":"horizontal"})
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    plot_pred.close()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Do it for all the wall distances
    # -------------------------------------------------------------------------------------------------------------------
    for index_y in np.arange(len(mat_comb[:,0,0])):
        titlefig    = "$y^+\simeq$"+'{0:.0f}'.format(yplustot[index_y])
        fig_name    = fig_name0+"_field"+str(index_ii)+"_y"+str(index_y)
        # ---------------------------------------------------------------------------------------------------------------
        # Create the plot
        # ---------------------------------------------------------------------------------------------------------------
        data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                      "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                      "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                      "xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"zmin":None,"zmax":None}
        plot_pred = plot_format(data_in=data_plot)
        plot_pred.create_figure()
        plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":mat_comb[index_y,:,:],"colormap":colormap,
                                      "plot_number_x":0,"plot_number_y":0,"vmax":15,"vmin":1,"Ncolor":None})
        plot_pred.plot_layout_pcolor(data_in={"title":titlefig,"colorbar":False,"b_text":None,
                                              "colorticks":[],
                                              "colorlabels":[],
                                              "equal":True,
                                              "xticks":[0,
                                                        flowfield.L_x/4*flowfield.rey,
                                                        2*flowfield.L_x/4*flowfield.rey,
                                                        3*flowfield.L_x/4*flowfield.rey,
                                                        flowfield.L_x*flowfield.rey],
                                              "yticks":[0,
                                                        flowfield.L_z/2*flowfield.rey,
                                                        flowfield.L_z*flowfield.rey],
                                              "xticklabels":["0",
                                                             str(int(np.round(flowfield.L_x/np.pi/4*
                                                                              flowfield.rey)))+"$\pi$",
                                                             str(int(np.round(2*flowfield.L_x/np.pi/4*
                                                                              flowfield.rey)))+"$\pi$",
                                                             str(int(np.round(3*flowfield.L_x/np.pi/4*
                                                                              flowfield.rey)))+"$\pi$",
                                                             str(int(np.round(flowfield.L_x/np.pi*
                                                                              flowfield.rey)))+"$\pi$"],
                                              "yticklabels":["0",
                                                             str(int(np.round(flowfield.L_z/np.pi/2*
                                                                              flowfield.rey)))+"$\pi$",
                                                             str(int(np.round(flowfield.L_z/np.pi*
                                                                              flowfield.rey)))+"$\pi$"]})
        plot_pred.plot_save_png()
        plot_pred.plot_save_pdf()
        plot_pred.close()
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create a colorbar with the percentage of each type of structure
        # ---------------------------------------------------------------------------------------------------------------
        fig_name       = fig_name1+"_field"+str(index_ii)+"_y"+str(index_y)
        frac_coinc_yii = np.zeros((len(frac_coinc_tot)))
        for ii_coinc in np.arange(len(frac_coinc_tot)):
            if index_y <= len(frac_coinc_tot[ii_coinc])-1:
                index_y_mod          = index_y
            else:
                index_y_mod          = len(frac_coinc_tot[ii_coinc])-index_y-2
            frac_coinc_yii[ii_coinc] = frac_coinc_tot[ii_coinc][index_y_mod]
        frac_coinc_shap = np.sum(frac_coinc_yii[::2])
        frac_coinc_yii /= frac_coinc_shap
        data_plot  = {"xlabel":xlabel,"ylabel":ylabelbar,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x+6,
                      "figsize_y":figsize_y+6,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                      "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,
                      "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":yminbar,"ymax":ymaxbar,
                      "zmin":None,"zmax":None}
        plot_pred = plot_format(data_in=data_plot)
        plot_pred.create_figure()
        plot_pred.add_cust_colorbar(data_in={"labels":lab_list,"data":frac_coinc_yii,"colors":colors,"title":titlefig,
                                             "ynum":ynumbar,"posplot":[0.15,0.95,0.8,0.6]})
        plot_pred.plot_save_png()
        plot_pred.plot_save_pdf()
        plot_pred.close()
    
      
def plot_coinc_mat_all_3types_withcontour(data_in={"plot_folder":"plots","plot_coin_folder":"uv_shap_coin_y",
                                                   "plot_coin_file":"uv_shap_coin_y","xlabel":"x","ylabel":"z",
                                                   "fontsize":18,"figsize_x":10,"figsize_y":8,"colornum":2,"dpi":60,
                                                   "struc1_lab":"uv","struc2_lab":"streak","struc3_lab":"chong",
                                                   "flowfield":[],"mat_comb":[],"index_ii":0,
                                                   "calc_coin_tot":"calc.txt","data_folder":"data","yminbar":0,
                                                   "ymaxbar":1,"ynumbar":4,"ylabelbar":"-","contourfield":[],
                                                   "linewidth":1,"xmin":None,"xmax":None,"zmin":None,"zmax":None,
                                                   "zoombox":[0,1,0,1]}):
    """
    .....................................................................................................................
    # plot_coinc_mat_all_3types_withcontour: Function to plot the coincidence between all the structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"plot_folder":"plots","plot_coin_folder":"uv_shap_coin_y",
                        "plot_coin_file":"uv_shap_coin_y","xlabel":"x","ylabel":"z","fontsize":18,
                        "figsize_x":10,"figsize_y":8,"colornum":2,"dpi":60,
                        "struc1_lab":"uv","struc2_lab":"streak","struc3_lab":"chong",
                        "flowfield":[],"mat_comb":[],"index_ii":0,"calc_coin_tot":"calc.txt",
                        "data_folder":"data","ylabelbar":"-","contourfield":[]}.
        Data:
            - plot_folder      : folder to store the plots
            - plot_coin_folder : folder to save the coincidence plots between structures
            - plot_coin_file   : file to save the coincidence plots between structures
            - xlabel           : label of the x axis
            - ylabel           : label of the y axis
            - fontsize         : font size used for the figure
            - figsize_x        : size of the figure in x
            - figsize_y        : size of the figure in y
            - colornum         : number of colors of the colormap, two curves are used. The number of levels of the 
                                 colormap needs to be higher than 2 
            - dpi              : dots per inch of the saved figure
            - struc1_lab       : label of the structure 1
            - struc2_lab       : label of the structure 2
            - struc3_lab       : label of the structure 3
            - flowfield        : data of the flow field
            - mat_comb         : matrix with the combination of the structures
            - index_ii         : index of the fields
            - calc_coin_tot    : file to read the coincidence between structures
            - data_folder      : folder to read data files
            - yminbar          : minimum value of the y axis of the bar plot
            - ymaxbar          : maximum value of the y axis of the bar plot
            - ynumbar          : number of ticks of the bar plot
            - ylabelbar        : label for the y axis of the bar plot
            - contourfield     : field used for the contours
            - linewidth        : linewidth of the contours
            - x_min            : Minimum position in x
            - x_max            : Maximum position in x
            - z_min            : Minimum position in z
            - z_max            : Maximum position in z
            - zoombox          : Box to print the zoom region [xmin,xmax,zmin,zmax]

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from matplotlib.colors import ListedColormap
    from py_bin.py_functions.calc_coinc import read_coinc_all_3types
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    plot_folder           = str(data_in["plot_folder"])
    plot_coin_folder      = str(data_in["plot_coin_folder"])
    plot_coin_file        = str(data_in["plot_coin_file"])
    xlabel                = str(data_in["xlabel"])
    ylabel                = str(data_in["ylabel"])
    fontsize              = int(data_in["fontsize"])
    figsize_x             = int(data_in["figsize_x"])
    figsize_y             = int(data_in["figsize_y"])
    colornum              = int(data_in["colornum"])
    dpi                   = float(data_in["dpi"])
    struc1_lab            = str(data_in["struc1_lab"])
    struc2_lab            = str(data_in["struc2_lab"])
    struc3_lab            = str(data_in["struc3_lab"])
    flowfield             = data_in["flowfield"]
    mat_comb              = np.array(data_in["mat_comb"],dtype="float")
    index_ii              = data_in["index_ii"]
    calc_coin_tot         = str(data_in["calc_coin_tot"])
    data_folder           = str(data_in["data_folder"])
    yminbar               = float(data_in["yminbar"])
    ymaxbar               = float(data_in["ymaxbar"])
    ynumbar               = int(data_in["ynumbar"])
    ylabelbar             = str(data_in["ylabelbar"])
    contourfield          = data_in["contourfield"]
    if "zoombox" in data_in:
        flagzoombox       = True
        zoombox           = np.array(data_in["zoombox"],dtype="int")
    else:
        flagzoombox       = False
    mat_comb[mat_comb==0] = np.nan
    fig_name0             = plot_coin_folder+'/'+plot_coin_file
    fig_name1             = plot_coin_folder+'/percbar_'+plot_coin_file
    linewidth             = float(data_in["linewidth"])
    if "x_min" in data_in.keys():
        x_min_input       = data_in["x_min"]
    else:
        x_min_input       = None
    if "x_max" in data_in.keys():
        x_max_input       = data_in["x_max"]
    else:
        x_max_input       = None
    if "z_min" in data_in.keys():
        z_min_input       = data_in["z_min"]
    else:
        z_min_input       = None
    if "z_max" in data_in.keys():
        z_max_input       = data_in["z_max"]
    else:
        z_max_input       = None
    
    if x_min_input is not None:
        x_min = int(x_min_input)
    else:
        x_min = 0
    if x_max_input is not None:
        x_max = int(x_max_input)
    else:
        x_max = int(flowfield.shpx)
    if z_min_input is not None:
        z_min = int(z_min_input)
    else:
        z_min = 0
    if z_max_input is not None:
        z_max = int(z_max_input)
    else:
        z_max = int(flowfield.shpz)
    
     
    xx,zz       = np.meshgrid(flowfield.xplus[x_min:x_max],flowfield.zplus[z_min:z_max])
    xmin        = flowfield.xplus[x_min]
    xmax        = flowfield.xplus[x_max-1]
    zmin        = flowfield.zplus[z_min]
    zmax        = flowfield.zplus[z_max-1]
    yplustot    = (1-abs(flowfield.y_h))*flowfield.rey
    
    # -------------------------------------------------------------------------------------------------------------------
    # Define the colors
    # -------------------------------------------------------------------------------------------------------------------
    colors   = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8',
                '#f58231', '#911eb4', '#46f0f0']
    lab_list = [struc1_lab+"\\("+struc2_lab+" $\cup$ "+struc3_lab+")",
                struc2_lab+"\\("+struc1_lab+" $\cup$ "+struc3_lab+")",
                "("+struc2_lab+" $\cup$ "+struc1_lab+")\\("+struc3_lab+")",
                struc3_lab+"\\("+struc1_lab+" $\cup$ "+struc2_lab+")",
                "("+struc3_lab+" $\cup$ "+struc1_lab+")\\("+struc2_lab+")",
                "("+struc3_lab+" $\cup$ "+struc2_lab+")\\("+struc1_lab+")",
                struc3_lab+" $\cup$ "+struc2_lab+" $\cup$ "+struc1_lab]
    colormap = ListedColormap(colors)
    
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    try:
        os.mkdir(plot_folder+'/'+plot_coin_folder)
    except:
        print("Existing folder...",flush=True)
    # -------------------------------------------------------------------------------------------------------------------
    # Read the coincidence file
    # -------------------------------------------------------------------------------------------------------------------
    data_coinc     = read_coinc_all_3types(data_in={"calc_coin_file":calc_coin_tot,"folder":data_folder})
    frac_coinc_tot = data_coinc["frac_coinc_tot"]
    
    mat_cont       = np.array(contourfield.mat_struc,dtype='int')[:,z_min:z_max,x_min:x_max]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create a legend with a figure
    # -------------------------------------------------------------------------------------------------------------------
    fig_name_leg = plot_coin_folder+'/legend_'+plot_coin_file
    data_plot    = {"xlabel":[],"ylabel":[],"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x-2,
                    "figsize_y":figsize_y+2,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                    "colornum":colornum,"legend":True,"fig_name":fig_name_leg,"dpi":dpi,"plot_folder":plot_folder,
                    "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure()
    plot_pred.legend_infig(data_in={"colors":colors,"labels":lab_list,"barstype":"horizontal"})
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    plot_pred.close()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Do it for all the wall distances
    # -------------------------------------------------------------------------------------------------------------------
    for index_y in np.arange(len(mat_comb[:,0,0])):
        titlefig    = "$y^+\simeq$"+'{0:.0f}'.format(yplustot[index_y])
        fig_name    = fig_name0+"_field"+str(index_ii)+"_y"+str(index_y)
        # ---------------------------------------------------------------------------------------------------------------
        # Create the plot
        # ---------------------------------------------------------------------------------------------------------------
        data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                      "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                      "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                      "xmin":xmin,"xmax":xmax,"ymin":zmin,"ymax":zmax,"zmin":None,"zmax":None}
        plot_pred = plot_format(data_in=data_plot)
        plot_pred.create_figure()
        plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":mat_comb[index_y,z_min:z_max,x_min:x_max],
                                      "colormap":colormap,
                                      "plot_number_x":0,"plot_number_y":0,"vmax":7,"vmin":1,"Ncolor":None})
        plot_pred.add_contline(data_in={"data_x":xx,"data_y":zz,"data_cont":mat_cont[index_y,:,:],"val":0.5,
                                        "colors":"k","linewidth":linewidth})
        if x_min_input is None and x_max_input is None and z_min_input is None and z_max_input is None:
            plot_pred.plot_layout_pcolor(data_in={"title":titlefig,"colorbar":False,"b_text":None,
                                                  "colorticks":[],
                                                  "colorlabels":[],
                                                  "equal":True,
                                                  "xticks":[xmin,
                                                            (xmax-xmin)/4+xmin,
                                                            2*(xmax-xmin)/4+xmin,
                                                            3*(xmax-xmin)/4+xmin,
                                                            xmax],
                                                  "yticks":[zmin,
                                                            (zmax-zmin)/2+zmin,
                                                            zmax],
                                                  "xticklabels":[str(int(np.round(xmin)/np.pi)), 
                                                                 str(int(np.round(((xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                                 str(int(np.round((2*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                                 str(int(np.round((3*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                                 str(int(np.round((xmax)/np.pi)))+"$\pi$"],
                                                  "yticklabels":[str(int(np.round((zmin)/np.pi))),
                                                                 str(int(np.round(((zmax-zmin)/2+zmin)/np.pi)))+"$\pi$",
                                                                 str(int(np.round(((zmax))/np.pi)))+"$\pi$"]})
        else:
            plot_pred.plot_layout_pcolor(data_in={"title":titlefig,"colorbar":False,"b_text":None,
                                                  "colorticks":[],
                                                  "colorlabels":[],
                                                  "equal":True,
                                                  "xticks":[(xmax-xmin)/3+xmin,
                                                            2*(xmax-xmin)/3+xmin,
                                                            xmax],
                                                  "yticks":[zmin,
                                                            (zmax-zmin)/2+zmin,
                                                            zmax],
                                                  "xticklabels":[str(int(np.round((xmax-xmin)/3+xmin))),
                                                                 str(int(np.round(2*(xmax-xmin)/3+xmin))),
                                                                 str(int(np.round(xmax)))],
                                                  "yticklabels":[str(int(np.round(zmin))),
                                                                 str(int(np.round((zmax-zmin)/2+zmin))),
                                                                 str(int(np.round((zmax))))]})
        
        if flagzoombox == True:
            plot_info1 = {"data_x":np.array([flowfield.xplus[zoombox[0]],flowfield.xplus[zoombox[1]]]),
                          "data_y":np.array([flowfield.zplus[zoombox[2]],flowfield.zplus[zoombox[2]]]),
                          "label":"Training Loss","color":"k","linewidth":2,"plot_number":0,"style":"--"}
            plot_info2 = {"data_x":np.array([flowfield.xplus[zoombox[0]],flowfield.xplus[zoombox[1]]]),
                          "data_y":np.array([flowfield.zplus[zoombox[3]],flowfield.zplus[zoombox[3]]]),
                          "label":"Training Loss","color":"k","linewidth":2,"plot_number":0,"style":"--"}
            plot_info3 = {"data_x":np.array([flowfield.xplus[zoombox[0]],flowfield.xplus[zoombox[0]]]),
                          "data_y":np.array([flowfield.zplus[zoombox[2]],flowfield.zplus[zoombox[3]]]),
                          "label":"Training Loss","color":"k","linewidth":2,"plot_number":0,"style":"--"}
            plot_info4 = {"data_x":np.array([flowfield.xplus[zoombox[1]],flowfield.xplus[zoombox[1]]]),
                          "data_y":np.array([flowfield.zplus[zoombox[2]],flowfield.zplus[zoombox[3]]]),
                          "label":"Training Loss","color":"k","linewidth":2,"plot_number":0,"style":"--"}
            plot_pred.add_plot_2d(data_in=plot_info1)
            plot_pred.add_plot_2d(data_in=plot_info2)
            plot_pred.add_plot_2d(data_in=plot_info3)
            plot_pred.add_plot_2d(data_in=plot_info4)
        plot_pred.plot_save_png()
        plot_pred.plot_save_pdf()
        plot_pred.close()
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create a colorbar with the percentage of each type of structure
        # ---------------------------------------------------------------------------------------------------------------
        fig_name       = fig_name1+"_field"+str(index_ii)+"_y"+str(index_y)
        frac_coinc_yii = np.zeros((len(frac_coinc_tot)))
        for ii_coinc in np.arange(len(frac_coinc_tot)):
            if index_y <= len(frac_coinc_tot[ii_coinc])-1:
                index_y_mod          = index_y
            else:
                index_y_mod          = len(frac_coinc_tot[ii_coinc])-index_y-2
            frac_coinc_yii[ii_coinc] = frac_coinc_tot[ii_coinc][index_y_mod]
        frac_coinc_shap = np.sum(frac_coinc_yii[::2])
        frac_coinc_yii /= frac_coinc_shap
        data_plot  = {"xlabel":xlabel,"ylabel":ylabelbar,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x+6,
                      "figsize_y":figsize_y+6,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                      "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,
                      "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":yminbar,"ymax":ymaxbar,
                      "zmin":None,"zmax":None}
        plot_pred = plot_format(data_in=data_plot)
        plot_pred.create_figure()
        plot_pred.add_cust_colorbar(data_in={"labels":lab_list,"data":frac_coinc_yii,"colors":colors,"title":titlefig,
                                             "ynum":ynumbar,"posplot":[0.15,0.95,0.8,0.6]})
        plot_pred.plot_save_png()
        plot_pred.plot_save_pdf()
        plot_pred.close()
    