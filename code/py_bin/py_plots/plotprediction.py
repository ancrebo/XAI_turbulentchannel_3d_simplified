# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plotprediction.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella

File to plot the prediction of the fields. The file contains the following functions:
    - Functions:
        - plotprediction : function to plot the predictions
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

def plotprediction(data_in={"plot_folder":"plots","xlabel":"Epoch","ylabel":"Loss function (-)",
                            "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                            "fig_name":"training_info","dpi":60,"index_ii":1000,"Unet":[],"flowfield":[],
                            "index_y":12,"xmin":0,"xmax":125,"ymin":0,"ymax":125,"errmax":0,"errmin":1,
                            "b_velo_sim":"$u_s^+$","b_velo_pred":"$u_p^+$","b_velo_err":"$\epsilon_u$"}):
    """
    .....................................................................................................................
    # plotprediction: Function to generate the plot of the predicted fields
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"plot_folder":"plots","xlabel":"Epoch","ylabel":"Loss function (-)",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                        "fig_name":"training_info","dpi":60,"index_ii":1000,"Unet":[],
                        "index_y":12,"xmin":0,"xmax":125,"ymin":0,"ymax":125,"errmax":0,"errmin":1,
                        "b_velo_sim":"$u_s^+$","b_velo_pred":"$u_p^+$","b_velo_err":"$\epsilon_u}.
        Data:
            - plot_folder : folder to store the plots
            - xlabel      : label of the x axis
            - ylabel      : label of the y axis
            - fontsize    : font size used for the figure
            - figsize_x   : size of the figure in x
            - figsize_y   : size of the figure in y
            - colormap    : colormap used for the figure
            - colornum    : number of colors of the colormap, two curves are used. The number of levels of the 
                            colormap needs to be higher than 2 
            - fig_name    : name of the saved figure
            - dpi         : dots per inch of the saved figure
            - index_ii    : index of the field
            - Unet        : class of the unet
            - flowfield   : class of the flow data
            - index_y     : index in the wall-normal direction
            - xmin        : minimum value of the x axis
            - xmax        : maximum value of the x axis
            - ymin        : minimum value of the y axis
            - ymax        : maximum value of the y axis
            - errmax      : maximum value of the error in the colorbar
            - errmin      : minimum value of the error in the colorbar
            - b_velo_sim  : text for the bar of the simulated velocity
            - b_velo_pred : text for the bar of the predicted velocity
            - b_velo_err  : text for the bar of the error of the velocity

    Returns
    -------
    None.

    """
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from py_bin.py_class.plot_format import plot_format
    from py_bin.py_functions.normalization import read_norm
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    plot_folder = str(data_in["plot_folder"])                      # plot to store the figures
    xlabel      = str(data_in["xlabel"])                           # label of the x axis
    ylabel      = str(data_in["ylabel"])                           # label of the y axis
    fontsize    = int(data_in["fontsize"])                         # size of the text in the plot
    figsize_x   = int(data_in["figsize_x"])                        # size of the figure in direction x
    figsize_y   = int(data_in["figsize_y"])                        # size of the figure in direction y
    colormap    = str(data_in["colormap"])                         # colormap of the figure
    colornum    = int(data_in["colornum"])                         # number of colors of the colormap
    fig_name    = str(data_in["fig_name"])                         # name of the figure to be saved
    dpi         = float(data_in["dpi"])                            # dots per inch to save the figure
    index_ii    = int(data_in["index_ii"])                         # index of the field
    Unet        = data_in["Unet"]                                  # data of the Unet
    flowfield   = data_in["flowfield"]                             # data of the flow field
    index_y     = int(data_in["index_y"])                          # index used in the wall-normal direction
    data_error  = Unet.field_error(data_in={"index_ii":index_ii})
    xmin        = float(data_in["xmin"])
    xmax        = float(data_in["xmax"])
    ymin        = float(data_in["ymin"])
    ymax        = float(data_in["ymax"])
    errmax      = float(data_in["errmax"])
    errmin      = float(data_in["errmin"])
    b_velo_sim  = str(data_in["b_velo_sim"])
    b_velo_pred = str(data_in["b_velo_pred"])
    b_velo_err  = str(data_in["b_velo_err"])
    
    data_norm   = read_norm(data_in={"folder":Unet.data_folder,"file":Unet.umax_file})
    umax        = data_norm["uumax"]
    umin        = data_norm["uumin"]
    erru        = np.sum(np.multiply(data_error["err_u"],flowfield.vol_plus))/np.sum(flowfield.vol_plus)
    errv        = np.sum(np.multiply(data_error["err_v"],flowfield.vol_plus))/np.sum(flowfield.vol_plus)
    errw        = np.sum(np.multiply(data_error["err_w"],flowfield.vol_plus))/np.sum(flowfield.vol_plus)
 
    xx,zz       = np.meshgrid(flowfield.xplus,flowfield.zplus)
    yyplus      = "y^+ = "+'{0:.2f}'.format(flowfield.yplus[index_y])+\
        ", $\epsilon_u$: "+'{0:.2f}'.format(erru*100)+'%'+", $\epsilon_v$: "+'{0:.2f}'.format(errv*100)+'%'+\
            ", $\epsilon_w$: "+'{0:.2f}'.format(errw*100)+'%'
            
    fig_name    = fig_name+"_field"+str(index_ii)+"_y"+'{0:.0f}'.format(flowfield.yplus[index_y])
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure_multiplot(data_in={"row":3,"col":2})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":data_error["sim_u"][index_y,:,:],
                                  "colormap":None,"plot_number_x":0,"plot_number_y":0,"vmax":umax,"vmin":umin,
                                  "Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":0,"indexplot":0,"b_text":b_velo_sim,
                                        "title":yyplus,"colorbar_x":True})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":data_error["sim_u"][-index_y,:,:],
                                  "colormap":None,"plot_number_x":1,"plot_number_y":0,"vmax":umax,"vmin":umin,
                                  "Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":1,"plot_number_y":0,"indexplot":1,"b_text":b_velo_sim,
                                        "title":yyplus,"colorbar_x":True})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":data_error["pre_u"][index_y,:,:],
                                  "colormap":None,"plot_number_x":0,"plot_number_y":1,"vmax":umax,"vmin":umin,
                                  "Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":1,"indexplot":2,"b_text":b_velo_pred,
                                        "title":yyplus,"colorbar_x":True})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":data_error["pre_u"][-index_y,:,:],
                                  "colormap":None,"plot_number_x":1,"plot_number_y":1,"vmax":umax,"vmin":umin,
                                  "Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":1,"plot_number_y":1,"indexplot":3,"b_text":b_velo_pred,
                                        "title":yyplus,"colorbar_x":True})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":data_error["err_u"][index_y,:,:],
                                  "colormap":None,"plot_number_x":0,"plot_number_y":2,"vmax":errmax,"vmin":errmin,
                                  "Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":2,"indexplot":4,"b_text":b_velo_err,
                                        "title":yyplus,"colorbar_x":True})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":data_error["err_u"][-index_y,:,:],
                                  "colormap":None,"plot_number_x":1,"plot_number_y":2,"vmax":errmax,"vmin":errmin,
                                  "Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":1,"plot_number_y":2,"indexplot":5,"b_text":b_velo_err,
                                        "title":yyplus,"colorbar_x":True})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    