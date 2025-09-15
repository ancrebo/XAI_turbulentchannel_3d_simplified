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

def plotshap(data_in={"plot_folder":"plots","xlabel":"Epoch","ylabel":"Loss function (-)",
                      "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                      "fig_name":"training_info","dpi":60,"index_ii":1000,"shap_data":[],"vel_data":[],"flowfield":[],
                      "index_y":12,"xmin":0,"xmax":125,"ymin":0,"ymax":125,"b_velo_u":"$u$","b_shap_u":"$\phi$",
                      "b_velo_v":"$v$","b_shap_v":"$\phi$","w":"$u^+$","b_shap_w":"$\phi$","b_velo_m":"u",
                      "b_shap_m":"$\phi$","padding":15,"scale_shap":1e5,"colormapsh":'PRGn',"normfield":True}):
    """
    .....................................................................................................................
    # plotshap: Function to generate the plot of the shap values
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"plot_folder":"plots","xlabel":"Epoch","ylabel":"Loss function (-)",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                        "fig_name":"training_info","dpi":60,"index_ii":1000,"shap_data":[],"vel_data":[],"flowfield":[],
                        "index_y":12,"xmin":0,"xmax":125,"ymin":0,"ymax":125,"b_velo_u":"$u$","b_shap_u":"$\phi$",
                        "b_velo_v":"$v$","b_shap_v":"$\phi$","w":"$u^+$","b_shap_w":"$\phi$","b_velo_m":"u",
                        "b_shap_m":"$\phi$","padding":15,"scale_shap":1e5,"colormapsh":'PRGn',"normfield":True}.
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
            - shap_data   : class of the shap values
            - vel_data    : class of the velocity
            - flowfield   : class of the flow data
            - index_y     : index in the wall-normal direction
            - xmin        : minimum value of the x axis
            - xmax        : maximum value of the x axis
            - ymin        : minimum value of the y axis
            - ymax        : maximum value of the y axis
            - b_velo_u    : text for the bar of the streamwise velocity
            - b_shap_u    : text for the bar of the streamwise shap
            - b_velo_v    : text for the bar of the wall_normal velocity
            - b_shap_v    : text for the bar of the wall_normal shap
            - b_velo_w    : text for the bar of the spanwise velocity
            - b_shap_w    : text for the bar of the spanwise shap
            - b_velo_m    : text for the bar of the magnitude of the velocity
            - b_shap_m    : text for the bar of the magnitude of the shap
            - padding     : padding of the domain
            - scale_shap  : value to scale the shap values
            - colormapsh  : colormap used for the shap
            - normfield   : normalization of the field (True: use field normalization, False: use slice normalization)

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
    shap_data   = data_in["shap_data"]                             # data of the shap values
    vel_data    = data_in["vel_data"]                              # data of the velocity
    flowfield   = data_in["flowfield"]                             # data of the flow field
    index_y     = int(data_in["index_y"])                          # index used in the wall-normal direction
    xmin        = float(data_in["xmin"])
    xmax        = float(data_in["xmax"])
    ymin        = float(data_in["ymin"])
    ymax        = float(data_in["ymax"])
    scale_shap  = float(data_in["scale_shap"])
    b_velo_u    = str(data_in["b_velo_u"])
    b_shap_u    = str(data_in["b_shap_u"])+r" $\times 10^{"+"{:.1f}".format(np.log10(scale_shap))+"}$"
    b_velo_v    = str(data_in["b_velo_v"])
    b_shap_v    = str(data_in["b_shap_v"])+r" $\times 10^{"+"{:.1f}".format(np.log10(scale_shap))+"}$"
    b_velo_w    = str(data_in["b_velo_w"])
    b_shap_w    = str(data_in["b_shap_w"])+r" $\times 10^{"+"{:.1f}".format(np.log10(scale_shap))+"}$" 
    b_velo_m    = str(data_in["b_velo_m"])
    b_shap_m    = str(data_in["b_shap_m"])+r" $\times 10^{"+"{:.1f}".format(np.log10(scale_shap))+"}$" 
    padding     = int(data_in["padding"])
    colormapsh  = str(data_in["colormapsh"])
    normfield   = bool(data_in["normfield"])
    shap_mod    = np.sqrt(shap_data["SHAP_u"]**2+
                          shap_data["SHAP_v"]**2+
                          shap_data["SHAP_w"]**2)
    vel_mod     = np.sqrt(vel_data["uu"]**2+
                          vel_data["vv"]**2+
                          vel_data["ww"]**2)
    if normfield:
        max_u       = np.max(np.abs([np.max(shap_data["SHAP_u"]),np.min(shap_data["SHAP_u"])]))
        max_v       = np.max(np.abs([np.max(shap_data["SHAP_v"]),np.min(shap_data["SHAP_v"])]))
        max_w       = np.max(np.abs([np.max(shap_data["SHAP_w"]),np.min(shap_data["SHAP_w"])]))
        vmax_u      = max_u*scale_shap
        vmin_u      = -max_u*scale_shap
        vmax_v      = max_v*scale_shap
        vmin_v      = -max_v*scale_shap
        vmax_w      = max_w*scale_shap
        vmin_w      = -max_w*scale_shap
        vmax_m      = np.max(shap_mod)*scale_shap
        vmin_m      = np.min(shap_mod)*scale_shap
        vmax_vel_u  = np.max(vel_data["uu"])
        vmin_vel_u  = np.min(vel_data["uu"])
        vmax_vel_v  = np.max(vel_data["vv"])
        vmin_vel_v  = np.min(vel_data["vv"])
        vmax_vel_w  = np.max(vel_data["ww"])
        vmin_vel_w  = np.min(vel_data["ww"])
        vmax_vel_m  = np.max(vel_mod)
        vmin_vel_m  = np.min(vel_mod)
    else:
        max_u       = np.max(np.abs([np.max(shap_data["SHAP_u"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_u"][index_y,padding:-padding,padding:-padding])]))
        max_v       = np.max(np.abs([np.max(shap_data["SHAP_v"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_v"][index_y,padding:-padding,padding:-padding])]))
        max_w       = np.max(np.abs([np.max(shap_data["SHAP_w"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_w"][index_y,padding:-padding,padding:-padding])]))
        vmax_u      = max_u*scale_shap
        vmin_u      = -max_u*scale_shap
        vmax_v      = max_v*scale_shap
        vmin_v      = -max_v*scale_shap
        vmax_w      = max_w*scale_shap
        vmin_w      = -max_w*scale_shap
        vmax_m      = np.max(shap_mod[index_y,padding:-padding,padding:-padding])*scale_shap
        vmin_m      = np.min(shap_mod[index_y,padding:-padding,padding:-padding])*scale_shap
        vmax_vel_u  = np.max(vel_data["uu"][index_y,padding:-padding,padding:-padding])
        vmin_vel_u  = np.min(vel_data["uu"][index_y,padding:-padding,padding:-padding])
        vmax_vel_v  = np.max(vel_data["vv"][index_y,padding:-padding,padding:-padding])
        vmin_vel_v  = np.min(vel_data["vv"][index_y,padding:-padding,padding:-padding])
        vmax_vel_w  = np.max(vel_data["ww"][index_y,padding:-padding,padding:-padding])
        vmin_vel_w  = np.min(vel_data["ww"][index_y,padding:-padding,padding:-padding])
        vmax_vel_m  = np.max(vel_mod[index_y,padding:-padding,padding:-padding])
        vmin_vel_m  = np.min(vel_mod[index_y,padding:-padding,padding:-padding])
    
     
    xx,zz       = np.meshgrid(flowfield.xplus,flowfield.zplus)
    titlefig    = "$y^+ = "+'{0:.2f}'.format(flowfield.yplus[index_y])+"$"
            
    fig_name    = fig_name+"_field"+str(index_ii)+"_y"+'{0:.0f}'.format(flowfield.yplus[index_y])
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure_multiplot(data_in={"row":4,"col":2})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":shap_data["SHAP_u"][index_y,
                                                                                           padding:-padding,
                                                                                           padding:-padding]*scale_shap,
                                  "colormap":colormapsh,"plot_number_x":0,"plot_number_y":0,"vmax":vmax_u,
                                  "vmin":vmin_u,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":0,"indexplot":0,"b_text":b_shap_u,
                                        "title":titlefig,"colorbar_x":False})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":shap_data["SHAP_v"][index_y,
                                                                                           padding:-padding,
                                                                                           padding:-padding]*scale_shap,
                                  "colormap":colormapsh,"plot_number_x":0,"plot_number_y":1,"vmax":vmax_v,
                                  "vmin":vmin_v,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":1,"indexplot":1,"b_text":b_shap_v,
                                        "title":titlefig,"colorbar_x":False})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":shap_data["SHAP_w"][index_y,
                                                                                           padding:-padding,
                                                                                           padding:-padding]*scale_shap,
                                  "colormap":colormapsh,"plot_number_x":0,"plot_number_y":2,"vmax":vmax_w,
                                  "vmin":vmin_w,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":2,"indexplot":2,"b_text":b_shap_w,
                                        "title":titlefig,"colorbar_x":False})
    shap_data_m = shap_mod[index_y,padding:-padding,padding:-padding]*scale_shap
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":shap_data_m,
                                  "colormap":None,"plot_number_x":0,"plot_number_y":3,"vmax":vmax_m,
                                  "vmin":vmin_m,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":3,"indexplot":3,"b_text":b_shap_m,
                                        "title":titlefig,"colorbar_x":False})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":vel_data["uu"][index_y,
                                                                                      padding:-padding,
                                                                                      padding:-padding],
                                  "colormap":None,"plot_number_x":1,"plot_number_y":0,"vmax":vmax_vel_u,
                                  "vmin":vmin_vel_u,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":1,"plot_number_y":0,"indexplot":4,"b_text":b_velo_u,
                                        "title":titlefig,"colorbar_x":False})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":vel_data["vv"][index_y,
                                                                                      padding:-padding,
                                                                                      padding:-padding],
                                  "colormap":None,"plot_number_x":1,"plot_number_y":1,"vmax":vmax_vel_v,
                                  "vmin":vmin_vel_v,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":1,"plot_number_y":1,"indexplot":5,"b_text":b_velo_v,
                                        "title":titlefig,"colorbar_x":False})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":vel_data["ww"][index_y,
                                                                                      padding:-padding,
                                                                                      padding:-padding],
                                  "colormap":None,"plot_number_x":1,"plot_number_y":2,"vmax":vmax_vel_w,
                                  "vmin":vmin_vel_w,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":1,"plot_number_y":2,"indexplot":6,"b_text":b_velo_w,
                                        "title":titlefig,"colorbar_x":False})
    vel_data_m  = vel_mod[index_y,padding:-padding,padding:-padding]
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":vel_data_m,
                                  "colormap":None,"plot_number_x":1,"plot_number_y":3,"vmax":vmax_vel_m,
                                  "vmin":vmin_vel_m,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":1,"plot_number_y":3,"indexplot":7,"b_text":b_velo_m,
                                        "title":titlefig,"colorbar_x":False})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    
    
    
def plotshap_noise(data_in={"plot_folder":"plots","xlabel":"Epoch","ylabel":"Loss function (-)",
                            "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                            "fig_name":"training_info","dpi":60,"index_ii":1000,"shap_data":[],"vel_data":[],"flowfield":[],
                            "index_y":12,"xmin":0,"xmax":125,"ymin":0,"ymax":125,"b_velo_u":"$u$","b_shap_u":"$\phi$",
                            "b_velo_v":"$v$","b_shap_v":"$\phi$","w":"$u^+$","b_shap_w":"$\phi$","b_velo_m":"u",
                            "b_shap_m":"$\phi$","padding":15,"scale_shap":1e5,"colormapsh":'PRGn',"normfield":True,
                            "nfield":21}):
    """
    .....................................................................................................................
    # plotshap_noise: Function to generate the plot of the shap values
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"plot_folder":"plots","xlabel":"Epoch","ylabel":"Loss function (-)",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                        "fig_name":"training_info","dpi":60,"index_ii":1000,"shap_data":[],"vel_data":[],"flowfield":[],
                        "index_y":12,"xmin":0,"xmax":125,"ymin":0,"ymax":125,"b_velo_u":"$u$","b_shap_u":"$\phi$",
                        "b_velo_v":"$v$","b_shap_v":"$\phi$","w":"$u^+$","b_shap_w":"$\phi$","b_velo_m":"u",
                        "b_shap_m":"$\phi$","padding":15,"scale_shap":1e5,"colormapsh":'PRGn',"normfield":True,
                        "nfield":"1 field for computing SHAP"}.
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
            - shap_data   : class of the shap values
            - vel_data    : class of the velocity
            - flowfield   : class of the flow data
            - index_y     : index in the wall-normal direction
            - xmin        : minimum value of the x axis
            - xmax        : maximum value of the x axis
            - ymin        : minimum value of the y axis
            - ymax        : maximum value of the y axis
            - b_velo_u    : text for the bar of the streamwise velocity
            - b_shap_u    : text for the bar of the streamwise shap
            - b_velo_v    : text for the bar of the wall_normal velocity
            - b_shap_v    : text for the bar of the wall_normal shap
            - b_velo_w    : text for the bar of the spanwise velocity
            - b_shap_w    : text for the bar of the spanwise shap
            - b_velo_m    : text for the bar of the magnitude of the velocity
            - b_shap_m    : text for the bar of the magnitude of the shap
            - padding     : padding of the domain
            - scale_shap  : value to scale the shap values
            - colormapsh  : colormap used for the shap
            - normfield   : normalization of the field (True: use field normalization, False: use slice normalization)
            - nfield      : title with the number of fields

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
    shap_data   = data_in["shap_data"]                             # data of the shap values
    vel_data    = data_in["vel_data"]                              # data of the velocity
    flowfield   = data_in["flowfield"]                             # data of the flow field
    index_y     = int(data_in["index_y"])                          # index used in the wall-normal direction
    xmin        = float(data_in["xmin"])
    xmax        = float(data_in["xmax"])
    ymin        = float(data_in["ymin"])
    ymax        = float(data_in["ymax"])
    scale_shap  = float(data_in["scale_shap"])
    b_velo_u    = str(data_in["b_velo_u"])
    b_shap_u    = str(data_in["b_shap_u"])+r" $\times 10^{"+"{:.1f}".format(np.log10(scale_shap))+"}$"
    b_velo_v    = str(data_in["b_velo_v"])
    b_shap_v    = str(data_in["b_shap_v"])+r" $\times 10^{"+"{:.1f}".format(np.log10(scale_shap))+"}$"
    b_velo_w    = str(data_in["b_velo_w"])
    b_shap_w    = str(data_in["b_shap_w"])+r" $\times 10^{"+"{:.1f}".format(np.log10(scale_shap))+"}$" 
    b_velo_m    = str(data_in["b_velo_m"])
    b_shap_m    = str(data_in["b_shap_m"])+r" $\times 10^{"+"{:.1f}".format(np.log10(scale_shap))+"}$" 
    padding     = int(data_in["padding"])
    colormapsh  = str(data_in["colormapsh"])
    normfield   = bool(data_in["normfield"])
    nfield      = str(data_in["nfield"])
    shap_mod    = np.sqrt(shap_data["SHAP_u"]**2+
                          shap_data["SHAP_v"]**2+
                          shap_data["SHAP_w"]**2)
    if normfield:
        max_u       = np.max(np.abs([np.max(shap_data["SHAP_u"]),np.min(shap_data["SHAP_u"])]))
        max_v       = np.max(np.abs([np.max(shap_data["SHAP_v"]),np.min(shap_data["SHAP_v"])]))
        max_w       = np.max(np.abs([np.max(shap_data["SHAP_w"]),np.min(shap_data["SHAP_w"])]))
        vmax_u      = max_u*scale_shap
        vmin_u      = -max_u*scale_shap
        vmax_v      = max_v*scale_shap
        vmin_v      = -max_v*scale_shap
        vmax_w      = max_w*scale_shap
        vmin_w      = -max_w*scale_shap
        vmax_m      = np.max(shap_mod)*scale_shap
        vmin_m      = np.min(shap_mod)*scale_shap
    else:
        max_u       = np.max(np.abs([np.max(shap_data["SHAP_u"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_u"][index_y,padding:-padding,padding:-padding])]))
        max_v       = np.max(np.abs([np.max(shap_data["SHAP_v"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_v"][index_y,padding:-padding,padding:-padding])]))
        max_w       = np.max(np.abs([np.max(shap_data["SHAP_w"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_w"][index_y,padding:-padding,padding:-padding])]))
        vmax_u      = max_u*scale_shap
        vmin_u      = -max_u*scale_shap
        vmax_v      = max_v*scale_shap
        vmin_v      = -max_v*scale_shap
        vmax_w      = max_w*scale_shap
        vmin_w      = -max_w*scale_shap
        vmax_m      = np.max(shap_mod[index_y,padding:-padding,padding:-padding])*scale_shap
        vmin_m      = np.min(shap_mod[index_y,padding:-padding,padding:-padding])*scale_shap
    
     
    xx,zz       = np.meshgrid(flowfield.xplus,flowfield.zplus)
    titlefig    = nfield
            
    fig_name    = fig_name+"_field"+str(index_ii)+"_y"+'{0:.0f}'.format(flowfield.yplus[index_y])
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure_multiplot(data_in={"row":4,"col":1})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":shap_data["SHAP_u"][index_y,
                                                                                           padding:-padding,
                                                                                           padding:-padding]*scale_shap,
                                  "colormap":colormapsh,"plot_number_x":0,"plot_number_y":0,"vmax":vmax_u,
                                  "vmin":vmin_u,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":0,"indexplot":0,"b_text":b_shap_u,
                                        "title":titlefig,"colorbar_x":False})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":shap_data["SHAP_v"][index_y,
                                                                                           padding:-padding,
                                                                                           padding:-padding]*scale_shap,
                                  "colormap":colormapsh,"plot_number_x":0,"plot_number_y":1,"vmax":vmax_v,
                                  "vmin":vmin_v,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":1,"indexplot":1,"b_text":b_shap_v,
                                        "title":titlefig,"colorbar_x":False})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":shap_data["SHAP_w"][index_y,
                                                                                           padding:-padding,
                                                                                           padding:-padding]*scale_shap,
                                  "colormap":colormapsh,"plot_number_x":0,"plot_number_y":2,"vmax":vmax_w,
                                  "vmin":vmin_w,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":2,"indexplot":2,"b_text":b_shap_w,
                                        "title":titlefig,"colorbar_x":False})
    shap_data_m = shap_mod[index_y,padding:-padding,padding:-padding]*scale_shap
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":shap_data_m,
                                  "colormap":None,"plot_number_x":0,"plot_number_y":3,"vmax":vmax_m,
                                  "vmin":vmin_m,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":3,"indexplot":3,"b_text":b_shap_m,
                                        "title":titlefig,"colorbar_x":False})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    
    
       
def plotshap_noise_normalized(data_in={"plot_folder":"plots","xlabel":"Epoch","ylabel":"Loss function (-)",
                                       "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                                       "fig_name":"training_info","dpi":60,"index_ii":1000,"shap_data":[],"vel_data":[],
                                       "flowfield":[],"index_y":12,"xmin":0,"xmax":125,"ymin":0,"ymax":125,
                                       "b_velo_u":"$u$","b_shap_u":"$\phi$","b_velo_v":"$v$","b_shap_v":"$\phi$",
                                       "w":"$u^+$","b_shap_w":"$\phi$","b_velo_m":"u","b_shap_m":"$\phi$","padding":15,
                                       "scale_shap":1e5,"colormapsh":'PRGn',"normfield":True,"nfield":21}):
    """
    .....................................................................................................................
    # plotshap_noise_normalized: Function to generate the plot of the shap values
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"plot_folder":"plots","xlabel":"Epoch","ylabel":"Loss function (-)",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                        "fig_name":"training_info","dpi":60,"index_ii":1000,"shap_data":[],"vel_data":[],"flowfield":[],
                        "index_y":12,"xmin":0,"xmax":125,"ymin":0,"ymax":125,"b_velo_u":"$u$","b_shap_u":"$\phi$",
                        "b_velo_v":"$v$","b_shap_v":"$\phi$","w":"$u^+$","b_shap_w":"$\phi$","b_velo_m":"u",
                        "b_shap_m":"$\phi$","padding":15,"scale_shap":1e5,"colormapsh":'PRGn',"normfield":True,
                        "nfield":"1 field for computing SHAP"}.
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
            - shap_data   : class of the shap values
            - vel_data    : class of the velocity
            - flowfield   : class of the flow data
            - index_y     : index in the wall-normal direction
            - xmin        : minimum value of the x axis
            - xmax        : maximum value of the x axis
            - ymin        : minimum value of the y axis
            - ymax        : maximum value of the y axis
            - b_velo_u    : text for the bar of the streamwise velocity
            - b_shap_u    : text for the bar of the streamwise shap
            - b_velo_v    : text for the bar of the wall_normal velocity
            - b_shap_v    : text for the bar of the wall_normal shap
            - b_velo_w    : text for the bar of the spanwise velocity
            - b_shap_w    : text for the bar of the spanwise shap
            - b_velo_m    : text for the bar of the magnitude of the velocity
            - b_shap_m    : text for the bar of the magnitude of the shap
            - padding     : padding of the domain
            - scale_shap  : value to scale the shap values
            - colormapsh  : colormap used for the shap
            - normfield   : normalization of the field (True: use field normalization, False: use slice normalization)
            - nfield      : title with the number of fields

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
    shap_data   = data_in["shap_data"]                             # data of the shap values
    vel_data    = data_in["vel_data"]                              # data of the velocity
    flowfield   = data_in["flowfield"]                             # data of the flow field
    index_y     = int(data_in["index_y"])                          # index used in the wall-normal direction
    xmin        = float(data_in["xmin"])
    xmax        = float(data_in["xmax"])
    ymin        = float(data_in["ymin"])
    ymax        = float(data_in["ymax"])
    scale_shap  = float(data_in["scale_shap"])
    b_velo_u    = str(data_in["b_velo_u"])
    b_shap_u    = str(data_in["b_shap_u"])
    b_velo_v    = str(data_in["b_velo_v"])
    b_shap_v    = str(data_in["b_shap_v"])
    b_velo_w    = str(data_in["b_velo_w"])
    b_shap_w    = str(data_in["b_shap_w"])
    b_velo_m    = str(data_in["b_velo_m"])
    b_shap_m    = str(data_in["b_shap_m"]) 
    padding     = int(data_in["padding"])
    colormapsh  = str(data_in["colormapsh"])
    normfield   = bool(data_in["normfield"])
    nfield      = str(data_in["nfield"])
    shap_mod    = np.sqrt(shap_data["SHAP_u"]**2+
                          shap_data["SHAP_v"]**2+
                          shap_data["SHAP_w"]**2)
    if normfield:
        max_u       = np.max(np.abs([np.max(shap_data["SHAP_u"]),np.min(shap_data["SHAP_u"])]))
        max_v       = np.max(np.abs([np.max(shap_data["SHAP_v"]),np.min(shap_data["SHAP_v"])]))
        max_w       = np.max(np.abs([np.max(shap_data["SHAP_w"]),np.min(shap_data["SHAP_w"])]))
        min_u       = np.min(np.abs([np.max(shap_data["SHAP_u"]),np.min(shap_data["SHAP_u"])]))
        min_v       = np.min(np.abs([np.max(shap_data["SHAP_v"]),np.min(shap_data["SHAP_v"])]))
        min_w       = np.min(np.abs([np.max(shap_data["SHAP_w"]),np.min(shap_data["SHAP_w"])]))
        vmax_u      = 1
        vmin_u      = 0
        vmax_v      = 1
        vmin_v      = 0
        vmax_w      = 1
        vmin_w      = 0
        max_m       = np.max(shap_mod)*scale_shap
        min_m       = np.min(shap_mod)*scale_shap
        vmax_m      = 1
        vmin_m      = 0
    else:
        max_u       = np.max(np.abs([np.max(shap_data["SHAP_u"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_u"][index_y,padding:-padding,padding:-padding])]))
        max_v       = np.max(np.abs([np.max(shap_data["SHAP_v"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_v"][index_y,padding:-padding,padding:-padding])]))
        max_w       = np.max(np.abs([np.max(shap_data["SHAP_w"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_w"][index_y,padding:-padding,padding:-padding])]))
        min_u       = np.min(np.abs([np.max(shap_data["SHAP_u"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_u"][index_y,padding:-padding,padding:-padding])]))
        min_v       = np.min(np.abs([np.max(shap_data["SHAP_v"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_v"][index_y,padding:-padding,padding:-padding])]))
        min_w       = np.min(np.abs([np.max(shap_data["SHAP_w"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_w"][index_y,padding:-padding,padding:-padding])]))
        vmax_u      = 1
        vmin_u      = 0
        vmax_v      = 1
        vmin_v      = 0
        vmax_w      = 1
        vmin_w      = 0
        max_m       = np.max(shap_mod[index_y,padding:-padding,padding:-padding])
        min_m       = np.min(shap_mod[index_y,padding:-padding,padding:-padding])
        vmax_m      = 1
        vmin_m      = 0
    
     
    xx,zz       = np.meshgrid(flowfield.xplus,flowfield.zplus)
    titlefig    = nfield
            
    fig_name    = fig_name+"_field"+str(index_ii)+"_y"+'{0:.0f}'.format(flowfield.yplus[index_y])
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure_multiplot(data_in={"row":4,"col":1})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":abs(shap_data["SHAP_u"][index_y,
                                                                                           padding:-padding,
                                                                                           padding:-padding]/max_u),
                                  "colormap":None,"plot_number_x":0,"plot_number_y":0,"vmax":vmax_u,
                                  "vmin":vmin_u,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":0,"indexplot":0,"b_text":b_shap_u,
                                        "title":titlefig,"colorbar_x":False,
                                        "xticks":[xmin,
                                                  (xmax-xmin)/4+xmin,
                                                  2*(xmax-xmin)/4+xmin,
                                                  3*(xmax-xmin)/4+xmin,
                                                  xmax],
                                        "yticks":[ymin,
                                                  (ymax-ymin)/2+ymin,
                                                  ymax],
                                        "xticklabels":[str(int(np.round(xmin)/np.pi)), 
                                                       str(int(np.round(((xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((2*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((3*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((xmax)/np.pi)))+"$\pi$"],
                                        "yticklabels":[str(int(np.round((ymin)/np.pi))),
                                                       str(int(np.round(((ymax-ymin)/2+ymin)/np.pi)))+"$\pi$",
                                                       str(int(np.round(((ymax))/np.pi)))+"$\pi$"]})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":abs(shap_data["SHAP_v"][index_y,
                                                                                           padding:-padding,
                                                                                           padding:-padding]/max_v),
                                  "colormap":None,"plot_number_x":0,"plot_number_y":1,"vmax":vmax_v,
                                  "vmin":vmin_v,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":1,"indexplot":1,"b_text":b_shap_v,
                                        "title":titlefig,"colorbar_x":False,
                                        "xticks":[xmin,
                                                  (xmax-xmin)/4+xmin,
                                                  2*(xmax-xmin)/4+xmin,
                                                  3*(xmax-xmin)/4+xmin,
                                                  xmax],
                                        "yticks":[ymin,
                                                  (ymax-ymin)/2+ymin,
                                                  ymax],
                                        "xticklabels":[str(int(np.round(xmin)/np.pi)), 
                                                       str(int(np.round(((xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((2*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((3*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((xmax)/np.pi)))+"$\pi$"],
                                        "yticklabels":[str(int(np.round((ymin)/np.pi))),
                                                       str(int(np.round(((ymax-ymin)/2+ymin)/np.pi)))+"$\pi$",
                                                       str(int(np.round(((ymax))/np.pi)))+"$\pi$"]})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":abs(shap_data["SHAP_w"][index_y,
                                                                                           padding:-padding,
                                                                                           padding:-padding]/max_w),
                                  "colormap":None,"plot_number_x":0,"plot_number_y":2,"vmax":vmax_w,
                                  "vmin":vmin_w,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":2,"indexplot":2,"b_text":b_shap_w,
                                        "title":titlefig,"colorbar_x":False,
                                        "xticks":[xmin,
                                                  (xmax-xmin)/4+xmin,
                                                  2*(xmax-xmin)/4+xmin,
                                                  3*(xmax-xmin)/4+xmin,
                                                  xmax],
                                        "yticks":[ymin,
                                                  (ymax-ymin)/2+ymin,
                                                  ymax],
                                        "xticklabels":[str(int(np.round(xmin)/np.pi)), 
                                                       str(int(np.round(((xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((2*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((3*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((xmax)/np.pi)))+"$\pi$"],
                                        "yticklabels":[str(int(np.round((ymin)/np.pi))),
                                                       str(int(np.round(((ymax-ymin)/2+ymin)/np.pi)))+"$\pi$",
                                                       str(int(np.round(((ymax))/np.pi)))+"$\pi$"]})
    shap_data_m = shap_mod[index_y,padding:-padding,padding:-padding]/max_m
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":shap_data_m,
                                  "colormap":None,"plot_number_x":0,"plot_number_y":3,"vmax":vmax_m,
                                  "vmin":vmin_m,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":3,"indexplot":3,"b_text":b_shap_m,
                                        "title":titlefig,"colorbar_x":False,
                                        "xticks":[xmin,
                                                  (xmax-xmin)/4+xmin,
                                                  2*(xmax-xmin)/4+xmin,
                                                  3*(xmax-xmin)/4+xmin,
                                                  xmax],
                                        "yticks":[ymin,
                                                  (ymax-ymin)/2+ymin,
                                                  ymax],
                                        "xticklabels":[str(int(np.round(xmin)/np.pi)), 
                                                       str(int(np.round(((xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((2*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((3*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((xmax)/np.pi)))+"$\pi$"],
                                        "yticklabels":[str(int(np.round((ymin)/np.pi))),
                                                       str(int(np.round(((ymax-ymin)/2+ymin)/np.pi)))+"$\pi$",
                                                       str(int(np.round(((ymax))/np.pi)))+"$\pi$"]})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    
    
           
def plotshap_noise_normalized_2(data_in={"plot_folder":"plots","xlabel":"Epoch","ylabel":"Loss function (-)",
                                         "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                                         "fig_name":"training_info","dpi":60,"index_ii":1000,"shap_data":[],"vel_data":[],
                                         "flowfield":[],"index_y":12,"xmin":0,"xmax":125,"ymin":0,"ymax":125,
                                         "b_velo_u":"$u$","b_shap_u":"$\phi$","b_velo_v":"$v$","b_shap_v":"$\phi$",
                                         "w":"$u^+$","b_shap_w":"$\phi$","b_velo_m":"u","b_shap_m":"$\phi$","padding":15,
                                         "scale_shap":1e5,"colormapsh":'PRGn',"normfield":True,"nfield":21}):
    """
    .....................................................................................................................
    # plotshap_noise_normalized_2: Function to generate the plot of the shap values
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"plot_folder":"plots","xlabel":"Epoch","ylabel":"Loss function (-)",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,
                        "fig_name":"training_info","dpi":60,"index_ii":1000,"shap_data":[],"vel_data":[],"flowfield":[],
                        "index_y":12,"xmin":0,"xmax":125,"ymin":0,"ymax":125,"b_velo_u":"$u$","b_shap_u":"$\phi$",
                        "b_velo_v":"$v$","b_shap_v":"$\phi$","w":"$u^+$","b_shap_w":"$\phi$","b_velo_m":"u",
                        "b_shap_m":"$\phi$","padding":15,"scale_shap":1e5,"colormapsh":'PRGn',"normfield":True,
                        "nfield":"1 field for computing SHAP"}.
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
            - shap_data   : class of the shap values
            - vel_data    : class of the velocity
            - flowfield   : class of the flow data
            - index_y     : index in the wall-normal direction
            - xmin        : minimum value of the x axis
            - xmax        : maximum value of the x axis
            - ymin        : minimum value of the y axis
            - ymax        : maximum value of the y axis
            - b_velo_u    : text for the bar of the streamwise velocity
            - b_shap_u    : text for the bar of the streamwise shap
            - b_velo_v    : text for the bar of the wall_normal velocity
            - b_shap_v    : text for the bar of the wall_normal shap
            - b_velo_w    : text for the bar of the spanwise velocity
            - b_shap_w    : text for the bar of the spanwise shap
            - b_velo_m    : text for the bar of the magnitude of the velocity
            - b_shap_m    : text for the bar of the magnitude of the shap
            - padding     : padding of the domain
            - scale_shap  : value to scale the shap values
            - colormapsh  : colormap used for the shap
            - normfield   : normalization of the field (True: use field normalization, False: use slice normalization)
            - nfield      : title with the number of fields

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
    shap_data   = data_in["shap_data"]                             # data of the shap values
    vel_data    = data_in["vel_data"]                              # data of the velocity
    flowfield   = data_in["flowfield"]                             # data of the flow field
    index_y     = int(data_in["index_y"])                          # index used in the wall-normal direction
    xmin        = float(data_in["xmin"])
    xmax        = float(data_in["xmax"])
    ymin        = float(data_in["ymin"])
    ymax        = float(data_in["ymax"])
    scale_shap  = float(data_in["scale_shap"])
    b_velo_u    = str(data_in["b_velo_u"])
    b_shap_u    = str(data_in["b_shap_u"])
    b_velo_v    = str(data_in["b_velo_v"])
    b_shap_v    = str(data_in["b_shap_v"])
    b_velo_w    = str(data_in["b_velo_w"])
    b_shap_w    = str(data_in["b_shap_w"])
    b_velo_m    = str(data_in["b_velo_m"])
    b_shap_m    = str(data_in["b_shap_m"]) 
    padding     = int(data_in["padding"])
    colormapsh  = str(data_in["colormapsh"])
    normfield   = bool(data_in["normfield"])
    nfield      = str(data_in["nfield"])
    shap_mod    = np.sqrt(shap_data["SHAP_u"]**2+
                          shap_data["SHAP_v"]**2+
                          shap_data["SHAP_w"]**2)
    if normfield:
        max_u       = np.max(np.abs([np.max(shap_data["SHAP_u"]),np.min(shap_data["SHAP_u"])]))
        max_v       = np.max(np.abs([np.max(shap_data["SHAP_v"]),np.min(shap_data["SHAP_v"])]))
        max_w       = np.max(np.abs([np.max(shap_data["SHAP_w"]),np.min(shap_data["SHAP_w"])]))
        min_u       = np.min(np.abs([np.max(shap_data["SHAP_u"]),np.min(shap_data["SHAP_u"])]))
        min_v       = np.min(np.abs([np.max(shap_data["SHAP_v"]),np.min(shap_data["SHAP_v"])]))
        min_w       = np.min(np.abs([np.max(shap_data["SHAP_w"]),np.min(shap_data["SHAP_w"])]))
        vmax_u      = 1
        vmin_u      = 0
        vmax_v      = 1
        vmin_v      = 0
        vmax_w      = 1
        vmin_w      = 0
        max_m       = np.max(shap_mod)*scale_shap
        min_m       = np.min(shap_mod)*scale_shap
        vmax_m      = 1
        vmin_m      = 0
    else:
        max_u       = np.max(np.abs([np.max(shap_data["SHAP_u"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_u"][index_y,padding:-padding,padding:-padding])]))
        max_v       = np.max(np.abs([np.max(shap_data["SHAP_v"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_v"][index_y,padding:-padding,padding:-padding])]))
        max_w       = np.max(np.abs([np.max(shap_data["SHAP_w"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_w"][index_y,padding:-padding,padding:-padding])]))
        min_u       = np.min(np.abs([np.max(shap_data["SHAP_u"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_u"][index_y,padding:-padding,padding:-padding])]))
        min_v       = np.min(np.abs([np.max(shap_data["SHAP_v"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_v"][index_y,padding:-padding,padding:-padding])]))
        min_w       = np.min(np.abs([np.max(shap_data["SHAP_w"][index_y,padding:-padding,padding:-padding]),
                                     np.min(shap_data["SHAP_w"][index_y,padding:-padding,padding:-padding])]))
        vmax_u      = 1
        vmin_u      = 0
        vmax_v      = 1
        vmin_v      = 0
        vmax_w      = 1
        vmin_w      = 0
        max_m       = np.max(shap_mod[index_y,padding:-padding,padding:-padding])
        min_m       = np.min(shap_mod[index_y,padding:-padding,padding:-padding])
        vmax_m      = 1
        vmin_m      = 0
    
     
    xx,zz       = np.meshgrid(flowfield.xplus,flowfield.zplus)
    titlefig    = nfield
            
    fig_name    = fig_name+"_field"+str(index_ii)+"_y"+'{0:.0f}'.format(flowfield.yplus[index_y])
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure_multiplot(data_in={"row":4,"col":1})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":abs(shap_data["SHAP_u"][index_y,
                                                                                           padding:-padding,
                                                                                           padding:-padding]/max_u)**0.5,
                                  "colormap":None,"plot_number_x":0,"plot_number_y":0,"vmax":vmax_u,
                                  "vmin":vmin_u,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":0,"indexplot":0,"b_text":b_shap_u,
                                        "title":titlefig,"colorbar_x":False,
                                        "xticks":[xmin,
                                                  (xmax-xmin)/4+xmin,
                                                  2*(xmax-xmin)/4+xmin,
                                                  3*(xmax-xmin)/4+xmin,
                                                  xmax],
                                        "yticks":[ymin,
                                                  (ymax-ymin)/2+ymin,
                                                  ymax],
                                        "xticklabels":[str(int(np.round(xmin)/np.pi)), 
                                                       str(int(np.round(((xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((2*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((3*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((xmax)/np.pi)))+"$\pi$"],
                                        "yticklabels":[str(int(np.round((ymin)/np.pi))),
                                                       str(int(np.round(((ymax-ymin)/2+ymin)/np.pi)))+"$\pi$",
                                                       str(int(np.round(((ymax))/np.pi)))+"$\pi$"]})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":abs(shap_data["SHAP_v"][index_y,
                                                                                           padding:-padding,
                                                                                           padding:-padding]/max_v)**0.5,
                                  "colormap":None,"plot_number_x":0,"plot_number_y":1,"vmax":vmax_v,
                                  "vmin":vmin_v,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":1,"indexplot":1,"b_text":b_shap_v,
                                        "title":titlefig,"colorbar_x":False,
                                        "xticks":[xmin,
                                                  (xmax-xmin)/4+xmin,
                                                  2*(xmax-xmin)/4+xmin,
                                                  3*(xmax-xmin)/4+xmin,
                                                  xmax],
                                        "yticks":[ymin,
                                                  (ymax-ymin)/2+ymin,
                                                  ymax],
                                        "xticklabels":[str(int(np.round(xmin)/np.pi)), 
                                                       str(int(np.round(((xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((2*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((3*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((xmax)/np.pi)))+"$\pi$"],
                                        "yticklabels":[str(int(np.round((ymin)/np.pi))),
                                                       str(int(np.round(((ymax-ymin)/2+ymin)/np.pi)))+"$\pi$",
                                                       str(int(np.round(((ymax))/np.pi)))+"$\pi$"]})
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":abs(shap_data["SHAP_w"][index_y,
                                                                                           padding:-padding,
                                                                                           padding:-padding]/max_w)**0.5,
                                  "colormap":None,"plot_number_x":0,"plot_number_y":2,"vmax":vmax_w,
                                  "vmin":vmin_w,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":2,"indexplot":2,"b_text":b_shap_w,
                                        "title":titlefig,"colorbar_x":False,
                                        "xticks":[xmin,
                                                  (xmax-xmin)/4+xmin,
                                                  2*(xmax-xmin)/4+xmin,
                                                  3*(xmax-xmin)/4+xmin,
                                                  xmax],
                                        "yticks":[ymin,
                                                  (ymax-ymin)/2+ymin,
                                                  ymax],
                                        "xticklabels":[str(int(np.round(xmin)/np.pi)), 
                                                       str(int(np.round(((xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((2*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((3*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((xmax)/np.pi)))+"$\pi$"],
                                        "yticklabels":[str(int(np.round((ymin)/np.pi))),
                                                       str(int(np.round(((ymax-ymin)/2+ymin)/np.pi)))+"$\pi$",
                                                       str(int(np.round(((ymax))/np.pi)))+"$\pi$"]})
    shap_data_m = shap_mod[index_y,padding:-padding,padding:-padding]/max_m
    plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":shap_data_m**0.5,
                                  "colormap":None,"plot_number_x":0,"plot_number_y":3,"vmax":vmax_m,
                                  "vmin":vmin_m,"Ncolor":None})
    plot_pred.plot_multilayout(data_in={"plot_number_x":0,"plot_number_y":3,"indexplot":3,"b_text":b_shap_m,
                                        "title":titlefig,"colorbar_x":False,
                                        "xticks":[xmin,
                                                  (xmax-xmin)/4+xmin,
                                                  2*(xmax-xmin)/4+xmin,
                                                  3*(xmax-xmin)/4+xmin,
                                                  xmax],
                                        "yticks":[ymin,
                                                  (ymax-ymin)/2+ymin,
                                                  ymax],
                                        "xticklabels":[str(int(np.round(xmin)/np.pi)), 
                                                       str(int(np.round(((xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((2*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((3*(xmax-xmin)/4+xmin)/np.pi)))+"$\pi$",
                                                       str(int(np.round((xmax)/np.pi)))+"$\pi$"],
                                        "yticklabels":[str(int(np.round((ymin)/np.pi))),
                                                       str(int(np.round(((ymax-ymin)/2+ymin)/np.pi)))+"$\pi$",
                                                       str(int(np.round(((ymax))/np.pi)))+"$\pi$"]})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    