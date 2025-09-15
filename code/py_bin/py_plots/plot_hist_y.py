# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plot_hist_y.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:37:35 2024

@author: Andres Cremades Botella

File to plot the shap pdf in the domain. The file contains the following functions:
    - Functions:
        - plot_hist_y : function to plot the pdf of the velocities in the structures
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

def plot_hist_y(data_in={"plot_folder":"plots","plot_fileu":"fileu","plot_filev":"filev","plot_filew":"filew",
                         "plot_filem":"filem","ylabel":"y","xlabelu":"u","xlabelv":"v","xlabelw":"w","xlabelm":"m",
                         "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,"dpi":60,
                         "uu":[],"vv":[],"ww":[],"mm":[],"yplus_struc":[],"yplusmesh":[],"bins":1,"lev_min":1e-2,
                         "lev_delta":None,"linewidth":2}):
    """""
    .....................................................................................................................
    # plot_histuvw_y: Function to plot the components of a vector in the field as a function of the wall-distance
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"plot_folder":"plots","plot_fileu":"fileu","plot_filev":"filev","plot_filew":"filew",
                        "plot_filem":"filem","ylabel":"y","xlabelu":"u","xlabelv":"v","xlabelw":"w","xlabelm":"m",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,"dpi":60,
                        "uu":[],"vv":[],"ww":[],"mm":[],"yplus_struc":[],"yplusmesh":[],"bins":1,"lev_min":1e-2,
                        "lev_delta":None,"linewidth":2}.
        Data:
            - plot_folder      : folder to store the plots
            - plot_fileu       : file to save the pdf of the streamwise field
            - plot_filev       : file to save the pdf of the wall-normal field
            - plot_filew       : file to save the pdf of the spanwise field
            - plot_filem       : file to save the pdf of the absolute value
            - ylabel           : label of the y axis
            - xlabelu          : label of the x axis for the streamwise field
            - xlabelv          : label of the x axis for the wall-normal field
            - xlabelw          : label of the x axis for the spanwise field
            - xlabelm          : label of the x axis for the absolute field
            - fontsize         : font size used for the figure
            - figsize_x        : size of the figure in x
            - figsize_y        : size of the figure in y
            - colormap         : colormap used for the figure
            - colornum         : number of colors of the colormap, two curves are used. The number of levels of the 
                                 colormap needs to be higher than 2 
            - dpi              : dots per inch of the saved figure
            - uu               : array of the streamwise field
            - vv               : array of the wall-normal field
            - ww               : array of the spanwise field
            - mm               : array of the absolute value of the field
            - yplus_struc      : array of the wall-distance 
            - yplus_mesh       : positions of the mesh in the wall-normal direction
            - bins             : bins in the pdf
            - lev_min          : minimum value of the levels of the pdf
            - lev_delta        : distance between levels (None if distance needs to be calculated by the code)
            - linewidth        : width of the line in the histogram

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
    plot_fileu            = str(data_in["plot_fileu"])
    plot_filev            = str(data_in["plot_filev"])
    plot_filew            = str(data_in["plot_filew"])
    plot_filem            = str(data_in["plot_filem"])
    ylabel                = str(data_in["ylabel"])
    xlabelu               = str(data_in["xlabelu"])
    xlabelv               = str(data_in["xlabelv"])
    xlabelw               = str(data_in["xlabelw"])
    xlabelm               = str(data_in["xlabelm"])
    fontsize              = int(data_in["fontsize"])
    figsize_x             = int(data_in["figsize_x"])
    figsize_y             = int(data_in["figsize_y"])
    colormap              = str(data_in["colormap"])
    colornum              = int(data_in["colornum"])
    dpi                   = float(data_in["dpi"])
    linewidth             = float(data_in["linewidth"])
    uu                    = np.array(data_in["uu"],dtype="float")
    vv                    = np.array(data_in["vv"],dtype="float")
    ww                    = np.array(data_in["ww"],dtype="float")
    mm                    = np.array(data_in["mm"],dtype="float")
    yplus_struc           = np.array(data_in["yplus_struc"],dtype="float")
    yplusmesh             = np.array(data_in["yplusmesh"],dtype="float")
    bins                  = int(data_in["bins"])
    lev_min               = float(data_in["lev_min"])
    lev_delta             = data_in["lev_delta"]
    ymax                  = np.max(yplusmesh)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the mesh for the pdf
    # -------------------------------------------------------------------------------------------------------------------
    diffy = np.diff(yplusmesh)/2
    binsy = np.concatenate(([-diffy[0]],yplusmesh[:-1]+diffy,[yplusmesh[-1]+diffy[-1]]))
      
    # ----------------------------------------------------------------------------------------------------------------
    # Calculate the number of levels
    # ----------------------------------------------------------------------------------------------------------------
    if lev_delta is None:
        explev0   = np.log10(lev_min)
        explevs   = np.linspace(explev0,0,5)
        levels    = 10**explevs 
        locator   = ticker.LogLocator(numticks=5)
    else:
        lev_delta = int(lev_delta)
        levels    = [lev_min,lev_min+lev_delta,lev_min+2*lev_delta,lev_min+3*lev_delta,lev_min+4*lev_delta]
        locator   = ticker.LinearLocator()
    
    # ----------------------------------------------------------------------------------------------------------------
    # Calculate the pdf for u
    # ----------------------------------------------------------------------------------------------------------------
    hist_uy,hist_u,hist_y = np.histogram2d(uu,yplus_struc,bins=(bins,binsy))
    hist_y                = hist_y[:-1]+np.diff(hist_y)/2
    hist_u                = hist_u[:-1]+np.diff(hist_u)/2
    grid_u,grid_y         = np.meshgrid(hist_u,hist_y)
    grid_uy               = hist_uy.T.copy()
    grid_uy              /= np.max(grid_uy)
    ucontent              = grid_u[np.where(grid_uy>=lev_min)]
    umin                  = np.min(ucontent)
    umax                  = np.max(ucontent)        
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot for u
    # -------------------------------------------------------------------------------------------------------------------
    titlefig   = xlabelu+"$ - y^+$"
    data_plot  = {"xlabel":xlabelu,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"log","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":plot_fileu,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":umin,"xmax":umax,"ymin":1,"ymax":ymax,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure()
    plot_pred.add_hist2d_y(data_in={"xx":grid_u,"yy":grid_y,"xxyy":grid_uy,"levels":levels,"colormap":None,
                                  "locator":locator,"alp":0.65,"linewidth":linewidth})
    labels = ['{0:.1e}'.format(levels[ii]) for ii in np.arange(len(levels))]
    plot_pred.plot_layout_pcolor(data_in={"title":titlefig,"colorbar":True,"b_text":None,
                                          "colorticks":levels,"colorlabels":labels,"equal":False})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    plot_pred.close()
   
    # ----------------------------------------------------------------------------------------------------------------
    # Calculate the pdf for v
    # ----------------------------------------------------------------------------------------------------------------
    hist_vy,hist_v,hist_y = np.histogram2d(vv,yplus_struc,bins=(bins,binsy))
    hist_y                = hist_y[:-1]+np.diff(hist_y)/2
    hist_v                = hist_v[:-1]+np.diff(hist_v)/2
    grid_v,grid_y         = np.meshgrid(hist_v,hist_y)
    grid_vy               = hist_vy.T.copy()
    grid_vy              /= np.max(grid_vy)
    vcontent              = grid_v[np.where(grid_vy>=lev_min)]
    vmin                  = np.min(vcontent)
    vmax                  = np.max(vcontent) 
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot for v
    # -------------------------------------------------------------------------------------------------------------------
    titlefig   = xlabelv+"$ - y^+$"
    data_plot  = {"xlabel":xlabelv,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"log","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":plot_filev,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":vmin,"xmax":vmax,"ymin":1,"ymax":ymax,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure()
    plot_pred.add_hist2d_y(data_in={"xx":grid_v,"yy":grid_y,"xxyy":grid_vy,"levels":levels,"colormap":None,
                                  "locator":locator,"alp":0.65,"linewidth":linewidth})
    labels = ['{0:.1e}'.format(levels[ii]) for ii in np.arange(len(levels))]
    plot_pred.plot_layout_pcolor(data_in={"title":titlefig,"colorbar":True,"b_text":None,
                                          "colorticks":levels,"colorlabels":labels,"equal":False})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    plot_pred.close()
    
    # ----------------------------------------------------------------------------------------------------------------
    # Calculate the pdf for w
    # ----------------------------------------------------------------------------------------------------------------
    hist_wy,hist_w,hist_y = np.histogram2d(ww,yplus_struc,bins=(bins,binsy))
    hist_y                = hist_y[:-1]+np.diff(hist_y)/2
    hist_w                = hist_w[:-1]+np.diff(hist_w)/2
    grid_w,grid_y         = np.meshgrid(hist_w,hist_y)
    grid_wy               = hist_wy.T.copy()
    grid_wy              /= np.max(grid_wy)
    wcontent              = grid_w[np.where(grid_wy>=lev_min)]
    wmin                  = np.min(wcontent)
    wmax                  = np.max(wcontent) 
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot for w
    # -------------------------------------------------------------------------------------------------------------------
    titlefig   = xlabelw+"$ - y^+$"
    data_plot  = {"xlabel":xlabelw,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"log","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":plot_filew,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":wmin,"xmax":wmax,"ymin":1,"ymax":ymax,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure()
    plot_pred.add_hist2d_y(data_in={"xx":grid_w,"yy":grid_y,"xxyy":grid_wy,"levels":levels,"colormap":None,
                                  "locator":locator,"alp":0.65,"linewidth":linewidth})
    labels = ['{0:.1e}'.format(levels[ii]) for ii in np.arange(len(levels))]
    plot_pred.plot_layout_pcolor(data_in={"title":titlefig,"colorbar":True,"b_text":None,
                                          "colorticks":levels,"colorlabels":labels,"equal":False})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    plot_pred.close()
    
    
    # ----------------------------------------------------------------------------------------------------------------
    # Calculate the pdf for absolute values
    # ----------------------------------------------------------------------------------------------------------------
    hist_my,hist_m,hist_y = np.histogram2d(mm,yplus_struc,bins=(bins,binsy))
    hist_y                = hist_y[:-1]+np.diff(hist_y)/2
    hist_m                = hist_m[:-1]+np.diff(hist_m)/2
    grid_m,grid_y         = np.meshgrid(hist_m,hist_y)
    grid_my               = hist_my.T.copy()
    grid_my              /= np.max(grid_my)
    mcontent              = grid_m[np.where(grid_my>=lev_min)]
    mmin                  = np.min(mcontent)
    mmax                  = np.max(mcontent) 
    
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot for m
    # -------------------------------------------------------------------------------------------------------------------
    titlefig   = xlabelm+"$ - y^+$"
    data_plot  = {"xlabel":xlabelm,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"log","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":plot_filem,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":mmin,"xmax":mmax,"ymin":1,"ymax":ymax,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure()
    plot_pred.add_hist2d_y(data_in={"xx":grid_m,"yy":grid_y,"xxyy":grid_my,"levels":levels,"colormap":None,
                                  "locator":locator,"alp":0.65,"linewidth":linewidth})
    labels = ['{0:.1e}'.format(levels[ii]) for ii in np.arange(len(levels))]
    plot_pred.plot_layout_pcolor(data_in={"title":titlefig,"colorbar":True,"b_text":None,
                                          "colorticks":levels,"colorlabels":labels,"equal":False})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    plot_pred.close()
    
    
    
    
def plot_hist_1d(data_in={"plot_folder":"plots","plot_file":"file","ylabel":"y","xlabelu":"u",
                          "xlabelv":"v","xlabelw":"w","xlabelm":"m","fontsize":18,"figsize_x":10,"figsize_y":8,
                          "colormap":"viridis","colornum":2,"dpi":60,"uu":[],"vv":[],"ww":[],"mm":[],
                          "yplus_struc":0,"bins":1,"lev_min":1e-2,"lev_delta":None}):
    """""
    .....................................................................................................................
    # plot_histuvw_y: Function to plot the components of a vector in the field as a function of the wall-distance
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"plot_folder":"plots","plot_file":"file","ylabel":"y","xlabelu":"u","xlabelv":"v",
                        "xlabelw":"w","xlabelm":"m","fontsize":18,"figsize_x":10,"figsize_y":8,
                        "colormap":"viridis","colornum":2,"dpi":60,"uu":[],"vv":[],"ww":[],"mm":[],
                        "yplus_struc":0,"bins":1,"lev_min":1e-2,"lev_delta":None}.
        Data:
            - plot_folder      : folder to store the plots
            - plot_file        : file to save the pdf 
            - ylabel           : label of the y axis
            - xlabelu          : label of the x axis for the streamwise field
            - xlabelv          : label of the x axis for the wall-normal field
            - xlabelw          : label of the x axis for the spanwise field
            - xlabelm          : label of the x axis for the absolute field
            - fontsize         : font size used for the figure
            - figsize_x        : size of the figure in x
            - figsize_y        : size of the figure in y
            - colormap         : colormap used for the figure
            - colornum         : number of colors of the colormap, two curves are used. The number of levels of the 
                                 colormap needs to be higher than 2 
            - dpi              : dots per inch of the saved figure
            - uu               : array of the streamwise field
            - vv               : array of the wall-normal field
            - ww               : array of the spanwise field
            - mm               : array of the absolute value of the field
            - yplus_struc      : wall-distance 
            - bins             : bins in the pdf
            - lev_min          : minimum value of the levels of the pdf
            - lev_delta        : distance between levels (None if distance needs to be calculated by the code)

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
    xlabelu               = str(data_in["xlabelu"])
    xlabelv               = str(data_in["xlabelv"])
    xlabelw               = str(data_in["xlabelw"])
    xlabelm               = str(data_in["xlabelm"])
    fontsize              = int(data_in["fontsize"])
    figsize_x             = int(data_in["figsize_x"])
    figsize_y             = int(data_in["figsize_y"])
    colormap              = str(data_in["colormap"])
    colornum              = int(data_in["colornum"])
    dpi                   = float(data_in["dpi"])
    uu                    = np.array(data_in["uu"],dtype="float")
    vv                    = np.array(data_in["vv"],dtype="float")
    ww                    = np.array(data_in["ww"],dtype="float")
    mm                    = np.array(data_in["mm"],dtype="float")
    yplus_struc           = float(data_in["yplus_struc"])
    bins                  = int(data_in["bins"])
    lev_min               = float(data_in["lev_min"])
    lev_delta             = data_in["lev_delta"]
    
        
    # ----------------------------------------------------------------------------------------------------------------
    # Calculate the pdf for u
    # ----------------------------------------------------------------------------------------------------------------
    hist_uy,hist_u        = np.histogram(uu,bins=bins)
    hist_u                = hist_u[:-1]+np.diff(hist_u)/2 
    hist_uy               = hist_uy/np.max(hist_uy)
    hist_vy,hist_v        = np.histogram(vv,bins=bins)
    hist_v                = hist_v[:-1]+np.diff(hist_v)/2
    hist_vy               = hist_vy/np.max(hist_vy)
    hist_wy,hist_w        = np.histogram(ww,bins=bins)
    hist_w                = hist_w[:-1]+np.diff(hist_w)/2
    hist_wy               = hist_wy/np.max(hist_wy)    
    
    umean                 = np.mean(uu)
    vmean                 = np.mean(vv)
    wmean                 = np.mean(ww)
    ustd                  = np.std(uu)
    vstd                  = np.std(vv)
    wstd                  = np.std(ww)
    print("log")
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot for u
    # -------------------------------------------------------------------------------------------------------------------
    data_plot  = {"xlabel":xlabelu,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"log","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":plot_file,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure()
    plot_pred.add_plot_2d(data_in={"data_x":hist_u,"data_y":hist_uy,"label":"$\phi_u$","color":None,
                                   "linewidth":2,"plot_number":0,"style":"-"})
    plot_pred.add_plot_2d(data_in={"data_x":hist_v,"data_y":hist_vy,"label":"$\phi_v$","color":None,
                                   "linewidth":2,"plot_number":1,"style":"-"})
    plot_pred.add_plot_2d(data_in={"data_x":hist_w,"data_y":hist_wy,"label":"$\phi_w$","color":None,
                                   "linewidth":2,"plot_number":2,"style":"-"})
    plot_pred.plot_layout()
    plot_pred.add_plot_2d(data_in={"data_x":[umean,umean],"data_y":[0,1],"label":"-","color":None,
                                   "linewidth":2,"plot_number":0,"style":"--"})
    plot_pred.add_plot_2d(data_in={"data_x":[vmean,vmean],"data_y":[0,1],"label":"-","color":None,
                                   "linewidth":2,"plot_number":1,"style":"--"})
    plot_pred.add_plot_2d(data_in={"data_x":[wmean,wmean],"data_y":[0,1],"label":"-","color":None,
                                   "linewidth":2,"plot_number":2,"style":"--"})
    plot_pred.add_plot_2d(data_in={"data_x":[umean+ustd,umean+ustd],"data_y":[0,1],"label":"-","color":None,
                                   "linewidth":2,"plot_number":0,"style":":"})
    plot_pred.add_plot_2d(data_in={"data_x":[umean-ustd,umean-ustd],"data_y":[0,1],"label":"-","color":None,
                                   "linewidth":2,"plot_number":0,"style":":"})
    plot_pred.add_plot_2d(data_in={"data_x":[vmean+vstd,vmean+vstd],"data_y":[0,1],"label":"-","color":None,
                                   "linewidth":2,"plot_number":1,"style":":"})
    plot_pred.add_plot_2d(data_in={"data_x":[vmean-vstd,vmean-vstd],"data_y":[0,1],"label":"-","color":None,
                                   "linewidth":2,"plot_number":1,"style":":"})
    plot_pred.add_plot_2d(data_in={"data_x":[wmean+wstd,wmean+wstd],"data_y":[0,1],"label":"-","color":None,
                                   "linewidth":2,"plot_number":2,"style":":"})
    plot_pred.add_plot_2d(data_in={"data_x":[wmean-wstd,wmean-wstd],"data_y":[0,1],"label":"-","color":None,
                                   "linewidth":2,"plot_number":2,"style":":"})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    plot_pred.close()
   
    
def plot_hist_y_withhist(data_in={"plot_folder":"plots","plot_file":"fileu","ylabel":"y","xlabel":"u",
                                  "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,"dpi":60,
                                  "hist_data":[],"hist_x":[],"hist_y":[],"yplusmesh":[],"lev_min":1e-2,
                                  "lev_delta":None,"linewidth":2,"datamin":0,"datamax":1}):
    """""
    .....................................................................................................................
    # plot_histuvw_y: Function to plot the components of a vector in the field as a function of the wall-distance
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"plot_folder":"plots","plot_fileu":"fileu","plot_filev":"filev","plot_filew":"filew",
                        "plot_filem":"filem","ylabel":"y","xlabelu":"u","xlabelv":"v","xlabelw":"w","xlabelm":"m",
                        "fontsize":18,"figsize_x":10,"figsize_y":8,"colormap":"viridis","colornum":2,"dpi":60,
                        "uu":[],"vv":[],"ww":[],"mm":[],"yplus_struc":[],"yplusmesh":[],"lev_min":1e-2,
                        "lev_delta":None,"linewidth":2,"datamin":0,"datamax":1}.
        Data:
            - plot_folder      : folder to store the plots
            - plot_file        : file to save the pdf of the field
            - ylabel           : label of the y axis
            - xlabel           : label of the x axis for the field
            - fontsize         : font size used for the figure
            - figsize_x        : size of the figure in x
            - figsize_y        : size of the figure in y
            - colormap         : colormap used for the figure
            - colornum         : number of colors of the colormap, two curves are used. The number of levels of the 
                                 colormap needs to be higher than 2 
            - dpi              : dots per inch of the saved figure
            - hist_data        : array of the streamwise field
            - hist_x           : array of the wall-normal field
            - hist_y           : array of the spanwise field
            - yplus_struc      : array of the wall-distance 
            - yplus_mesh       : positions of the mesh in the wall-normal direction
            - lev_min          : minimum value of the levels of the pdf
            - lev_delta        : distance between levels (None if distance needs to be calculated by the code)
            - linewidth        : width of the line in the histogram
            - datamin          : minimum value of the plot of the data in x
            - datamax          : maximum value of the plot of the data in x

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
    colormap              = str(data_in["colormap"])
    colornum              = int(data_in["colornum"])
    dpi                   = float(data_in["dpi"])
    linewidth             = float(data_in["linewidth"])
    hist_data             = np.array(data_in["hist_data"],dtype="float")
    hist_x                = np.array(data_in["hist_x"],dtype="float")
    hist_y                = np.array(data_in["hist_y"],dtype="float")
    yplusmesh             = np.array(data_in["yplusmesh"],dtype="float")
    datamin               = float(data_in["datamin"])
    datamax               = float(data_in["datamax"])
    lev_min               = float(data_in["lev_min"])
    lev_delta             = data_in["lev_delta"]
    ymax                  = np.max(yplusmesh)
      
    # ----------------------------------------------------------------------------------------------------------------
    # Calculate the number of levels
    # ----------------------------------------------------------------------------------------------------------------
    if lev_delta is None:
        explev0   = np.log10(lev_min)
        explevs   = np.linspace(explev0,0,5)
        levels    = 10**explevs 
        locator   = ticker.LogLocator(numticks=5)
    else:
        lev_delta = int(lev_delta)
        levels    = [lev_min,lev_min+lev_delta,lev_min+2*lev_delta,lev_min+3*lev_delta,lev_min+4*lev_delta]
        locator   = ticker.LinearLocator()
            
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot for u
    # -------------------------------------------------------------------------------------------------------------------
    titlefig   = xlabel+"$ - y^+$"
    data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
                  "figsize_y":figsize_y,"xscale":"linear","yscale":"log","zscale":"linear","colormap":colormap,
                  "colornum":colornum,"legend":True,"fig_name":plot_file,"dpi":dpi,"plot_folder":plot_folder,
                  "xmin":datamin,"xmax":datamax,"ymin":1,"ymax":ymax,"zmin":None,"zmax":None}
    plot_pred = plot_format(data_in=data_plot)
    plot_pred.create_figure()
    plot_pred.add_hist2d_y(data_in={"xx":hist_x,"yy":hist_y,"xxyy":hist_data,"levels":levels,"colormap":None,
                                  "locator":locator,"alp":0.65,"linewidth":linewidth})
    labels = ['{0:.1e}'.format(levels[ii]) for ii in np.arange(len(levels))]
    plot_pred.plot_layout_pcolor(data_in={"title":titlefig,"colorbar":True,"b_text":None,
                                          "colorticks":levels,"colorlabels":labels,"equal":False})
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    plot_pred.plot_save_png()
    plot_pred.plot_save_pdf()
    plot_pred.close()
   