# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plot_coinc_mat_3d.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Oct  3 11:09:01 2024

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

      
def plot_coinc_mat_3types_withcontours_3d(data_in={"struc1":[],"struc2":[],"struc3":[],"struc_cont":[],
                                                   "plot_folder":"plots","plot_coin_folder":"uv_shap_coin_y",
                                                   "plot_coin_file":"uv_shap_coin_y","xlabel":"x","ylabel":"z",
                                                   "zlabel":"y",
                                                   "fontsize":18,"figsize_x":10,"figsize_y":8,
                                                   "fig_name":"struc3d","dpi":60,
                                                   "dy":1,"dx":1,"dz":1,"uvw_folder":"../../P125_21pi_vu/",
                                                   "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                                                   "struc1_lab":"uv","struc2_lab":"streak","struc3_lab":"chong",
                                                   "mat_comb":[],"mat_comb":[],"data_folder":"data",
                                                   "linewidth":1,"flowfield":[],"yplane":1,"linewidth":1}):
    """
    .....................................................................................................................
    # plot_coinc_mat_all_3types_withcontour: Function to plot the coincidence between all the structures
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data required for generating the plot. 
        The default is {"struc1":[],"struc2":[],"struc3":[],"struc_cont":[],
                        "plot_folder":"plots","plot_coin_folder":"uv_shap_coin_y",
                        "plot_coin_file":"uv_shap_coin_y","xlabel":"x","ylabel":"z","zlabel":"y","fontsize":18,
                        "figsize_x":10,"figsize_y":8,"fig_name":"struc3d","dpi":60,"dy":1,"dx":1,"dz":1,
                        "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","cmap_flag":False,
                        "struc1_lab":"uv","struc2_lab":"streak","struc3_lab":"chong","mat_comb":[],"mat_comb":[],
                        "data_folder":"data","flowfield":[],"yplane":1,"linewidth":1}.
        Data:
            - struc1           : data of the structure 1
            - struc2           : data of the structure 2
            - struc3           : data of the structure 3
            - struc_cont       : data of the structure of the contour
            - plot_folder      : folder to store the plots
            - plot_coin_folder : folder to save the coincidence plots between structures
            - plot_coin_file   : file to save the coincidence plots between structures
            - xlabel           : label of the x axis
            - ylabel           : label of the y axis
            - zlabel           : label of the z axis
            - fontsize         : font size used for the figure
            - figsize_x        : size of the figure in x
            - figsize_y        : size of the figure in y
            - fig_name    : name of the saved figure
            - dpi              : dots per inch of the saved figure
            - dy               : downsampling in the wall-normal direction
            - dx               : downsampling in the streamwise direction
            - dz               : downsampling in the spanwise direction
            - uvw_folder       : folder of the flow fields
            - uvw_file         : file of the flow fields
            - cmap_flag        : flag to use a color map (True) or solid color (False)
            - struc1_lab       : label of the structure 1
            - struc2_lab       : label of the structure 2
            - struc3_lab       : label of the structure 3
            - mat_comb         : matrix with the combination of the structures
            - data_folder      : folder to read data files
            - linewidth        : linewidth of the contours
            - flowfield        : data of the flow field
            - yplane           : plane to show
            - linewidth        : linewidth of the contour

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
    from py_bin.py_functions.calc_coinc import read_coinc_all_3types
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the parameters of the plot
    # -------------------------------------------------------------------------------------------------------------------
    struc1      = data_in["struc1"]
    struc2      = data_in["struc2"]
    struc3      = data_in["struc3"]
    struc_cont  = data_in["struc_cont"]
    plot_folder = str(data_in["plot_folder"]) 
    plot_coin_folder      = str(data_in["plot_coin_folder"])
    plot_coin_file        = str(data_in["plot_coin_file"])
    xlabel      = str(data_in["xlabel"]) 
    ylabel      = str(data_in["ylabel"])
    zlabel      = str(data_in["zlabel"])
    fontsize    = int(data_in["fontsize"])
    figsize_x   = int(data_in["figsize_x"])
    figsize_y   = int(data_in["figsize_y"])
    fig_name    = str(data_in["fig_name"])
    dpi         = float(data_in["dpi"])
    dy          = int(data_in["dy"])
    dx          = int(data_in["dx"])
    dz          = int(data_in["dz"])
    uvw_folder  = str(data_in["uvw_folder"])
    uvw_file    = str(data_in["uvw_file"])
    struc1_lab            = str(data_in["struc1_lab"])
    struc2_lab            = str(data_in["struc2_lab"])
    struc3_lab            = str(data_in["struc3_lab"])
    mat_comb              = np.array(data_in["mat_comb"],dtype="float")
    data_folder           = str(data_in["data_folder"])
    cmapflag    =  False
    flowfield             = data_in["flowfield"]
    linewidth          = int(data_in["linewidth"])
    L_x = flowfield.L_x
    L_y = flowfield.L_y
    L_z = flowfield.L_z
    rey = flowfield.rey
    utau = flowfield.utau
    yplane_vec = np.array(data_in["yplane"],dtype="int")
    ii_x_min = 150
    ii_x_max = 300
    ii_z_min = 100
    ii_z_max = 250
    yplane_min = 0
    yplane_max = int(flowfield.shpy/2-1)
    # alpha = 0.1
    alpha_cont = 0.25
    
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
    colormap1 = colors[0]
    colormap2 = colors[1]
    colormap3 = colors[3]
    colormap_cont = '#E0E0E0'
    try:
        os.mkdir(plot_folder)
    except:
        print("Existing folder...",flush=True)
    try:
        os.mkdir(plot_folder+'/'+plot_coin_folder)
    except:
        print("Existing folder...",flush=True)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the file mean velocity
    # -------------------------------------------------------------------------------------------------------------------
    Data_flow = {"folder":uvw_folder,"file":uvw_file,"down_x":dx,"down_y":dy,"down_z":dz,
                 "L_x":L_x,"L_y":L_y,"L_z":L_z,"rey":rey,"utau":utau}
    flow_data = flow_field(data_in=Data_flow)
    flow_data.shape_tensor()
    flow_data.flow_grid()
    xx = flow_data.xplus[ii_x_min:ii_x_max]
    zz = flow_data.zplus[ii_z_min:ii_z_max]
    yy = (flow_data.y_h[yplane_min:yplane_max]*rey)
    yylabel = rey-abs(yy)
    xx_grid,zz_grid       = np.meshgrid(xx,zz)
    
    # # -------------------------------------------------------------------------------------------------------------------
    # # Read the coincidence file
    # # -------------------------------------------------------------------------------------------------------------------
    
    mat_cont       = np.array(struc_cont.mat_struc,dtype='int')
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data of the structures
    # -------------------------------------------------------------------------------------------------------------------
    mat_segment1 = struc1.structures.mat_segment_filtered[yplane_min:yplane_max,ii_z_min:ii_z_max,ii_x_min:ii_x_max]
    mat_segment2 = struc2.structures.mat_segment_filtered[yplane_min:yplane_max,ii_z_min:ii_z_max,ii_x_min:ii_x_max]
    mat_segment3 = struc3.structures.mat_segment_filtered[yplane_min:yplane_max,ii_z_min:ii_z_max,ii_x_min:ii_x_max]
    mat_segment_cont = struc_cont.structures.mat_segment_filtered[yplane_min:yplane_max,ii_z_min:ii_z_max,
                                                                  ii_x_min:ii_x_max]

    # -------------------------------------------------------------------------------------------------------------------
    # Create the plot
    # -------------------------------------------------------------------------------------------------------------------
    for yplane in yplane_vec:
        if yplane > yplane_max:
            yplane = (flowfield.shpy-1)-yplane
        data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":zlabel,"fontsize":fontsize,"figsize_x":figsize_x,
                      "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap1,
                      "colornum":0,"legend":False,"fig_name":fig_name+"_field_"+str(yplane),"dpi":dpi,"plot_folder":plot_folder,
                      "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
        plot_train = plot_format(data_in=data_plot)
        plot_train.create_figure3d()
        # plot_info1 = {"data_x":xx,"data_y":yy,"data_z":zz,"struc":mat_segment1,
        #               "color":colormap1,"cmap_flag":cmapflag,"vmax":None,"vmin":None,"alpha":alpha}
        # plot_train.add_plot_3d_structure(data_in=plot_info1)
        # plot_info2 = {"data_x":xx,"data_y":yy,"data_z":zz,"struc":mat_segment2,
        #               "color":colormap2,"cmap_flag":cmapflag,"vmax":None,"vmin":None,"alpha":alpha}
        # plot_train.add_plot_3d_structure(data_in=plot_info2)
        # plot_info3 = {"data_x":xx,"data_y":yy,"data_z":zz,"struc":mat_segment3,
        #               "color":colormap3,"cmap_flag":cmapflag,"vmax":None,"vmin":None,"alpha":alpha}
        # plot_train.add_plot_3d_structure(data_in=plot_info3)
        plot_info_cont = {"data_x":xx,"data_y":yy,"data_z":zz,"struc":mat_segment_cont,
                      "color":colormap_cont,"cmap_flag":cmapflag,"vmax":None,"vmin":None,"alpha":alpha_cont}
        plot_train.add_plot_3d_structure(data_in=plot_info_cont)
        mat_comb_zoom = mat_comb[yplane,ii_z_min:ii_z_max,ii_x_min:ii_x_max]
        # mat_comb_zoom[mat_comb_zoom==0] = np.nan
        mat_cont_zoom = mat_cont[yplane,ii_z_min:ii_z_max,ii_x_min:ii_x_max]
        
        
        plot_train.add_pcolor_3d(data_in={"data_x":xx_grid,"data_y":zz_grid,"data_color":mat_comb_zoom,"colormap":colormap,
                                          "plot_number_x":0,"plot_number_y":0,"vmax":7,"vmin":1,"Ncolor":None,
                                          "zpos":yy[yplane],"alpha":0.5})
        plot_train.add_contline_3d(data_in={"data_x":xx,"data_y":zz,"data_cont":mat_cont_zoom,"val":0.8,
                                           "colors":"k","linewidth":linewidth,"zpos":yy[yplane]+1})
        plot_train.plot_layout_3d(data_in={"xticks":[xx[0],
                                                      np.mean(xx),
                                                      xx[-1]],
                                            "yticks":[zz[0],
                                                      np.mean(zz),
                                                      zz[-1]],
                                            "zticks":[yy[yplane]],
                                            "xticklabels":[str(int(np.round(xx[0]))),
                                                          str(int(np.round(np.mean(xx)))),
                                                          str(int(np.round(xx[-1])))],
                                            "yticklabels":[str(int(np.round(zz[0]))),
                                                          str(int(np.round(np.mean(zz)))),
                                                          str(int(np.round(zz[-1])))],
                                            "zticklabels":[str(int(yylabel[yplane]))],
                                            "L_x":xx[-1]-xx[0],"L_y":yy[-1]-yy[0],"L_z":zz[-1]-zz[0],"xpad":30,"ypad":10,
                                            "zpad":7})
    
    
        # -------------------------------------------------------------------------------------------------------------------
        # Plot the plane
        # -------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------
        # Create the plot
        # ---------------------------------------------------------------------------------------------------------------
    
        
        
        try:
            os.mkdir(plot_folder)
        except:
            print("Existing folder...",flush=True)
        plot_train.plot_save_png()
        plot_train.plot_save_pdf()
    
    # # -------------------------------------------------------------------------------------------------------------------
    # # Import packages
    # # -------------------------------------------------------------------------------------------------------------------
    # from py_bin.py_class.plot_format import plot_format
    # from matplotlib.colors import ListedColormap
    # from py_bin.py_functions.calc_coinc import read_coinc_all_3types
    
    # # -------------------------------------------------------------------------------------------------------------------
    # # Read the parameters of the plot
    # # -------------------------------------------------------------------------------------------------------------------
    # plot_folder           = str(data_in["plot_folder"])
    # plot_coin_folder      = str(data_in["plot_coin_folder"])
    # plot_coin_file        = str(data_in["plot_coin_file"])
    # xlabel                = str(data_in["xlabel"])
    # ylabel                = str(data_in["ylabel"])
    # fontsize              = int(data_in["fontsize"])
    # figsize_x             = int(data_in["figsize_x"])
    # figsize_y             = int(data_in["figsize_y"])
    # colornum              = int(data_in["colornum"])
    # dpi                   = float(data_in["dpi"])
    # struc1_lab            = str(data_in["struc1_lab"])
    # struc2_lab            = str(data_in["struc2_lab"])
    # struc3_lab            = str(data_in["struc3_lab"])
    # flowfield             = data_in["flowfield"]
    # mat_comb              = np.array(data_in["mat_comb"],dtype="float")
    # index_ii              = data_in["index_ii"]
    # calc_coin_tot         = str(data_in["calc_coin_tot"])
    # data_folder           = str(data_in["data_folder"])
    # yminbar               = float(data_in["yminbar"])
    # ymaxbar               = float(data_in["ymaxbar"])
    # ynumbar               = int(data_in["ynumbar"])
    # ylabelbar             = str(data_in["ylabelbar"])
    # contourfield          = data_in["contourfield"]
    # mat_comb[mat_comb==0] = np.nan
    # fig_name0             = plot_coin_folder+'/'+plot_coin_file
    # fig_name1             = plot_coin_folder+'/percbar_'+plot_coin_file
    # linewidth             = float(data_in["linewidth"])
    
     
    # xx,zz       = np.meshgrid(flowfield.xplus,flowfield.zplus)
    # xmin        = 0
    # xmax        = flowfield.L_x*flowfield.rey
    # ymin        = 0
    # ymax        = flowfield.L_z*flowfield.rey
    # yplustot    = (1-abs(flowfield.y_h))*flowfield.rey
    
    # # -------------------------------------------------------------------------------------------------------------------
    # # Define the colors
    # # -------------------------------------------------------------------------------------------------------------------
    # colors   = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8',
    #             '#f58231', '#911eb4', '#46f0f0']
    # lab_list = [struc1_lab+"\\("+struc2_lab+" $\cup$ "+struc3_lab+")",
    #             struc2_lab+"\\("+struc1_lab+" $\cup$ "+struc3_lab+")",
    #             "("+struc2_lab+" $\cup$ "+struc1_lab+")\\("+struc3_lab+")",
    #             struc3_lab+"\\("+struc1_lab+" $\cup$ "+struc2_lab+")",
    #             "("+struc3_lab+" $\cup$ "+struc1_lab+")\\("+struc2_lab+")",
    #             "("+struc3_lab+" $\cup$ "+struc2_lab+")\\("+struc1_lab+")",
    #             struc3_lab+" $\cup$ "+struc2_lab+" $\cup$ "+struc1_lab]
    # colormap = ListedColormap(colors)
    
    # try:
    #     os.mkdir(plot_folder)
    # except:
    #     print("Existing folder...",flush=True)
    # try:
    #     os.mkdir(plot_folder+'/'+plot_coin_folder)
    # except:
    #     print("Existing folder...",flush=True)
    # # -------------------------------------------------------------------------------------------------------------------
    # # Read the coincidence file
    # # -------------------------------------------------------------------------------------------------------------------
    # data_coinc     = read_coinc_all_3types(data_in={"calc_coin_file":calc_coin_tot,"folder":data_folder})
    # frac_coinc_tot = data_coinc["frac_coinc_tot"]
    
    # mat_cont       = np.array(contourfield.mat_struc,dtype='int')
    
    # # -------------------------------------------------------------------------------------------------------------------
    # # Create a legend with a figure
    # # -------------------------------------------------------------------------------------------------------------------
    # fig_name_leg = plot_coin_folder+'/legend_'+plot_coin_file
    # data_plot    = {"xlabel":[],"ylabel":[],"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x-2,
    #                 "figsize_y":figsize_y+2,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
    #                 "colornum":colornum,"legend":True,"fig_name":fig_name_leg,"dpi":dpi,"plot_folder":plot_folder,
    #                 "xmin":None,"xmax":None,"ymin":None,"ymax":None,"zmin":None,"zmax":None}
    # plot_pred = plot_format(data_in=data_plot)
    # plot_pred.create_figure()
    # plot_pred.legend_infig(data_in={"colors":colors,"labels":lab_list,"barstype":"horizontal"})
    # plot_pred.plot_save_png()
    # plot_pred.plot_save_pdf()
    # plot_pred.close()
    
    # # -------------------------------------------------------------------------------------------------------------------
    # # Do it for all the wall distances
    # # -------------------------------------------------------------------------------------------------------------------
    # for index_y in np.arange(len(mat_comb[:,0,0])):
    #     titlefig    = "$y^+\simeq$"+'{0:.0f}'.format(yplustot[index_y])
    #     fig_name    = fig_name0+"_field"+str(index_ii)+"_y"+str(index_y)
    #     # ---------------------------------------------------------------------------------------------------------------
    #     # Create the plot
    #     # ---------------------------------------------------------------------------------------------------------------
    #     data_plot  = {"xlabel":xlabel,"ylabel":ylabel,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x,
    #                   "figsize_y":figsize_y,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
    #                   "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,"plot_folder":plot_folder,
    #                   "xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"zmin":None,"zmax":None}
    #     plot_pred = plot_format(data_in=data_plot)
    #     plot_pred.create_figure()
    #     plot_pred.add_pcolor(data_in={"data_x":xx,"data_y":zz,"data_color":mat_comb[index_y,:,:],"colormap":colormap,
    #                                   "plot_number_x":0,"plot_number_y":0,"vmax":7,"vmin":1,"Ncolor":None})
    #     plot_pred.add_contline(data_in={"data_x":xx,"data_y":zz,"data_cont":mat_cont[index_y,:,:],"val":0.5,
    #                                     "colors":"k","linewidth":linewidth})
    #     plot_pred.plot_layout_pcolor(data_in={"title":titlefig,"colorbar":False,"b_text":None,
    #                                           "colorticks":[],
    #                                           "colorlabels":[],
    #                                           "equal":True,
    #                                           "xticks":[0,
    #                                                     flowfield.L_x/4*flowfield.rey,
    #                                                     2*flowfield.L_x/4*flowfield.rey,
    #                                                     3*flowfield.L_x/4*flowfield.rey,
    #                                                     flowfield.L_x*flowfield.rey],
    #                                           "yticks":[0,
    #                                                     flowfield.L_z/2*flowfield.rey,
    #                                                     flowfield.L_z*flowfield.rey],
    #                                           "xticklabels":["0",
    #                                                          str(int(np.round(flowfield.L_x/np.pi/4*flowfield.rey)))+
    #                                                          "$\pi$",
    #                                                          str(int(np.round(2*flowfield.L_x/np.pi/4*flowfield.rey)))+
    #                                                          "$\pi$",
    #                                                          str(int(np.round(3*flowfield.L_x/np.pi/4*flowfield.rey)))+
    #                                                          "$\pi$",
    #                                                          str(int(np.round(flowfield.L_x/np.pi*flowfield.rey)))+
    #                                                          "$\pi$"],
    #                                           "yticklabels":["0",
    #                                                          str(int(np.round(flowfield.L_z/np.pi/2*flowfield.rey)))+
    #                                                          "$\pi$",
    #                                                          str(int(np.round(flowfield.L_z/np.pi*flowfield.rey)))+
    #                                                          "$\pi$"]})
    #     plot_pred.plot_save_png()
    #     plot_pred.plot_save_pdf()
    #     plot_pred.close()
        
    #     # ---------------------------------------------------------------------------------------------------------------
    #     # Create a colorbar with the percentage of each type of structure
    #     # ---------------------------------------------------------------------------------------------------------------
    #     fig_name       = fig_name1+"_field"+str(index_ii)+"_y"+str(index_y)
    #     frac_coinc_yii = np.zeros((len(frac_coinc_tot)))
    #     for ii_coinc in np.arange(len(frac_coinc_tot)):
    #         if index_y <= len(frac_coinc_tot[ii_coinc])-1:
    #             index_y_mod          = index_y
    #         else:
    #             index_y_mod          = len(frac_coinc_tot[ii_coinc])-index_y-2
    #         frac_coinc_yii[ii_coinc] = frac_coinc_tot[ii_coinc][index_y_mod]
    #     frac_coinc_shap = np.sum(frac_coinc_yii[::2])
    #     frac_coinc_yii /= frac_coinc_shap
    #     data_plot  = {"xlabel":xlabel,"ylabel":ylabelbar,"zlabel":[],"fontsize":fontsize,"figsize_x":figsize_x+6,
    #                   "figsize_y":figsize_y+6,"xscale":"linear","yscale":"linear","zscale":"linear","colormap":colormap,
    #                   "colornum":colornum,"legend":True,"fig_name":fig_name,"dpi":dpi,
    #                   "plot_folder":plot_folder,"xmin":None,"xmax":None,"ymin":yminbar,"ymax":ymaxbar,
    #                   "zmin":None,"zmax":None}
    #     plot_pred = plot_format(data_in=data_plot)
    #     plot_pred.create_figure()
    #     plot_pred.add_cust_colorbar(data_in={"labels":lab_list,"data":frac_coinc_yii,"colors":colors,"title":titlefig,
    #                                          "ynum":ynumbar,"posplot":[0.15,0.95,0.8,0.6]})
    #     plot_pred.plot_save_png()
    #     plot_pred.plot_save_pdf()
    #     plot_pred.close()