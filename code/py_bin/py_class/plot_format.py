# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
plot_format.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 28 12:39:52 2024

@author:  Andres Cremades Botella

File to define plots format. The file includes a class:
    - Class:
        - plot_format : class containing the figure options.
"""
# ---------------------------------------------------------------------------------------------------------------------
# Import packages for all the functions
# ---------------------------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import sys

# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# Define the functions
# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

class plot_format():
    """
    .....................................................................................................................
    # plot_format: Class generating the format for the plots.
        * Functions:
            - __init__      : initialization of the class
            - create_figure : function to create the figure
            - add_plot_2d   : function to add a curve to a plot
            - plot_layout   : function to add format to a plot
            - plot_save_png : function to save the figure in png
            - plot_save_pdf : function to save the figure in pdf
        * Variables:
            - xlabel        : label of the x axis
            - ylabel        : label of the y axis
            - zlabel        : label of the z axis
            - fontsize      : size of the text in the plot
            - figsize_x     : size of the figure in the x direction
            - figsize_y     : size of the figure in the y direction
            - xscale        : scale of the x axis
            - yscale        : scale of the y axis
            - colormap      : colormap of the plot
            - colornum      : number of colors used in the colormap
            - legend        : select if using a legend
            - fig_name      : name of the figure to be saved
            - dpi           : dots per inch of the image saved
            - plot_folder   : folder to save the plots
            - figure        : figure of the plot
            - plot          : list of the plots added to the figure
    """
    def __init__(self,data_in={"xlabel":"x","ylabel":"y","zlabel":"z","fontsize":18,"figsize_x":10,
                               "figsize_y":8,"xscale":"linear","yscale":"linear","colormap":"viridis",
                               "colornum":1,"legend":False,"fig_name":"fig","dpi":60,"plot_folder":"plots",
                               "xmin":0,"xmax":125,"ymin":0,"ymax":125,"zmin":0,"zmax":125}):
        """
        .................................................................................................................
        # __init__
        .................................................................................................................
        Function to initialize the format of the plots

        Parameters
        ----------
        data_in : dict, optional
            Data required for generating the plots.
            The default is {"xlabel":"x","ylabel":"y","zlabel":"z","fontsize":18,"figsize_x":10,
                            "figsize_y":8,"xscale":"linear","yscale":"linear","colormap":"viridis",
                            "colornum":1,"legend":False,"fig_name":"fig","dpi":60,"plot_folder":"plots",
                            "xmin":0,"xmax":125,"ymin":0,"ymax":125,"zmin":0,"zmax":125}.
            Data:
                - xlabel      : label of the x axis
                - ylabel      : label of the y axis
                - zlabel      : label of the z axis
                - fontsize    : size of the text in the plot
                - figsize_x   : size of the figure in the x direction
                - figsize_y   : size of the figure in the y direction
                - xscale      : scale of the x axis
                - yscale      : scale of the y axis
                - zscale      : scale of the z axis
                - colormap    : colormap of the plot
                - colornum    : number of colors used in the colormap
                - legend      : select if using a legend
                - fig_name    : name of the figure to be saved
                - dpi         : dots per inch of the image saved
                - plot_folder : folder to save the plots
                - xmin        : minimum value of x
                - xmax        : maximum value of x 
                - ymin        : minimum value of y
                - ymax        : maximum value of y 
                - zmin        : minimum value of z
                - zmax        : maximum value of z 

        Returns
        -------
        None.

        """      
        # ---------------------------------------------------------------------------------------------------------------
        # Read the data
        # ---------------------------------------------------------------------------------------------------------------
        self.xlabel       = str(data_in["xlabel"])      # label of the x axis
        self.ylabel       = str(data_in["ylabel"])      # label of the y axis
        self.zlabel       = str(data_in["zlabel"])      # label of the z axis
        self.fontsize     = int(data_in["fontsize"])    # size of the text of the figure
        self.figsize_x    = int(data_in["figsize_x"])   # size of the figure in x
        self.figsize_y    = int(data_in["figsize_y"])   # size of the figure in y
        self.xscale       = str(data_in["xscale"])      # scaling of the x axis
        self.yscale       = str(data_in["yscale"])      # scaling of the y axis
        self.zscale       = str(data_in["zscale"])      # scaling of the z axis
        self.colormap     = str(data_in["colormap"])    # colormap selection
        self.legend       = bool(data_in["legend"])     # choose to create a legend
        self.fig_name     = str(data_in["fig_name"])    # name of the figure to be saved
        self.dpi          = float(data_in["dpi"])       # dots per inch
        self.plot_folder  = str(data_in["plot_folder"]) # folder to save the plots
        if data_in["xmin"] is not None:
            self.xmin     = float(data_in["xmin"]) 
        if data_in["xmax"] is not None:
            self.xmax     = float(data_in["xmax"]) 
        if data_in["ymin"] is not None:
            self.ymin     = float(data_in["ymin"])
        if data_in["ymax"] is not None: 
            self.ymax     = float(data_in["ymax"])
        if data_in["zmin"] is not None:
            self.zmin     = float(data_in["zmin"])
        if data_in["zmax"] is not None: 
            self.zmax     = float(data_in["zmax"])
        colornum_read     = data_in["colornum"]
        if colornum_read is None:
            self.colornum = None
        else:
            self.colornum = int(colornum_read)          # number of colors of the colormap
        
    def create_figure(self):
        """
        .................................................................................................................
        # create_figure
        .................................................................................................................
        Function to create the figure

        Returns
        -------
        None.

        """
        matplotlib.rc('font',size=self.fontsize)
        matplotlib.rc('axes',titlesize=self.fontsize)
        self.figure = plt.figure(figsize=(self.figsize_x,self.figsize_y))
        self.axes   = plt.axes()
        self.plot   = [] 
        self.multi  = False
            
    def create_figure_multiplot(self,data_in={"row":1,"col":1}):
        """
        .................................................................................................................
        # create_figure_multiplot
        .................................................................................................................
        Function to create the figure with multiple plots

        Parameters
        ----------
        data_in : dict, optional
            Data required for creating the figure.
            The default is {"row":1,"col":1}.
            Data:
                - row : row of the plot
                - col : columns of the plot

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Read the data
        # ---------------------------------------------------------------------------------------------------------------
        self.row = int(data_in["row"])
        self.col = int(data_in["col"])
        
        # ---------------------------------------------------------------------------------------------------------------
        # Define the options
        # ---------------------------------------------------------------------------------------------------------------
        matplotlib.rc('font',size=self.fontsize)
        matplotlib.rc('axes',titlesize=self.fontsize)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create the figure
        # ---------------------------------------------------------------------------------------------------------------
        self.figure,self.axes = plt.subplots(nrows=self.row,ncols=self.col,figsize=(self.figsize_x,self.figsize_y))
        self.plot             = [] 
        self.multi            = True
        
    def add_plot_2d(self,data_in={"data_x":[],"data_y":[],"label":"plot","color":None,"linewidth":1,
                                  "plot_number":0,"style":"-"}):
        """
        .................................................................................................................
        # add_plot_2d
        .................................................................................................................
        Function to add a plot to a figure

        Parameters
        ----------
        data_in : dict, optional
            Data required for the plot.
            The default is {"data_x":[],"data_y":[],"label":"plot","color":None}.
            Data:
                - data_x      : data for the x variable
                - data_y      : data for the y variable
                - label       : name of the curve plotted
                - color       : color of the curve
                - linewidth   : width of the lines
                - plot_number : number of the plot
                - style       : style of the line of the plot

        Returns
        -------
        None.

        """
        # --------------------------------------------------------------------------------------------------------------
        # Read the data
        # --------------------------------------------------------------------------------------------------------------
        data_x      = np.array(data_in["data_x"],dtype='float').reshape(-1) # data for the variable x
        data_y      = np.array(data_in["data_y"],dtype='float').reshape(-1) # data for the variable y
        label       = str(data_in["label"])                                 # label of the plot
        color_read  = data_in["color"]                                      # color of the plot
        linewidth   = float(data_in["linewidth"])                           # width of the line
        plot_number = int(data_in["plot_number"])                           # number of the plot (used for the colors)
        style       = str(data_in["style"])                                 # style of the line
        if plot_number >= self.colornum:
            plot_number = self.colornum-1
        if color_read is None:
            color  = matplotlib.cm.get_cmap(self.colormap,self.colornum).colors[plot_number,:]
        else:
            color  = str(color_read)
        
        # --------------------------------------------------------------------------------------------------------------
        # Create the plot
        # --------------------------------------------------------------------------------------------------------------
        self.figure
        plt_id = plt.plot(data_x,data_y,color=color,label=label,linewidth=linewidth,linestyle=style)
        self.plot.append(plt_id)
        

        
    def plot_layout(self): 
        """
        .................................................................................................................
        # plot_layout
        .................................................................................................................
        Function to generate the format of the plots

        Returns
        -------
        None.

        """
        self.figure
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.xscale(self.xscale)
        plt.yscale(self.yscale)
        try:
            plt.xlim([self.xmin,self.xmax])
        except:
            pass
        try:
            plt.ylim([self.ymin,self.ymax])
        except:
            pass
        legend = plt.legend()
        plt.grid()
        
        # ---------------------------------------------------------------------------------------------------------------
        # Check if the legend overlaps with the plot and if so move it outsid
        # ---------------------------------------------------------------------------------------------------------------
        renderer   = self.figure.canvas.get_renderer()
        legend_box = legend.get_window_extent(renderer)
        ax_box     = self.axes.get_window_extent(renderer)
        
        # ---------------------------------------------------------------------------------------------------------------
        # If the legend overlaps with the plot, move it outside
        # ---------------------------------------------------------------------------------------------------------------
        if legend_box.width > ax_box.width * 0.4 and legend_box.height > ax_box.height * 0.4:
            self.axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                       
        plt.tight_layout()
        

        
        
    def plot_save_png(self):
        """
        .................................................................................................................
        # plot_save_png
        .................................................................................................................
        Function to save the plots with png format

        Returns
        -------
        None.

        """
        self.figure
        save_name = self.plot_folder+'/'+self.fig_name+'.png'
        plt.savefig(save_name,dpi=self.dpi)
        
    def plot_save_pdf(self):
        """
        .................................................................................................................
        # plot_save_pdf
        .................................................................................................................
        Function to save the plots with pdf format

        Returns
        -------
        None.

        """
        self.figure
        save_name = self.plot_folder+'/'+self.fig_name+'.pdf'
        plt.savefig(save_name)

        
    def add_pcolor(self,data_in={"data_x":[],"data_y":[],"data_color":[],"colormap":None,"plot_number_x":0,
                                 "plot_number_y":0,"vmax":None,"vmin":None,"Ncolor":None}):
        """
        .................................................................................................................
        # add_pcolor
        .................................................................................................................
        Function to add a colormap to a figure

        Parameters
        ----------
        data_in : dict, optional
            Data required for the plot.
            The default is {"data_x":[],"data_y":[],"label":"plot","color":None}.
            Data:
                - data_x        : data for the x variable
                - data_y        : data for the y variable
                - data_color    : data for the colormap
                - colormap      : colormap used for the plot
                - plot_number_x : number of the plot in the x direction
                - plot_number_y : number of the plot in the y direction
                - Ncolor        : number of colors in the colorbar

        Returns
        -------
        None.

        """
        # --------------------------------------------------------------------------------------------------------------
        # Read packages
        # --------------------------------------------------------------------------------------------------------------
        from matplotlib.colors import ListedColormap
        
        # --------------------------------------------------------------------------------------------------------------
        # Read the data
        # --------------------------------------------------------------------------------------------------------------
        data_x        = np.array(data_in["data_x"],dtype='float')     # data for the variable x
        data_y        = np.array(data_in["data_y"],dtype='float')     # data for the variable y
        data_color    = np.array(data_in["data_color"],dtype='float') # data for the color
        cmap_read     = data_in["colormap"]                           # color of the plot
        plot_number_x = int(data_in["plot_number_x"])                 # number of the plot (used for the position x)
        plot_number_y = int(data_in["plot_number_y"])                 # number of the plot (used for the position y)
        vmax          = data_in["vmax"]
        vmin          = data_in["vmin"]
        Ncolor        = data_in["Ncolor"]
        if Ncolor is None:
            flag_ncol = False
        else:
            flag_ncol = True
            Ncolor    = int(Ncolor)
        if vmax is None:
            vmax      = float(np.max(data_color))
        else:
            vmax      = float(vmax)                                   # number of the plot (used for the position y)
        if vmin is None:
            vmin      = float(np.min(data_color))
        else:
            vmin      = float(vmin)
        if self.multi:
            if self.col == 1:
                ax = self.axes[plot_number_y]
            elif self.row == 1:
                ax = self.axes[plot_number_x]
            else:
                ax = self.axes[plot_number_y,plot_number_x]
            if plot_number_x >= self.col:
                print("Increase the number of rows. Exit...",flush=True)
                sys.exit()
            elif plot_number_y >= self.row:
                print("Increase the number of columns. Exit...",flush=True)
                sys.exit()
        else:
            ax = self.axes
        if cmap_read is None:
            colormap  = self.colormap
        elif isinstance(cmap_read,ListedColormap):
            colormap  = cmap_read
        else:
            colormap  = str(cmap_read)
        if flag_ncol:
            colormap  = plt.cm.get_cmap(colormap,Ncolor)
        
        # --------------------------------------------------------------------------------------------------------------
        # Create the plot
        # --------------------------------------------------------------------------------------------------------------
        self.figure
        plt_id = ax.pcolor(data_x,data_y,data_color,cmap=colormap,vmax=vmax,vmin=vmin)
        self.plot.append(plt_id)
        
                
    def add_pcolor_3d(self,data_in={"data_x":[],"data_y":[],"data_color":[],"colormap":None,"plot_number_x":0,
                                    "plot_number_y":0,"vmax":None,"vmin":None,"Ncolor":None,"zpos":0,"alpha":1}):
        """
        .................................................................................................................
        # add_pcolor
        .................................................................................................................
        Function to add a colormap to a figure

        Parameters
        ----------
        data_in : dict, optional
            Data required for the plot.
            The default is {"data_x":[],"data_y":[],"label":"plot","color":None}.
            Data:
                - data_x        : data for the x variable
                - data_y        : data for the y variable
                - data_color    : data for the colormap
                - colormap      : colormap used for the plot
                - plot_number_x : number of the plot in the x direction
                - plot_number_y : number of the plot in the y direction
                - Ncolor        : number of colors in the colorbar
                - zpos          : position in z
                - alpha         : transparency

        Returns
        -------
        None.

        """
        # --------------------------------------------------------------------------------------------------------------
        # Read packages
        # --------------------------------------------------------------------------------------------------------------
        from matplotlib.colors import ListedColormap, Normalize
        
        # --------------------------------------------------------------------------------------------------------------
        # Read the data
        # --------------------------------------------------------------------------------------------------------------
        data_x        = np.array(data_in["data_x"],dtype='float')     # data for the variable x
        data_y        = np.array(data_in["data_y"],dtype='float')     # data for the variable y
        data_color    = np.array(data_in["data_color"],dtype='float') # data for the color
        cmap_read     = data_in["colormap"]                           # color of the plot
        plot_number_x = int(data_in["plot_number_x"])                 # number of the plot (used for the position x)
        plot_number_y = int(data_in["plot_number_y"])                 # number of the plot (used for the position y)
        vmax          = data_in["vmax"]
        vmin          = data_in["vmin"]
        Ncolor        = data_in["Ncolor"]
        zpos          = float(data_in["zpos"])
        alpha_val     = float(data_in["alpha"])
        if Ncolor is None:
            flag_ncol = False
        else:
            flag_ncol = True
            Ncolor    = int(Ncolor)
        if vmax is None:
            vmax      = float(np.max(data_color))
        else:
            vmax      = float(vmax)                                   # number of the plot (used for the position y)
        if vmin is None:
            vmin      = float(np.min(data_color))
        else:
            vmin      = float(vmin)
        if self.multi:
            if self.col == 1:
                ax = self.axes[plot_number_y]
            elif self.row == 1:
                ax = self.axes[plot_number_x]
            else:
                ax = self.axes[plot_number_y,plot_number_x]
            if plot_number_x >= self.col:
                print("Increase the number of rows. Exit...",flush=True)
                sys.exit()
            elif plot_number_y >= self.row:
                print("Increase the number of columns. Exit...",flush=True)
                sys.exit()
        else:
            ax = self.axes
        if cmap_read is None:
            colormap  = self.colormap
        elif isinstance(cmap_read,ListedColormap):
            colormap  = cmap_read
        else:
            colormap  = plt.get_cmap(str(cmap_read))
        if flag_ncol:
            colormap  = plt.cm.get_cmap(colormap,Ncolor)
        
        data_z = np.ones_like(data_x)*zpos
        # --------------------------------------------------------------------------------------------------------------
        # Create the plot
        # --------------------------------------------------------------------------------------------------------------
        self.figure
        norm_c = colormap((data_color-vmin)/(vmax-vmin))#Normalize(vmin=vmin, vmax=vmax)
        alpham = np.where((data_color < vmin) | (data_color > vmax), 0, alpha_val)
        norm_c[...,3] = alpham
        plt_id = ax.plot_surface(data_x,data_y,data_z,facecolors=norm_c, rstride=1, cstride=1,
                                 antialiased=True, linewidth=0)
        self.plot.append(plt_id)
               
        
    def add_contline(self,data_in={"data_x":[],"data_y":[],"data_cont":[],"val":0.5,"colors":"k","linewidth":2}):
        """
        .................................................................................................................
        # add_contline
        .................................................................................................................
        Function to add a contour line to a figure

        Parameters
        ----------
        data_in : dict, optional
            Data required for the plot.
            The default is {"data_x":[],"data_y":[],"label":"plot","color":None,"colors":"k","linewidth":2}.
            Data:
                - data_x        : data for the x variable
                - data_y        : data for the y variable
                - data_cont     : data for the contour
                - val           : value of the contour
                - colors        : color for the contour
                - linewidth     : linewidth for the contour

        Returns
        -------
        None.

        """
        # --------------------------------------------------------------------------------------------------------------
        # Read packages
        # --------------------------------------------------------------------------------------------------------------
        from matplotlib.colors import ListedColormap
        
        # --------------------------------------------------------------------------------------------------------------
        # Read the data
        # --------------------------------------------------------------------------------------------------------------
        data_x        = np.array(data_in["data_x"],dtype='float')     # data for the variable x
        data_y        = np.array(data_in["data_y"],dtype='float')     # data for the variable y
        data_cont     = np.array(data_in["data_cont"],dtype='float')  # data for the color
        val           = float(data_in["val"])
        colors        = str(data_in["colors"])
        linewidth     = float(data_in["linewidth"])
        if self.multi:
            if self.col == 1:
                ax = self.axes[plot_number_y]
            elif self.row == 1:
                ax = self.axes[plot_number_x]
            else:
                ax = self.axes[plot_number_y,plot_number_x]
            if plot_number_x >= self.col:
                print("Increase the number of rows. Exit...",flush=True)
                sys.exit()
            elif plot_number_y >= self.row:
                print("Increase the number of columns. Exit...",flush=True)
                sys.exit()
        else:
            ax = self.axes
        
        # --------------------------------------------------------------------------------------------------------------
        # Create the plot
        # --------------------------------------------------------------------------------------------------------------
        self.figure
        plt_id = ax.contour(data_x,data_y,data_cont,colors=colors,linewidths=linewidth,levels=[val])
        self.plot.append(plt_id)
           
    def add_contline_3d(self,data_in={"data_x":[],"data_y":[],"data_cont":[],"val":0.5,"colors":"k","linewidth":2,
                                      "zpos":1}):
        """
        .................................................................................................................
        # add_contline_3d
        .................................................................................................................
        Function to add a contour line to a figure

        Parameters
        ----------
        data_in : dict, optional
            Data required for the plot.
            The default is {"data_x":[],"data_y":[],"label":"plot","color":None,"colors":"k","linewidth":2,"zpos":1}.
            Data:
                - data_x        : data for the x variable
                - data_y        : data for the y variable
                - data_cont     : data for the contour
                - val           : value of the contour
                - colors        : color for the contour
                - linewidth     : linewidth for the contour
                - zpos          : position in z of the contour

        Returns
        -------
        None.

        """
        # --------------------------------------------------------------------------------------------------------------
        # Read packages
        # --------------------------------------------------------------------------------------------------------------
        from matplotlib.colors import ListedColormap
        
        # --------------------------------------------------------------------------------------------------------------
        # Read the data
        # --------------------------------------------------------------------------------------------------------------
        data_x        = np.array(data_in["data_x"],dtype='float')     # data for the variable x
        data_y        = np.array(data_in["data_y"],dtype='float')     # data for the variable y
        data_cont     = np.array(data_in["data_cont"],dtype='float')  # data for the color
        val           = float(data_in["val"])
        colors        = str(data_in["colors"])
        linewidth     = float(data_in["linewidth"])
        zpos          = float(data_in["zpos"])
        if self.multi:
            if self.col == 1:
                ax = self.axes[plot_number_y]
            elif self.row == 1:
                ax = self.axes[plot_number_x]
            else:
                ax = self.axes[plot_number_y,plot_number_x]
            if plot_number_x >= self.col:
                print("Increase the number of rows. Exit...",flush=True)
                sys.exit()
            elif plot_number_y >= self.row:
                print("Increase the number of columns. Exit...",flush=True)
                sys.exit()
        else:
            ax = self.axes
        
        # --------------------------------------------------------------------------------------------------------------
        # Create the plot
        # --------------------------------------------------------------------------------------------------------------
        self.figure
        plt_id = ax.contour(data_x,data_y,data_cont,zdir='z',offset=zpos,
                            colors=colors,linewidths=linewidth,levels=[val])
        self.plot.append(plt_id)
        
    def plot_layout_pcolor(self,data_in={"title":"plot","colorbar":True,"b_text":"c","colorticks":[],
                                         "colorlabels":[],"equal":True,"xticks":[0,250*np.pi/3,500*np.pi/3,250*np.pi],
                                         "yticks":[0,125*np.pi],"xticklabels":["0","$250/3\pi$","$500/3\pi$","$250\pi$"],
                                         "yticklabels":["0","$150\pi$"]}): 
        """
        .................................................................................................................
        # plot_layout_pcolor
        .................................................................................................................
        Function to generate the format of the plots

        Parameters
        ----------
        data_in : dict, optional
            Data required for the plot.
            The default is {"title":"plot","colorbar":True,"b_text":"c","colorticks":[],"colorlabels":[]}.
            Data:
                - title       : title of the figure
                - colorbar    : flag for adding the colormap
                - b_text      : label of the figures, colorbar
                - colorticks  : ticks in which to add labels in the colorbar
                - colorlabels : labels of the ticks of the colorbar
                - equal       : set equal dimension in axis
                - xticks      : ticks in axis x
                - yticks      : ticks in axis y
                - xticklabels : label of the ticks in axis x
                - yticklabels : label of the ticks in axis y

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Read data
        # ---------------------------------------------------------------------------------------------------------------
        title       = data_in["title"]
        colorbar    = bool(data_in["colorbar"])
        b_text      = data_in["b_text"]
        colorticks  = np.array(data_in["colorticks"],dtype="float")
        colorlabels = data_in["colorlabels"]
        equal       = bool(data_in["equal"])
        xticks      = data_in["xticks"]
        yticks      = data_in["yticks"]
        xticklabels = data_in["xticklabels"]
        yticklabels = data_in["yticklabels"]
        if b_text is None:
            flag_b = False
        else:
            flag_b = True
            b_text = str(b_text)
        if len(colorticks):
            flag_ctick = True
        else:
            flag_ctick = False
        
        # ---------------------------------------------------------------------------------------------------------------
        # Add the format
        # ---------------------------------------------------------------------------------------------------------------
        self.figure
        self.axes.set_xlabel(self.xlabel)
        self.axes.set_ylabel(self.ylabel)
        self.axes.set_xscale(self.xscale)
        self.axes.set_yscale(self.yscale)
        if xticks is not None:
            self.axes.set_xticks(xticks)
            self.axes.set_xticklabels(xticklabels)
        if yticks is not None:
            self.axes.set_yticks(yticks)
            self.axes.set_yticklabels(yticklabels)
        try:
            self.axes.set_xlim([self.xmin,self.xmax])
        except:
            pass
        try:
            self.axes.set_ylim([self.ymin,self.ymax])
        except:
            pass
        if colorbar:
            self.cb = plt.colorbar(self.plot[0],ax=self.axes,orientation="vertical",aspect=10)
            if flag_b:
                self.cb.set_label(label=b_text)
            if flag_ctick:
                self.cb.ax.set_yticks(colorticks)
                self.cb.ax.set_yticklabels(colorlabels)
         
                
        if title is not None:
            title = str(title)
            self.axes.set_title(title)
        if equal:
            self.axes.set_box_aspect((self.ymax-self.ymin)/(self.xmax-self.xmin))
            # self.axes.set_aspect('equal')
        self.figure.tight_layout()
        
    def plot_multilayout(self,data_in={"plot_number_x":0,"plot_number_y":0,"indexplot":0,"b_text":"c","title":"plot",
                                       "colorbar_x":True,"xticks":[0,250*np.pi/3,500*np.pi/3,250*np.pi],
                                       "yticks":[0,125*np.pi],"xticklabels":["0","$250/3\pi$","$500/3\pi$","$250\pi$"],
                                       "yticklabels":["0","$150\pi$"]}): 
        """
        .................................................................................................................
        # plot_layout
        .................................................................................................................
        Function to generate the format of the plots

        Parameters
        ----------
        data_in : dict, optional
            Data required for the plot.
            The default is {"title":"plot","colorbar":True}.
            Data:
                - plot_number_x : position of x of the plot
                - plot_number_y : posotion of y of the plot
                - indexplot     : index of the plot
                - b_text        : label of the figures, colorbar
                - title         : title of the figure
                - colorbar_x    : flag for adding the colormap
                - xticks        : ticks in axis x
                - yticks        : ticks in axis y
                - xticklabels   : label of the ticks in axis x
                - yticklabels   : label of the ticks in axis y
        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Read data
        # ---------------------------------------------------------------------------------------------------------------
        xx         = int(data_in["plot_number_x"])
        yy         = int(data_in["plot_number_y"])
        indexplot  = int(data_in["indexplot"])
        b_text     = str(data_in["b_text"])
        title      = str(data_in["title"])
        colorbar_x = bool(data_in["colorbar_x"])
        if "xticks" in data_in:
            xticks      = data_in["xticks"]
        else:
            xticks      = None
        if "yticks" in data_in:
            yticks      = data_in["yticks"]
        else:
            yticks      = None
        if "xticklabels" in data_in:
            xticklabels = data_in["xticklabels"]
        else:
            xticklabels = None
        if "yticklabels" in data_in:
            yticklabels = data_in["yticklabels"]
        else:
            yticklabels = None
        
        # ---------------------------------------------------------------------------------------------------------------
        # Add the format
        # ---------------------------------------------------------------------------------------------------------------
        if self.col == 1:
            flag_1 = True
            nn     = yy
        elif self.row == 1:
            flag_1 = True
            nn     = xx
        else:
            flag_1 = False
        if flag_1:
            dimy = self.axes.shape[0]
            self.figure
            if yy == dimy-1:
                self.axes[nn].set_xlabel(self.xlabel)
            else:
                self.axes[nn].tick_params(bottom = False, labelbottom = False)
            if xx == 0:
                self.axes[nn].set_ylabel(self.ylabel)
            else:
                self.axes[nn].tick_params(left = False, labelleft = False)
            self.axes[nn].set_xscale(self.xscale)
            self.axes[nn].set_yscale(self.yscale)
            self.axes[nn].set_xlim([self.xmin,self.xmax])
            self.axes[nn].set_ylim([self.ymin,self.ymax])
            if xticks is not None:
                self.axes[nn].set_xticks(xticks)
                self.axes[nn].set_xticklabels(xticklabels)
            if yticks is not None:
                self.axes[nn].set_yticks(yticks)
                self.axes[nn].set_yticklabels(yticklabels)
            if not colorbar_x:
                self.cb = plt.colorbar(self.plot[indexplot],ax=self.axes[nn],orientation="vertical",aspect=5)
                self.cb.set_label(label=b_text)
            elif xx == dimx-1:
                self.cb = plt.colorbar(self.plot[indexplot],ax=self.axes[nn].ravel().tolist(),orientation="vertical",aspect=5)
                self.cb.set_label(label=b_text)
            if nn==0:
                plt.suptitle(title)
        else:
            dimy,dimx = self.axes.shape
            self.figure
            if yy == dimy-1:
                self.axes[yy,xx].set_xlabel(self.xlabel)
            else:
                self.axes[yy,xx].tick_params(bottom = False, labelbottom = False)
            if xx == 0:
                self.axes[yy,xx].set_ylabel(self.ylabel)
            else:
                self.axes[yy,xx].tick_params(left = False, labelleft = False)
            self.axes[yy,xx].set_xscale(self.xscale)
            self.axes[yy,xx].set_yscale(self.yscale)
            self.axes[yy,xx].set_xlim([self.xmin,self.xmax])
            self.axes[yy,xx].set_ylim([self.ymin,self.ymax])
            if xticks is not None:
                self.axes[nn].set_xticks(xticks)
                self.axes[nn].set_xticklabels(xticklabels)
            if yticks is not None:
                self.axes[nn].set_yticks(yticks)
                self.axes[nn].set_yticklabels(yticklabels)
            if not colorbar_x:
                self.cb = plt.colorbar(self.plot[indexplot],ax=self.axes[yy,xx],orientation="vertical",aspect=5)
                self.cb.set_label(label=b_text)
            elif xx == dimx-1:
                self.cb = plt.colorbar(self.plot[indexplot],ax=self.axes[yy,:].ravel().tolist(),orientation="vertical",aspect=5)
                self.cb.set_label(label=b_text)
            if xx == 0 and yy==0:
                plt.suptitle(title)        
    
    def add_plot_3d_structure(self,data_in={"data_x":[],"data_y":[],"data_z":[],"struc":[],"color":None,
                                            "cmap_flag":False,"vmax":None,"vmin":None,"alpha":1}):
        """
        .................................................................................................................
        # add_plot_3d_structure
        .................................................................................................................
        Function to add the structure isosurface to a figure

        Parameters
        ----------
        data_in : dict, optional
            Data required for the plot.
            The default is {"data_x":[],"data_y":[],"data_z":[],"struc":[],"color":None,"plot_number":0}.
            Data:
                - data_x : data for the x variable
                - data_y : data for the y variable
                - data_z : data for the z variable
                - struc  : data for the structure field to calculate the isosurface
                - color  : color of the curve
                - vmax   : maximum value for the color
                - vmin   : minimum value for the color
                - alpha  : transparency of the plot

        Returns
        -------
        None.

        """
        # --------------------------------------------------------------------------------------------------------------
        # Import package
        # --------------------------------------------------------------------------------------------------------------
        from skimage import measure
        from scipy import interpolate
        
        # --------------------------------------------------------------------------------------------------------------
        # Read the data
        # --------------------------------------------------------------------------------------------------------------
        data_x      = np.array(data_in["data_x"],dtype='float')
        data_y      = np.array(data_in["data_y"],dtype='float') 
        data_z      = np.array(data_in["data_z"],dtype='float')
        struc       = np.array(data_in["struc"],dtype='float')
        vmax        = data_in["vmax"]
        vmin        = data_in["vmin"]
        color_read  = data_in["color"]   
        cmap_flag   = bool(data_in["cmap_flag"])
        if "alpha" in data_in.keys():
            alpha   = float(data_in["alpha"])
        else:
            alpha   = 1
        if color_read is None:
            color  = matplotlib.cm.get_cmap(self.colormap)
        else:
            if isinstance(color_read,str):
                color = str(color_read)
            else:
                color = tuple(color_read)
        if vmax is None:
            vmax = np.max(data_y)
        else:
            vmax = float(vmax)
        if vmin is None:
            vmin = np.min(data_y)
        else:
            vmin = float(vmin)
            
            
        # --------------------------------------------------------------------------------------------------------------
        # Calculate the maximum value of a structure
        # --------------------------------------------------------------------------------------------------------------
        for jj in np.unique(struc)[1:]:
            print(jj)
            mat_struct              = np.zeros_like(struc)
            mat_struct[struc==jj] = 1
            marcubes                = measure.marching_cubes(mat_struct,0.5,spacing=(1,1,1),step_size=1)
            verts0                  = marcubes[0]
            faces                   = marcubes[1]
            shpverts                = verts0.shape
            verts                   = np.zeros(shpverts)
            f_int_x                 = interpolate.interp1d(np.arange(len(data_x)),data_x)
            verts[:,0]              = f_int_x(verts0[:,2])
            f_int_y                 = interpolate.interp1d(np.arange(len(data_y)),data_y)
            verts[:,2]              = f_int_y(verts0[:,0])
            f_int_z                 = interpolate.interp1d(np.arange(len(data_z)),data_z)
            verts[:,1]              = f_int_z(verts0[:,1])
            
            # -----------------------------------------------------------------------------------------------------------
            # Plot the data
            # -----------------------------------------------------------------------------------------------------------
            if cmap_flag:
                plt_id = self.axes.plot_trisurf(verts[:,0],verts[:,1],verts[:,2],triangles=faces,cmap=color,
                                                vmin=vmin,vmax=vmax,alpha=alpha)
            else:
                plt_id = self.axes.plot_trisurf(verts[:,0],verts[:,1],verts[:,2],triangles=faces,color=color,alpha=alpha)
            self.plot.append(plt_id)
            
                
    def plot_layout_3d(self,data_in={"xticks":[0,250*np.pi/3,500*np.pi/3,250*np.pi],"yticks":[0,125*np.pi],
                                     "zticks":[-125,0,125],"xticklabels":["0","$250/3\pi$","$500/3\pi$","$250\pi$"],
                                     "yticklabels":["0","$150\pi$"],"zticklabels":["-125","0"+"125"],"L_x":2*np.pi,
                                     "L_y":1,"L_z":np.pi,"xpad":1,"ypad":1,"zpad":1}): 
        """
        .................................................................................................................
        # plot_layout_3d
        .................................................................................................................
        Function to generate the format of the plots

        Parameters
        ----------
        data_in : dict, optional
            Data required for the plot.
            The default is {"xticks":[0,250*np.pi/3,500*np.pi/3,250*np.pi],"yticks":[0,125*np.pi],
                            "zticks":[-125,0,125],"xticklabels":["0","$250/3\pi$","$500/3\pi$","$250\pi$"],
                            "yticklabels":["0","$150\pi$"],"zticklabels":["-125","0"+"125"],"L_x":2*np.pi,
                            "L_y":1,"L_z":np.pi}.
            Data:
                - xticks      : ticks in axis x
                - yticks      : ticks in axis y
                - zticks      : ticks in axis z
                - xticklabels : label of the ticks in axis x
                - yticklabels : label of the ticks in axis y
                - zticklabels : label of the ticks in axis z
                - L_x         : streamwise dimension of the channel
                - L_y         : wall-normal dimension of the channel
                - L_z         : spanwise dimension of the channel
        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Read data
        # ---------------------------------------------------------------------------------------------------------------
        xticks      = data_in["xticks"]
        yticks      = data_in["yticks"]
        zticks      = data_in["zticks"]
        xticklabels = data_in["xticklabels"]
        yticklabels = data_in["yticklabels"]
        zticklabels = data_in["zticklabels"]
        L_x         = float(data_in["L_x"])
        L_y         = float(data_in["L_y"])
        L_z         = float(data_in["L_z"])
        xpad        = int(data_in["xpad"])
        ypad        = int(data_in["ypad"])
        zpad        = int(data_in["zpad"])
        
        # ---------------------------------------------------------------------------------------------------------------
        # Apply format
        # ---------------------------------------------------------------------------------------------------------------
        self.figure
        self.axes.xaxis.set_rotate_label(False) 
        self.axes.yaxis.set_rotate_label(False) 
        self.axes.zaxis.set_rotate_label(False) 
        self.axes.set_xlabel(self.xlabel,labelpad=xpad)
        self.axes.set_ylabel(self.ylabel,labelpad=ypad)
        self.axes.set_zlabel(self.zlabel,labelpad=zpad)
        self.axes.set_xscale(self.xscale)
        self.axes.set_yscale(self.yscale)
        self.axes.set_zscale(self.zscale)
        self.axes.set_box_aspect([1,L_z/L_x,L_y/L_x])
        self.axes.set_xticks(xticks)
        self.axes.set_yticks(yticks)
        self.axes.set_zticks(zticks)
        self.axes.set_xticklabels(xticklabels)
        self.axes.set_yticklabels(yticklabels)
        self.axes.set_zticklabels(zticklabels,va="center")
        self.axes.tick_params(axis='both',which='major')
        self.axes.xaxis.pane.fill = False
        self.axes.yaxis.pane.fill = False
        self.axes.zaxis.pane.fill = False
        plt.grid()
        plt.tight_layout()
        
    def create_figure3d(self):
        """
        .................................................................................................................
        # create_figure3d
        .................................................................................................................
        Function to create the figure for 3d representation

        Returns
        -------
        None.

        """
        matplotlib.rc('font',size=self.fontsize)
        matplotlib.rc('axes',titlesize=self.fontsize)
        self.figure = plt.figure(figsize=(self.figsize_x,self.figsize_y))
        self.axes   = plt.axes(projection='3d')
        self.plot   = []
        self.multi  = False 
        
    def close(self):
        """
        .................................................................................................................
        # close
        .................................................................................................................
        Function to close the figure

        Returns
        -------
        None.

        """
        plt.close()
    
    def legend_infig(self,data_in={"colors":['#1f77b4','#ff7f0e','#2ca02c'],"labels":["a","b","c"],
                                   "barstype":"horizontal"}):
        """
        .................................................................................................................
        # legend_infig
        .................................................................................................................
        Function to generate the legend of the plot in a diferent figure
        
        Parameters
        ----------
        data_in : dict, optional
            Data required for the plot.
            The default is {"colors":['#1f77b4','#ff7f0e','#2ca02c'],"labels":["a","b","c"]}.
            Data:
                - colors      : list of the colors
                - labels      : labels of the plot
                - barstype    : check if horizontal or vertical bars
        Returns
        -------
        None.
        
        """
        # ----------------------------------------------------------------------------------------------------------------
        # Read the data
        # ----------------------------------------------------------------------------------------------------------------
        colors   = data_in["colors"]
        labels   = data_in["labels"]
        barstype = str(data_in["barstype"])
        
        # ----------------------------------------------------------------------------------------------------------------
        # Plot labels
        # ----------------------------------------------------------------------------------------------------------------
        if barstype == "vertical":
            plotid = plt.bar(np.arange(len(colors)),np.ones((len(colors),)),color=colors)
            flag   = True
            plt.xticks(np.arange(len(colors)), labels, rotation=90)
            plt.yticks([])
            plt.subplots_adjust(left=0.05, right=0.95, top=0.8, bottom=0.7)
        elif barstype == "horizontal":
            plotid = plt.barh(np.arange(len(colors)),np.ones((len(colors),)),color=colors)
            flag   = True
            plt.yticks(np.arange(len(colors)), labels)
            plt.xticks([])
            plt.subplots_adjust(left=0.85, right=0.95, top=1, bottom=0)
        else:
            print("Error, define type of bars")
            flag   = False
        if flag:
            self.plot.append(plotid)
            self.axes.spines['top'].set_visible(False)
            self.axes.spines['right'].set_visible(False)
            self.axes.spines['bottom'].set_visible(False)
            self.axes.spines['left'].set_visible(False)
            plt.show()
        
        
    def add_cust_colorbar(self,data_in={"labels":["a","b","c"],"data":[],"colors":['#1f77b4','#ff7f0e','#2ca02c'],
                                        "title":"yplus","ynum":4,"posplot":[0.15,0.95,0.8,0.6]}):
        """
        .................................................................................................................
        # add_cust_colorbar
        .................................................................................................................
        Function to generate a customized colorbar with labels
        
        Parameters
        ----------
        data_in : dict, optional
            Data required for the plot.
            The default is {"labels":["a","b","c"],"data":[],"colors":['#1f77b4','#ff7f0e','#2ca02c'],
                            "title":"yplus","ynum":4,"posplot":[0.15,0.95,0.8,0.6]}.
            Data:
                - labels      : labels of the plot
                - data        : data of the colorbars
                - colors      : list of the colors
                - title       : title of the figure
                - ynum        : number of ticks
                - posplot     : position of the figure
        Returns
        -------
        None.
        
        """
        # ---------------------------------------------------------------------------------------------------------------
        # Read the data
        # ---------------------------------------------------------------------------------------------------------------
        labels  = data_in["labels"]
        data    = np.array(data_in["data"],dtype="float")
        colors  = data_in["colors"]
        title   = str(data_in["title"])
        ynum    = int(data_in["ynum"])
        posplot = np.array(data_in["posplot"],dtype="float")
        
        # ---------------------------------------------------------------------------------------------------------------
        # Plot labels
        # ---------------------------------------------------------------------------------------------------------------
        plotid = plt.bar(np.arange(len(labels)),data,color=colors)
        self.plot.append(plotid)
        plt.xticks(np.arange(len(labels)), labels, rotation=90)
        plt.subplots_adjust(left=posplot[0],right=posplot[1],top=posplot[2],bottom=posplot[3])
        plt.ylabel(self.ylabel)
        yticks        = np.linspace(self.ymin,self.ymax,ynum)
        ytickslab     = ['{0:.2f}'.format(yticks[ii]) for ii in np.arange(ynum)]
        ytickslab[-1] = ">"+ytickslab[-1]
        self.axes.set_yticks(yticks)
        self.axes.set_yticklabels(ytickslab)
        try:
            plt.ylim([self.ymin,self.ymax])
        except:
            pass
        plt.xlim([-1,len(labels)])
        self.axes.set_title(title)
        plt.grid()
        plt.show()
            
    def add_hist2d_y(self,data_in={"xx":[],"yy":[],"xxyy":[],"levels":[1,2,3,4],"colormap":"viridis",
                                   "locator":[],"alp":1,"linewidth":2}):
        """
        .................................................................................................................
        # add_hist2d_y
        .................................................................................................................
        Function to plot the pdf

        Parameters
        ----------
        data_in : dict, optional
            Data required for the plot.
            The default is {"xx":[],"yy":[],"xxyy":[],"levels":[1,2,3,4],"colormap":"viridis",
                            "locator":[],"alp":1,"linewidth":2}.
            Data:
                - xx        : data x of the histogram
                - yy        : data y of the histogram
                - xxyy      : data of the histogram
                - levels    : levels of the histogram
                - colormap  : colormap used in the histogram
                - locator   : locator to specify the levels of the contours
                - alp       : alpha channel of the plot
                - linewidth : linewidth of the contour

        Returns
        -------
        None.

        """
        # ----------------------------------------------------------------------------------------------------------------
        # Read data
        # ----------------------------------------------------------------------------------------------------------------
        xx         = np.array(data_in["xx"],dtype="float")
        yy         = np.array(data_in["yy"],dtype="float")
        xxyy       = np.array(data_in["xxyy"],dtype="float")
        levels     = np.array(data_in["levels"],dtype="float")
        color_read = data_in["colormap"]
        locator    = data_in["locator"]
        alp        = float(data_in["alp"])
        linewidth  = float(data_in["linewidth"])
        nlev       = len(levels)-1
        lev_min    = levels[0]
        lev_max    = levels[-1]
        if color_read is None:
            color  = self.colormap
        else:
            color  = str(color_read)
            
        # ----------------------------------------------------------------------------------------------------------------
        # Plot the contours
        # ----------------------------------------------------------------------------------------------------------------
        if self.multi:
            if self.col == 1:
                ax = self.axes[plot_number_y]
            elif self.row == 1:
                ax = self.axes[plot_number_x]
            else:
                ax = self.axes[plot_number_y,plot_number_x]
            if plot_number_x >= self.col:
                print("Increase the number of rows. Exit...",flush=True)
                sys.exit()
            elif plot_number_y >= self.row:
                print("Increase the number of columns. Exit...",flush=True)
                sys.exit()
        else:
            ax = self.axes
        self.figure
        im1 = ax.contourf(xx,yy,xxyy,levels=levels,cmap=matplotlib.cm.get_cmap(color,nlev),
                          alpha=alp,vmin=lev_min,vmax=levels[-1],locator=locator)
        im2 = ax.contour(xx,yy,xxyy,levels=levels,cmap=matplotlib.cm.get_cmap(color,nlev),
                         vmin=lev_min,vmax=levels[-1],locator=locator,linewidths=linewidth)
        self.plot.append(im1)
        self.plot.append(im2)
        
            
    def add_hist2d_cont(self,data_in={"xx":[],"yy":[],"xxyy":[],"level":1,"colormap":"viridis","alp":1,
                                      "linewidth":2,"cmap_flag":False}):
        """
        .................................................................................................................
        # add_hist2d_cont
        .................................................................................................................
        Function to plot the pdf of the structures no levels

        Parameters
        ----------
        data_in : dict, optional
            Data required for the plot.
            The default is {"xx":[],"yy":[],"xxyy":[],"levels":1,"colormap":"viridis",
                            "alp":1,"linewidth":2,"cmap_flag":False}.
            Data:
                - xx        : data x of the histogram
                - yy        : data y of the histogram
                - xxyy      : data of the histogram
                - levels    : level of the histogram
                - colormap  : colormap used in the histogram
                - alp       : alpha channel of the plot
                - linewidth : linewidth of the contour
                - cmap_flag : flag deciding if using color map

        Returns
        -------
        None.

        """
        # ----------------------------------------------------------------------------------------------------------------
        # Read data
        # ----------------------------------------------------------------------------------------------------------------
        xx         = np.array(data_in["xx"],dtype="float")
        yy         = np.array(data_in["yy"],dtype="float")
        xxyy       = np.array(data_in["xxyy"],dtype="float")
        levels     = float(data_in["levels"])
        color_read = data_in["colormap"]
        alp        = float(data_in["alp"])
        linewidth  = float(data_in["linewidth"])
        cmap_flag  = bool(data_in["cmap_flag"])
        if color_read is None:
            color  = matplotlib.cm.get_cmap(self.colormap)
        else:
            if isinstance(color_read,str):
                color = str(color_read)
            else:
                color = tuple(color_read)
            
        # ----------------------------------------------------------------------------------------------------------------
        # Plot the contours
        # ----------------------------------------------------------------------------------------------------------------
        if self.multi:
            if self.col == 1:
                ax = self.axes[plot_number_y]
            elif self.row == 1:
                ax = self.axes[plot_number_x]
            else:
                ax = self.axes[plot_number_y,plot_number_x]
            if plot_number_x >= self.col:
                print("Increase the number of rows. Exit...",flush=True)
                sys.exit()
            elif plot_number_y >= self.row:
                print("Increase the number of columns. Exit...",flush=True)
                sys.exit()
        else:
            ax = self.axes
        self.figure
        if cmap_flag:
            im1 = ax.contourf(xx,yy,xxyy,levels=[levels,10*np.max(xxyy)],cmap=matplotlib.cm.get_cmap(color),
                              alpha=alp)
            im2 = ax.contour(xx,yy,xxyy,levels=levels,cmap=matplotlib.cm.get_cmap(color),
                             linewidths=linewidth)
        else:
            im1 = ax.contourf(xx,yy,xxyy,levels=[levels,10*np.max(xxyy)],colors=color,
                              alpha=alp)
            im2 = ax.contour(xx,yy,xxyy,levels=levels,colors=color,
                             linewidths=linewidth)
        self.plot.append(im1)
        self.plot.append(im2)
        
        
        