# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
flow_field.py
-------------------------------------------------------------------------------------------------------------------------
Created on Mon Mar 25 11:07:26 2024

@author: Andres Cremades Botella

File to read the geometric characteristics of the data. The file contains a class:
    Class:
        - flow_field: class containing the information of the flow field. The class has the following functions
"""
# -----------------------------------------------------------------------------------------------------------------------
# Import packages for all the functions
# -----------------------------------------------------------------------------------------------------------------------
import glob
import h5py
import numpy as np

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define the class
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

class flow_field():
    """
    .....................................................................................................................
    # flow_field: Class to calculate and store the geometrical characteristics of the flow field. The class contains
                  the following
    functions:
        * Functions:
            - __init__     : initialization function
            - shape_tensor : function to calculate the shape of the tensors of the flow field
            - flow_grid    : calculate the geometry of the mesh of the fluid flow
        * Variables:
            - folder       : folder of the flow fields
            - file         : file of the flow fields
            - dx           : downsampling in the streamwise direction
            - dy           : downsampling in the wall-normal direction
            - dz           : downsampling in the spanwise direction
            - L_x          : dimension of the channel in the streamwise direction
            - L_y          : dimension of the channel in the wall-normal direction
            - L_z          : dimension of the channel in the spanwise direction
            - rey          : friction reynolds number
            - utau         : friction velocity
            - mx           : grid points in x
            - my           : grid points in y
            - mz           : grid points in z
            - ygrid        : grid points along the wall-normal direction
            - shpx         : shape of the tensors in the streamwise direction
            - shpy         : shape of the tensors in the wall-normal direction
            - shpz         : shape of the tensors in the spanwise direction
            - delta_x      : size of the grid in the streamwise direction
            - delta_y      : size of the grid in the wall-normal direction
            - delta_z      : size of the grid in the spanwise direction
            - delta_x_plus : size of the grid in the streamwise direction in wall units
            - delta_y_plus : size of the grid in the wall-normal direction in wall units
            - delta_z_plus : size of the grid in the spanwise direction in wall units
            - y_h          : grid in the wall-normal direction
            - x_h          : grid in the streamwise direction
            - z_h          : grid in the spanwise direction
            - yl_s         : index of the midchannel for the lower part. Use it like this for extracting the points
                             located in the lower part [:self.yl_s]
            - yu_s         : index of the midchannel for the upper part. Use it like this for extracting the points
                             located in the upper part [self.yu_s:]
            - yplus        : wall-normal coordinates
            - xplus        : streamwise coordinates
            - zplus        : spanwise coordinates
            - vol_h        : volume of the grid points
            - vol_plus     : volume of the grid points in viscous units
            - voltot       : total volume of the channel
            - voltot_plus  : total volume of the channel in viscous units
    .....................................................................................................................
    """
    def __init__(self,data_in={"folder":"../../P125_21pi_vu/","file":"P125_21pi_vu.$INDEX$.h5.uvw",
                               "down_x":1,"down_y":1,"down_z":1,"L_x":2*np.pi,"L_y":1,"L_z":np.pi,
                               "rey":125,"utau":0.060523258443963}):
        """
        .................................................................................................................
        # __init__
        .................................................................................................................
        Function to initialize the flow class. The funtion reads the input information required for obtaining the
        geometrical characteristics.

        Parameters
        ----------
        data_in : dict, optional
            Data required for obtaining the geometry of the flow.
            The default is {"folder":"../P125_21pi_vu/","file":"P125_21pi_vu.$INDEX$.h5.uvw",
                            "dx":1,"dy":1,"dz":1,"L_x":2*np.pi,"L_z":np.pi,
                            "rey":125,"utau":0.060523258443963}.
            Data:
                - folder : folder of the flow fields
                - file   : file of the flow fields
                - dx     : downsampling in the streamwise direction
                - dy     : downsampling in the wall-normal direction
                - dz     : downsampling in the spanwise direction
                - L_x    : dimension of the channel in the streamwise direction
                - L_y    : dimension of the channel in the wall-normal direction
                - L_z    : dimension of the channel in the spanwise direction
                - rey    : friction reynolds number
                - utau   : friction velocity

        Returns
        -------
        None.

        """
        # --------------------------------------------------------------------------------------------------------------
        # Read the data
        # --------------------------------------------------------------------------------------------------------------
        folder       = str(data_in["folder"]) # Folder of the flow field data
        file         = str(data_in["file"])   # File of the flow field data without the index
        self.down_x  = int(data_in["down_x"]) # Downsampling in the streamwise direction
        self.down_y  = int(data_in["down_y"]) # Downsampling in the wall-normal direction
        self.down_z  = int(data_in["down_z"]) # Downsampling in the spanwise direction
        self.L_x     = float(data_in["L_x"])  # Streamwise length of the channel
        self.L_y     = float(data_in["L_y"])  # Half width of the channel
        self.L_z     = float(data_in["L_z"])  # Spanwise length of the channel
        self.rey     = float(data_in["rey"])  # Reynolds tau of the flow
        self.utau    = float(data_in["utau"]) # Shear velocity
        
        # --------------------------------------------------------------------------------------------------------------
        # Choose a file from the directory and read the relevant data
        # --------------------------------------------------------------------------------------------------------------
        file_ref   = file.replace("$INDEX$","*")
        file_com   = folder+'/'+file_ref
        file_base  = glob.glob(file_com)[0]
        print("File for measuring the flow: "+file_base,flush=True)
        file       = h5py.File(file_base,'r')
        print("Reading flow field",flush=True)
        self.mx    = np.array(file["mx"])[0]            # grid points in x
        self.my    = np.array(file["my"])[0]            # grid points in y
        self.mz    = np.array(file["mz"])[0]            # grid points in z
        self.ygrid = np.array(file["y"])                # grid along y
        file.close()
        print("Flow field read",flush=True)
        
    def shape_tensor(self):
        """
        .................................................................................................................
        # shape_tensor
        .................................................................................................................
        Function to calculate the shape of the tensors after applying the downsampling

        Returns
        -------
        None.

        """
        self.shpy = int((self.my-1)/self.down_y)+1   # Shape in y
        self.shpz = int((self.mz-1)/self.down_z)+1   # Shape in z
        self.shpx = int((self.mx-1)/self.down_x)+1   # Shape in x
    
    def flow_grid(self):
        """
        .................................................................................................................
        # flow_grid
        .................................................................................................................
        Function to calculate the shape of the tensors after applying the downsampling

        Returns
        -------
        None.

        """
        # -------------------------------------------------------------------------------------------------------------
        # Calculate the grid size in the streamwise and spanwise directions
        # -------------------------------------------------------------------------------------------------------------
        self.delta_x = self.L_x/(self.shpx-1)          # grid size in the spanwise direction
        self.delta_z = self.L_z/(self.shpz-1)          # grid size in the streamwise direction
        self.delta_y = np.zeros((self.shpy,))      # create a vector for the grid size in the wall-normal direction
        self.delta_x_plus = self.delta_x*self.rey  # grid size in the streamwise direction in viscous units
        self.delta_z_plus = self.delta_z*self.rey  # grid size in the spanwise direction in viscous units
        
        # -------------------------------------------------------------------------------------------------------------
        # Calculate the mesh in channel units
        # -------------------------------------------------------------------------------------------------------------
        self.y_h = self.ygrid[::self.down_y]         # grid in the wall-normal direction
        self.x_h = self.delta_x*np.arange(self.shpx) # grid in the streamwise direction
        self.z_h = self.delta_z*np.arange(self.shpz) # grid in the spanwise direction
        
        # -------------------------------------------------------------------------------------------------------------
        # Calculate the grid size in the wall normal direction
        #     - delta_y      : grid size in the wall-normal direction
        #     - delta_y_plus : grid size in the wall-normal direction in viscous units
        # -------------------------------------------------------------------------------------------------------------
        mid_grid  = np.diff(self.y_h)/2
        mid_point = self.y_h[:-1]+mid_grid
        for ii in np.arange(self.shpy):
            if ii==0:
                self.delta_y[ii] = mid_point[0]-self.y_h[0]
            elif ii==self.shpy-1:
                self.delta_y[ii] = self.y_h[-1]-mid_point[-1]
            else:
                self.delta_y[ii] = mid_point[ii]-mid_point[ii-1]
        self.delta_y_plus = self.delta_y*self.rey
        
        # -------------------------------------------------------------------------------------------------------------
        # Calculate the separation between the upper and the lower channel
        #     - yl_s : index of the midchannel for the lower part. Use it like this for extracting the points
        #              located in the lower part [:self.yl_s]
        #     - yu_s : index of the midchannel for the upper part. Use it like this for extracting the points
        #              located in the upper part [self.yu_s:]
        # -------------------------------------------------------------------------------------------------------------
        if np.mod(self.shpy,2) == 0:
            self.yl_s = int(self.shpy*0.5)
            self.yu_s = int(self.shpy*0.5)
        else:
            self.yl_s = int(self.shpy*0.5)+1
            self.yu_s = int(self.shpy*0.5)
        
        # -------------------------------------------------------------------------------------------------------------
        # Grid in the viscous units
        # -------------------------------------------------------------------------------------------------------------
        self.y_h_plus = self.y_h*self.rey
        self.yplus    = (1+self.y_h[:self.yl_s])*self.rey
        self.xplus    = self.x_h*self.rey
        self.zplus    = self.z_h*self.rey
        
        # -------------------------------------------------------------------------------------------------------------
        # Calculate the volume of the grid points
        #     - vol_h       : volume of the grid points
        #     - vol_plus    : volume of the grid points in viscous units
        #     - voltot      : total volume of the channel
        #     - voltot_plus : total volume of the channel in viscous units
        # -------------------------------------------------------------------------------------------------------------
        vol_vec          = self.delta_x*self.delta_z*self.delta_y
        self.vol_h       = np.zeros((1,self.shpz,self.shpx),dtype=vol_vec.dtype)+vol_vec.reshape(-1,1,1)
        self.vol_plus    = self.vol_h*self.rey**3
        self.voltot      = self.L_x*self.L_y*2*self.L_z
        self.voltot_plus = self.voltot*self.rey**3
        
 
    
        