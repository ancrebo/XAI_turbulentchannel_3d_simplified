# -*- coding: utf-8 -*-GlobalAveragePooling3D()
"""
-------------------------------------------------------------------------------------------------------------------------
shap_config.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Apr  4 17:29:12 2024

@author:  Andres Cremades Botella

File to define the deep learning model for the calculation of the SHAP values.
The file contains a class for the deep learning model for the SHAP values:
    Class:
        - deep_model : Class of the deep learning model. The model has the following functions:
"""
import numpy as np
from os import mkdir
from glob import glob

class shap_config():    
    """
    .....................................................................................................................
    # shap_config: Class containing the information of the shap model.
        * Functions:
            - __init__           : initialization of the class. Read data of the flow, the gpus used, path to the 
                                   data...  
            - calc_gradientSHAP  : function to calculate the GradientExplainer. Defines the model and obtains the SHAP
                                   values.
            - write_shap         : function to save the SHAP values in a file.
            - read_shap          : function to read the SHAP values stored in a file
            - gradientSHAP_model : function to define the SHAP model
            - model_base_shap    : function to define the base model for the SHAP without the weights
            - background         : function to calculate the background of the SHAP values model
            - architecture_Unet  : function for defining the strategy
        * Variables:
            - shap_folder        : folder for the shap values
            - shap_file          : file for the shap values
            - uvw_folder         : folder of the flow fields
            - uvw_file           : file of the flow fields
            - padding            : padding of the flow field
            - dx                 : downsampling of the flow field in the streamwise direction
            - dy                 : downsampling of the flow field in the wall-normal direction
            - dz                 : downsampling of the flow field in the spanwise direction
            - data_folder        : folder of the generated data
            - umean_file         : file of the mean velocity
            - unorm_file         : file of the normalization
            - L_x                : size of the channel in the streamwise direction
            - L_z                : size of the channel in the spanwise direction
            - L_y                : size of the channel in the wall-normal direction
            - rey                : friction Reynolds number
            - utau               : friction velocity
            - ngpu               : number of gpus
            - field_ini          : initial field of the SHAPs
            - field_fin          : final field of the SHAPs
            - field_delta        : separation between the fields
            - model_folder       : folder for storing the models
            - model_read         : name of the model to load
            - nfil               : number of filters of the model CNN
            - stride             : stride of the model CNN
            - activation         : activation function of the model CNN
            - kernel             : kernel of the model CNN
            - pooling            : size of the poolings
            - delta_pred         : number of fields to advance the predictions
            - nsamples           : number of samples to take to calculate the SHAP values in the gradient explainer
            - nsamples_max       : maximum number of samples to calculate at the same time in the gradient explainer
            - shpx               : shape of the tensors in the streamwise direction
            - shpy               : shape of the tensors in the wall-normal direction
            - shpz               : shape of the tensors in the spanwise direction
            - weights            : weights of the trained model
            - inputs             : inputs for the definition of the model
            - outputs            : outputs for the definition of the model
        * Classes:
            - strategy           : segmentation strategy of the model used for the SHAP calculation
            - model_train        : model trained for the flow prediction
            - model              : model for the SHAP values calculation
    .....................................................................................................................
    """
    def __init__(self,data_in = {"shap_folder":"../../P125_21pi_vu_SHAP_gradient/",
                                 "shap_file":"P125_21pi_vu_nsamples$NSAMPLES$.$INDEX$.h5.shap",
                                 "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                                 "padding":15,"dx":1,"dy":1,"dz":1,"data_folder":"data","umean_file":"Umean.txt",
                                 "unorm_file":"Unorm.txt","L_x":2*np.pi,"L_z":np.pi,"L_y":1,
                                 "rey":125,"utau":0.060523258443963,"ngpu":None,"field_ini":1000,"field_fin":7000,
                                 "field_delta":1,"model_folder":"models","model_read":"trained_model.h5","nfil":16,
                                 "stride":1,"activation":"relu","kernel":3,"pooling":2,"delta_pred":1,"nsamples":200,
                                 "nsamples_max":100,"data_type":"float32","error_file":"error.txt",
                                 "umax_file":"umax_file.txt","urmspred_file":"Urms_pred.txt","mean_norm":False,
                                 "tfrecord_folder":'../../tfrecord/',"nrep_field":None,"shap_batch":1,
                                 "repeat_exist":False,"flag_model":True,"read_model":True}):
        """
        .................................................................................................................
        # __init__
        .................................................................................................................
        Function to initialize the shap model.

        Parameters
        ----------
        data_in : dict, optional
            Function to initialize the SHAP model.
            The default is {"shap_folder":"../../P125_21pi_vu_SHAP_gradient/",
                            "shap_file":"P125_21pi_vu_nsamples$NSAMPLES$.$INDEX$.h5.shap",
                            "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                            "padding":15,"dx":1,"dy":1,"dz":1,"data_folder":"data","umean_file":"Umean.txt",
                            "unorm_file":"Unorm.txt","L_x":2*np.pi,"L_z":np.pi,"L_y":1,
                            "rey":125,"utau":0.060523258443963,"ngpu":None,"field_ini":1000,"field_fin":7000,
                            "field_delta":1,"model_folder":"models","model_read":"trained_model.h5","nfil":16,
                            "stride":1,"activation":"relu","kernel":3,"pooling":2,"delta_pred":1,"nsamples":200,
                            "nsamples_max":100,"data_type":"float32","error_file":"error.txt",
                            "umax_file":"umax_file.txt","urmspred_file":"Urms_pred.txt","mean_norm":False,
                            "tfrecord_folder":'../../tfrecord/',"nrep_field":None,"shap_batch":1,"repeat_exist":False,
                            "flag_model":True,"read_model":True}.
            Data:
                - shap_folder     : folder for the shap values
                - shap_file       : file for the shap values
                - uvw_folder      : folder of the flow fields
                - uvw_file        : file of the flow fields
                - padding         : padding of the flow field
                - dx              : downsampling of the flow field in the streamwise direction
                - dy              : downsampling of the flow field in the wall-normal direction
                - dz              : downsampling of the flow field in the spanwise direction
                - data_folder     : folder of the generated data
                - umean_file      : file of the mean velocity
                - unorm_file      : file of the normalization
                - L_x             : size of the channel in the streamwise direction
                - L_z             : size of the channel in the spanwise direction
                - L_y             : size of the channel in the wall-normal direction
                - rey             : friction Reynolds number
                - utau            : friction velocity
                - ngpu            : number of gpus
                - field_ini       : initial field of the SHAPs
                - field_fin       : final field of the SHAPs
                - field_delta     : separation between the fields
                - model_folder    : folder for storing the models
                - model_read      : name of the model to load
                - nfil            : number of filters of the model CNN
                - stride          : stride of the model CNN
                - activation      : activation function of the model CNN
                - kernel          : kernel of the model CNN
                - pooling         : size of the poolings
                - delta_pred      : number of fields to advance the predictions
                - nsamples        : number of samples to take to calculate the SHAP values in the gradient explainer
                - nsamples_max    : maximum number of samples to calculate at the same time in the gradient explainer
                - data_type       : type of data used in the model
                - error_file      : file to store the prediction errors
                - umax_file       : file containing maximum and minimum velocities
                - urmspred_file   : file contatining the urms predicted by the model
                - mean_norm       : Flag to normalize using mean and std (True: normalizes with mean and std, False:
                                                                          normalizes with min and max)
                - tfrecord_folder : folder storing the tfrecord
                - nrep_field      : number of repetitions of the field for calculating the SHAP values. If none do 
                                    not apply any repetition
                - shap_batch      : batch size used for the shap calculation
                - repeat exist    : flag for repeating an existing file (True: recalculate, False: skip)
                - flag_model      : flag to load a model
                - read_model      : flag to read the model (True: read the model, False: not read the model)

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Import packages
        # ---------------------------------------------------------------------------------------------------------------
        from py_bin.py_class.ann_config import deep_model
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the data
        # ---------------------------------------------------------------------------------------------------------------
        self.shap_folder     = str(data_in["shap_folder"])       # Folder path to the shap values
        self.shap_file       = str(data_in["shap_file"])         # File to the shap values
        self.uvw_folder      = str(data_in["uvw_folder"])        # Folder path from the main file
        self.uvw_file        = str(data_in["uvw_file"])          # File name without the index
        self.padding         = int(data_in["padding"])           # Padding of the field
        self.dx              = int(data_in["dx"])                # Downsampling in x
        self.dy              = int(data_in["dy"])                # Downsampling in y
        self.dz              = int(data_in["dz"])                # Downsampling in z
        self.data_folder     = str(data_in["data_folder"])       # Folder for the generated data
        self.umean_file      = str(data_in["umean_file"])        # file of the mean velocity
        self.unorm_file      = str(data_in["unorm_file"])        # file for the normalization of the velocity
        self.L_x             = float(data_in["L_x"])             # size of the streamwise dimension of the channel
        self.L_z             = float(data_in["L_z"])             # size of the streamwise dimension of the channel
        self.L_y             = float(data_in["L_y"])             # size of half of the width of the channel
        self.rey             = float(data_in["rey"])             # Friction reynolds number
        self.utau            = float(data_in["utau"])            # Friction velocity
        if data_in["ngpu"] is None:
            self.ngpu        = None                              # Take all available GPUs
        else:
            self.ngpu        = int(data_in["ngpu"])              # Choose a number of GPUs 
        self.field_ini       = int(data_in["field_ini"])         # Initial field for the SHAP values
        self.field_fin       = int(data_in["field_fin"])         # Final field for the SHAP values
        self.field_delta     = int(data_in["field_delta"])       # Separation between the fields for the SHAP values
        self.model_folder    = str(data_in["model_folder"])      # Name of the folder to store the trained model
        self.model_read      = str(data_in["model_read"])        # Name of the file to store the trained model
        self.nfil            = int(data_in["nfil"])              # number of filters of the first layer of the unet
        self.stride          = int(data_in["stride"])            # stride of the unet
        self.activation      = str(data_in["activation"])        # activation function
        self.kernel          = int(data_in["kernel"])            # size of the filters
        self.pooling         = int(data_in["pooling"])           # size of the poolings of the unet
        self.delta_pred      = int(data_in["delta_pred"])        # separation between the fields
        self.nsamples        = int(data_in["nsamples"])          # number of samples to calculate the shap
        self.nsamples_max    = int(data_in["nsamples_max"])      # maximum number of samples to take at the same time
        self.data_type       = str(data_in["data_type"])         # data type for the tensors of the model
        self.error_file      = str(data_in["error_file"])        # file containing the errors of the predictions
        self.umax_file       = str(data_in["umax_file"])         # file containing the maximum and minimum velocities
        self.urmspred_file   = str(data_in["urmspred_file"])     # file containing the rms of the predicted velocity
        self.mean_norm       = bool(data_in["mean_norm"])        # flag for choosing the normalization
        self.tfrecord_folder = str(data_in["tfrecord_folder"])
        nrep_field           = data_in["nrep_field"]
        if nrep_field is None:
            self.nrep_field  = 0
        else:
            self.nrep_field  = int(nrep_field)
        self.shap_batch      = int(data_in["shap_batch"])
        self.print_summary   = True
        self.repeat_exist    = bool(data_in["repeat_exist"])
        self.flag_model      = bool(data_in["flag_model"])
        if "read_model" in data_in.keys():
            read_model       = bool(data_in["read_model"])
        else:
            read_model       = True
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create the model
        # ---------------------------------------------------------------------------------------------------------------
        unet_data  = {"uvw_folder":self.uvw_folder,"uvw_file":self.uvw_file,"padding":self.padding,"dx":self.dx,
                      "dy":self.dy,"dz":self.dz,"data_folder":self.data_folder,"umean_file":self.umean_file,
                      "unorm_file":self.unorm_file,"L_x":self.L_x,"L_z":self.L_z,"L_y":self.L_y,
                      "uvw_folder_tf":"-","uvw_folderii_tf":"-","rey":self.rey,"utau":self.utau,
                      "ssh_flag_train":False,"uvw_folder_temp":"-","ssh_server":"-","ssh_username":"-",
                      "ssh_password":"-","error_file":self.error_file,"umax_file":self.umax_file,
                      "urmspred_file":self.urmspred_file}
        model_data = {"ngpu":self.ngpu,"learat":1e-3,"optmom":0.9,"batch_size":0,"field_ini":self.field_ini,
                      "field_fin":self.field_fin,"field_delta":self.field_delta,"field_mem":0,"epoch_save":0,
                      "epoch_max":0,"read_model":read_model,"model_folder":self.model_folder,"model_write":"-",
                      "model_read":self.model_read,"nfil":self.nfil,"stride":self.stride,"activation":self.activation,
                      "kernel":self.kernel,"pooling":self.pooling,"delta_pred":self.delta_pred,"hist_file":"-",
                      "test_size":0,"adapt_batch":False,"prep_data":False,"flag_model":self.flag_model,
                      "flag_central":False,"data_type":self.data_type,"multi_worker":False,"prefetch":1,
                      "mean_norm":self.mean_norm,"check":False,"tfrecord_folder":self.tfrecord_folder,
                      "flag_tfrecord":False,"save_fields":False,"traintest_index":"-"}
        
        # ---------------------------------------------------------------------------------------------------------------
        # Define the model
        # ---------------------------------------------------------------------------------------------------------------
        Unet = deep_model(data_in = unet_data)
        Unet.define_model(data_in = model_data)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Store data
        # ---------------------------------------------------------------------------------------------------------------
        self.shpx              = Unet.shpx
        self.shpy              = Unet.shpy
        self.shpz              = Unet.shpz
        if self.flag_model:
            self.strategy          = Unet.strategy
            self.architecture_Unet = Unet.architecture_Unet
            self.weights           = Unet.model.get_weights()
            self.model_train       = Unet.model
    
    def _movenpad(self,data_in={"uu":[],"vv":[],"ww":[],"x0":0,"z0":0,"flag_pad":False}):
        """
        .................................................................................................................
        # _movenpad: Function to move the field to the position to analyze and then apply the padding
        .................................................................................................................
        
        Parameters
        ----------
        data_in : dict, optional
            Data to read the field and apply the transformation. The default is {"uu":[],"vv":[],"ww":[],
                                                                                 "x0":0,"z0":0}.
            Data: 
                - uu : field in the streamwise direction
                - vv : field in the wall-normal direction
                - ww : field in the spanwise direction
                - x0 : position to start in the streamwise direction
                - z0 : position to start in the spanwise direction

        Returns
        -------
        dict
            Field after applying the transformations
            Data:
                - uu : field in the streamwise direction
                - vv : field in the wall-normal direction
                - ww : field in the spanwise direction
        """
        # ---------------------------------------------------------------------------------------------------------------
        # Import packages
        # ---------------------------------------------------------------------------------------------------------------
        from py_bin.py_functions.padding_field import padding_field
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the data
        # ---------------------------------------------------------------------------------------------------------------
        uu       = np.array(data_in["uu"],dtype=self.data_type)
        vv       = np.array(data_in["vv"],dtype=self.data_type)
        ww       = np.array(data_in["ww"],dtype=self.data_type)
        x0       = int(data_in["x0"])
        z0       = int(data_in["z0"])
        flag_pad = bool(data_in["flag_pad"])
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create the matrix of the whole domain
        # ---------------------------------------------------------------------------------------------------------------
        field_uu = np.zeros((self.shpy,self.shpz,self.shpx),dtype=self.data_type) 
        field_vv = np.zeros((self.shpy,self.shpz,self.shpx),dtype=self.data_type)
        field_ww = np.zeros((self.shpy,self.shpz,self.shpx),dtype=self.data_type)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Locate in the index 0,padding,padding without the positions x0,y0,z0 of the original fields
        # ---------------------------------------------------------------------------------------------------------------
        # Area D: after x0, z0
        # ---------------------------------------------------------------------------------------------------------------
        field_uu[:,:self.shpz-z0,:self.shpx-x0] = uu[:,z0:,x0:] 
        field_vv[:,:self.shpz-z0,:self.shpx-x0] = vv[:,z0:,x0:] 
        field_ww[:,:self.shpz-z0,:self.shpx-x0] = ww[:,z0:,x0:] 
        # ---------------------------------------------------------------------------------------------------------------
        # Area A: before x0, z0
        # ---------------------------------------------------------------------------------------------------------------
        field_uu[:,self.shpz-z0:,self.shpx-x0:] = uu[:,:z0,:x0] 
        field_vv[:,self.shpz-z0:,self.shpx-x0:] = vv[:,:z0,:x0] 
        field_ww[:,self.shpz-z0:,self.shpx-x0:] = ww[:,:z0,:x0] 
        # ---------------------------------------------------------------------------------------------------------------
        # Area B: after x0, before z0
        # ---------------------------------------------------------------------------------------------------------------
        field_uu[:,self.shpz-z0:,:self.shpx-x0] = uu[:,:z0,x0:] 
        field_vv[:,self.shpz-z0:,:self.shpx-x0] = vv[:,:z0,x0:] 
        field_ww[:,self.shpz-z0:,:self.shpx-x0] = ww[:,:z0,x0:] 
        # ---------------------------------------------------------------------------------------------------------------
        # Area C: before x0, after z0
        # ---------------------------------------------------------------------------------------------------------------
        field_uu[:,:self.shpz-z0,self.shpx-x0:] = uu[:,z0:,:x0] 
        field_vv[:,:self.shpz-z0,self.shpx-x0:] = vv[:,z0:,:x0] 
        field_ww[:,:self.shpz-z0,self.shpx-x0:] = ww[:,z0:,:x0]  
        
        if flag_pad:
            # ---------------------------------------------------------------------------------------------------------------
            # Apply the padding
            # ---------------------------------------------------------------------------------------------------------------
            field_uu = padding_field(data_in={"field":field_uu,"shpx":self.shpx,"shpy":self.shpy,
                                                  "shpz":self.shpz,"padding":self.padding})["field"]
            field_vv = padding_field(data_in={"field":field_vv,"shpx":self.shpx,"shpy":self.shpy,
                                                  "shpz":self.shpz,"padding":self.padding})["field"]
            field_ww = padding_field(data_in={"field":field_ww,"shpx":self.shpx,"shpy":self.shpy,
                                                  "shpz":self.shpz,"padding":self.padding})["field"]
        
        # ---------------------------------------------------------------------------------------------------------------
        # Save the output
        # ---------------------------------------------------------------------------------------------------------------
        data_out = {"uu":field_uu,"vv":field_vv,"ww":field_ww}
        return data_out
        
        
    def _recover_movenpad(self,data_in={"shap_u":[],"shap_v":[],"shap_w":[],"x0":0,"z0":0}):
        """
        .................................................................................................................
        # _recover_movenpad: Function to move back the field to the original position and delete the padding
        .................................................................................................................
        
        Parameters
        ----------
        data_in : dict, optional
            Data to read the field and apply the transformation. The default is {"shap_u":[],"shap_v":[],
                                                                                 "shap_w":[],"x0":0,"z0":0}.
            Data: 
                - shap_u : field in the streamwise direction
                - shap_v : field in the wall-normal direction
                - shap_w : field in the spanwise direction
                - x0     : position to start in the streamwise direction
                - z0     : position to start in the spanwise direction

        Returns
        -------
        dict
            Field after applying the transformations
            Data:
                - shap_u : field in the streamwise direction
                - shap_v : field in the wall-normal direction
                - shap_w : field in the spanwise direction
        """
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the data
        # ---------------------------------------------------------------------------------------------------------------
        shap_u = np.array(data_in["shap_u"],dtype=self.data_type)
        shap_v = np.array(data_in["shap_v"],dtype=self.data_type)
        shap_w = np.array(data_in["shap_w"],dtype=self.data_type)
        x0     = int(data_in["x0"])
        z0     = int(data_in["z0"])
        
        # ---------------------------------------------------------------------------------------------------------------
        # Restore the transformation of the shap fields
        # ---------------------------------------------------------------------------------------------------------------
        shap_field_u = np.zeros((self.shpy,self.shpz,self.shpx))
        shap_field_v = np.zeros((self.shpy,self.shpz,self.shpx))
        shap_field_w = np.zeros((self.shpy,self.shpz,self.shpx))
        # ---------------------------------------------------------------------------------------------------------------
        # Area D: after x0, z0
        # ---------------------------------------------------------------------------------------------------------------
        shap_field_u[:,z0:,x0:] = shap_u[:,self.padding:self.padding+self.shpz-z0,
                                         self.padding:self.padding+self.shpx-x0]
        shap_field_v[:,z0:,x0:] = shap_v[:,self.padding:self.padding+self.shpz-z0,
                                         self.padding:self.padding+self.shpx-x0]
        shap_field_w[:,z0:,x0:] = shap_w[:,self.padding:self.padding+self.shpz-z0,
                                         self.padding:self.padding+self.shpx-x0]
        # ---------------------------------------------------------------------------------------------------------------
        # Area A: before x0, z0
        # ---------------------------------------------------------------------------------------------------------------
        shap_field_u[:,:z0,:x0] = shap_u[:,self.padding+self.shpz-z0:-self.padding,
                                         self.padding+self.shpx-x0:-self.padding]
        shap_field_v[:,:z0,:x0] = shap_v[:,self.padding+self.shpz-z0:-self.padding,
                                         self.padding+self.shpx-x0:-self.padding]
        shap_field_w[:,:z0,:x0] = shap_w[:,self.padding+self.shpz-z0:-self.padding,
                                         self.padding+self.shpx-x0:-self.padding] 
        # ---------------------------------------------------------------------------------------------------------------
        # Area B: after x0, before z0
        # ---------------------------------------------------------------------------------------------------------------
        shap_field_u[:,:z0,x0:] = shap_u[:,self.padding+self.shpz-z0:-self.padding,
                                         self.padding:self.padding+self.shpx-x0]
        shap_field_v[:,:z0,x0:] = shap_v[:,self.padding+self.shpz-z0:-self.padding,
                                         self.padding:self.padding+self.shpx-x0]
        shap_field_w[:,:z0,x0:] = shap_w[:,self.padding+self.shpz-z0:-self.padding,
                                         self.padding:self.padding+self.shpx-x0]
        # ---------------------------------------------------------------------------------------------------------------
        # Area C: before x0, after z0
        # ---------------------------------------------------------------------------------------------------------------
        shap_field_u[:,z0:,:x0] = shap_u[:,self.padding:self.padding+self.shpz-z0,
                                         self.padding+self.shpx-x0:-self.padding]
        shap_field_v[:,z0:,:x0] = shap_v[:,self.padding:self.padding+self.shpz-z0,
                                         self.padding+self.shpx-x0:-self.padding]
        shap_field_w[:,z0:,:x0] = shap_w[:,self.padding:self.padding+self.shpz-z0,
                                         self.padding+self.shpx-x0:-self.padding] 
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create the output
        # ---------------------------------------------------------------------------------------------------------------
        data_out = {"shap_u":shap_field_u,"shap_v":shap_field_v,"shap_w":shap_field_w}
        return data_out
      
    def _calculate_gradientshaps(self,data_in={"norm_velocity_in":[],"norm_velocity_out":[],"x0":0,"z0":0}):
        """
        .................................................................................................................
        # _calculate_gradientshaps: Function to calculate the SHAP values for a certain field and location
        .................................................................................................................
        
        Parameters
        ----------
        data_in : dict, optional
            Data to calculate the shap values. The default is {"index_ii":0,"x0":0,"z0":0}.
            Data: 
                - norm_velocity_in  : input field
                - norm_velocity_out : output field
                - x0                : position to start in the streamwise direction
                - z0                : position to start in the spanwise direction

        Returns
        -------
        dict
            Shap values in all the directions
            Data:
                - shap_u : field in the streamwise direction
                - shap_v : field in the wall-normal direction
                - shap_w : field in the spanwise direction
        """        
        # ---------------------------------------------------------------------------------------------------------------
        # Import packages
        # The shap package is imported from an edited folder to save the memory required for calculating the mean values
        # when the number of fields is too high.
        # ---------------------------------------------------------------------------------------------------------------
        from py_bin.py_functions.read_norm_velocity import read_norm_velocity
        from py_bin.py_packages import shap
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the data
        # ---------------------------------------------------------------------------------------------------------------
        norm_velocity_in  = data_in["norm_velocity_in"]
        norm_velocity_out = data_in["norm_velocity_out"]
        x0                = int(data_in["x0"])
        z0                = int(data_in["z0"])
        
        
        # ---------------------------------------------------------------------------------------------------------------
        # Apply the transformations
        # ---------------------------------------------------------------------------------------------------------------
        field_in             = np.zeros((1,self.shpy,self.shpz+2*self.padding,self.shpx+2*self.padding,3), 
                                        dtype=self.data_type)    
        field_transformed_in = self._movenpad(data_in={"uu":norm_velocity_in['unorm'],"vv":norm_velocity_in['vnorm'],
                                                       "ww":norm_velocity_in['wnorm'],"x0":x0,"z0":z0,
                                                       "flag_pad":True})
        field_in[0,:,:,:,0]  = field_transformed_in["uu"]
        field_in[0,:,:,:,1]  = field_transformed_in["vv"]
        field_in[0,:,:,:,2]  = field_transformed_in["ww"]
        del norm_velocity_in,field_transformed_in
        
        # ---------------------------------------------------------------------------------------------------------------
        # Apply the transformations
        # ---------------------------------------------------------------------------------------------------------------
        field_out             = np.zeros((1,self.shpy,self.shpz,self.shpx,3),dtype=self.data_type)
        field_transformed_out = self._movenpad(data_in={"uu":norm_velocity_out['unorm'],"vv":norm_velocity_out['vnorm'],
                                                        "ww":norm_velocity_out['wnorm'],"x0":x0,"z0":z0,
                                                        "flag_pad":False})
        field_out[0,:,:,:,0]  = field_transformed_out['uu']
        field_out[0,:,:,:,1]  = field_transformed_out['vv']
        field_out[0,:,:,:,2]  = field_transformed_out['ww']
        del norm_velocity_out,field_transformed_out
        
        # ---------------------------------------------------------------------------------------------------------------
        # Define the gradient explainer model. This model needs to be updated for each file because the tensor of
        # the output needs to be updated with each file.
        # ---------------------------------------------------------------------------------------------------------------
        self.gradientSHAP_model(data_in={"field_out":field_out})
        
        # ---------------------------------------------------------------------------------------------------------------
        # Define the explainer and calculate the SHAP values
        #     - explainer     : definition of the Gradient Explainer
        #     - shap_values   : SHAP values of the field
        #     - shap_values_u : SHAP values of the component of u
        #     - shap_values_v : SHAP values of the component of v
        #     - shap_values_w : SHAP values of the component of w
        # ---------------------------------------------------------------------------------------------------------------
        print(self.model(self.backmat),flush=True)
        explainer     = shap.GradientExplainer(self.model,self.backmat,batch_size=self.shap_batch)            
        shap_values   = explainer.shap_values(field_in,nsamples=self.nsamples)#,nsamplesmax=self.nsamples_max) 
        print(shap_values.shape,flush=True)
        shap_values_u = shap_values[0,:,:,:,0]
        shap_values_v = shap_values[0,:,:,:,1]
        shap_values_w = shap_values[0,:,:,:,2]
        print(np.mean(shap_values_u),flush=True)
        print(np.mean(shap_values_v),flush=True)
        print(np.mean(shap_values_w),flush=True)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Recover the original position of the shap values in the field
        # ---------------------------------------------------------------------------------------------------------------
        shap_recovered = self._recover_movenpad(data_in={"shap_u":shap_values_u,"shap_v":shap_values_v,
                                                         "shap_w":shap_values_w,"x0":x0,"z0":z0})
        shap_valreco_u = shap_recovered["shap_u"]
        shap_valreco_v = shap_recovered["shap_v"]
        shap_valreco_w = shap_recovered["shap_w"]
        
        # ---------------------------------------------------------------------------------------------------------------
        # Store the output data
        # ---------------------------------------------------------------------------------------------------------------
        data_out = {"shap_u":shap_valreco_u,"shap_v":shap_valreco_v,"shap_w":shap_valreco_w}
        return data_out
    
    def _calculate_kernelshaps(self,data_in={"norm_velocity_in":[],"norm_velocity_out":[]}):
        """
        .................................................................................................................
        # _calculate_kernelshaps: Function to calculate the SHAP values for a certain field and location
        .................................................................................................................
        
        Parameters
        ----------
        data_in : dict, optional
            Data to calculate the shap values. The default is {"norm_velocity_in":[],"norm_velocity_out":[]}.
            Data: 
                - norm_velocity_in  : input field
                - norm_velocity_out : output field

        Returns
        -------
        dict
            Shap values of the structures
            Data:
                - shap : shap of the structures
        """        
        # ---------------------------------------------------------------------------------------------------------------
        # Import packages
        # The shap package is imported from an edited folder to save the memory required for calculating the mean values
        # when the number of fields is too high.
        # ---------------------------------------------------------------------------------------------------------------
        from py_bin.py_functions.read_norm_velocity import read_norm_velocity
        from py_bin.py_packages import shap
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the data
        # ---------------------------------------------------------------------------------------------------------------
        norm_velocity_in                       = data_in["norm_velocity_in"]
        norm_velocity_out                      = data_in["norm_velocity_out"]        
        
        # ---------------------------------------------------------------------------------------------------------------
        # Apply the transformations
        # ---------------------------------------------------------------------------------------------------------------
        field_in             = np.zeros((1,self.shpy,self.shpz+2*self.padding,self.shpx+2*self.padding,3), 
                                        dtype=self.data_type)    
        field_in[0,:,:,:,0]  = norm_velocity_in['unorm']
        field_in[0,:,:,:,1]  = norm_velocity_in['vnorm']
        field_in[0,:,:,:,2]  = norm_velocity_in['wnorm']
        del norm_velocity_in
        
        # ---------------------------------------------------------------------------------------------------------------
        # Apply the transformations
        # ---------------------------------------------------------------------------------------------------------------
        field_out             = np.zeros((1,self.shpy,self.shpz,self.shpx,3),dtype=self.data_type)
        field_out[0,:,:,:,0]  = norm_velocity_out['unorm']
        field_out[0,:,:,:,1]  = norm_velocity_out['vnorm']
        field_out[0,:,:,:,2]  = norm_velocity_out['wnorm']
        del norm_velocity_out
        
        # ---------------------------------------------------------------------------------------------------------------
        # Define the gradient explainer model. This model needs to be updated for each file because the tensor of
        # the output needs to be updated with each file.
        # ---------------------------------------------------------------------------------------------------------------
        self.gradientSHAP_model(data_in={"field_out":field_out})
        
        # ---------------------------------------------------------------------------------------------------------------
        # Define functions
        # --------------------------------------------------------------------------------------------------------------- 
        def get_structure_indices():
            """
            .................................................................................................................
            # get_structure_indices: Function to calculate indices of the structures
            .................................................................................................................
            
            Returns
            -------
            array
                index of structures
            """ 
            struc_indx = []
            for ii in range(self.struc_num):
                indx = np.array(np.where(self.segmentation == ii)).transpose()
                struc_indx.append(indx.astype(int))
            array_struc_indx = np.array(struc_indx, dtype=object)
            return array_struc_indx
        
        def model_kernel(zs):
            """
            .................................................................................................................
            # model_kernel: Function to calculate the agnostic model required for the kernel
            .................................................................................................................
            
            Parameters
            ----------
            zs : array
                Data to select the coallition (1 existing, 0 absent).

            Returns
            -------
            array
                MSE of the prediction
            """        
            ii = 0
            lm = zs.shape[0]
            mse = np.zeros((lm,1))
            print("Starting kernel SHAP:",flush=True)
            for ii in np.arange(lm):
                if ii<lm-1:
                    print("Calculation "+str(ii)+" of "+str(lm),end='\r',flush=True)
                else:
                    print("Calculation "+str(ii)+" of "+str(lm),flush=True)
                zii         = zs[ii]
                model_input = mask_dom(zii)
                mse[ii,0]   = self.model(model_input)
            return mse
        
        def mask_dom(zs):
            """
            .................................................................................................................
            # mask_dom: Function to mask the structures in the coallition
            .................................................................................................................
            
            Parameters
            ----------
            zs : array
                Data to select the coallition (1 existing, 0 absent).

            Returns
            -------
            array
                Masking of the domain
            """ 
            
            mask_out = field_in.copy()
            if 0 not in zs:
                return mask_out
                      
            struc_selected = np.where(zs==0)[0].astype(int)
            indx = np.vstack(array_struc_indx[struc_selected]).astype(int)
            mask_out[:,indx[:,0],indx[:,1],indx[:,2], :] = self.backmat[:,indx[:,0],indx[:,1],indx[:,2],:] 
            return mask_out
        
        # ---------------------------------------------------------------------------------------------------------------
        # Define the explainer and calculate the SHAP values
        #     - explainer     : definition of the Gradient Explainer
        #     - shap_values   : SHAP values of the field
        # ---------------------------------------------------------------------------------------------------------------
        array_struc_indx = get_structure_indices()
        explainer        = shap.KernelExplainer(model_kernel,np.zeros((1,self.struc_num)))            
        shap_values      = explainer.shap_values(np.ones((1,self.struc_num)),nsamples="auto")[0][0]
        
        # ---------------------------------------------------------------------------------------------------------------
        # Store the output data
        # ---------------------------------------------------------------------------------------------------------------
        data_out = {"shap":shap_values}
        return data_out

    
    def _file_name(self,data_in={"index_ii":1000}):
        """
        .................................................................................................................
        # _file_name: Function to generate the name of the specific file
        .................................................................................................................

        Parameters
        ----------
        data_in : dict, optional
            Data required for the name generation. The default is {"index_ii":1000}.
            Data:
                - index_ii : index of the field

        Returns
        -------
        dict
            Name of the file
            Data:
                - file_shap : relative path to the file of the field

        """
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the data
        # ---------------------------------------------------------------------------------------------------------------
        index_ii      = str(data_in["index_ii"])
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create the name
        # ---------------------------------------------------------------------------------------------------------------
        file_shap     = self.shap_folder+'/'+self.shap_file
        file_shap     = file_shap.replace("$INDEX$",index_ii)
        if self.nsamples is None:
            file_shap = file_shap.replace("$NSAMPLES$","")
        else:
            file_shap = file_shap.replace("$NSAMPLES$",str(self.nsamples))
        # ---------------------------------------------------------------------------------------------------------------
        # Return the name
        # ---------------------------------------------------------------------------------------------------------------
        data_out = {"file_shap":file_shap}
        return data_out
    
    def calc_gradientSHAP(self):
        """
        .................................................................................................................
        # calc_gradientSHAP
        .................................................................................................................
        Function to calculate the gradient SHAP values

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Import packages
        # ---------------------------------------------------------------------------------------------------------------
        from py_bin.py_functions.read_norm_velocity import read_norm_velocity
        from py_bin.py_functions.padding_field import padding_field
        
        # ---------------------------------------------------------------------------------------------------------------
        # Calculate the background values
        # ---------------------------------------------------------------------------------------------------------------
        self.background()
        
        # ---------------------------------------------------------------------------------------------------------------
        # Calculate the SHAP values of all the flow fields
        # ---------------------------------------------------------------------------------------------------------------
        interval = range(self.field_ini,self.field_fin,self.field_delta)
        for index_ii in interval:
            
            # -----------------------------------------------------------------------------------------------------------
            # Check if the file has already been calculated
            # -----------------------------------------------------------------------------------------------------------
            file_shap = self._file_name(data_in={"index_ii":index_ii})["file_shap"]
            exist_file    = bool(len(glob(file_shap)))
            if exist_file and not self.repeat_exist:
                print("Existing field",flush=True)
                continue
            else:
                print("New field",flush=True)
            
            print("-"*100,flush=True)
            print('Calculating the SHAP for field: '+str(index_ii),flush=True)
            # -----------------------------------------------------------------------------------------------------------
            # Read the input field 
            # -----------------------------------------------------------------------------------------------------------   
            data_norm_in     = {"folder":self.uvw_folder,"file":self.uvw_file,"padding":0,"shpx":self.shpx,
                                "shpy":self.shpy,"shpz":self.shpz,"dx":self.dx,"dy":self.dy,"dz":self.dz,
                                "data_folder":self.data_folder,"umean_file":self.umean_file,
                                "unorm_file":self.unorm_file,"index":index_ii,"data_type":self.data_type,
                                "mean_norm":self.mean_norm}
            data_veloc_norm  = read_norm_velocity(data_in=data_norm_in)
            norm_velocity_in = data_veloc_norm["norm_velocity"]
            print("Time for reading the field: "+str(data_veloc_norm["time_read"]),flush=True)
            print("Time for normalizing the field: "+str(data_veloc_norm["time_norm"]),flush=True)
            del data_norm_in,data_veloc_norm
            
            # -----------------------------------------------------------------------------------------------------------
            # Read the output field 
            # -----------------------------------------------------------------------------------------------------------
            data_norm_out     = {"folder":self.uvw_folder,"file":self.uvw_file,"padding":0,"shpx":self.shpx,
                                 "shpy":self.shpy,"shpz":self.shpz,"dx":self.dx,"dy":self.dy,"dz":self.dz,
                                 "data_folder":self.data_folder,"umean_file":self.umean_file,
                                 "unorm_file":self.unorm_file,"index":index_ii+self.delta_pred,
                                 "data_type":self.data_type,"mean_norm":self.mean_norm}
            data_veloc_norm   = read_norm_velocity(data_in=data_norm_out)
            norm_velocity_out = data_veloc_norm["norm_velocity"]
            print("Time for reading the field: "+str(data_veloc_norm["time_read"]),flush=True)
            print("Time for normalizing the field: "+str(data_veloc_norm["time_norm"]),flush=True)
            del data_norm_out,data_veloc_norm
        
            
            # -----------------------------------------------------------------------------------------------------------
            # Calculate the SHAP values
            # -----------------------------------------------------------------------------------------------------------
            shap_values   = self._calculate_gradientshaps(data_in={"norm_velocity_in":norm_velocity_in,
                                                           "norm_velocity_out":norm_velocity_out,
                                                           "x0":0,"z0":0})
            shap_values_u = shap_values["shap_u"]
            shap_values_v = shap_values["shap_v"]
            shap_values_w = shap_values["shap_w"]
            if self.nrep_field > 0:
                for ii_rep in np.arange(self.nrep_field):
                    print("Repetition:"+str(ii_rep)+"/"+str(self.nrep_field),flush=True)
                    x0             = int(np.round((self.shpx-1)*np.random.rand()))
                    z0             = int(np.round((self.shpz-1)*np.random.rand()))
                    shap_values    = self._calculate_gradientshaps(data_in={"norm_velocity_in":norm_velocity_in,
                                                                            "norm_velocity_out":norm_velocity_out,
                                                                            "x0":x0,"z0":z0})
                    shap_values_u += shap_values["shap_u"]
                    shap_values_v += shap_values["shap_v"]
                    shap_values_w += shap_values["shap_w"]
            
            # -----------------------------------------------------------------------------------------------------------
            # Calculate the mean value of the SHAP
            # -----------------------------------------------------------------------------------------------------------
            shap_values_u /= self.nrep_field+1
            shap_values_v /= self.nrep_field+1
            shap_values_w /= self.nrep_field+1
            
            if self.padding>0:
                # ---------------------------------------------------------------------------------------------------------------
                # Apply the padding
                # ---------------------------------------------------------------------------------------------------------------
                shap_values_u = padding_field(data_in={"field":shap_values_u,"shpx":self.shpx,"shpy":self.shpy,
                                                       "shpz":self.shpz,"padding":self.padding})["field"]
                shap_values_v = padding_field(data_in={"field":shap_values_v,"shpx":self.shpx,"shpy":self.shpy,
                                                       "shpz":self.shpz,"padding":self.padding})["field"]
                shap_values_w = padding_field(data_in={"field":shap_values_w,"shpx":self.shpx,"shpy":self.shpy,
                                                       "shpz":self.shpz,"padding":self.padding})["field"]
            
            # -----------------------------------------------------------------------------------------------------------
            # Save the SHAP values
            # -----------------------------------------------------------------------------------------------------------
            data_shap = {"shap_values_u":shap_values_u,"shap_values_v":shap_values_v,"shap_values_w":shap_values_w,
                         "index":index_ii}
            self.write_shap(data_in=data_shap)
            print('-'*100,flush=True)
            
            
                
    def calc_kernelSHAP(self,data_in={"structurefolder":[],"struct_type":[],"filvol":0}):
        """
        .................................................................................................................
        # calc_kernelSHAP
        .................................................................................................................
        
        Parameters
        ----------
        data_in : dict, optional
            Data to calculate the kernel shap values. The default is {"struc_folder":[],"struc_file":[]}.
            Data: 
                - structurefolder : information of the path to structures
                - struct_type     : type of structure
                - filvol          : volume filter

        Returns
        -------
        dict
            Shap values in the structures
            Data:
                - shap : Shap values in the structures
        """
        # ---------------------------------------------------------------------------------------------------------------
        # Import packages
        # ---------------------------------------------------------------------------------------------------------------
        from py_bin.py_functions.read_norm_velocity import read_norm_velocity
        from py_bin.py_functions.padding_field import padding_field
        import importlib
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read information
        # ---------------------------------------------------------------------------------------------------------------
        structurefolder = data_in["structurefolder"]
        struct_type     = data_in["struct_type"] 
        filvol          = data_in["filvol"]
        struct_folder   = struct_type+"_folder"
        struct_file     = struct_type+"_file"
        struct_fun      = struct_type+"_fun"
        strucsel_folder = structurefolder[struct_folder]
        strucsel_file   = structurefolder[struct_file]
        strucsel_fun    = structurefolder[struct_fun]
        
        # ---------------------------------------------------------------------------------------------------------------
        # Calculate the background values
        # ---------------------------------------------------------------------------------------------------------------
        self.background()
        
        # ---------------------------------------------------------------------------------------------------------------
        # Calculate the SHAP values of all the flow fields
        # ---------------------------------------------------------------------------------------------------------------
        interval = range(self.field_ini,self.field_fin,self.field_delta)
        for index_ii in interval:
            
            # -----------------------------------------------------------------------------------------------------------
            # Load the structures module
            # -----------------------------------------------------------------------------------------------------------
            data_struc         = {"uvw_folder":self.uvw_folder,"uvw_file":self.uvw_file,"Hperc":0,"index":index_ii,
                                  "dx":self.dx,"dy":self.dy,"dz":self.dz,"L_x":self.L_x,"L_y":self.L_y,"L_z":self.L_z,
                                  "rey":self.rey,"utau":self.utau,"padding":self.padding,"data_folder":self.data_folder,
                                  "umean_file":self.umean_file,"urms_file":"-","sym_quad":True,
                                  "filvol":filvol,"shap_folder":"-","shap_file":"-","folder":strucsel_folder,
                                  "file":strucsel_file,"padding":self.padding,"data_type":self.data_type}
            module             = importlib.import_module(f"py_bin.py_class.{strucsel_fun}")
            structure_function = getattr(module, strucsel_fun)
            
            # -----------------------------------------------------------------------------------------------------------
            # Calculate segmentation
            # -----------------------------------------------------------------------------------------------------------
            segment_struc                            = structure_function(data_in=data_struc)
            segment_struc.read_struc()
            self.segmentation                        = segment_struc.structures.mat_segment_filtered
            self.struc_num                           = np.max(self.segmentation)
            self.segmentation                        = self.segmentation-1
            self.segmentation[self.segmentation==-1] = self.struc_num
            self.index_filtered                      = segment_struc.structures.filt_index
            
            # -----------------------------------------------------------------------------------------------------------
            # Check if the file has already been calculated
            # -----------------------------------------------------------------------------------------------------------
            file_shap     = self._file_name(data_in={"index_ii":index_ii})["file_shap"]
            exist_file    = bool(len(glob(file_shap)))
            if exist_file and not self.repeat_exist:
                print("Existing field",flush=True)
                continue
            else:
                print("New field",flush=True)
            
            print("-"*100,flush=True)
            print('Calculating the SHAP for field: '+str(index_ii),flush=True)
            # -----------------------------------------------------------------------------------------------------------
            # Read the input field 
            # -----------------------------------------------------------------------------------------------------------   
            data_norm_in     = {"folder":self.uvw_folder,"file":self.uvw_file,"padding":self.padding,"shpx":self.shpx,
                                "shpy":self.shpy,"shpz":self.shpz,"dx":self.dx,"dy":self.dy,"dz":self.dz,
                                "data_folder":self.data_folder,"umean_file":self.umean_file,
                                "unorm_file":self.unorm_file,"index":index_ii,"data_type":self.data_type,
                                "mean_norm":self.mean_norm}
            data_veloc_norm  = read_norm_velocity(data_in=data_norm_in)
            norm_velocity_in = data_veloc_norm["norm_velocity"]
            print("Time for reading the field: "+str(data_veloc_norm["time_read"]),flush=True)
            print("Time for normalizing the field: "+str(data_veloc_norm["time_norm"]),flush=True)
            del data_norm_in,data_veloc_norm
            
            # -----------------------------------------------------------------------------------------------------------
            # Read the output field 
            # -----------------------------------------------------------------------------------------------------------
            data_norm_out     = {"folder":self.uvw_folder,"file":self.uvw_file,"padding":0,"shpx":self.shpx,
                                 "shpy":self.shpy,"shpz":self.shpz,"dx":self.dx,"dy":self.dy,"dz":self.dz,
                                 "data_folder":self.data_folder,"umean_file":self.umean_file,
                                 "unorm_file":self.unorm_file,"index":index_ii+self.delta_pred,
                                 "data_type":self.data_type,"mean_norm":self.mean_norm}
            data_veloc_norm   = read_norm_velocity(data_in=data_norm_out)
            norm_velocity_out = data_veloc_norm["norm_velocity"]
            print("Time for reading the field: "+str(data_veloc_norm["time_read"]),flush=True)
            print("Time for normalizing the field: "+str(data_veloc_norm["time_norm"]),flush=True)
            del data_norm_out,data_veloc_norm
        
            
            # -----------------------------------------------------------------------------------------------------------
            # Calculate the SHAP values
            # -----------------------------------------------------------------------------------------------------------
            shap_values   = self._calculate_kernelshaps(data_in={"norm_velocity_in":norm_velocity_in,
                                                                 "norm_velocity_out":norm_velocity_out
                                                                 })["shap"]
            
            # -----------------------------------------------------------------------------------------------------------
            # Save the SHAP values
            # -----------------------------------------------------------------------------------------------------------
            data_shap = {"shap_values":shap_values,"index":index_ii}
            self.write_shap_kernel(data_in=data_shap)
            print('-'*100,flush=True)
            
            
                                  
    def write_shap(self,data_in={"shap_values_u":[],"shap_values_v":[],"shap_values_w":[],"index":0}):
        """
        .................................................................................................................
        # write_shap
        .................................................................................................................
        Function to save the SHAP values in a file.

        Parameters
        ----------
        data_in : dict, optional
            Data required for saving the SHAP values.
            The default is {"shap_values_u":[],"shap_values_v":[],"shap_values_w":[],"index":0}.
            Data:
                - shap_values_u : component u of the SHAP values
                - shap_values_v : component v of the SHAP values
                - shap_values_w : component w of the SHAP values
                - index         : index of the flow field

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Import packages
        # ---------------------------------------------------------------------------------------------------------------        
        from h5py import File
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the data
        # ---------------------------------------------------------------------------------------------------------------
        shap_values_u = np.array(data_in["shap_values_u"],dtype='float')
        shap_values_v = np.array(data_in["shap_values_v"],dtype='float')
        shap_values_w = np.array(data_in["shap_values_w"],dtype='float')
        index_ii      = int(data_in["index"])
        file_shap     = self._file_name(data_in={"index_ii":index_ii})["file_shap"]
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create the folder
        # ---------------------------------------------------------------------------------------------------------------
        try:
            mkdir(self.shap_folder)
        except:
            print("Folder is already created.",flush=True)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create the file and store the shap values
        # ---------------------------------------------------------------------------------------------------------------
        hf = File(file_shap,'w')
        hf.create_dataset('SHAP_u',data=shap_values_u)
        hf.create_dataset('SHAP_v',data=shap_values_v)
        hf.create_dataset('SHAP_w',data=shap_values_w)
        hf.close()
        
                                          
    def write_shap_kernel(self,data_in={"shap_values":[],"index":0}):
        """
        .................................................................................................................
        # write_shap_kernel
        .................................................................................................................
        Function to save the SHAP values in a file.

        Parameters
        ----------
        data_in : dict, optional
            Data required for saving the SHAP values.
            The default is {"shap_values":[],"index":0}.
            Data:
                - shap_values  :  SHAP values
                - index        : index of the flow field

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Import packages
        # ---------------------------------------------------------------------------------------------------------------        
        from h5py import File
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the data
        # ---------------------------------------------------------------------------------------------------------------
        shap_values  = np.array(data_in["shap_values"],dtype='float')
        index_ii     = int(data_in["index"])
        file_shap    = self._file_name(data_in={"index_ii":index_ii})["file_shap"]
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create the folder
        # ---------------------------------------------------------------------------------------------------------------
        try:
            mkdir(self.shap_folder)
        except:
            print("Folder is already created.",flush=True)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create the file and store the shap values
        # ---------------------------------------------------------------------------------------------------------------
        hf = File(file_shap,'w')
        hf.create_dataset('SHAP',data=shap_values)
        hf.create_dataset('index_filtered',data=self.index_filtered)
        hf.close()
        
    def read_shap(self,data_in = {"index":0}):
        """
        .................................................................................................................
        # read_shap
        .................................................................................................................
        Function to read the SHAP values

        Parameters
        ----------
        data_in : dict, optional
            Data to read the SHAP values file.
            The default is {"index":0}.
            Data:
                - index : index of the file to save

        Returns
        -------
        data_out : dict
            Data from the SHAP values file.
            Data:
                - SHAP_u : SHAP values of the field u
                - SHAP_v : SHAP values of the field v
                - SHAP_w : SHAP values of the field w

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Import packages
        # ---------------------------------------------------------------------------------------------------------------
        from h5py import File
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read data
        # ---------------------------------------------------------------------------------------------------------------
        index_ii = int(data_in["index"])
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the file
        # ---------------------------------------------------------------------------------------------------------------
        file_shap     = self._file_name(data_in={"index_ii":index_ii})["file_shap"]
        hf            = File(file_shap,'r+')
        shap_values_u = np.array(hf['SHAP_u'],dtype=self.data_type)
        shap_values_v = np.array(hf['SHAP_v'],dtype=self.data_type)
        shap_values_w = np.array(hf['SHAP_w'],dtype=self.data_type)
        hf.close()
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create the output data
        # ---------------------------------------------------------------------------------------------------------------
        data_out = {"SHAP_u":shap_values_u,"SHAP_v":shap_values_v,"SHAP_w":shap_values_w}
        return data_out
    
            
    def read_shap_kernel(self,data_in = {"index":0}):
        """
        .................................................................................................................
        # read_shap_kernel
        .................................................................................................................
        Function to read the SHAP values calculated in the segmented domain

        Parameters
        ----------
        data_in : dict, optional
            Data to read the SHAP values file.
            The default is {"index":0}.
            Data:
                - index : index of the file to save

        Returns
        -------
        data_out : dict
            Data from the SHAP values file.
            Data:
                - SHAP           : SHAP values 
                - index_filtered : index of the structures used for segmenting the domain

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Import packages
        # ---------------------------------------------------------------------------------------------------------------
        from h5py import File
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read data
        # ---------------------------------------------------------------------------------------------------------------
        index_ii = int(data_in["index"])
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the file
        # ---------------------------------------------------------------------------------------------------------------
        file_shap      = self._file_name(data_in={"index_ii":index_ii})["file_shap"]
        hf             = File(file_shap,'r+')
        shap_values    = np.array(hf['SHAP'],dtype=self.data_type)
        index_filtered = np.array(hf["index_filtered"],dtype=self.data_type)
        hf.close()
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create the output data
        # ---------------------------------------------------------------------------------------------------------------
        data_out = {"SHAP":shap_values,"index_filtered":index_filtered}
        return data_out
        
    def gradientSHAP_model(self,data_in={"field_out":[]}):
        """
        .................................................................................................................
        # gradientSHAP_model
        .................................................................................................................
        Function to define the gradientSHAP model

        Parameters
        ----------
        data_in : dict, optional
            data for the shap model.
            The default is {"field_out":[]}.
            Data:
                - field_out : ouput field taken from dataset

        Returns
        -------
        None.

        """
        
        # ---------------------------------------------------------------------------------------------------------------
        # Import packages
        # ---------------------------------------------------------------------------------------------------------------
        import tensorflow as tf
        from tensorflow.keras import Model
        from tensorflow.keras.optimizers import RMSprop
        import psutil
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create the model
        # ---------------------------------------------------------------------------------------------------------------
        with self.strategy.scope(): 
            self.model_base_shap(data_in=data_in)
            optimizer  = RMSprop() 
            self.model = Model(self.inputs, self.outputs)
            self.model.set_weights(self.weights)
            self.model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=optimizer)
        if self.print_summary:
            self.model.summary()  
            self.print_summary = False
            memory_data = psutil.virtual_memory()
            print('Total RAM (GB): '+str(memory_data[0]/1e9),flush=True)
        print("-"*100,flush=True)
        
        
                
    def model_base_shap(self,data_in={"field_out":[]}):
        """
        .................................................................................................................
        # model_base_shap
        .................................................................................................................
        Function to define the model used for the problem

        Parameters
        ----------
        data_in : dict, optional
            data for the shap model.
            The default is {"field_out":[]}.
            Data:
                - field_out : ouput field taken from dataset

        Returns
        -------
        None.

        """
        # -------------------------------------------------------------------------------------------------------------
        # Define the required packages
        # -------------------------------------------------------------------------------------------------------------
        from tensorflow.keras.layers import Input
        from tensorflow.math import subtract, multiply, reduce_mean
        from tensorflow import reshape, constant, cast, math, float32    
        # from tensorflow.keras.layers import Input,subtract,multiply,GlobalAveragePooling3D,add #Layer,
        # # from tensorflow.math import subtract, multiply, reduce_mean
        # from tensorflow import reshape, constant, cast, math, float32
        
        # -------------------------------------------------------------------------------------------------------------
        # Read data
        # -------------------------------------------------------------------------------------------------------------
        field_out = constant(data_in["field_out"])
        
        # -------------------------------------------------------------------------------------------------------------
        # The dimensions of the input field are selected from the size of the fields, adding the padding
        # -------------------------------------------------------------------------------------------------------------
        dim0 = self.shpy
        dim1 = self.shpz+2*self.padding
        dim2 = self.shpx+2*self.padding
        dim3 = 3
        shp = (dim0,dim1,dim2,dim3)
        
        # -------------------------------------------------------------------------------------------------------------
        # Define the input and the output of the model
        # -------------------------------------------------------------------------------------------------------------
        self.inputs  = Input(shape=shp,dtype=self.data_type)
        x_in         = self.inputs
        x_out        = self.architecture_Unet(data_in={"x_in":x_in,"flag_print":self.print_summary})["x_out"]
        x_out        = cast(x_out,dtype=float32)
        field_out    = cast(field_out,dtype=x_out.dtype)
        outsubs      = subtract(x_out,field_out)
        outsubs2     = multiply(outsubs,outsubs)
        out_mse      = reduce_mean(outsubs2,keepdims=True,axis=(1,2,3,4))
        scale_ctn    = cast(constant(1e4),dtype=out_mse.dtype)
        out_mse      = math.scalar_mul(scale_ctn,reshape(out_mse,[-1]))
        self.outputs = out_mse
        # try:
        # outsubs      = subtract([x_out,field_out],dtype=float32)
        # outsubs2     = multiply([outsubs,outsubs])
        # out_mse_comp = GlobalAveragePooling3D()(outsubs2)
        # out_mse      = add([add([out_mse_comp[:,0],out_mse_comp[:,1]]),out_mse_comp[:,2]])
        # except:
        #     # ---------------------------------------------------------------------------------------------------------
        #     # Define the layers so tensorflow latest versions can operate
        #     # ---------------------------------------------------------------------------------------------------------
        #     # Define the cast function
        #     # ---------------------------------------------------------------------------------------------------------
        #     def custom_cast(x_tf,dtype=float32):
        #         return cast(x_tf,dtype=dtype)
        #     class CustomLayer_cast(Layer):
        #         def call(self,inputs,dtype=float32):
        #             return custom_cast(inputs,dtype=dtype)
        #     # ---------------------------------------------------------------------------------------------------------
        #     # Define the subtract function
        #     # ---------------------------------------------------------------------------------------------------------
        #     def custom_subtract(x_tf,y_tf):
        #         return subtract(x_tf,y_tf)
        #     class CustomLayer_subtract(Layer):
        #         def call(self,inputs):
        #             x_in,y_in = inputs
        #             return custom_subtract(x_in,y_in)
        #     # ---------------------------------------------------------------------------------------------------------
        #     # Define the multiply function
        #     # ---------------------------------------------------------------------------------------------------------
        #     def custom_multiply(x_tf,y_tf):
        #         return multiply(x_tf,y_tf)
        #     class CustomLayer_multiply(Layer):
        #         def call(self,inputs):
        #             x_in,y_in = inputs
        #             return custom_multiply(x_in,y_in)
        #     # ---------------------------------------------------------------------------------------------------------
        #     # Define the reducemean function
        #     # ---------------------------------------------------------------------------------------------------------
        #     def custom_reducemean(x_tf):
        #         return reduce_mean(x_tf,keepdims=True,axis=(1,2,3,4))
        #     class CustomLayer_reducemean(Layer):
        #         def call(self,inputs):
        #             return custom_reducemean(inputs)
        #     # ---------------------------------------------------------------------------------------------------------
        #     # Define the reshape function
        #     # ---------------------------------------------------------------------------------------------------------
        #     def custom_reshape(x_tf):
        #         return reshape(x_tf,[-1])
        #     class CustomLayer_reshape(Layer):
        #         def call(self,inputs):
        #             return custom_reshape(inputs)
        #     # ---------------------------------------------------------------------------------------------------------
        #     # Define the scalar multiplication function
        #     # ---------------------------------------------------------------------------------------------------------
        #     def custom_scalar_mul(scalar_tf,tensor_tf):
        #         return math.scalar_mul(scalar_tf,tensor_tf)
        #     class CustomLayer_scalar_mul(Layer):
        #         def call(self,inputs):
        #             scalar_in,tensor_in = inputs
        #             return custom_scalar_mul(scalar_in,tensor_in)
        #     # ---------------------------------------------------------------------------------------------------------
        #     # Operate the model
        #     # ---------------------------------------------------------------------------------------------------------
        #     x_out     = CustomLayer_cast()(x_out)
        #     field_out = CustomLayer_cast()(field_out)
        #     outsubs   = CustomLayer_subtract()([x_out,field_out])
        #     outsubs2  = CustomLayer_multiply()([outsubs,outsubs])
        #     out_mse   = CustomLayer_reducemean()(outsubs2)
        #     scale_ctn = CustomLayer_cast()(constant(1e4),dtype=out_mse.dtype)
        #     out_mse   = CustomLayer_scalar_mul()([scale_ctn,CustomLayer_reshape()(out_mse)])
        self.outputs = out_mse
        
        
    def background(self):
        """
        .................................................................................................................
        # background
        .................................................................................................................
        Function to calculate the background values

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Import the packages
        # ---------------------------------------------------------------------------------------------------------------
        from py_bin.py_functions.normalization import read_norm
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create the background thensor as a null fluctuation velocity.
        # Both the tensor and the values of the normalized 0 fluctuation velocities are stored using float 16 to 
        # reduce the use of memory.
        # ---------------------------------------------------------------------------------------------------------------
        data_norm       = read_norm(data_in={"folder":self.data_folder,"file":self.unorm_file})
        self.backmat    = np.ones((1,self.shpy,self.shpz+2*self.padding,self.shpx+2*self.padding,3),
                                  dtype=self.data_type)
        u_zerofluc      = np.array((-data_norm["uumin"])/(data_norm["uumax"]-data_norm["uumin"]),dtype=self.data_type)
        v_zerofluc      = np.array((-data_norm["vvmin"])/(data_norm["vvmax"]-data_norm["vvmin"]),dtype=self.data_type)
        w_zerofluc      = np.array((-data_norm["wwmin"])/(data_norm["wwmax"]-data_norm["wwmin"]),dtype=self.data_type)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Store the null fluctuations in the background tensor
        # ---------------------------------------------------------------------------------------------------------------
        self.backmat[0,:,:,:,0] *= u_zerofluc
        self.backmat[0,:,:,:,1] *= v_zerofluc
        self.backmat[0,:,:,:,2] *= w_zerofluc
        
        
    def structure_shap(self,data_in={"mat_segment":[],"index_ii":1000}):
        """
        .................................................................................................................
        # structure_shap
        .................................................................................................................
        Function for calculating the total SHAP of the structure
    
        Parameters
        ----------
        data_in : dict, optional
            data for calculating the total SHAP of the structure.
            The default is {"mat_segment":[],"index_ii":1000}.
            Data:
                - mat_segment : matrix of the segmented domain
                - index_ii    : index of the field
                
        Returns
        -------
        data_out : dict
            Data of the shap value of the structures.
            Data:
                - SHAP : SHAP values of the structures
    
        """         
        # ---------------------------------------------------------------------------------------------------------------
        # Read the data
        # ---------------------------------------------------------------------------------------------------------------
        mat_segment = data_in["mat_segment"]
        index_ii    = int(data_in["index_ii"])
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the SHAP value fields
        # ---------------------------------------------------------------------------------------------------------------
        shap_read_data     = self.read_shap(data_in = {"index":index_ii})
        shap_data          = np.zeros((self.shpy,self.shpz,self.shpx,3))
        shap_data[:,:,:,0] = shap_read_data["SHAP_u"][:,self.padding:-self.padding,self.padding:-self.padding] 
        shap_data[:,:,:,1] = shap_read_data["SHAP_v"][:,self.padding:-self.padding,self.padding:-self.padding] 
        shap_data[:,:,:,2] = shap_read_data["SHAP_w"][:,self.padding:-self.padding,self.padding:-self.padding]                          
                                    
        # ---------------------------------------------------------------------------------------------------------------
        # For every structure sum all the norms of the SHAP values of the nodes of the structure
        #     - nn : index of the structure
        #     - nodes_nn : indices of the nodes of the structure nn
        # ---------------------------------------------------------------------------------------------------------------
        max_struc = int(np.max(mat_segment))
        shap_str  = np.zeros((max_struc,))
        for nn  in np.arange(max_struc-1):
            nodes_nn      = np.where(mat_segment==nn+1)
            shap_str[nn] += np.linalg.norm(shap_data[nodes_nn])
            
        # ---------------------------------------------------------------------------------------------------------------
        # Store the output data
        # ---------------------------------------------------------------------------------------------------------------
        data_out = {"SHAP":shap_str}
        return data_out
    
    def check_repetitions_independence(self,data_in={"index_ii":0,"repetitions":[],"file_repetition":"-"}):
        """
        .................................................................................................................
        # check_repetitions_independence
        .................................................................................................................        

        Parameters
        ----------
        data_in : dict, optional
            data for calculating the error of the shap field as a function of the number of repetitions.
            The default is {"index_ii":0,"repetitions":[]}.
            Data:
                - index_ii        : index of the field
                - repetitions     : vector containing the number of repetitions of the analysis
                - file_repetition : file to store the repetition

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Read data
        # ---------------------------------------------------------------------------------------------------------------
        index           = int(data_in["index"])
        repetitions     = data_in["repetitions"]
        file_repetition = str(data_in["file_repetition"])
        
        # ---------------------------------------------------------------------------------------------------------------
        # Save the shap folder original name
        # ---------------------------------------------------------------------------------------------------------------
        self.shap_folder_base = self.shap_folder
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the field for the number of repetitions
        # ---------------------------------------------------------------------------------------------------------------
        nrep   = len(repetitions)
        shap_u = np.zeros((nrep,self.shpy,self.shpz+2*self.padding,self.shpx+2*self.padding))
        shap_v = np.zeros((nrep,self.shpy,self.shpz+2*self.padding,self.shpx+2*self.padding))
        shap_w = np.zeros((nrep,self.shpy,self.shpz+2*self.padding,self.shpx+2*self.padding))
        for ii in np.arange(nrep):
            ii_rep           = int(repetitions[ii])
            self.shap_folder = self.shap_folder_base+"_"+str(ii_rep)
            field_ii         = self.read_shap(data_in={"index":index})
            shap_u[ii,:,:,:] = field_ii["SHAP_u"]
            shap_v[ii,:,:,:] = field_ii["SHAP_v"]
            shap_w[ii,:,:,:] = field_ii["SHAP_w"]
        
        # ---------------------------------------------------------------------------------------------------------------
        # Calculate the errors
        # ---------------------------------------------------------------------------------------------------------------
        error_u      = np.zeros((nrep-1,self.shpy,self.shpz+2*self.padding,self.shpx+2*self.padding))
        error_v      = np.zeros((nrep-1,self.shpy,self.shpz+2*self.padding,self.shpx+2*self.padding))
        error_w      = np.zeros((nrep-1,self.shpy,self.shpz+2*self.padding,self.shpx+2*self.padding))
        max_u        = np.max(shap_u[-1,:,:,:])
        max_v        = np.max(shap_v[-1,:,:,:])
        max_w        = np.max(shap_w[-1,:,:,:])
        for ii in np.arange(nrep-1):
            error_u[ii,:,:,:] = (shap_u[ii,:,:,:]-shap_u[-1,:,:,:])/max_u
            error_v[ii,:,:,:] = (shap_v[ii,:,:,:]-shap_v[-1,:,:,:])/max_v
            error_w[ii,:,:,:] = (shap_w[ii,:,:,:]-shap_w[-1,:,:,:])/max_w
        error_u_max  = np.max(error_u,axis=(1,2,3))
        error_v_max  = np.max(error_v,axis=(1,2,3))
        error_w_max  = np.max(error_w,axis=(1,2,3))
        error_u_mean = np.mean(error_u,axis=(1,2,3))
        error_v_mean = np.mean(error_v,axis=(1,2,3))
        error_w_mean = np.mean(error_w,axis=(1,2,3))
        error_u_std  = np.std(error_u,axis=(1,2,3))
        error_v_std  = np.std(error_v,axis=(1,2,3))
        error_w_std  = np.std(error_w,axis=(1,2,3))
        # ---------------------------------------------------------------------------------------------------------------
        # Print and save the results
        # ---------------------------------------------------------------------------------------------------------------
        for ii in np.arange(nrep-1):
            print('Maximum error for '+str(repetitions[ii])+' repetitions: $\epsilon_u=$'+str(error_u_max[ii])+
                  '; $\epsilon_v=$'+str(error_v_max[ii])+'; $\epsilon_w=$'+str(error_w_max[ii]))
            print('Mean error for '+str(repetitions[ii])+' repetitions: $\epsilon_u=$'+str(error_u_mean[ii])+
                  '; $\epsilon_v=$'+str(error_v_mean[ii])+'; $\epsilon_w=$'+str(error_w_mean[ii]))
            print('Standard deviation of the error for '+str(repetitions[ii])+
                  ' repetitions: $\epsilon_u=$'+str(error_u_std[ii])+'; $\epsilon_v=$'+
                  str(error_v_std[ii])+'; $\epsilon_w=$'+str(error_w_std[ii]))
        file_repetition = self.data_folder+'/'+file_repetition
        file_save       = open(file_repetition, "w+")           
        content         = str(error_u_max)+'\n'
        file_save.write(content)           
        content         = str(error_v_max)+'\n'
        file_save.write(content)           
        content         = str(error_w_max)+'\n'
        file_save.write(content)           
        content         = str(error_u_mean)+'\n'
        file_save.write(content)           
        content         = str(error_v_mean)+'\n'
        file_save.write(content)           
        content         = str(error_w_mean)+'\n'
        file_save.write(content)           
        content         = str(error_u_std)+'\n'
        file_save.write(content)           
        content         = str(error_v_std)+'\n'
        file_save.write(content)           
        content         = str(error_w_std)+'\n'
        file_save.write(content) 
        file_save.close()
        
        
    def check_repetitions_snr(self,data_in={"index_ii":0,"repetitions":[],"ngp":2,"file_snr":"-"}):
        """
        .................................................................................................................
        # check_repetitions_independence
        .................................................................................................................        

        Parameters
        ----------
        data_in : dict, optional
            data for calculating the error of the shap field as a function of the number of repetitions.
            The default is {"index_ii":0,"repetitions":[]}.
            Data:
                - index_ii    : index of the field
                - repetitions : vector containing the number of repetitions of the analysis
                - ngp         : number of grid points of the cutting frequency
                - file_snr    : file of the snr

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Read data
        # ---------------------------------------------------------------------------------------------------------------
        index           = int(data_in["index"])
        repetitions     = data_in["repetitions"]
        ngp             = int(data_in["ngp"])
        file_snr        = str(data_in["file_snr"])
        
        # ---------------------------------------------------------------------------------------------------------------
        # Save the shap folder original name
        # ---------------------------------------------------------------------------------------------------------------
        self.shap_folder_base = self.shap_folder
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the field for the number of repetitions
        # ---------------------------------------------------------------------------------------------------------------
        nrep   = len(repetitions)
        shap_u = np.zeros((nrep,self.shpy,self.shpz+2*self.padding,self.shpx+2*self.padding))
        shap_v = np.zeros((nrep,self.shpy,self.shpz+2*self.padding,self.shpx+2*self.padding))
        shap_w = np.zeros((nrep,self.shpy,self.shpz+2*self.padding,self.shpx+2*self.padding))
        for ii in np.arange(nrep):
            ii_rep           = int(repetitions[ii])
            self.shap_folder = self.shap_folder_base+"_"+str(ii_rep)
            field_ii         = self.read_shap(data_in={"index":index})
            shap_u[ii,:,:,:] = field_ii["SHAP_u"]
            shap_v[ii,:,:,:] = field_ii["SHAP_v"]
            shap_w[ii,:,:,:] = field_ii["SHAP_w"]
            
        # ---------------------------------------------------------------------------------------------------------------
        # Analyze the noise/signal ratio
        # ---------------------------------------------------------------------------------------------------------------
        snr_u  = np.zeros((nrep,))
        snr_v  = np.zeros((nrep,))
        snr_w  = np.zeros((nrep,))
        file_snr_tot = self.data_folder+'/'+file_snr
        file_save       = open(file_snr_tot, "w+")      
        for ii in np.arange(nrep):
            # -----------------------------------------------------------------------------------------------------------
            # Apply 3d fourier transform
            # -----------------------------------------------------------------------------------------------------------
            print("Apply fourier transform",flush=True)
            data_fft_u         = np.fft.fftn(shap_u[ii,:,:,:]) 
            data_fft_v         = np.fft.fftn(shap_v[ii,:,:,:]) 
            data_fft_w         = np.fft.fftn(shap_w[ii,:,:,:]) 
            # -----------------------------------------------------------------------------------------------------------
            # Shift the 0 frequency to the center of the spectrum
            # -----------------------------------------------------------------------------------------------------------
            print("Shift the frequency transform",flush=True)
            data_fft_shifted_u = np.fft.fftshift(data_fft_u)
            data_fft_shifted_v = np.fft.fftshift(data_fft_v)
            data_fft_shifted_w = np.fft.fftshift(data_fft_w)         
            # -----------------------------------------------------------------------------------------------------------
            # Calculate filter mask
            # -----------------------------------------------------------------------------------------------------------
            shape  = [self.shpy,self.shpz+2*self.padding,self.shpx+2*self.padding]
            center = [s // 2 for s in shape]
            mask   = np.zeros(shape)
            ymat   = np.arange(shape[0]).reshape(-1,1,1)*np.ones((shape[0],shape[1],shape[2]))
            zmat   = np.arange(shape[1]).reshape(1,-1,1)*np.ones((shape[0],shape[1],shape[2]))
            xmat   = np.arange(shape[2]).reshape(1,1,-1)*np.ones((shape[0],shape[1],shape[2]))
            dist_y = abs(ymat-center[0])
            dist_z = abs(zmat-center[1])
            dist_x = abs(xmat-center[2])
            cuto_y = shape[0]/2/ngp
            cuto_z = shape[1]/2/ngp
            cuto_x = shape[2]/2/ngp
            mat_co = (dist_y<cuto_y)*(dist_z<cuto_z)*(dist_x<cuto_x)
            mask[mat_co] = 1
            # -----------------------------------------------------------------------------------------------------------
            # Low pass filtering
            # -----------------------------------------------------------------------------------------------------------
            print("Low pass filtering",flush=True)
            filtered_fft_u     = data_fft_shifted_u*mask
            filtered_fft_v     = data_fft_shifted_v*mask
            filtered_fft_w     = data_fft_shifted_w*mask
            # -----------------------------------------------------------------------------------------------------------
            # Filtered signal
            # -----------------------------------------------------------------------------------------------------------
            print("Filtered signal",flush=True)
            signal_filtered_u  = np.abs(np.fft.ifftn(np.fft.fftshift(filtered_fft_u)))**2
            signal_filtered_v  = np.abs(np.fft.ifftn(np.fft.fftshift(filtered_fft_v)))**2
            signal_filtered_w  = np.abs(np.fft.ifftn(np.fft.fftshift(filtered_fft_w)))**2
            # -----------------------------------------------------------------------------------------------------------
            # Calculate the noise
            # -----------------------------------------------------------------------------------------------------------
            print("Calculate the noise",flush=True)
            noise_u            = shap_u[ii,:,:,:]**2 - signal_filtered_u
            noise_v            = shap_v[ii,:,:,:]**2 - signal_filtered_v
            noise_w            = shap_w[ii,:,:,:]**2 - signal_filtered_w
            # -----------------------------------------------------------------------------------------------------------
            # Calculate the SNR
            # -----------------------------------------------------------------------------------------------------------
            print("Calculate SNR",flush=True)
            signal_power_u     = np.mean(signal_filtered_u)
            signal_power_v     = np.mean(signal_filtered_v)
            signal_power_w     = np.mean(signal_filtered_w)
            noise_power_u      = np.mean(noise_u)
            noise_power_v      = np.mean(noise_v)
            noise_power_w      = np.mean(noise_w)
            snr_u[ii]          = 10 * np.log10(signal_power_u / noise_power_u)
            snr_v[ii]          = 10 * np.log10(signal_power_v / noise_power_v)
            snr_w[ii]          = 10 * np.log10(signal_power_w / noise_power_w)
            print('SNR '+str(repetitions[ii])+' repetitions: u='+str(snr_u[ii])+
                  '; v='+str(snr_v[ii])+'; w='+str(snr_w[ii]))   
        file_snr_tot = self.data_folder+'/'+file_snr
        file_save    = open(file_snr_tot, "w+")   
        content      = str(snr_u)+'\n'
        file_save.write(content)     
        content      = str(snr_v)+'\n'
        file_save.write(content) 
        content      = str(snr_w)+'\n'
        file_save.write(content)       
        file_save.close()   
        for ii in np.arange(nrep-1):
            print('SNR percentage'+str(repetitions[ii])+' repetitions: u='+str(snr_u[ii]/snr_u[-1]*100)+
                  '%; v='+str(snr_v[ii]/snr_v[-1]*100)+'%; w='+str(snr_w[ii]/snr_w[-1]*100)+'%') 
    
            
            