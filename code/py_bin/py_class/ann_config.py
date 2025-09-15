# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
ann_config.py
-------------------------------------------------------------------------------------------------------------------------
Created on Mon Mar 18 10:18:38 2024

@author: Andres Cremades Botella

File to define the deep learning model. The file contains a class for the deep learning model:
    Class:
        - deep_model : Class of the deep learning model. 
"""
# -----------------------------------------------------------------------------------------------------------------------
# Import packages for all the functions
# -----------------------------------------------------------------------------------------------------------------------
import os
import numpy as np
import sys

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

class deep_model():
    """
    .....................................................................................................................
    # deep_model: Class containing the information of the deep learning model.
        * Functions:
            - __init__          : initialization of the class. Read data of the flow, path to the data...
            - define_model      : defines the Deep Learning model
            - create_model      : creates the Deep Learning model
            - model_base        : defines the model layers
            - train_model       : trains the DL model
            - prepare_data      : function to create the data in the tensorflow format
            - architecture_Unet : definition of the architecture of the U-net
            - _save_training    : function for saving the training data
            - check_data        : check the database
            - pred_field        : function to use the model for predicting the next field
            - field_error       : function to calculate field containing the error of the prediction in all the 
                                  velocity components
            - pred_error        : function to calculate the error of the prediction weighted by the volume in a set of
                                  fields
        * Variables:
            - uvw_folder        : folder of the velocity fields
            - uvw_file          : file name of the velocity fileds without index
            - padding           : padding of the fields
            - dx                : downsampling in x
            - dy                : downsampling in y
            - dz                : downsampling in z
            - data_folder       : folder for storing the data of the model
            - umean_file        : file of the mean velocity
            - unorm_file        : file for the normalization of the velocity
            - L_x               : size of the channel in the streamwise direction
            - L_z               : size of the channel in the spanwise dirction
            - L_y               : half of the width of the channel
            - uvw_folder_tf     : folder of the velocity fields with the tensorflow format
            - uvw_folderii_tf   : file of the velocity fields with the tensorflow format
            - rey               : friction reynolds number
            - utau              : friction velocity
            - ssh_flag_train    : flag determining if ssh connection should be activated
            - uvw_folder_temp   : temporal forlder for storing the files
            - ssh_server        : server to read using the ssh connection
            - ssh_username      : username of the server
            - ssh_password      : password of the server user
            - shpx              : shape of the channel in the streamwise direction
            - shpy              : shape of the channel in the wall-normal direction
            - shpz              : shape of the channel in the spanwise direction
            - ngpu              : number of GPUs
            - learat            : learning ratio of the training
            - optmom            : momentum of the RMSprop algorithm
            - batch_size        : batch size of the training
            - field_ini         : initial field for the training
            - field_fin         : final fields for the training
            - field_mem         : number of fields load in memory
            - epoch_save        : number of epoch for saving the solution during the training
            - epoch_max         : number of epoch of the training before refreshing the data loaded in the memory
            - read_model        : flag to define if the model has to be created or read
            - model_folder      : folder for the models
            - model_write       : name of the model file to save
            - model_read        : name of the model file to load
            - nfil              : number of filters of the first layer of the unet
            - stride            : stride of the unet
            - activation        : activation function
            - kernel            : kernel size
            - pooling           : size of the poolings
            - delta_pred        : distance between the input field and the output field for the default database 
                                  the distance is 5 viscous units
            - hist_file         : file for saving the history of the training
            - test_size         : percentage of the data used for testing during training (value between 0 and 1)
            - adapt_batch       : flag for adapting the fields read to the batch size of the strategy
            - prep_data         : flag for selecting prepared data (True) or non prepared data (False)
            - flag_model        : flag for selecting if the model needs to be created (True: create it,
                                                                                       Flase: do not create it)
            - flag_central      : flag to choose the segmentation strategy (True: CentralStorageStrategy,
                                                                            False: MirroredDistributedStrategy)
            - data_type         : format of the training data
            - multi_worker      : flag to decide if the multiple worker training is activated
            - options           : options of the model
            - nworkers          : number of workers of the model
            - cluster_resolver  : SLURM cluster resolver of the multiworker
            - cluster_spec      : specification of the nodes sharing the multiworker and their atributio as chief,
                                  worker...
            - prefetch          : number of batches to load in memory
            - error_file        : file to store the prediction error
        * Classes
            - strategy          : strategy to use for the multiple gpu. Three different strategies can be used
                                  (MirroredStrategy, CentralStorageStrategy and MultiWorkerMirroredStrategy)
            - model             : Deep Learning model
    .....................................................................................................................
    """
    def __init__(self,data_in = {"uvw_folder":"../../P125_21pi_vu/",
                                 "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","padding":15,
                                 "dx":1,"dy":1,"dz":1,"data_folder":"data","umean_file":"Umean.txt",
                                 "unorm_file":"Unorm.txt","L_x":2*np.pi,"L_z":np.pi,"L_y":1,
                                 "uvw_folder_tf":"../../P125_21pi_vu_tf","uvw_folderii_tf":"P125_21pi_vu.$INDEX$",
                                 "rey":125,"utau":0.060523258443963,"ssh_flag_train":False,
                                 "uvw_folder_temp":"-","ssh_server":"-","ssh_username":"-","ssh_password":"-",
                                 "error_file":"error.txt","umax_file":"umax_file.txt",
                                 "urmspred_file":"Urms_pred.txt"}):
        """
        .................................................................................................................
        # __init__
        .................................................................................................................
        Parameters
        ----------
        data_in : dict, dictionary containing all the information required for the neural network
            DESCRIPTION. The default is {"uvw_folder":"../P125_21pi_vu/",
                                         "uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw","padding":15,
                                         "dx":1,"dy":1,"dz":1,"data_folder":"data","umeanfile":"Umean.txt"
                                         "unorm_file":"Unorm.txt","L_x":2*np.pi,"L_z":np.pi,"L_y":1,
                                         "uvw_folder_tf":"../../P125_21pi_vu_tf",
                                         "uvw_folderii_tf":"P125_21pi_vu.$INDEX$",
                                         "rey":125,"utau":0.060523258443963,"ssh_flag_train":False,
                                         "uvw_folder_temp":"-","ssh_server":"-","ssh_username":"-","ssh_password":"-",
                                         "error_file":"error.txt","umax_file":"umax_file.txt",
                                         "urmspred_file":"Urms_pred.txt"}.
            Data:
                - uvw_folder      : folder of the velocity fields
                - uvw_file        : file name of the velocity fileds without index
                - padding         : padding of the fields
                - dx              : downsampling in x
                - dy              : downsampling in y
                - dz              : downsampling in z
                - data_folder     : folder for storing the data of the model
                - umean_file      : file of the mean velocity
                - unorm_file      : file for the normalization of the velocity
                - L_x             : size of the channel in the streamwise direction
                - L_z             : size of the channel in the spanwise dirction
                - L_y             : half of the width of the channel
                - uvw_folder_tf   : folder of the velocity fields with the tensorflow format
                - uvw_folderii_tf : file of the velocity fields with the tensorflow format
                - rey             : friction reynolds number
                - utau            : friction velocity
                - ssh_flag_train  : flag determining if ssh connection should be activated. If the connection is 
                                    activated the files used in the training are read from the remote server
                - uvw_folder_temp : temporal forlder for storing the files
                - ssh_server      : server to read using the ssh connection
                - ssh_username    : username of the server
                - ssh_password    : password of the server user
                - error_file      : file to store the prediction errors
                - umax_file       : file containing maximum and minimum velocities
                - urmspred_file   : file contatining the urms predicted by the model

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Import packages
        # ---------------------------------------------------------------------------------------------------------------
        from py_bin.py_class.flow_field import flow_field
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the data
        # ---------------------------------------------------------------------------------------------------------------   
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
        self.uvw_folder_tf   = str(data_in["uvw_folder_tf"])     # Folder path for the tf format data
        self.uvw_folderii_tf = str(data_in["uvw_folderii_tf"])   # File for the tf format data
        self.rey             = float(data_in["rey"])             # Friction reynolds number
        self.utau            = float(data_in["utau"])            # Friction velocity
        self.ssh_flag_train  = bool(data_in["ssh_flag_train"])   # Flag for reading the files via ssh
        self.uvw_folder_temp = str(data_in["uvw_folder_temp"])   # Folder to temporally store the files
        self.ssh_server      = str(data_in["ssh_server"])        # Server to read the files
        self.ssh_username    = str(data_in["ssh_username"])      # username of the server
        self.ssh_password    = str(data_in["ssh_password"])      # password of the server
        self.error_file      = str(data_in["error_file"])        # file to store the prediction errors
        self.umax_file       = str(data_in["umax_file"])
        self.urmspred_file   = str(data_in["urmspred_file"])
        
        # ---------------------------------------------------------------------------------------------------------------
        # Calculate the shape of the tensors in the different directions this is useful in the case
        # of downsampling the fields
        # ---------------------------------------------------------------------------------------------------------------
        Data_flow={"folder":self.uvw_folder,"file":self.uvw_file,"down_x":self.dx,"down_y":self.dy,
                   "down_z":self.dz,"L_x":self.L_x,"L_y":self.L_y,"L_z":self.L_z,"rey":self.rey,"utau":self.utau}
        flowfield = flow_field(data_in=Data_flow)
        flowfield.shape_tensor()
        
        # ---------------------------------------------------------------------------------------------------------------
        # Store the shape of the grid in the class
        # ---------------------------------------------------------------------------------------------------------------
        self.shpy = flowfield.shpy   # Shape in y
        self.shpz = flowfield.shpz   # Shape in z
        self.shpx = flowfield.shpx   # Shape in x
        print("Shape of the tensors: ("+str(self.shpy)+","+str(self.shpz)+","+str(self.shpx)+")",flush=True)
           
    def define_model(self,data_in = {"ngpu":None,"learat":1e-3,"optmom":0.9,"batch_size":8,"field_ini":1000,
                                     "field_fin":7000,"field_delta":1,"field_mem":320,"epoch_save":200,"epoch_max":2e4,
                                     "read_model":False,"model_folder":"models","model_write":"trained_model.h5",
                                     "model_read":"trained_model.h5","nfil":16,"stride":1,"activation":"relu",
                                     "kernel":3,"pooling":2,"delta_pred":1,"hist_file":"hist.txt","test_size":0.2,
                                     "adapt_batch":True,"prep_data":True,"flag_model":True,"flag_train":True,
                                     "data_type":"float32","multi_worker":True,"prefetch":1,"check":True,
                                     "tfrecord_folder":'../../tfrecord/',"flag_tfrecord":False}): 
        """
        .................................................................................................................
        # define_model
        .................................................................................................................
        Function for defining the model, read all the information regarding the training and the unet
        and then defines the model and distributes the training in the different GPUs. In this function the model
        is created for multiworker strategies.
        Parameters
        ----------
        data_in : dict, dictionary containing the data required for the training
            DESCRIPTION. The default is {"ngpu":None,"learat":1e-3,"optmom":0.9,"batch_size":8,"field_ini":1000,
                                         "field_fin":7000,"field_delta":1,"field_mem":320,"epoch_save":200,
                                         "epoch_max":2e4,"read_model":False,"model_folder":"models",
                                         "model_write":"trained_model.h5","model_read":"trained_model.h5","nfil":16,
                                         "stride":1,"activation":"relu","kernel":3,"pooling":2,"delta_pred":1,
                                         "hist_file":"hist.txt","test_size":0.2,"adapt_batch":True,"prep_data":True,
                                         "flag_model":True,"flag_central":True,"data_type":"float32",
                                         "multi_worker":True,"prefetch":1,"check":True,
                                         "tfrecord_folder":'../../tfrecord/',"flag_tfrecord":False}.
            Data:
                - ngpu            : number of GPUs
                - learat          : learning ratio of the training
                - optmom          : momentum of the RMSprop algorithm
                - batch_size      : batch size of the training
                - field_ini       : initial field for the training
                - field_fin       : final fields for the training
                - field_delta     : distance between the fields used for the model
                - field_mem       : number of fields load in memory
                - epoch_save      : number of epoch for saving the solution during the training
                - epoch_max       : number of epoch of the training before refreshing the data loaded in the memory
                - read_model      : flag to define if the model has to be created or read
                - model_folder    : folder for the models
                - model_write     : name of the model file
                - nfil            : number of filters of the first layer of the unet
                - stride          : stride of the unet
                - activation      : activation function
                - kernel          : kernel size
                - pooling         : size of the poolings
                - delta_pred      : distance between the input field and the output field for the default database the
                                    distance is 5 viscous units
                - hist_file       : file for saving the history of the training
                - test_size       : percentage of the data used for testing during training (value between 0 and 1)
                - adapt_batch     : flag for adapting the fields read to the batch size of the strategy
                - prep_data       : flag for selecting prepared data (True) or non prepared data (False)
                - flag_model      : flag for selecting if the model needs to be created (True: create it,
                                                                                         Flase: do not create it)
                - flag_central    : flag to choose the segmentation strategy (True: CentralStorageStrategy,
                                                                              False: MirroredDistributedStrategy)
                - data_type       : format of the training data
                - multi_worker    : flag to decide if the multiple worker training is activated
                - prefetch        : number of batches loaded in memory
                - mean_norm       : Flag to normalize using mean and std (True: normalizes with mean and std, False:
                                                                          normalizes with min and max)
                - check           : Flag for checking the data
                - tfrecord_folder : folder storing the tfrecord
                - flag_tfrecord   : flag to read the tfrecord file (True: read , False: do not read)
                - save_fields     : flag to activate if the fields used for training and validation need to be stored
                                    (True to save, False not save)
                - traintest_index : file to store training and test files
                
        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Define packages
        # ---------------------------------------------------------------------------------------------------------------
        import tensorflow as tf
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the input dictionary
        # ---------------------------------------------------------------------------------------------------------------
        if data_in["ngpu"] is None:
            self.ngpu        = None                                  # Take all available GPUs
        else :
            self.ngpu        = int(data_in["ngpu"])                  # Choose a number of GPUs 
        self.learat          = float(data_in["learat"])              # Learning ratio
        self.optmom          = float(data_in["optmom"])              # Momentum of the RMSprop
        self.batch_size      = int(data_in["batch_size"])            # Batch size
        self.field_ini       = int(data_in["field_ini"])             # Initial field of the training
        self.field_fin       = int(data_in["field_fin"])             # Final field of the training
        self.field_delta     = int(data_in["field_delta"])           # Separation between fields of the model
        self.field_mem       = int(data_in["field_mem"])             # Number of fields stored in memory 
        self.epoch_save      = int(data_in["epoch_save"])            # Number of epoch before saving the model
        self.epoch_max       = int(data_in["epoch_max"])             # Number of epoch before refreshing the data
        self.read_model      = bool(data_in["read_model"])           # Flag to create or read the model
        self.model_folder    = str(data_in["model_folder"])          # Name of the folder to store the trained model
        self.model_write     = str(data_in["model_write"])           # Name of the file to store the trained model
        self.model_read      = str(data_in["model_read"])            # Name of the file load the trained model
        self.nfil            = int(data_in["nfil"])                  # number of filters of the first layer of the unet
        self.stride          = int(data_in["stride"])                # stride of the unet
        self.activation      = str(data_in["activation"])            # activation function
        self.kernel          = int(data_in["kernel"])                # size of the filters
        self.pooling         = int(data_in["pooling"])               # size of the poolings of the unet
        self.delta_pred      = int(data_in["delta_pred"])            # separation between the fields
        self.hist_file       = str(data_in["hist_file"])             # evolution of the training loss function
        self.test_size       = float(data_in["test_size"])           # percentage of data used for test (between 0 
                                                                     # and 1)
        self.test_size       = np.max([0,self.test_size])
        self.test_size       = np.min([1,self.test_size])
        self.adapt_batch     = bool(data_in["adapt_batch"])          # flag for adapting the number of fields to batch
        self.prep_data       = str(data_in["prep_data"])             # flag for using prepared data or not
        self.flag_model      = bool(data_in["flag_model"])           # flag for creating the model
        self.flag_central    = bool(data_in["flag_central"])         # flag for choosing the distributed strategy
        self.data_type       = str(data_in["data_type"])             # format of the training data
        self.multi_worker    = bool(data_in["multi_worker"])         # flag to decide if the multiple worker is active
        self.prefetch        = int(data_in["prefetch"])              # number of batches loaded in memory
        self.mean_norm       = bool(data_in["mean_norm"])            # flag for choosing normalization
        self.check           = bool(data_in["check"])                # flag for checking the data
        self.tfrecord_folder = str(data_in["tfrecord_folder"])
        self.flag_tfrecord   = bool(data_in["flag_tfrecord"])
        self.save_fields     = bool(data_in["save_fields"])
        self.traintest_index = str(data_in["traintest_index"])
        
        print("Start the model definition.",flush=True)
        # ---------------------------------------------------------------------------------------------------------------
        # Define the multi_worker strategy
        # ---------------------------------------------------------------------------------------------------------------
        self.options = tf.data.Options()
        if self.multi_worker:
            print("MultiWorker strategy selected: ...")
            
            # -----------------------------------------------------------------------------------------------------------
            # Identify the workers (nodes of the cluster)
            # If the code is not evaluated in a cluster then stop the calculation
            # If it is evaluated in a cluster print the nodes used by the cluster and get the number of nodes
            # -----------------------------------------------------------------------------------------------------------       
            if 'SLURM_JOB_NODELIST' in os.environ:
                nodelist      = os.environ['SLURM_JOB_NODELIST']
                number_nodes  = int(os.environ['SLURM_JOB_NUM_NODES'])
                print(nodelist,flush=True)
            
                # -----------------------------------------------------------------------------------------------------------
                # Determine the number of workers and scale the batch size and the fields in memory
                # -----------------------------------------------------------------------------------------------------------
                print("Original batch size (batch size per node): "+str(self.batch_size),flush=True)
                print("Original number of files in memory (fields per node): "+str(self.field_mem),flush=True)
                self.nworkers    = number_nodes
                self.batch_size *= self.nworkers
                self.field_mem  *= self.nworkers
                print("Number of workers: "+str(self.nworkers),flush=True)
                print("Final batch size (batch size): "+str(self.batch_size),flush=True)
                print("Final number of files in memory: "+str(self.field_mem),flush=True)
                   
                # -----------------------------------------------------------------------------------------------------------
                # Create the multiworker environment taking the SLURM variables. Define a port of connection
                # -----------------------------------------------------------------------------------------------------------
                self.cluster_resolver =  tf.distribute.cluster_resolver.SlurmClusterResolver()  
                self.cluster_spec     = self.cluster_resolver.cluster_spec()
                print("Cluster has been resolved.",flush=True)
                print("Cluster specification: ",flush=True)
                print(self.cluster_spec,flush=True)
             
                # -----------------------------------------------------------------------------------------------------------
                # Define the protocol use for the communication.
                # RING communication protocol is selected
                # ----------------------------------------------------------------------------------------------------------- 
                implementation        = tf.distribute.experimental.CommunicationImplementation.NCCL
                communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation)
                print("Comunication has been defined.",flush=True)
                
                # -----------------------------------------------------------------------------------------------------------         
                # Define the MultiWorkerMirroredStrategy
                # -----------------------------------------------------------------------------------------------------------
                self.strategy = tf.distribute.MultiWorkerMirroredStrategy(cluster_resolver=self.cluster_resolver,
                                                                          communication_options=communication_options)
                print("MultiWorker strategy created",flush=True)
                self.options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
                
                # ---------------------------------------------------------------------------------------------------------
                # Change the temporal file folder
                # ---------------------------------------------------------------------------------------------------------
                self.uvw_folder_temp = str(self.uvw_folder_temp+'_'+self.strategy.cluster_resolver.task_type+
                                           str(self.strategy.cluster_resolver.task_id))
            else:
                print("SLURM_JOB_NODELIST environment variable not found.",flush=True)
                print("Cannot train with multi-worker. Changing to single worker model...",flush=True)
                self.multi_worker = False
        
        if not self.multi_worker:
            # -----------------------------------------------------------------------------------------------------------
            # Define the list of devices used for the training. If the number of GPUs is None, choose all the 
            # available GPUs. In the case of int values of the number of GPUs enter the condition and limit the number 
            # of GPUs that are taken.
            # -----------------------------------------------------------------------------------------------------------
            if self.ngpu is not None:
                if self.ngpu > 0:
                    dev_list                           = str(np.arange(self.ngpu).tolist())
                    cudadevice                         = dev_list.replace('[','').replace(']','')        
                    os.environ["CUDA_VISIBLE_DEVICES"] = cudadevice  
            
            # -----------------------------------------------------------------------------------------------------------
            # In the case of limiting to a number the GPUs, from this point in advance, only these GPUs are taken
            # -----------------------------------------------------------------------------------------------------------
            physical_devices = tf.config.list_physical_devices('GPU') # Physical devices
            available_gpus   = len(physical_devices)                  # Number of physical devices
            self.ngpu        = available_gpus
            print("-"*100,flush=True)
            print('Using TensorFlow version: ',tf.__version__, ', available GPU:',available_gpus,flush=True) 
            print("-"*100,flush=True)
            if physical_devices:
                try:
                    for gpu in physical_devices:
                        
                        # -----------------------------------------------------------------------------------------------
                        # Allocate only as much GPU memory as needed for the runtime allocations
                        # -----------------------------------------------------------------------------------------------
                        tf.config.experimental.set_memory_growth(gpu,True)
                        print("Memory growth for GPU: "+str(gpu),flush=True)
                except RuntimeError as ee:
                    print(ee,flush=True)
            print("-"*100,flush=True)
                    
            # -----------------------------------------------------------------------------------------------------------
            # Update number of GPUs and select the devices for the strategy
            # -----------------------------------------------------------------------------------------------------------
            list_compute   = ['CPU:0']
            list_parameter = 'CPU:0'
            for ii in np.arange(self.ngpu,dtype='int'):
                list_compute.append('GPU:'+str(ii))
                    
            # -----------------------------------------------------------------------------------------------------------
            # Define the strategy for distributing the training
            # we use central storage strategy because the size of the fields requires loading them on the CPU RAM
            # and then take inside the GPUs the batch size. With other strategies such as mirrored we are having 
            # overflows in the memory. Central storage is experimental, there may be compatibility problems
            # -----------------------------------------------------------------------------------------------------------
            if self.flag_central:
                self.strategy = tf.distribute.experimental.CentralStorageStrategy(compute_devices=list_compute,
                                                                                  parameter_device=list_parameter)
            else:
                self.strategy = tf.distribute.MirroredStrategy()
            self.options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
        print("-"*100,flush=True)
        print('Number of devices in the strategy: {}'.format(self.strategy.num_replicas_in_sync),flush=True)
        print("List of CPUs in use:",flush=True)
        for cpu in tf.config.list_logical_devices('CPU'):
            print(cpu.name,flush=True)
        print("List of GPUs in use:",flush=True)
        for gpu in tf.config.list_logical_devices('GPU'):
            print(gpu.name,flush=True)        
        print("-"*100,flush=True)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Define precision tensorflow
        # ---------------------------------------------------------------------------------------------------------------
        tf.keras.backend.set_floatx(self.data_type)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create the model
        # ---------------------------------------------------------------------------------------------------------------
        if self.flag_model:
            self.create_model()
        else:
            print("Model is not created. Activate the flag flag_model.",flush=True)
            
    def create_model(self):
        """
        .................................................................................................................
        # create_model
        .................................................................................................................
        Function to create the model within the segmentation

        Returns
        -------
        None.

        """
        
        # --------------------------------------------------------------------------------------------------------------
        # Import packages
        # --------------------------------------------------------------------------------------------------------------
        import tensorflow as tf
        from tensorflow.keras import Model
        from tensorflow.keras.optimizers import RMSprop
        import psutil
        from tensorflow.keras import mixed_precision
        
        # --------------------------------------------------------------------------------------------------------------
        # Create the model
        # --------------------------------------------------------------------------------------------------------------
        if self.ngpu > 0:
            mixed_precision.set_global_policy('mixed_float16')
            
        if self.ngpu == 0:
            model0     = tf.keras.models.load_model(self.model_folder+'/'+self.model_read)
            weight0    = model0.get_weights()
            self.model_base()	
            optimizer  = RMSprop(learning_rate=self.learat,momentum=self.optmom) 																 
            self.model = Model(self.inputs, self.outputs)
            self.model.set_weights(weight0) 
            self.model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=optimizer)
        else:
            with self.strategy.scope():
                if self.read_model:
                    try:
                        self.model = tf.keras.models.load_model(self.model_folder+'/'+self.model_read)
                    except:
                        self.model = tf.keras.models.load_model(self.model_folder+'/'+self.model_read,compile=False) 
                        optimizer  = RMSprop(learning_rate=self.learat,momentum=self.optmom) 
                        self.model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=optimizer)
                else:
                    self.model_base()
                    optimizer  = RMSprop(learning_rate=self.learat,momentum=self.optmom) 
                    self.model = Model(self.inputs, self.outputs)
                    self.model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=optimizer)
        self.model.summary()    
        memory_data  = psutil.virtual_memory()
        print('Total RAM (GB): '+str(memory_data[0]/1e9),flush=True)
        print("-"*100,flush=True)
        
    def model_base(self):
        """
        .................................................................................................................
        # model_base
        .................................................................................................................
        Function to define the model used for the problem

        Returns
        -------
        None.

        """    
        # --------------------------------------------------------------------------------------------------------------
        # Define the required packages
        # --------------------------------------------------------------------------------------------------------------    
        from tensorflow.keras.layers import Input
        
        # --------------------------------------------------------------------------------------------------------------
        # The dimensions of the input field are selected from the size of the fields, adding the padding
        # --------------------------------------------------------------------------------------------------------------
        dim0 = self.shpy
        dim1 = self.shpz+2*self.padding
        dim2 = self.shpx+2*self.padding
        dim3 = 3
        shp = (dim0,dim1,dim2,dim3)
        
        # --------------------------------------------------------------------------------------------------------------
        # Define the input and the output of the model
        # --------------------------------------------------------------------------------------------------------------
        self.inputs  = Input(shape=shp)
        x_in         = self.inputs
        self.outputs = self.architecture_Unet(data_in={"x_in":x_in,"flag_print":True})["x_out"]
              
    def train_model(self):
        """
        .................................................................................................................
        # train_model
        .................................................................................................................
        Function to generate the training of the model. This function calls the functions for reading the data
        normalizes it and then trains the model
        
        Returns
        -------
        None.

        """
        # --------------------------------------------------------------------------------------------------------------
        # Import packages
        # --------------------------------------------------------------------------------------------------------------
        from py_bin.py_functions.trainvali_data import read_inout_notprepared, data_traintest_tf, read_data_tf
        from py_bin.py_functions.multiworker_checkpoint import write_filepath, remove_workers
        import tensorflow as tf
        from py_bin.py_remote.read_remote import recursivedelete
        from py_bin.py_functions.read_tfrecord import read_tfrecord
        
        # --------------------------------------------------------------------------------------------------------------
        # Modifgy the number of fields loaded in memory and the percentage used for validation to fit the batch and 
        # number gpus requirement
        # --------------------------------------------------------------------------------------------------------------
        if not self.flag_tfrecord:
            if self.adapt_batch:
                train_size     = (1-self.test_size)*self.field_mem
                train_factor   = train_size/(self.batch_size)
                train_size_m   = self.batch_size*np.round(train_factor)
                vali_size      = self.test_size*self.field_mem
                vali_factor    = vali_size/(self.batch_size)
                vali_size_m    = self.batch_size*np.round(vali_factor)
                print("Train size: "+str(train_size_m),flush=True)
                print("Validation size: "+str(vali_size_m),flush=True)
                if vali_size_m == 0:
                    print("Increase the value of fields loaded in memory to fulfill the distribution strategy."+
                          "Exit calculation!",flush=True)
                    sys.exit()
                fmem0 = self.field_mem
                tsiz0 = self.test_size
                self.field_mem = int(train_size_m+vali_size_m)
                self.test_size = vali_size_m/(self.field_mem)
                if fmem0 != self.field_mem:
                    print("The number of fields have been modified to "+str(self.field_mem)+
                          " to fullfil the training requirements.",flush=True)
                if tsiz0 != self.test_size:
                    print("The percentage of fields used for the validation have been modified to "+str(self.test_size)+
                          " to fullfil the training requirements.",flush=True)
            if self.ssh_flag_train:
                recursivedelete(self.uvw_folder_temp) 
            
        # --------------------------------------------------------------------------------------------------------------
        # The training randomizes the fields in order to take smaller dataset to load them in the memory. 
        # Then this dataset is divided in smaller batchs using the field_mem data. 
        # ind_vec : randomized index for the training
        # ii_ini  : initial index of the subset of data
        # ii_fin  : final index of the subset of data
        # --------------------------------------------------------------------------------------------------------------
        if self.save_fields and self.read_model:
            ind_vec = self._read_fieldsvec()["ind_vec"]
        else:
            ind_vec = np.array(range(self.field_ini,self.field_fin,self.field_delta),dtype='int')
            np.random.shuffle(ind_vec)
            self._save_fieldsvec(data_in={"ind_vec":ind_vec})
        ii_ini   = 0
        ii_fin   = ii_ini+self.field_mem
        epochcum = 0
        ind_tfr  = 0    
        while ii_ini < self.field_fin-self.field_ini-1:
            if ii_fin < self.field_fin-self.field_ini:
                interval = ind_vec[ii_ini:ii_fin]
            else:
                interval = np.concatenate((ind_vec[ii_ini:],ind_vec[:ii_fin-len(ind_vec)]))
                
            # ----------------------------------------------------------------------------------------------------------
            # The data is read
            # train_data : data for the training, segmented using the batch size and with the options loaded
            # vali_data  : data for the validation, segemented usign the  batch size and with the options loaded
            # ----------------------------------------------------------------------------------------------------------
            if self.flag_tfrecord:
                data_trainvali = read_tfrecord(data_in={"tfrecord_folder":self.tfrecord_folder,"interval":interval,
                                                        "test_size":self.test_size,"padding":self.padding,
                                                        "shpx":self.shpx,"shpy":self.shpy,"shpz":self.shpz,
                                                        "data_type":self.data_type})
                ind_tfr = 0
            else:
                if self.prep_data:
                    data_tensor    = {"folder_tf":self.uvw_folder_tf,"folderii_tf":self.uvw_folderii_tf,
                                      "interval":interval,"test_size":self.test_size,"printflag":True,
                                      "ssh_flag_train":self.ssh_flag_train,"uvw_folder_temp":self.uvw_folder_temp,
                                      "ssh_server":self.ssh_server,"ssh_username":self.ssh_username,
                                      "ssh_password":self.ssh_password,"check":self.check}
                    data_trainvali = read_data_tf(data_in=data_tensor)
                else:
                    data_trainval  = {"folder":self.uvw_folder,"file":self.uvw_file,"interval":interval,
                                      "delta_pred":self.delta_pred,"padding":self.padding,
                                      "shpx":self.shpx,"shpy":self.shpy,"shpz":self.shpz,
                                      "dx":self.dx,"dy":self.dy,"dz":self.dz,"data_folder":self.data_folder,
                                      "umean_file":self.umean_file,"unorm_file":self.unorm_file,
                                      "mean_norm":self.mean_norm,"data_type":data_type}
                    data_base_mem  = read_inout_notprepared(data_in=data_trainval)
                    data_tensor    = {"data_X":data_base_mem["data_X"],"data_Y":data_base_mem["data_Y"],
                                      "interval":None,"test_size":self.test_size,"shpx":self.shpx,"shpy":self.shpy,
                                      "shpz":self.shpz,"padding":self.padding,"data_type":self.data_type}
                    data_trainvali = data_traintest_tf(data_in=data_tensor)
            train_data = data_trainvali["data_train"]
            vali_data  = data_trainvali["data_vali"]
            del data_trainvali
            
            if self.prefetch < 0:
                self.prefetch = tf.data.AUTOTUNE
            train_data = train_data.batch(self.batch_size)
            vali_data  = vali_data.batch(self.batch_size)
            train_data = train_data.prefetch(self.prefetch)
            vali_data  = vali_data.prefetch(self.prefetch)
            train_data = train_data.with_options(self.options)
            vali_data  = vali_data.with_options(self.options)
            
            # ----------------------------------------------------------------------------------------------------------
            # Start training. Train during epoch_save epochs and save the results (model and training).
            # After epoch_max, change the data stored in the memory.
            #      - epoch : epoch of the training
            #      - hmat  : array to store the epoch, the training loss and the validation loss
            # ----------------------------------------------------------------------------------------------------------
            epoch = 0
            while epoch < self.epoch_max: 
                print('Training... '+str(ii_ini/(self.field_fin-self.field_ini)*100)+'%',flush=True)
                data_training = self.model.fit(train_data,batch_size=self.batch_size,verbose=2,
                                               epochs=self.epoch_save,validation_data=vali_data) 
                print('Number of epochs...',flush=True)
                
                # ------------------------------------------------------------------------------------------------------
                # Depending of the multiworker flag, saving the model uses different functions. For a single node
                # the default function can be used. In the case of the multiworker, the tutorial presented by
                # tensorflow in: https://www.tensorflow.org/tutorials/distribute/multi_worker_with_ctl
                # ------------------------------------------------------------------------------------------------------
                print("Save the model...",flush=True)
                if self.multi_worker:
                    task_type, task_id = (self.strategy.cluster_resolver.task_type,
                                          self.strategy.cluster_resolver.task_id)
                    checkpoint_dir     = self.model_folder+'/'+self.model_write
                    write_model_path   = write_filepath(checkpoint_dir,task_type,task_id,self.cluster_spec)
                    self.model.save(write_model_path)
                    remove_workers(write_model_path,task_type,task_id,self.cluster_spec)
                    data_save_epoch    = {"data_training":data_training,"task_type":task_type,"task_id":task_id,
                                          "epochcum":epochcum}
                else:
                    self.model.save(self.model_folder+'/'+self.model_write)
                    data_save_epoch    = {"data_training":data_training,"epochcum":epochcum}
                self._save_training(data_in=data_save_epoch)
                print("Model saved",flush=True)
                print("Epochs saved: "+str(epochcum),flush=True)
                epoch    += self.epoch_save
                epochcum += self.epoch_save
                del data_training
            ii_ini   = ii_ini+int(self.field_mem/2)
            ii_fin   = ii_fin+int(self.field_mem/2)
            ind_tfr += 1
            del train_data, vali_data
            if self.ssh_flag_train:
                print("Delete folder: "+self.uvw_folder_temp,flush=True)
                recursivedelete(self.uvw_folder_temp)
            
    def prepare_data(self):
        """
        ................................................................................................................
        # prepare_data
        ................................................................................................................
        Function to prepare the data for the training. This function creates a new folder with the data of the uvw 
        files with the tensorflow format.

        Returns
        -------
        None.

        """
        # --------------------------------------------------------------------------------------------------------------
        # Import packages
        # --------------------------------------------------------------------------------------------------------------
        from py_bin.py_functions.trainvali_data import prepare_data_tf
        
        # --------------------------------------------------------------------------------------------------------------
        # Prepare the data
        # --------------------------------------------------------------------------------------------------------------
        interval      = np.array(range(self.field_ini,self.field_fin,self.field_delta),dtype='int')
        data_trainval = {"folder":self.uvw_folder,"file":self.uvw_file,"interval":interval,
                         "delta_pred":self.delta_pred,"padding":self.padding,"shpx":self.shpx,"shpy":self.shpy,
                         "shpz":self.shpz,"dx":self.dx,"dy":self.dy,"dz":self.dz,"data_folder":self.data_folder,
                         "umean_file":self.umean_file,"unorm_file":self.unorm_file,"test_size":self.test_size,
                         "folder_tf":self.uvw_folder_tf,"folderii_tf":self.uvw_folderii_tf,
                         "data_type":self.data_type,"mean_norm":self.mean_norm}
        prepare_data_tf(data_in=data_trainval)

    
    def _save_training(self,data_in={"data_training":[],"task_type":"worker","task_id":0,"epochcum":0}):
        """      
        ................................................................................................................
        # _save_training
        ................................................................................................................
        Funtion to save the training loss function with the epochs

        Parameters
        ----------
        data_in : dict, dictionary containing the data required for the training
            DESCRIPTION. The default is {"check_chief":False,"ii_ini":0,"epoch":0}.
            Data:
                - data_training : class containing the training information
                - task_type     : type of task of the node
                - task_id       : identifier of the task
                - epochcum      : index of the cummulated epoch

        Returns
        -------
        None.

        """
        # --------------------------------------------------------------------------------------------------------------
        # Import packages
        # --------------------------------------------------------------------------------------------------------------
        from py_bin.py_functions.multiworker_checkpoint import is_chief
        
        # --------------------------------------------------------------------------------------------------------------
        # Read the data
        # --------------------------------------------------------------------------------------------------------------
        data_training = data_in["data_training"]
        epochcum      = int(data_in["epochcum"]) 
        if self.multi_worker:
            task_type = str(data_in["task_type"])
            task_id   = int(data_in["task_id"])
            task_flag = is_chief(task_type, task_id, self.cluster_spec)
        else:
            task_flag = False
        
        # --------------------------------------------------------------------------------------------------------------
        # Save if a single worker is used or is the chief of the multiworker
        # --------------------------------------------------------------------------------------------------------------
        if (self.multi_worker and task_flag) or not self.multi_worker:
            hmat = np.zeros((self.epoch_save,3))
            if epochcum == 0 and not self.read_model:
                print("Create the file for the training epochs: "+str(epochcum), flush=True)
                hmat[:,0] = np.arange(self.epoch_save)
                hmat[:,1] = data_training.history['loss']
                hmat[:,2] = data_training.history['val_loss']
                with open(self.data_folder+'/'+self.hist_file,'w') as filehist:
                    for line in hmat:
                        filehist.write(str(line[0])+','+str(line[1])+','+str(line[2])+'\n')
            else:
                print("Save the training for the epoch: "+str(epochcum), flush=True)
                hmat[:,0] = np.arange(self.epoch_save)+epochcum
                hmat[:,1] = data_training.history['loss']
                hmat[:,2] = data_training.history['val_loss']
                with open(self.data_folder+'/'+self.hist_file,'a') as filehist:
                    for line in hmat:
                        filehist.write(str(line[0])+','+str(line[1])+','+str(line[2])+'\n') 
             
    def check_data(self):
        """
        ................................................................................................................
        # check_data
        ................................................................................................................
        Function to check the data for the training do not present errors. This function creates a new folder with the
        data of the uvw files with the tensorflow format.

        Returns
        -------
        None.

        """
        # --------------------------------------------------------------------------------------------------------------
        # Import packages
        # --------------------------------------------------------------------------------------------------------------
        from py_bin.py_functions.trainvali_data import check_data_tf
        from py_bin.py_remote.read_remote import recursivedelete
        import glob
        
        # --------------------------------------------------------------------------------------------------------------
        # Prepare the data
        # --------------------------------------------------------------------------------------------------------------
        field_ini    = self.field_ini
        field_fin    = self.field_fin
        files_folder = glob.glob(self.uvw_folder_tf+'/*')
        interval     = np.array(range(self.field_ini,self.field_fin,self.field_delta),dtype='int')
        for index in interval:
            file        = self.uvw_folder_tf+'/'+self.uvw_folderii_tf
            file        = file.replace("$INDEX$",str(index))
            quant_files = len(glob.glob(file+'/*'))
            if quant_files != 3:
                print("Remove the folder: "+file,flush=True)
                recursivedelete(file)
                self.field_ini = index
                self.field_fin = index+1
                self.prepare_data()
        self.field_ini = field_ini
        self.field_fin = field_fin
        while True:
            interval    = np.array(range(self.field_ini,self.field_fin,self.field_delta),dtype='int')
            data_tensor = {"folder_tf":self.uvw_folder_tf,"folderii_tf":self.uvw_folderii_tf,
                           "interval":interval,"test_size":self.test_size,"printflag":True,
                           "ssh_flag_train":self.ssh_flag_train,"uvw_folder_temp":self.uvw_folder_temp,
                           "ssh_server":self.ssh_server,"ssh_username":self.ssh_username,
                           "ssh_password":self.ssh_password,"check":self.check}
            data_out    = check_data_tf(data_in=data_tensor)
            if data_out["flag_return"]:
                break
            else:
                self.field_ini = data_out["index_file"]
                if self.field_ini < self.field_fin:
                    print("Recalculate corrupted file...",flush=True)
                    field_fin      = self.field_fin
                    self.field_fin = self.field_ini+1
                    delete_folder  = self.uvw_folder_tf+'/'+self.uvw_folderii_tf
                    recursivedelete(delete_folder.replace("$INDEX$",str(self.field_ini)))
                    self.prepare_data()
                    self.field_fin = field_fin
                    
    def pred_field(self,data_in={"index_ii":1000}):
        """
        ................................................................................................................
        # pred_field
        ................................................................................................................
        Function to use the model to predict the evolution of the flow
        Parameters
        ----------
        data_in : dict, dictionary containing the data required for predicting the field
            DESCRIPTION. The default is {"index_ii":1000}.
            Data:
                - index_ii : Index of the field

        Returns
        -------
        dict
            Predicted velocity fluctuation in the streamwise, wall-normal and spanwise directions.
            Data:
                - uu : velocity fluctuation in the streamwise direction
                - vv : velocity fluctuation in the wall-normal direction
                - ww : velocity fluctuation in the spanwise direction
        """
        # --------------------------------------------------------------------------------------------------------------
        # Import packages
        # --------------------------------------------------------------------------------------------------------------
        from py_bin.py_functions.read_norm_velocity import read_norm_velocity
        from py_bin.py_functions.norm_velocity import dim_velocity
        
        # --------------------------------------------------------------------------------------------------------------
        # Read the data
        # --------------------------------------------------------------------------------------------------------------
        index_ii = int(data_in["index_ii"])
        
        # --------------------------------------------------------------------------------------------------------------
        # Read the input field
        # --------------------------------------------------------------------------------------------------------------
        field_in         = np.zeros((1,self.shpy,self.shpz+2*self.padding,self.shpx+2*self.padding,3),
                                    dtype=self.data_type)
        data_norm_in     = {"folder":self.uvw_folder,"file":self.uvw_file,"padding":self.padding,"shpx":self.shpx,
                            "shpy":self.shpy,"shpz":self.shpz,"dx":self.dx,"dy":self.dy,"dz":self.dz,
                            "data_folder":self.data_folder,"umean_file":self.umean_file,
                            "unorm_file":self.unorm_file,"index":index_ii,
                            "data_type":self.data_type,"mean_norm":self.mean_norm}
        data_veloc_norm  = read_norm_velocity(data_in=data_norm_in)
        norm_velocity_in = data_veloc_norm["norm_velocity"]
        print("Time for reading the field: "+str(data_veloc_norm["time_read"]),flush=True)
        print("Time for normalizing the field: "+str(data_veloc_norm["time_norm"]),flush=True)
        del data_norm_in,data_veloc_norm
        field_in[0,:,:,:,0] = norm_velocity_in['unorm']
        field_in[0,:,:,:,1] = norm_velocity_in['vnorm']
        field_in[0,:,:,:,2] = norm_velocity_in['wnorm']
        del norm_velocity_in
        
        # --------------------------------------------------------------------------------------------------------------
        # Calculate the predicted field
        # --------------------------------------------------------------------------------------------------------------
        field_pred = self.model.predict(field_in)
        del field_in
        data_out   = dim_velocity(data_in={"unorm":field_pred[0,:,:,:,0],"vnorm":field_pred[0,:,:,:,1],
                                           "wnorm":field_pred[0,:,:,:,2],"folder_data":self.data_folder,
                                           "unorm_file":self.unorm_file,"data_type":self.data_type,
                                           "mean_norm":self.mean_norm})
        return data_out

    def field_error(self,data_in={"index_ii":1000}):
        """
        ................................................................................................................
        # field_error
        ................................................................................................................
        Function to use the model to calculate the error in the prediction. 
        Parameters
        ----------
        data_in : dict, dictionary containing the data required for predicting the field
            DESCRIPTION. The default is {"index_ii":1000}.
            Data:
                - index_ii : Index of the field

        Returns
        -------
        dict
            Predicted velocity fluctuation in the streamwise, wall-normal and spanwise directions.
            Data:
                - err_u : error in the streamwise velocity
                - err_v : error in the wall-normal velocity
                - err_w : error in the spanwise velocity
        """  
        # --------------------------------------------------------------------------------------------------------------
        # Import packages
        # --------------------------------------------------------------------------------------------------------------
        from py_bin.py_functions.read_velocity import read_velocity
        from py_bin.py_functions.normalization import read_norm
        
        # --------------------------------------------------------------------------------------------------------------
        # Read the data
        # --------------------------------------------------------------------------------------------------------------
        index_ii = int(data_in["index_ii"])
        
        # ----------------------------------------------------------------------------------------------------------
        # Predict the field
        # ----------------------------------------------------------------------------------------------------------
        dim_pred                = self.pred_field(data_in={"index_ii":index_ii})
        field_out_pred          = np.zeros((self.shpy,self.shpz,self.shpx,3),dtype=self.data_type)
        field_out_pred[:,:,:,0] = dim_pred["uu"]
        field_out_pred[:,:,:,1] = dim_pred["vv"]
        field_out_pred[:,:,:,2] = dim_pred["ww"]
        
        # --------------------------------------------------------------------------------------------------------------
        # Read the output file
        # --------------------------------------------------------------------------------------------------------------
        field_out          = np.zeros((self.shpy,self.shpz,self.shpx,3),dtype=self.data_type)
        data_velocity_out  = {"folder":self.uvw_folder,"file":self.uvw_file,"index":index_ii+self.delta_pred,
                              "dx":self.dx,"dy":self.dy,"dz":self.dz,"shpx":self.shpx,"shpy":self.shpy,
                              "shpz":self.shpz,"padding":0,"data_folder":self.data_folder,
                              "umean_file":self.umean_file}
        velocity_out       = read_velocity(data_velocity_out)
        field_out[:,:,:,0] = velocity_out['uu']
        field_out[:,:,:,1] = velocity_out['vv']
        field_out[:,:,:,2] = velocity_out['ww']
        del data_velocity_out
        
        # --------------------------------------------------------------------------------------------------------------
        # Calculate the error
        # --------------------------------------------------------------------------------------------------------------
        data_norm = read_norm(data_in={"folder":self.data_folder,"file":self.umax_file})
        uref      = np.max([abs(data_norm["uumax"]),abs(data_norm["uumin"])])
        vref      = np.max([abs(data_norm["vvmax"]),abs(data_norm["vvmin"])])
        wref      = np.max([abs(data_norm["wwmax"]),abs(data_norm["wwmin"])])
        errorfun  = abs(field_out-field_out_pred)
        
        # --------------------------------------------------------------------------------------------------------------
        # Generate the output
        # --------------------------------------------------------------------------------------------------------------
        data_out          = {}
        data_out["err_u"] = errorfun[:,:,:,0]/uref
        data_out["err_v"] = errorfun[:,:,:,1]/vref
        data_out["err_w"] = errorfun[:,:,:,2]/wref
        del errorfun
        data_out["pre_u"] = dim_pred["uu"]
        data_out["pre_v"] = dim_pred["vv"]
        data_out["pre_w"] = dim_pred["ww"]
        del dim_pred
        data_out["sim_u"] = velocity_out["uu"]
        data_out["sim_v"] = velocity_out["vv"]
        data_out["sim_w"] = velocity_out["ww"]
        del velocity_out
        return data_out
                         
    def pred_error(self):
        """
        ................................................................................................................
        # pred_error
        ................................................................................................................
        Function to calculate the error between the predicted field and the expected field.
        The error is calculated weighting the localerror with the volume of each cell. 
        The relative error is calculated with respect to the maximum velocity of each wall distance.
																 
        Returns
        -------
        None.

        """
        # --------------------------------------------------------------------------------------------------------------
        # Import packages
        # --------------------------------------------------------------------------------------------------------------
        from py_bin.py_class.flow_field import flow_field
        
        # --------------------------------------------------------------------------------------------------------------																												
        # Get the volume of the mesh
        # --------------------------------------------------------------------------------------------------------------
        Data_flow = {"folder":self.uvw_folder,"file":self.uvw_file,"down_x":self.dx,"down_y":self.dy,
                     "down_z":self.dz,"L_x":self.L_x,"L_y":self.L_y,"L_z":self.L_z,"rey":self.rey,"utau":self.utau}
        flowfield = flow_field(data_in=Data_flow)
        flowfield.shape_tensor()
        flowfield.flow_grid()
        vol = flowfield.vol_plus
        del flowfield, Data_flow
        
        # --------------------------------------------------------------------------------------------------------------
        # Create the array containing the indices of the fields to check
        # --------------------------------------------------------------------------------------------------------------
        interval = np.array(range(self.field_ini,self.field_fin,self.field_delta),dtype='int')
        error_u  = 0
        error_v  = 0
        error_w  = 0
        n_error  = 0
        
        for index_ii in interval:
            data_error = self.field_error(data_in={"index_ii":index_ii,"vol":vol})
            erru       = np.sum(np.multiply(data_error["err_u"],vol))
            errv       = np.sum(np.multiply(data_error["err_v"],vol))
            errw       = np.sum(np.multiply(data_error["err_w"],vol))
            errvol     = np.sum(vol)
            error_u   += erru
            error_v   += errv
            error_w   += errw
            n_error   += errvol
            print("Error u:"+str(erru/errvol*100)+"%",flush=True)
            print("Error v:"+str(errv/errvol*100)+"%",flush=True)
            print("Error w:"+str(errw/errvol*100)+"%",flush=True)
            del erru,errv,errw,errvol,data_error
            
        error_u /= n_error
        error_v /= n_error
        error_w /= n_error
        
        # ----------------------------------------------------------------------------------------------------------
        # Save in file
        # ---------------------------------------------------------------------------------------------------------- 
        file_error = self.data_folder+'/'+self.error_file                     
        file_save  = open(file_error, "w+")           
        content    = str(error_u.tolist())+'\n'
        file_save.write(content)           
        content    = str(error_v.tolist())+'\n'
        file_save.write(content)           
        content    = str(error_w.tolist())+'\n'
        file_save.write(content) 
        file_save.close()

                         
    def pred_error_y(self):
        """
        ................................................................................................................
        # pred_error_y
        ................................................................................................................
        Function to calculate the error between the predicted field and the expected field.
        The error is calculated weighting the localerror with the volume of each cell. 
        The relative error is calculated with respect to the maximum velocity of each wall distance.
        The error is distributed along y+
																 
        Returns
        -------
        None.

        """
        # --------------------------------------------------------------------------------------------------------------
        # Import packages
        # --------------------------------------------------------------------------------------------------------------
        from py_bin.py_class.flow_field import flow_field
        
        # --------------------------------------------------------------------------------------------------------------																												
        # Get the volume of the mesh
        # --------------------------------------------------------------------------------------------------------------
        Data_flow = {"folder":self.uvw_folder,"file":self.uvw_file,"down_x":self.dx,"down_y":self.dy,
                     "down_z":self.dz,"L_x":self.L_x,"L_y":self.L_y,"L_z":self.L_z,"rey":self.rey,"utau":self.utau}
        flowfield = flow_field(data_in=Data_flow)
        flowfield.shape_tensor()
        flowfield.flow_grid()
        vol = flowfield.vol_plus
        del flowfield, Data_flow
        
        # --------------------------------------------------------------------------------------------------------------
        # Create the array containing the indices of the fields to check
        # --------------------------------------------------------------------------------------------------------------
        interval = np.array(range(self.field_ini,self.field_fin,self.field_delta),dtype='int')
        error_u  = np.zeros((self.shpy,))
        error_v  = np.zeros((self.shpy,))
        error_w  = np.zeros((self.shpy,))
        n_error  = np.zeros((self.shpy,))
        
        for index_ii in interval:
            data_error = self.field_error(data_in={"index_ii":index_ii,"vol":vol})
            erru       = np.sum(np.multiply(data_error["err_u"],vol),axis=(1,2))
            errv       = np.sum(np.multiply(data_error["err_v"],vol),axis=(1,2))
            errw       = np.sum(np.multiply(data_error["err_w"],vol),axis=(1,2))
            errvol     = np.sum(vol,axis=(1,2))
            error_u   += erru
            error_v   += errv
            error_w   += errw
            n_error   += errvol
            print("Error u:"+str(erru/errvol*100)+"%",flush=True)
            print("Error v:"+str(errv/errvol*100)+"%",flush=True)
            print("Error w:"+str(errw/errvol*100)+"%",flush=True)
        error_u /= n_error
        error_v /= n_error
        error_w /= n_error
        
        # ----------------------------------------------------------------------------------------------------------
        # Save in file
        # ---------------------------------------------------------------------------------------------------------- 
        file_error = self.data_folder+'/'+self.error_file
        file_error = file_error.replace(".txt","_y.txt")                     
        file_save  = open(file_error, "w+")           
        content    = str(error_u.tolist())+'\n'
        file_save.write(content)           
        content    = str(error_v.tolist())+'\n'
        file_save.write(content)           
        content    = str(error_w.tolist())+'\n'
        file_save.write(content) 
        file_save.close()                       
          
    def pred_urms(self):
        """
        ................................................................................................................
        # pred_error
        ................................................................................................................
        Function to calculate the error between the predicted field and the expected field.
        The error is calculated weighting the localerror with the volume of each cell. 
        The relative error is calculated with respect to the maximum velocity of each wall distance.
																 
        Returns
        -------
        None.

        """
        # --------------------------------------------------------------------------------------------------------------
        # Import packages
        # --------------------------------------------------------------------------------------------------------------
        from py_bin.py_class.flow_field import flow_field
        from py_bin.py_functions.urms import save_rms
        
        # --------------------------------------------------------------------------------------------------------------
        # Calculate the rms of the predictions
        # --------------------------------------------------------------------------------------------------------------
        interval = np.array(range(self.field_ini,self.field_fin,self.field_delta),dtype='int')
        error_u  = 0
        error_v  = 0
        error_w  = 0
        n_error  = 0
        
        for index_ii in interval:
            pred_vel = self.pred_field(data_in={"index_ii":index_ii})
            uu       = pred_vel["uu"]
            vv       = pred_vel["vv"]
            ww       = pred_vel["ww"]
            if index_ii == interval[0]:
                uu_cum   = np.sum(uu**2,axis=(1,2))
                vv_cum   = np.sum(vv**2,axis=(1,2))
                ww_cum   = np.sum(ww**2,axis=(1,2))
                uv_cum   = np.sum(uu*vv,axis=(1,2))
                uw_cum   = np.sum(uu*ww,axis=(1,2))
                vw_cum   = np.sum(vv*ww,axis=(1,2))
                nn_cum   = self.shpx*self.shpz
            else:
                uu_cum   += np.sum(uu**2,axis=(1,2))
                vv_cum   += np.sum(vv**2,axis=(1,2))
                ww_cum   += np.sum(ww**2,axis=(1,2))
                uv_cum   += np.sum(uu*vv,axis=(1,2))
                uw_cum   += np.sum(uu*ww,axis=(1,2))
                vw_cum   += np.sum(vv*ww,axis=(1,2))
                nn_cum   += self.shpx*self.shpz
        uu_cum /= nn_cum
        vv_cum /= nn_cum
        ww_cum /= nn_cum
        uv_cum /= nn_cum
        uw_cum /= nn_cum
        vw_cum /= nn_cum
        uu_rms  = np.sqrt(uu_cum)
        vv_rms  = np.sqrt(vv_cum)
        ww_rms  = np.sqrt(ww_cum)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Save the data
        # ---------------------------------------------------------------------------------------------------------------
        save_rms(data_in={"file":self.urmspred_file,"folder":self.data_folder,"uurms":uu_rms,"vvrms":vv_rms,
                          "wwrms":ww_rms,"uv":uv_cum,"vw":vw_cum,"uw":uw_cum})
        
        
                
    def architecture_Unet(self,data_in={"x_in":[],"flag_print":True}):
        """
        .................................................................................................................
        # architecture_Unet
        .................................................................................................................

        Parameters
        ----------
        data_in : dict, optional
            Data for define the architecture. The default is {"x_in":[],"flag_print":True}.
            Data:
                - x_in       : input of the model
                - flag_print : flag for printing the data type of the layers (True: print, False: do not print)

        Returns
        -------
        dict
            Output of the model
            Data:
                - x_out : output of the model
        """        
        # --------------------------------------------------------------------------------------------------------------
        # Read the data
        # --------------------------------------------------------------------------------------------------------------
        x_in       = data_in["x_in"]
        flag_print = bool(data_in["flag_print"])
        
        # --------------------------------------------------------------------------------------------------------------
        # Define the required packages
        # --------------------------------------------------------------------------------------------------------------
        from tensorflow.keras.layers import AveragePooling3D,Concatenate
        from py_bin.py_functions.CNNblock_definition import block,invblock
        
        # --------------------------------------------------------------------------------------------------------------
        # Define the number of filters of each layer
        # --------------------------------------------------------------------------------------------------------------
        nfil1 = self.nfil
        nfil2 = 2*nfil1
        nfil3 = 2*nfil2
        nfil4 = 2*nfil3
        
        # --------------------------------------------------------------------------------------------------------------
        # Define the Unet by layers in the following lines the fields are defined following the format:
        # xij_k: being i the index of the level (similar sizes of the unet) 
        #        and j the number of the field (increase after the convolutional neurons)
        #        k defines if the level is encoding (e) or decoding (d)
        # --------------------------------------------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------------------------------------------
        # Definintion of the encoder first layer
        # --------------------------------------------------------------------------------------------------------------
        data_x11_e = {"input":x_in,"nfil":nfil1,"stride":self.stride,"activ":self.activation,"kernel":self.kernel,
                      "dtype":None}
        x11_e      = block(data_in=data_x11_e)["output"]
        data_x12_e = {"input":x11_e,"nfil":nfil1,"stride":self.stride,"activ":self.activation,"kernel":self.kernel,
                      "dtype":None}
        x12_e      = block(data_in=data_x12_e)["output"]
        # --------------------------------------------------------------------------------------------------------------
        # Add an average pooling to go to the second layer (reducing the size of the fields)
        # --------------------------------------------------------------------------------------------------------------
        x20_e = AveragePooling3D(self.pooling)(x12_e)
        
        # --------------------------------------------------------------------------------------------------------------
        # Definition of the encoder second layer
        # --------------------------------------------------------------------------------------------------------------
        data_x21_e = {"input":x20_e,"nfil":nfil2,"stride":self.stride,"activ":self.activation,"kernel":self.kernel,
                      "dtype":None}
        x21_e      = block(data_in=data_x21_e)["output"]
        data_x22_e = {"input":x21_e,"nfil":nfil2,"stride":self.stride,"activ":self.activation,"kernel":self.kernel,
                      "dtype":None}
        x22_e      = block(data_in=data_x22_e)["output"]
        # --------------------------------------------------------------------------------------------------------------
        # Add an average pooling to go to the third layer
        # --------------------------------------------------------------------------------------------------------------
        x30_e = AveragePooling3D(self.pooling)(x22_e)
        
        # --------------------------------------------------------------------------------------------------------------
        # Definition of the encoder third layer
        # --------------------------------------------------------------------------------------------------------------
        data_x31_e = {"input":x30_e,"nfil":nfil3,"stride":self.stride,"activ":self.activation,"kernel":self.kernel,
                      "dtype":None}
        x31_e      = block(data_in=data_x31_e)["output"]
        data_x32_e = {"input":x31_e,"nfil":nfil3,"stride":self.stride,"activ":self.activation,"kernel":self.kernel,
                      "dtype":None}
        x32_e      = block(data_in=data_x32_e)["output"]
        # --------------------------------------------------------------------------------------------------------------
        # Add an average pooling to go to the fourth layer
        # --------------------------------------------------------------------------------------------------------------
        x40_e = AveragePooling3D(self.pooling)(x32_e) 
        
        # --------------------------------------------------------------------------------------------------------------
        # Definition of the encoder fourth layer
        # --------------------------------------------------------------------------------------------------------------
        data_x41_e = {"input":x40_e,"nfil":nfil4,"stride":self.stride,"activ":self.activation,"kernel":self.kernel,
                      "dtype":None}
        x41_e      = block(data_in=data_x41_e)["output"]
        data_x42_e = {"input":x41_e,"nfil":nfil4,"stride":self.stride,"activ":self.activation,"kernel":self.kernel,
                      "dtype":None}
        x42_e      = block(data_in=data_x42_e)["output"]
        
        # --------------------------------------------------------------------------------------------------------------
        # Definition of the decoder third layer
        # Note: if modifiying the default values of the layer, the outpad and the size taken of x30_d 
        # may need to be modified
        # --------------------------------------------------------------------------------------------------------------
        data_x30_d = {"input":x42_e,"nfil":nfil3,"stride":self.pooling,"activ":self.activation,"kernel":self.kernel}#,
                      # "outpad":(0,0,0)}
        x30_d      = invblock(data_in=data_x30_d)["output"]
        x31_d      = Concatenate()([x32_e,x30_d[:,:-1,:,:,:]]) 
        data_x32_d = {"input":x31_d,"nfil":nfil3,"stride":self.stride,"activ":self.activation,"kernel":self.kernel,
                      "dtype":None}
        x32_d      = block(data_in=data_x32_d)["output"]
        
        # --------------------------------------------------------------------------------------------------------------
        # Definition of the decoder second layer
        # Note: if modifiying the default values of the layer, the outpad and the size taken of x20_d 
        # may need to be modified
        # --------------------------------------------------------------------------------------------------------------
        data_x20_d = {"input":x32_d,"nfil":nfil2,"stride":self.pooling,"activ":self.activation,"kernel":self.kernel}#,
                      # "outpad":(0,0,0)}
        x20_d      = invblock(data_in=data_x20_d)["output"]
        x21_d      = Concatenate()([x22_e,x20_d[:,:-1,:,:,:]]) 
        data_x22_d = {"input":x21_d,"nfil":nfil2,"stride":self.stride,"activ":self.activation,"kernel":self.kernel,
                      "dtype":None}
        x22_d      = block(data_in=data_x22_d)["output"]
        
        # --------------------------------------------------------------------------------------------------------------
        # Definition of the decoder first layer
        # Note: if modifiying the default values of the layer, the outpad and the size taken of x10_d 
        # may need to be modified
        # --------------------------------------------------------------------------------------------------------------
        data_x10_d = {"input":x22_d,"nfil":nfil1,"stride":self.pooling,"activ":self.activation,"kernel":self.kernel}#,
                      # "outpad":(0,0,0)}
        x10_d      = invblock(data_in=data_x10_d)["output"] 
        x11_d      = Concatenate()([x12_e,x10_d[:,:,:-1,:-1,:]])
        data_x12_d = {"input":x11_d,"nfil":nfil1,"stride":self.stride,"activ":self.activation,"kernel":self.kernel,
                      "dtype":None}
        x12_d      = block(data_in=data_x12_d)["output"]
        if self.mean_norm:
            data_x13_d = {"input":x12_d,"nfil":3,"stride":self.stride,"activ":"tanh","kernel":self.kernel,
                          "dtype":self.data_type}
        else:
            data_x13_d = {"input":x12_d,"nfil":3,"stride":self.stride,"activ":"sigmoid","kernel":self.kernel,
                          "dtype":self.data_type}
        x13_d      = block(data_in=data_x13_d)["output"]
        
        # --------------------------------------------------------------------------------------------------------------
        # Crop the solution to delete the padding
        # --------------------------------------------------------------------------------------------------------------
        x_out    = x13_d[:,:,self.padding:-self.padding,self.padding:-self.padding,:]
        data_out = {"x_out":x_out}
        
        # --------------------------------------------------------------------------------------------------------------
        # Print data type
        # --------------------------------------------------------------------------------------------------------------
        if flag_print:
            print("x_in type:"+str(x_in.dtype),flush=True)
            print("x11_e type:"+str(x11_e.dtype),flush=True)
            print("x12_e type:"+str(x12_e.dtype),flush=True)
            print("x20_e type:"+str(x20_e.dtype),flush=True)
            print("x21_e type:"+str(x21_e.dtype),flush=True)
            print("x22_e type:"+str(x22_e.dtype),flush=True)
            print("x30_e type:"+str(x30_e.dtype),flush=True)
            print("x31_e type:"+str(x31_e.dtype),flush=True)
            print("x32_e type:"+str(x32_e.dtype),flush=True)      
            print("x40_e type:"+str(x40_e.dtype),flush=True)
            print("x41_e type:"+str(x41_e.dtype),flush=True)
            print("x42_e type:"+str(x42_e.dtype),flush=True)
            print("x30_d type:"+str(x30_d[:,:-1,:,:,:].dtype),flush=True) 
            print("x31_d type:"+str(x31_d.dtype),flush=True) 
            print("x32_d type:"+str(x32_d.dtype),flush=True)
            print("x20_d type:"+str(x20_d[:,:-1,:,:,:].dtype),flush=True) 
            print("x21_d type:"+str(x21_d.dtype),flush=True)
            print("x22_d type:"+str(x22_d.dtype),flush=True)
            print("x10_d type:"+str(x10_d[:,:,:-1,:-1,:].dtype),flush=True)
            print("x11_d type:"+str(x11_d.dtype),flush=True)
            print("x12_d type:"+str(x12_d.dtype),flush=True)
            print("x13_d type:"+str(x13_d.dtype),flush=True)
            print("x_out type:"+str(x_out.dtype),flush=True)
        
        # --------------------------------------------------------------------------------------------------------------
        # Return the output
        # --------------------------------------------------------------------------------------------------------------
        return data_out
        
    def prepare_tfrecords(self):
        """
        .................................................................................................................
        # prepare_tfrecords
        .................................................................................................................
        Function to prepare the data for the training. This function creates a new folder with the data of the uvw 
        files with the tensorflow format.
    
        Returns
        -------
        None.
    
        """
        # --------------------------------------------------------------------------------------------------------------
        # Import packages
        # --------------------------------------------------------------------------------------------------------------
        from py_bin.py_functions.merge_data_fromfields import merge_data
        
        # --------------------------------------------------------------------------------------------------------------
        # Create the folder
        # --------------------------------------------------------------------------------------------------------------
        try:
            os.mkdir(self.tfrecord_folder)
        except:
            pass
        
        # --------------------------------------------------------------------------------------------------------------
        # The training randomizes the fields in order to take smaller dataset to load them in the memory. 
        # Then this dataset is divided in smaller batchs using the field_mem data. 
        # ind_vec : randomized index for the training
        # ii_ini  : initial index of the subset of data
        # ii_fin  : final index of the subset of data
        # --------------------------------------------------------------------------------------------------------------
        ind_vec = np.array(range(self.field_ini,self.field_fin,self.field_delta),dtype='int')
        from_tf_file = False
        if from_tf_file:
            merge_data(data_in={"base_directory":self.uvw_folder_tf+"/"+self.uvw_folderii_tf,"padding":self.padding,
                                "shpx":self.shpx,"shpy":self.shpy,"shpz":self.shpz,"data_type":self.data_type,
                                "datasets":ind_vec,"output_path":self.tfrecord_folder})
        else:
            merge_data(data_in={"base_directory":self.uvw_folder,"base_file":self.uvw_file,"padding":self.padding,
                                "shpx":self.shpx,"shpy":self.shpy,"shpz":self.shpz,"dx":self.dx,"dy":self.dy,
                                "dz":self.dz,"data_type":self.data_type,"datasets":ind_vec,
                                "output_path":self.tfrecord_folder,"data_folder":self.data_folder,
                                "umean_file":self.umean_file,"unorm_file":self.unorm_file,
                                "mean_norm":self.mean_norm,"delta_pred":self.delta_pred})
            
    def _read_fieldsvec(self):
        """
        .................................................................................................................
        # _read_fieldsvec
        .................................................................................................................
        Function to read the fields used for the training and validation        

        Returns
        -------
        dict
            Indices of the training and validation fields
            Data:
                - ind_vec : list of index of the training and test data

        """
        file_traintest = self.data_folder+'/'+self.traintest_index
        file_read      = open(file_traintest,"r")
        ind_vec        = np.array(file_read.readline().replace('[','').replace(']','').split(','),dtype='int')
        data_out       = {"ind_vec":ind_vec}
        file_read.close()
        return data_out
            
        
    def _save_fieldsvec(self,data_in={"ind_vec":[]}):
        """
        .................................................................................................................
        # _save_fieldsvec
        .................................................................................................................
        Function to save the fields used for the training and validation        

        Parameters
        ----------
        data_in : dict, optional
            Data of the indices of the training and test fields.
            The default is {"ind_vec":[]}.
            Data:
                - ind_vec : indices of the training and test data

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Read the data
        # ---------------------------------------------------------------------------------------------------------------
        ind_vec = np.array(data_in["ind_vec"],dtype="int")
        
        # ---------------------------------------------------------------------------------------------------------------
        # Save the information
        # ---------------------------------------------------------------------------------------------------------------
        file_traintest = self.data_folder+'/'+self.traintest_index                    
        file_save      = open(file_traintest, "w+")           
        content        = str(ind_vec.tolist())+'\n'
        file_save.write(content) 
        file_save.close()
        
