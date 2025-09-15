# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
trainvali_data.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 21 14:47:23 2024

@author: Andres Cremades Botella

File to prepare the data for the training of the neural network. The file contains the following functions:
    Functions:
        - prepare_data_tf        : function to prepare the data with the tensorflow format
        - read_data_tf           : function to read the data with the tensorflow format
        - read_inout_notprepared : function to read the data for the training and the test directly from the flow
                                   files
        - data_traintest_tf      : function to separe training and test data from the flow files and convert them 
                                   into the tensorflow format
"""
# -----------------------------------------------------------------------------------------------------------------------
# Import packages for all functions
# -----------------------------------------------------------------------------------------------------------------------
import numpy as np
import os
import sys
from py_bin.py_functions.read_norm_velocity import read_norm_velocity

# -----------------------------------------------------------------------------------------------------------------------
# Define read function
# -----------------------------------------------------------------------------------------------------------------------
def _load_datasets(data_in={"base_directory":"../../P125_21pi_vu_tf_float32_mean/","datasets":[],
                            "version_tf":[2,10,1]}):
    """
    .....................................................................................................................
    # _load_datasets: Function for reading the data
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        data required for selecting the fields for the training process.
        The default is {"base_directory":"../../P125_21pi_vu_tf_float32_mean/","datasets":[],
                        "elem_spec":[],"version_tf":[2,10,1]}.
        Data:
            - base_directory : folder in which the data is stored
            - datasets       : fields to read
            - version_tf     : version of the tensorflow
    
    Returns
    -------
    dict
        Structure containing the data for the training and the data for the test
        Data:
            - data_train : data for training
            - data_vali  : data for test
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Load packages
    # -------------------------------------------------------------------------------------------------------------------
    import tensorflow as tf
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    base_directory = str(data_in["base_directory"])
    num_datasets   = np.array(data_in["datasets"],dtype="int")
    version_tf     = np.array(data_in["version_tf"],dtype="int")
    
    # -------------------------------------------------------------------------------------------------------------------
    # Select the datasets
    # -------------------------------------------------------------------------------------------------------------------
    all_files     = sorted(os.listdir(base_directory))
    index_files   = [int(file.split('.')[1]) for file in all_files]
    index_data    = np.nonzero(np.in1d(index_files,num_datasets))[0]
    dataset_paths = [os.path.join(base_directory, all_files[index_field]) for index_field in index_data]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Assuming datasets are in TFRecord or a compatible format
    # -------------------------------------------------------------------------------------------------------------------
    def fun_load(iifile,totalfiles,dataset_paths,version_tf):
        """
        Function to load the information with verbose

        Parameters
        ----------
        iifile : int
            Index of the file.
        totalfiles : int
            Maximum number of files.
        dataset_paths : list 
            List of the dataset.
        version_tf : array of int
            Version of tensorflow.

        Returns
        -------
        data_out : tf.data.Dataset
            Loaded dataset.

        """
        file = dataset_paths[iifile]
        print("-"*100,flush=True)
        print(str(file)+" field: percentage: "+str(iifile/totalfiles*100)+"%",flush=True)
        print("-"*100,flush=True)
        if version_tf[0] >= 2 and version_tf[1] >= 10 and version_tf[2] >= 0:
            data_out = tf.data.Dataset.load(file)
        else:
            data_out = tf.data.experimental.load(file)
        return data_out
    totalfiles         = len(dataset_paths)
    datasets           = [fun_load(iifile,totalfiles,dataset_paths,version_tf) for iifile in np.arange(totalfiles)]
    merged_dataset     = tf.data.Dataset.from_tensor_slices(datasets)
    merged_dataset_map = merged_dataset.flat_map(lambda xx:xx)
    return merged_dataset_map


def _read_datatf_function(data_in={"index":[],"folder_tf":"../../P125_21pi_vu_tf","folderii_tf":"P125_21pi_vu.$INDEX$",
                                   "train_test":"Test","printflag":False,"version_tf":[2,10,1],"check":True,
                                   "ssh_flag_train":False,"uvw_folder_temp":"-","ssh_server":"-","ssh_username":"-",
                                   "ssh_password":"-"}):
    """
    .....................................................................................................................
    # _read_datatf_function: Subroutine for reading the dataset. The function creates the tensorflow structure in case
                             of the first file and concatenates it in the case of the following. The function is
                             separated to the main workflow as it is repeated for the training and the validation.
    .....................................................................................................................    
    Parameters
    ----------
    data_in : dict, optional
        data required for selecting the fields of the training and prepare it for the training process.
        The default is {"index":[],"folder_savetf":"../../P125_21pi_vu_tf","train_test":"Test",
                        "printflag":False,"version_tf":[2,10,1]}.
        Data:
            - index           : Index of the field to read.
            - folder_tf       : folder for saving the data in the tensorflow format
            - folderii_tf     : folder for saving the data in the tensorflow format for each flow field
            - train_test      : Specifies if the field corresponds to the training or the validation
            - version_tf      : version of the tensorflow module
            - check           : check if database is correct
            - ssh_flag_train  : flag determining if ssh connection should be activated
            - uvw_folder_temp : temporal forlder for storing the files
            - ssh_server      : server to read using the ssh connection
            - ssh_username    : username of the server
            - ssh_password    : password of the server user

    Returns
    -------
    data_out : dict
        Dictionary containing the tensorflow dataset for training or test and errors in case of checking.
        Data:
            - data_tf     : tensor containing the data
            - flag_return : flag indicating if there is an error (1:no error, 0: error)
            - index_file  : index of the flag generating the error

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    import time
    import tensorflow as tf
    from py_bin.py_remote.read_remote import read_from_server
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    index           = np.array(data_in["index"],dtype='int')
    folder_tf       = str(data_in["folder_tf"])
    folderii_tf     = str(data_in["folderii_tf"])
    folder_savetf   = folder_tf+'/'+folderii_tf
    train_test      = str(data_in["train_test"])
    printflag       = bool(data_in["printflag"])
    version_tf      = np.array(data_in["version_tf"],dtype='int')
    check           = bool(data_in["check"])
    flag_return     = True
    ssh_flag_train  = bool(data_in["ssh_flag_train"])              # Flag for reading the files via ssh
    uvw_folder_temp = str(data_in["uvw_folder_temp"])              # Folder to temporally store the files
    ssh_server      = str(data_in["ssh_server"])                   # Server to read the files
    ssh_username    = str(data_in["ssh_username"])                 # username of the server
    ssh_password    = str(data_in["ssh_password"])                 # password of the server
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the files
    # -------------------------------------------------------------------------------------------------------------------
    lenind = len(index)
    for ii in np.arange(lenind):
        flag_try         = 0
        while True:
            index_file       = index[ii]
            tread_0          = time.time()
            folder_savetf_ii = folder_savetf.replace("$INDEX$",str(index_file))
            if printflag:
                print("-"*100,flush=True)
                print(str(train_test)+" field: percentage: "+str(ii/lenind*100)+"%",flush=True)
                print("Reading tensorflow field: "+folder_savetf_ii,flush=True)
            if ssh_flag_train:
                folder_savetf_temp = uvw_folder_temp+'/'+folderii_tf
                folder_temp_ii     = folder_savetf_temp.replace("$INDEX$",str(index_file))
                read_from_server(data_in={"remotedir":folder_savetf_ii,"localdir":folder_temp_ii,"server":ssh_server,
                                          "username":ssh_username,"password":ssh_password})
                folder_savetf_ii   = folder_temp_ii
            if version_tf[0] >= 2 and version_tf[1] >= 10 and version_tf[2] >= 0:
                if index_file == index[0]:
                    rdat_tf = tf.data.Dataset.load(folder_savetf_ii)
                    data_tf = rdat_tf
                else:
                    rdat_tf = tf.data.Dataset.load(folder_savetf_ii)
                    data_tf = data_tf.concatenate(rdat_tf)
            else:
                if index_file == index[0]:
                    rdat_tf = tf.data.experimental.load(folder_savetf_ii)
                    data_tf = rdat_tf
                else:
                    rdat_tf = tf.data.experimental.load(folder_savetf_ii)
                    data_tf = data_tf.concatenate(rdat_tf)
            tread_1 = time.time()
            if check:
                try:
                    print("Checking data file...",flush=True)
                    list(rdat_tf.take(1).as_numpy_iterator())
                    tread_2 = time.time()
                    print("Checked data file in: "+str(tread_2-tread_1)+"s",flush=True)
                    break
                except:
                    print("Error in database. Field: "+folder_savetf_ii+" is corrupt",flush=True)
                    flag_try   += 1
                    if flag_try > 3:
                        flag_return = False
                        break
            else:
                break
            if printflag:
                print("Time to read field "+str(index_file)+": "+str(tread_1-tread_0),flush=True)
                print("-"*100,flush=True)
        if flag_return == False:
            break
    data_out = {"data_tf":data_tf,"flag_return":flag_return,"index_file":index_file}
    return data_out

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define the functions
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

def prepare_data_tf(data_in={"folder":"../../P125_21pi_vu","file":"P125_21pi_vu.$INDEX$.h5.uvw",
                             "interval":[],"delta_pred":1,"padding":15,"shpx":1,"shpy":1,"shpz":1,
                             "dx":1,"dy":1,"dz":1,"data_folder":"Data","umean_file":"Umean.txt",
                             "unorm_file":"Unorm.txt","folder_tf":"../../P125_21pi_vu_tf",
                             "folderii_tf":"P125_21pi_vu.$INDEX$","data_type":"float32","mean_norm":False}):
    """
    .....................................................................................................................
    # prepare_data_tf: Function for preparing the training and validation data
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        data required for selecting the fields of the training and prepare it for the training process.
        The default is {"folder":"../P125_21pi_vu","file":"P125_21pi_vu.$INDEX$.h5.uvw",
                        "interval":[],"delta_pred":1,"padding":15,
                        "shpx":1,"shpy":1,"shpz":1,"dx":1,"dy":1,"dz":1,"data_folder":"Data",
                        "umean_file":"Umean.txt","unorm_file":"Unorm.txt","folder_tf":"../P125_21pi_vu_tf",
                        "folderii_tf":"P125_21pi_vu.$INDEX$","data_type":"float32","mean_norm":False}.
        Data:
            - folder      : folder to read the data of the velocity fields
            - file        : file to read the data of the velocity fields
            - interval    : index of the fields required for the training
            - delta_pred  : distance between the fields
            - padding     : padding of the fields
            - shpx        : shape of the fields in x
            - shpy        : shape of the fields in y
            - shpz        : shape of the fields in z
            - dx          : downsampling in x
            - dy          : downsampling in y
            - dz          : downsampling in z
            - data_folder : folder to store the data generated by the code
            - umean_file  : mean velocity file
            - unorm_file  : file for the normalization of the 
            - folder_tf   : folder for saving the data in the tensorflow format
            - folderii_tf : folder for saving the data in the tensorflow format for each flow field
            - data_type   : type of data of the tensors
            - mean_norm   : flag for using the mean and std for normalizing (True: use mean and std
                                                                             False: use min and max)

    Returns
    -------
    dict
        Structure containing the data base for the training and the data for the test
        Data:
            - data_X      : training data
            - data_Y      : validation data

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Load packages
    # -------------------------------------------------------------------------------------------------------------------
    import tensorflow as tf
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    folder      = str(data_in["folder"])                       # folder to read the data of the velocity fields
    file        = str(data_in["file"])                         # file to read the data of the velocity fields
    interval    = np.array(data_in["interval"],dtype='int')    # array with the fields of the training
    delta_pred  = int(data_in["delta_pred"])                   # separation between the fields
    padding     = int(data_in["padding"])                      # padding of the fields
    shpx        = int(data_in["shpx"])                         # shape of the fields in x
    shpy        = int(data_in["shpy"])                         # shape of the fields in y
    shpz        = int(data_in["shpz"])                         # shape of the fields in z
    dx          = int(data_in["dx"])                           # downsamplin in x
    dy          = int(data_in["dy"])                           # downsamplin in y
    dz          = int(data_in["dz"])                           # downsamplin in z
    data_folder = str(data_in["data_folder"])                  # folder to store generated data
    umean_file  = str(data_in["umean_file"])                   # file to read the mean velocity
    unorm_file  = str(data_in["unorm_file"])                   # file to read the normalization values
    folder_tf   = str(data_in["folder_tf"])                    # folder of the tensorflow format 
    folderii_tf = str(data_in["folderii_tf"])                  # file of the tensorflow format
    mean_norm   = bool(data_in["mean_norm"])                   # flag for normalizing with mean and std
    if "data_type" in data_in.keys():
        data_type = str(data_in["data_type"])                  # definition of the data type.
        if not (data_type=="float32" or data_type=="float16"):
            data_type = "float32"
    else:
        print("[trainvali_data.py:data_traintest_tf] Data type needs to be selected.")
        sys.exit()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the forlder for the tensorflow format data. This data is normalized and prepared for the training
    # -------------------------------------------------------------------------------------------------------------------
    try:
        os.mkdir(folder_tf)
    except:
        print("Folder: "+folder_tf+" is already created",flush=True)
    folder_savetf = folder_tf+'/'+folderii_tf
    
    # -------------------------------------------------------------------------------------------------------------------
    # Define the input and the output data matrices, the precision of the matrices is reduced to data type to
    # adapt the memory required for them. Then the dataset is converted to the tensorflow dataset format.
    # data_X : input data for training and validation. Read the file with the index in the interval uses the padding
    #          of the model
    # data_Y : output data for training and validation. Read the file with the following index in the interval. It
    #          does not use padding.
    # The variables are clear as soon as possible to save memory.
    # -------------------------------------------------------------------------------------------------------------------
    for ii in interval: 
        data_X               = np.zeros((1,shpy,shpz+2*padding,shpx+2*padding,3),dtype=data_type)
        data_Y               = np.zeros((1,shpy,shpz,shpx,3),dtype=data_type)
        folder_savetf_ii     = folder_savetf.replace("$INDEX$",str(ii))
        try:
            os.mkdir(folder_savetf_ii)
        except:
            print("Folder: "+folder_savetf_ii+" is already created",flush=True)
            
        # ---------------------------------------------------------------------------------------------------------------
        # Read the input field
        # ---------------------------------------------------------------------------------------------------------------
        data_norm_X     = {"folder":folder,"file":file,"padding":padding,"shpx":shpx,
                           "shpy":shpy,"shpz":shpz,"dx":dx,"dy":dy,"dz":dz,"data_folder":data_folder,
                           "umean_file":umean_file,"unorm_file":unorm_file,"index":ii,"data_type":data_type,
                           "mean_norm":mean_norm}
        data_veloc_norm = read_norm_velocity(data_in=data_norm_X)
        norm_velocity_X = data_veloc_norm["norm_velocity"]
        print("Time for reading the field: "+str(data_veloc_norm["time_read"]),flush=True)
        print("Time for normalizing the field: "+str(data_veloc_norm["time_norm"]),flush=True)
        del data_norm_X,data_veloc_norm
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the output field
        # ---------------------------------------------------------------------------------------------------------------
        data_norm_Y     = {"folder":folder,"file":file,"padding":0,"shpx":shpx,
                           "shpy":shpy,"shpz":shpz,"dx":dx,"dy":dy,"dz":dz,"data_folder":data_folder,
                           "umean_file":umean_file,"unorm_file":unorm_file,"index":ii+delta_pred,
                           "data_type":data_type,"mean_norm":mean_norm}
        data_veloc_norm = read_norm_velocity(data_in=data_norm_Y)
        norm_velocity_Y = data_veloc_norm["norm_velocity"]
        print("Time for reading the field: "+str(data_veloc_norm["time_read"]),flush=True)
        print("Time for normalizing the field: "+str(data_veloc_norm["time_norm"]),flush=True)
        del data_norm_Y,data_veloc_norm
        print("-"*100,flush=True)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Store the input and output data
        # ---------------------------------------------------------------------------------------------------------------
        data_X[0,:,:,:,0]    = norm_velocity_X['unorm']
        data_X[0,:,:,:,1]    = norm_velocity_X['vnorm']
        data_X[0,:,:,:,2]    = norm_velocity_X['wnorm']
        del norm_velocity_X
        data_Y[0,:,:,:,0]    = norm_velocity_Y['unorm']
        data_Y[0,:,:,:,1]    = norm_velocity_Y['vnorm']
        data_Y[0,:,:,:,2]    = norm_velocity_Y['wnorm']
        del norm_velocity_Y
        print("-"*100,flush=True)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Save the data in a file
        # ---------------------------------------------------------------------------------------------------------------
        data_XY = tf.data.Dataset.from_tensor_slices((data_X,data_Y))
        tf.data.Dataset.save(data_XY,folder_savetf_ii)
        del data_X, data_Y
        
        

        
def read_data_tf(data_in={"folder_tf":"../../P125_21pi_vu_tf","folderii_tf":"P125_21pi_vu.$INDEX$",
                          "interval":[],"test_size":0.2,"printflag":True,"ssh_flag_train":False,
                          "uvw_folder_temp":"-","ssh_server":"-","ssh_username":"-",
                          "ssh_password":"-","check":True}):
    """
    .....................................................................................................................
    # read_data_tf: Function for reading the training and validation data in the tensorflow format
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        data required for selecting the fields for the training process.
        The default is {"folder_tf":"../P125_21pi_vu_tf","folderii_tf":"P125_21pi_vu.$INDEX$","interval":[],
                        "test_size":0.2,"check":True}.
        Data:
            - folder_tf       : folder for saving the data in the tensorflow format
            - folderii_tf     : folder for saving the data in the tensorflow format for each flow field
            - interval        : index of the fields required for the training
            - test_size       : size in percentage of the test dataset (value between 0 and 1)
            - printflag       : flag for printing the file that is read
            - ssh_flag_train  : flag determining if ssh connection should be activated
            - uvw_folder_temp : temporal forlder for storing the files
            - ssh_server      : server to read using the ssh connection
            - ssh_username    : username of the server
            - ssh_password    : password of the server user
            - check           : flag for checking the data
    
    Returns
    -------
    dict
        Structure containing the data for the training and the data for the test
        Data:
            - data_train : data for training
            - data_vali  : data for test

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    import tensorflow as tf
    
    # -------------------------------------------------------------------------------------------------------------------
    # Get the tensorflow version
    # -------------------------------------------------------------------------------------------------------------------
    version_tf = np.array(tf.__version__.split('.'),dtype="int")
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    folder_tf       = str(data_in["folder_tf"])                    # folder containing the tensorflow data
    folderii_tf     = str(data_in["folderii_tf"])                  # folder containing the tf data of the field ii
    interval        = np.array(data_in["interval"],dtype='int')    # indices of the dataset to load in the training
    test_size       = float(data_in["test_size"])                  # size of the test data set                                                                   # (value between 0 and 1)
    test_size       = np.min([1,test_size])
    test_size       = np.max([0,test_size])
    len_train       = int(len(interval)*(1-test_size))
    printflag       = bool(data_in["printflag"])                   # flag for printing read
    ssh_flag_train  = bool(data_in["ssh_flag_train"])              # Flag for reading the files via ssh
    uvw_folder_temp = str(data_in["uvw_folder_temp"])              # Folder to temporally store the files
    ssh_server      = str(data_in["ssh_server"])                   # Server to read the files
    ssh_username    = str(data_in["ssh_username"])                 # username of the server
    ssh_password    = str(data_in["ssh_password"])                 # password of the server
    check           = bool(data_in["check"])                       # flag for checking the data
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the tensorflow datasets
    # -------------------------------------------------------------------------------------------------------------------
    # The condition concat_flag is set to False as the function is deprecated and a newer one is used. In case of
    # reading the information through ssh then, the older function is used
    # -------------------------------------------------------------------------------------------------------------------
    concat_flag = False
    if concat_flag or ssh_flag_train:
        errfile = open("err_read.log", 'w')
        data_readtf   = _read_datatf_function(data_in={"index":interval[:len_train],"folder_tf":folder_tf,
                                                        "folderii_tf":folderii_tf,"train_test":"Training",
                                                        "printflag":printflag,"version_tf":version_tf,
                                                        "check":check,"ssh_flag_train":ssh_flag_train,
                                                        "uvw_folder_temp":uvw_folder_temp,
                                                        "ssh_server":ssh_server,"ssh_username":ssh_username,
                                                        "ssh_password":ssh_password})
        data_train_tf = data_readtf["data_tf"]
        flag_train    = data_readtf["flag_return"]
        if not flag_train:
            print("Error in field: "+str(data_readtf["index_file"]),file=errfile)
            print("Exiting the calculation...",flush=True)
            sys.exit()
        data_readtf   = _read_datatf_function(data_in={"index":interval[len_train:],"folder_tf":folder_tf,
                                                        "folderii_tf":folderii_tf,"train_test":"Validation",
                                                        "printflag":printflag,"version_tf":version_tf,"check":check,
                                                        "ssh_flag_train":ssh_flag_train,
                                                        "uvw_folder_temp":uvw_folder_temp,"ssh_server":ssh_server,
                                                        "ssh_username":ssh_username,"ssh_password":ssh_password})
        data_vali_tf  = data_readtf["data_tf"]
        flag_vali     = data_readtf["flag_return"]
        if not flag_vali:
            print("Error in field: "+str(data_readtf["index_file"]),file=errfile)
            print("Exiting the calculation...",flush=True)
            sys.exit()
        errfile.close()
    else:    
        data_train_tf = _load_datasets(data_in={"base_directory":folder_tf,"datasets":interval[:len_train],
                                                "version_tf":version_tf})
        data_vali_tf = _load_datasets(data_in={"base_directory":folder_tf,"datasets":interval[len_train:],
                                               "version_tf":version_tf})
    # -------------------------------------------------------------------------------------------------------------------
    # Generate the output
    # -------------------------------------------------------------------------------------------------------------------
    data_out = {"data_train":data_train_tf,"data_vali":data_vali_tf}
    return data_out
        

        
def check_data_tf(data_in={"folder_tf":"../../P125_21pi_vu_tf","folderii_tf":"P125_21pi_vu.$INDEX$",
                           "interval":[],"printflag":True,"ssh_flag_train":False,
                           "uvw_folder_temp":"-","ssh_server":"-","ssh_username":"-",
                           "ssh_password":"-","check":True}):
    """
    .....................................................................................................................
    # check_data_tf: Function for reading the training and validation data in the tensorflow format
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        data required for selecting the fields for the training process.
        The default is {"folder_tf":"../P125_21pi_vu_tf","folderii_tf":"P125_21pi_vu.$INDEX$","interval":[],
                        "test_size":0.2,"check":True}.
        Data:
            - folder_tf       : folder for saving the data in the tensorflow format
            - folderii_tf     : folder for saving the data in the tensorflow format for each flow field
            - interval        : index of the fields required for the training
            - printflag       : flag for printing the file that is read
            - ssh_flag_train  : flag determining if ssh connection should be activated
            - uvw_folder_temp : temporal forlder for storing the files
            - ssh_server      : server to read using the ssh connection
            - ssh_username    : username of the server
            - ssh_password    : password of the server user
            - check           : flag for checking data
    
    Returns
    -------
    dict
        Structure containing the data for the training and the data for the test
        Data:
            - data_train : data for training
            - data_vali  : data for test

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    import tensorflow as tf
    
    # -------------------------------------------------------------------------------------------------------------------
    # Get the tensorflow version
    # -------------------------------------------------------------------------------------------------------------------
    version_tf = np.array(tf.__version__.split('.'),dtype="int")

    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    folder_tf       = str(data_in["folder_tf"])                    # folder containing the tensorflow data
    folderii_tf     = str(data_in["folderii_tf"])                  # folder containing the tf data of the field ii
    interval        = np.array(data_in["interval"],dtype='int')    # indices of the dataset to load in the training
    folder_savetf   = folder_tf+'/'+folderii_tf
    printflag       = bool(data_in["printflag"])                   # flag for printing read
    ssh_flag_train  = bool(data_in["ssh_flag_train"])              # Flag for reading the files via ssh
    uvw_folder_temp = str(data_in["uvw_folder_temp"])              # Folder to temporally store the files
    ssh_server      = str(data_in["ssh_server"])                   # Server to read the files
    ssh_username    = str(data_in["ssh_username"])                 # username of the server
    ssh_password    = str(data_in["ssh_password"])                 # password of the server
    check           = bool(data_in["check"])                       # flag for checking the data
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the tensorflow datasets
    # -------------------------------------------------------------------------------------------------------------------
    data_readtf = _read_datatf_function(data_in={"index":interval,"folder_tf":folder_tf,
                                                 "folderii_tf":folderii_tf,"train_test":"Validation",
                                                 "printflag":printflag,"version_tf":version_tf,"check":check,
                                                 "ssh_flag_train":ssh_flag_train,"uvw_folder_temp":uvw_folder_temp,
                                                 "ssh_server":ssh_server,"ssh_username":ssh_username,
                                                 "ssh_password":ssh_password})
    return data_readtf
    

        

def read_inout_notprepared(data_in={"folder":"../../P125_21pi_vu","file":"P125_21pi_vu.$INDEX$.h5.uvw",
                                    "interval":[],"delta_pred":1,"padding":15,
                                    "shpx":1,"shpy":1,"shpz":1,"dx":1,"dy":1,"dz":1,"data_folder":"Data",
                                    "umean_file":"Umean.txt","unorm_file":"Unorm.txt","test_size":0.2,
                                    "data_type":"float32","mean_norm":False}):
    """
    .....................................................................................................................
    # read_inout_notprepared: Function for preparing the training and validation data
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        data required for selecting the fields of the training and prepare it for the training process.
        The default is {"folder":"../P125_21pi_vu","file":"P125_21pi_vu.$INDEX$.h5.uvw",
                        "interval":[],"delta_pred":1,"padding":15,
                        "shpx":1,"shpy":1,"shpz":1,"dx":1,"dy":1,"dz":1,"data_folder":"Data",
                        "umean_file":"Umean.txt","unorm_file":"Unorm.txt","data_type":"float32"}.
        Data:
            - folder      : folder to read the data of the velocity fields
            - file        : file to read the data of the velocity fields
            - interval    : index of the fields required for the training
            - delta_pred  : distance between the fields
            - padding     : padding of the fields
            - shpx        : shape of the fields in x
            - shpy        : shape of the fields in y
            - shpz        : shape of the fields in z
            - dx          : downsampling in x
            - dy          : downsampling in y
            - dz          : downsampling in z
            - data_folder : folder to store the data generated by the code
            - umean_file  : mean velocity file
            - unorm_file  : file for the normalization of the velocity
            - data_type   : type of the data (float16, float32)
            - mean_norm    : Flag to normalize using mean and std (True: use mean and std
                                                                   False: use min and max)

    Returns
    -------
    dict
        Structure containing the data base for the training and the data for the test
        Data:
            - data_X      : training data
            - data_Y      : validation data

    """
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    folder      = str(data_in["folder"])                       # folder to read the data of the velocity fields
    file        = str(data_in["file"])                         # file to read the data of the velocity fields
    interval    = np.array(data_in["interval"],dtype='int')    # array with the fields of the training
    delta_pred  = int(data_in["delta_pred"])                   # separation between the fields
    padding     = int(data_in["padding"])                      # padding of the fields
    shpx        = int(data_in["shpx"])                         # shape of the fields in x
    shpy        = int(data_in["shpy"])                         # shape of the fields in y
    shpz        = int(data_in["shpz"])                         # shape of the fields in z
    dx          = int(data_in["dx"])                           # downsamplin in x
    dy          = int(data_in["dy"])                           # downsamplin in y
    dz          = int(data_in["dz"])                           # downsamplin in z
    data_folder = str(data_in["data_folder"])                  # folder to store generated data
    umean_file  = str(data_in["umean_file"])                   # file to read the mean velocity
    unorm_file  = str(data_in["unorm_file"])                   # file to read the normalization values
    mean_norm   = bool(data_in["mean_norm"])                   # choose thipe of normalization
    if "data_type" in data_in.keys():
        data_type = str(data_in["data_type"])                  # definition of the data type.
        if not (data_type=="float32" or data_type=="float16"):
            data_type = "float32"
    else:
        print("[trainvali_data.py:data_traintest_tf] Data type needs to be selected.")
        sys.exit()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Define the input and the output data matrices, the precision of the matrices is reduced to data type to
    # adapt the memory required for them.
    # data_X : input data for training and validation. Read the file with the index in the interval uses the padding
    #          of the model
    # data_Y : output data for training and validation. Read the file with the following index in the interval. It
    #          does not use padding.
    # The variables are clear as soon as possible to save memory.
    # -------------------------------------------------------------------------------------------------------------------
    data_X = np.zeros((len(interval),shpy,shpz+2*padding,shpx+2*padding,3),dtype=data_type)
    data_Y = np.zeros((len(interval),shpy,shpz,shpx,3),dtype=data_type)
    for ii in np.arange(len(interval)):  
        field_interval = interval[ii]  
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the input field
        # ---------------------------------------------------------------------------------------------------------------
        data_norm_X     = {"folder":folder,"file":file,"padding":padding,"shpx":shpx,
                           "shpy":shpy,"shpz":shpz,"dx":dx,"dy":dy,"dz":dz,"data_folder":data_folder,
                           "umean_file":umean_file,"unorm_file":unorm_file,"index":field_interval,
                           "data_type":data_type,"mean_norm":mean_norm}
        data_veloc_norm = read_norm_velocity(data_in=data_norm_X)
        norm_velocity_X = data_veloc_norm["norm_velocity"]
        print("Time for reading the field: "+str(data_veloc_norm["time_read"]),flush=True)
        print("Time for normalizing the field: "+str(data_veloc_norm["time_norm"]),flush=True)
        del data_norm_X,data_veloc_norm
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read the output field
        # ---------------------------------------------------------------------------------------------------------------
        data_norm_Y     = {"folder":folder,"file":file,"padding":0,"shpx":shpx,
                           "shpy":shpy,"shpz":shpz,"dx":dx,"dy":dy,"dz":dz,"data_folder":data_folder,
                           "umean_file":umean_file,"unorm_file":unorm_file,"index":field_interval+delta_pred,
                           "data_type":data_type,"mean_norm":mean_norm}
        data_veloc_norm = read_norm_velocity(data_in=data_norm_Y)
        norm_velocity_Y = data_veloc_norm["norm_velocity"]
        print("Time for reading the field: "+str(data_veloc_norm["time_read"]),flush=True)
        print("Time for normalizing the field: "+str(data_veloc_norm["time_norm"]),flush=True)
        del data_norm_Y,data_veloc_norm
        print("-"*100,flush=True)
        
        # ---------------------------------------------------------------------------------------------------------------
        # Store the input and output data
        # ---------------------------------------------------------------------------------------------------------------
        data_X[ii,:,:,:,0] = norm_velocity_X['unorm']
        data_X[ii,:,:,:,1] = norm_velocity_X['vnorm']
        data_X[ii,:,:,:,2] = norm_velocity_X['wnorm']
        del norm_velocity_X
        data_Y[ii,:,:,:,0] = norm_velocity_Y['unorm']
        data_Y[ii,:,:,:,1] = norm_velocity_Y['vnorm']
        data_Y[ii,:,:,:,2] = norm_velocity_Y['wnorm']
        del norm_velocity_Y
    # -------------------------------------------------------------------------------------------------------------------
    # Store the database
    # -------------------------------------------------------------------------------------------------------------------
    data_out           = {}
    data_out["data_X"] = data_X
    del data_X
    data_out["data_Y"] = data_Y
    del data_Y
    return data_out


def data_traintest_tf(data_in={"data_X":[],"data_Y":[],"interval":None,"test_size":0.2,"data_type":"float32"}):
    """
    .....................................................................................................................
    # data_traintest_tf: Function for preparing the training and validation data
    .....................................................................................................................
    Parameters
    ----------
    data_trainval : dict, optional
        data required for creating the training and validation tensors. The database has been already read and here 
        the arrays of data are converted in the tensors used in the training
        The default is {"data_X":[],"data_Y":[],"interval":None,"test_size":0.2,"data_type":"float32"}.
        Data:
            - data_X      : training data
            - data_Y      : validation data
            - interval    : index of the fields required for the training
            - test_size   : size in percentage of the test dataset (value between 0 and 1)
            - data_type   : type of data of the tensors (float32,float16)
    
    Returns
    -------
    dict
        Structure containing the data for the training and the data for the test
        Data:
            - data_train : data for training
            - data_vali  : data for test
    
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import packages
    # -------------------------------------------------------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    if "data_type" in data_in.keys():
        data_type = str(data_in["data_type"])                    # definition of the data type.
        if not (data_type=="float32" or data_type=="float16"):
            data_type = "float32"
    else:
        print("[trainvali_data.py:data_traintest_tf] Data type needs to be selected.")
        sys.exit()
    data_X        = np.array(data_in["data_X"],dtype=data_type)  # train data
    data_Y        = np.array(data_in["data_Y"],dtype=data_type)  # validation data
    interval_read = data_in["interval"]                          # indices of the dataset to load in the training
    test_size     = float(data_in["test_size"])                  # size of the test data set (value between 0 and 1)
    test_size     = np.min([1,test_size])
    test_size     = np.max([0,test_size])
    if interval_read is not None:
        interval = np.array(interval_read,dtype='int')
        data_X   = data_X[interval,:,:,:,:]
        data_Y   = data_Y[interval,:,:,:,:]
        
    # -------------------------------------------------------------------------------------------------------------------
    # Split the data between the training and the test subsets. Then delete all the big variables to save memory
    # The data needs to be reshaped to the tensorflow format
    # Then store training and test data in a different dataset
    # Returns:
    #    - data_traintest : structure containing the training and the test data
    # -------------------------------------------------------------------------------------------------------------------
    train_X,valid_X,train_Y,valid_Y = train_test_split(data_X, data_Y,test_size=test_size,
                                                       shuffle=False,random_state=13) 
    del data_X, data_Y
    data_out               = {}
    data_out["data_train"] = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
    del train_X,train_Y
    data_out["data_vali"]  = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y))
    del valid_X, valid_Y
    return data_out

