"""
-------------------------------------------------------------------------------------------------------------------------
read_tfrecord.py
-------------------------------------------------------------------------------------------------------------------------
Created on Wed May 15 11:58:23 2024

@author: Andres Cremades Botella - modification of the functions provided by Xuan Gu (Berzelius support)

File containing the function to improve the reading of the data in Berzelius. The file contains the following functions:
    Functions:
        - read_tfrecord   : function to read the tfrecord
        - load_dataset    : function to read the data with the tensorflow format
"""
# -----------------------------------------------------------------------------------------------------------------------
# Import packages
# -----------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import sys
import numpy as np
import os

def load_dataset(data_in={"tfrecord_files":'/tfrecord/dataset_0000.tfrecord',"padding":15,"shpx":1,"shpy":1,"shpz":1,
                           "data_type":"float32","index":[]}):
    """
    .....................................................................................................................
    # _load_dataset: Function for loading the data
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        data required for loading the data
        The default is {"tfrecord":'/tfrecord/merged_train_data.tfrecord'}.
        Data:
            - tfrecord_files : list containing the path to the data
            - padding        : padding of the fields
            - shpx           : shape of the fields in the streamwise direction
            - shpy           : shape of the fields in the wall-normal direction
            - shpz           : shape of the fields in the spanwise direction
            - data_type      : type of data
    Returns
    -------
    dict
        Read data
        Data:
            - data : merged data
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    tfrecord_files = data_in["tfrecord_files"]
    padding        = int(data_in["padding"])
    shpx           = int(data_in["shpx"])
    shpy           = int(data_in["shpy"])
    shpz           = int(data_in["shpz"])
    data_type      = str(data_in["data_type"])
    index          = np.array(data_in["index"],dtype="int")
    if data_type == "float32":
        dtype = tf.float32
    elif data_type == "float16":
        dtype = tf.float16
    else:
        print("Exit the calculation due to invalid data type.",flush=True)
        sys.exit()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Define the function to parse
    # -------------------------------------------------------------------------------------------------------------------
    def parse_function(proto):
        """
        
        .................................................................................................................
        # _parse_function: Function for parsing the data
        .................................................................................................................
    
        Parameters
        ----------
        proto : TFRecords data
            Information to parse.

        Returns
        -------
        feature : TFRecords data
            Parsed features.
        label : TYPE
            Parsed labels.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Define your `features` dictionary here:
        # Adjust shape based on how data was flattened
        # ---------------------------------------------------------------------------------------------------------------
        feature_description = {'feature': tf.io.FixedLenFeature([shpy*(shpz+2*padding)*(shpx+2*padding)*3],dtype),
                               'label': tf.io.FixedLenFeature([shpy*shpz*shpx*3],dtype)}
        parsed_features     = tf.io.parse_single_example(proto,feature_description)
        feature             = tf.reshape(parsed_features['feature'],[shpy,shpz+2*padding,shpx+2*padding,3])
        label               = tf.reshape(parsed_features['label'],[shpy,shpz,shpx,3])
        return feature,label
    
            
    # -------------------------------------------------------------------------------------------------------------------
    # This function reads multiple TFRecord files and returns a parsed dataset.
    # -------------------------------------------------------------------------------------------------------------------
    files   = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.experimental.AUTOTUNE)
    return dataset.map(parse_function)  # Parse each example using the specified parsing function

    # -------------------------------------------------------------------------------------------------------------------
    # This function reads a TFRecord file and returns a parsed dataset.
    # Parse each example using the specified parsing function
    # -------------------------------------------------------------------------------------------------------------------
    dataset = tf.data.TFRecordDataset(tfrecord)
    data_out = {"data":dataset.map(parse_function)} 
    return data_out

def read_tfrecord(data_in={"tfrecord_folder":'/tfrecord/',"interval":[],"test_size":0.2,"padding":15,"shpx":1,
                           "shpy":1,"shpz":1,"data_type":"float32"}):
    """
    .....................................................................................................................
    # read_tfrecord: Function for reading the training and validation data of the tfrecord
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        data required for selecting the fields for the training process.
        The default is {"tfrecord_train":'/tfrecord/merged_train_data.tfrecord',
                        "tfrecord_vali":'/tfrecord/merged_vali_data.tfrecord',
                        "padding":15,"shpx":1,"shpy":1,"shpz":1,"data_type":"float32"}.
        Data:
            - tfrecord_folder : folder containing tfrecord data
            - interval        : fields used in the training
            - test_size       : percentage used for validation
            - padding         : padding of the field
            - shpx            : shape of the field in the streamwise direction
            - shpy            : shape of the field in the wall-normal direction
            - shpz            : shape of the field in the spanwise direction
            - data_type       : type of the data
  
    Returns
    -------
    dict
        Structure containing the data for the training and the data for the test
        Data:
            - data_train : data for training
            - data_vali  : data for test
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    tfrecord_folder = str(data_in["tfrecord_folder"])
    interval        = np.array(data_in["interval"],dtype="int")
    test_size       = float(data_in["test_size"])
    padding         = int(data_in["padding"])
    shpx            = int(data_in["shpx"])
    shpy            = int(data_in["shpy"])
    shpz            = int(data_in["shpz"])
    data_type       = str(data_in["data_type"])
    if data_type == "float32":
        dtype = tf.float32
    elif data_type == "float16":
        dtype = tf.float16
    else:
        print("Exit the calculation due to invalid data type.",flush=True)
        sys.exit()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Define the function to parse
    # -------------------------------------------------------------------------------------------------------------------
    def parse_function(proto):
        """
        
        .................................................................................................................
        # _parse_function: Function for parsing the data
        .................................................................................................................
    
        Parameters
        ----------
        proto : TFRecords data
            Information to parse.

        Returns
        -------
        feature : TFRecords data
            Parsed features.
        label : TYPE
            Parsed labels.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Define your `features` dictionary here:
        # Adjust shape based on how data was flattened
        # ---------------------------------------------------------------------------------------------------------------
        feature_description = {'feature': tf.io.FixedLenFeature([shpy*(shpz+2*padding)*(shpx+2*padding)*3],dtype),
                               'label': tf.io.FixedLenFeature([shpy*shpz*shpx*3],dtype)}
        parsed_features     = tf.io.parse_single_example(proto,feature_description)
        feature             = tf.reshape(parsed_features['feature'],[shpy,shpz+2*padding,shpx+2*padding,3])
        label               = tf.reshape(parsed_features['label'],[shpy,shpz,shpx,3])
        return feature,label
    
    # -------------------------------------------------------------------------------------------------------------------
    # Check data
    # -------------------------------------------------------------------------------------------------------------------
    check = False
    if check:
        for ind in np.arange(len(interval)):
            index = interval[ind]
            print("-"*100,flush=True)
            print("Checking file: "+str(index),flush=True)
            print("-"*100,flush=True)
            index1  = index
            fileii  = tfrecord_folder+"dataset_"+str(index1)+".tfrecord"
            while True:
                try:
                    filessl = tf.data.Dataset.from_tensor_slices([fileii])
                    dataset = filessl.interleave(tf.data.TFRecordDataset,cycle_length=tf.data.experimental.AUTOTUNE)
                    mapdata = dataset.map(parse_function) 
                    data_tf = list(mapdata.take(1).as_numpy_iterator())[0][0]
                    data_tf_out = list(mapdata.take(1).as_numpy_iterator())[0][1]
                    break
                except:
                    index1 += 1
                    fileii  = tfrecord_folder+"dataset_"+str(index1)+".tfrecord"
                    print("-"*100,flush=True)
                    print("Field corrupted, changing it by:  file: "+fileii,flush=True)
                    print("-"*100,flush=True)
            if index != index1:
                interval[ind] = index1
    
    all_files = sorted([os.path.join(tfrecord_folder, f) 
                        for f in os.listdir(tfrecord_folder) if f.endswith('.tfrecord')])
    
    # -------------------------------------------------------------------------------------------------------------------
    # List all tfrecord files
    # -------------------------------------------------------------------------------------------------------------------
    num_train            = int((1-test_size)*len(interval))
    tfrecord_files_train = [tfrecord_folder+"dataset_"+str(index)+".tfrecord" for index in interval[:num_train]]
    tfrecord_files_vali  = [tfrecord_folder+"dataset_"+str(index)+".tfrecord" for index in interval[num_train:]]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Load the data
    # -------------------------------------------------------------------------------------------------------------------
    data_train = load_dataset(data_in={"tfrecord_files":tfrecord_files_train,"padding":padding,"shpx":shpx,"shpy":shpy,
                                       "shpz":shpz,"data_type":data_type,"index":interval[:num_train]})
    data_vali  = load_dataset(data_in={"tfrecord_files":tfrecord_files_vali,"padding":padding,"shpx":shpx,"shpy":shpy,
                                       "shpz":shpz,"data_type":data_type,"index":interval[num_train:]})
    data_out   = {"data_train":data_train,"data_vali":data_vali}
    return data_out
