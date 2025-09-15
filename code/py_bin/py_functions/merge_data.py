"""
-------------------------------------------------------------------------------------------------------------------------
merge_data.py
-------------------------------------------------------------------------------------------------------------------------
Created on Wed May 15 11:58:23 2024

@author: Andres Cremades Botella - Inherited from Xuan Gu (Berzelius support)

File containing the function to improve the reading of the data in Berzelius. The file contains the following functions:
    Functions:
        - load_datasets : function for loading the data
        - read_data_tf           : function to read the data with the tensorflow format
        - read_inout_notprepared : function to read the data for the training and the test directly from the flow
                                   files
        - data_traintest_tf      : function to separe training and test data from the flow files and convert them 
                                   into the tensorflow format
"""
# -----------------------------------------------------------------------------------------------------------------------
# Import packages
# -----------------------------------------------------------------------------------------------------------------------
import sys
import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm

def load_datasets(data_in={"file":'/tfrecord/merged_vali_data/dataset_0000.tfrecord',
                           "elem_spec":[]}):
    """
    .....................................................................................................................
    # load_datasets: Function for reading the data
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        data required for selecting the fields for the training process.
        The default is {"file_path":'/tfrecord/merged_vali_data/dataset_0000.tfrecord',
                        "elem_spec":[],"version_tf":[2,10,1]}.
        Data:
            - file       : path to the file to save
            - elem_spec  : specification of the format of the elements
            - version_tf : version of the tensorflow
    
    Returns
    -------
    dict
        Structure containing the data for the training and the data for the test
        Data:
            - loaddata : data loaded
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    file       = str(data_in["file"])
    elem_spec  = data_in["elem_spec"]
    version_tf = np.array(tf.__version__.split('.'),dtype="int")
    
    # -------------------------------------------------------------------------------------------------------------------
    # Load the data
    # -------------------------------------------------------------------------------------------------------------------
    if version_tf[0] >= 2 and version_tf[1] >= 10 and version_tf[2] >= 0:
        loaddata = tf.data.Dataset.load(file,element_spec=elem_spec)
    else:
        loaddata = tf.data.experimental.load(file,element_spec=elem_spec)
    data_out = {"loaddata":loaddata}
    return data_out

def serialize_example(data_in={"feature":[],"label":[]}):
    """
    .....................................................................................................................
    # _serialize_example: Create a tf.train.Example message ready to be written to a file.
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        data containing the input information.
        The default is {"feature":[],"label":[]}.
        Data:
            - feature : features of the tfrecord
            - label   : name of the tfrecord
    
    Returns
    -------
    dict
        Structure containing mapped example
        Data:
            - example : Structure containing mapped example
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    feature = data_in["feature"]
    label   = data_in["label"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
    # -------------------------------------------------------------------------------------------------------------------
    feature = {'feature': tf.train.Feature(float_list=tf.train.FloatList(value=feature.flatten())),
               'label': tf.train.Feature(float_list=tf.train.FloatList(value=label.flatten()))}

    # -------------------------------------------------------------------------------------------------------------------
    # Create a Features message using tf.train.Example.
    # -------------------------------------------------------------------------------------------------------------------
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    data_out      = {"example":example_proto.SerializeToString()}
    return data_out

def write_tfrecord(data_in={"output_path":'/tfrecord/merged_vali_data/dataset_0000.tfrecord',"dataset":[]}):
    """
    ---------------------------------------------------------------------------------------------------------------------
    # write_tfrecord : Writes dataset to a TFRecord file.
    ---------------------------------------------------------------------------------------------------------------------
    Parameters
    ----------
    data_in : TYPE, optional
        DESCRIPTION. The default is {"file_path":'/tfrecord/merged_vali_data/dataset_0000.tfrecord',
                                     "dataset":[]}.
        Data:
            - file      : path to the file to write
            - dataset   : Data to save

    Returns
    -------
    None.

    """
    output_path = str(data_in["output_path"])
    dataset     = data_in["dataset"]
    with tf.io.TFRecordWriter(output_path) as writer:
        for features, labels in tqdm(dataset, desc=f"Writing {output_path}"):
            data_serialize = {"feature":features.numpy(),"label":labels.numpy()}
            example = serialize_example(data_in=data_serialize)["example"]
            writer.write(example)
    

def create_tfrecords(data_in={"output_directory":'/tfrecord/merged_vali_data/',
                              "base_directory":"../../P125_21pi_vu_tf_float32/",
                              "datasets":[],"elem_spec":[]}):
    """
    ---------------------------------------------------------------------------------------------------------------------
    # create_tfrecords : Function to create the tfrecords.
    ---------------------------------------------------------------------------------------------------------------------
    Parameters
    ----------
    data_in : TYPE, optional
        DESCRIPTION. The default is {"output_directory":'/tfrecord/merged_vali_data/',
                                     "base_directory":"../../P125_21pi_vu_tf_float32/",
                                     "datasets":[],"elem_spec":[]}.

    Returns
    -------
    None.

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    output_directory = str(data_in["output_directory"])
    base_directory   = str(data_in["base_directory"])
    datasets         = np.array(data_in["datasets"],dtype="int")
    elem_spec        = data_in["elem_spec"]
    num_datasets     = len(datasets)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Get the list of dataset files
    # -------------------------------------------------------------------------------------------------------------------
    dataset_paths = [base_directory.replace("$INDEX$",str(index)) for index in datasets] 
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create output directory if it does not exist
    # -------------------------------------------------------------------------------------------------------------------
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Process each dataset file one by one
    # -------------------------------------------------------------------------------------------------------------------
    for idx, file in enumerate(tqdm(dataset_paths, desc="Processing datasets")):
        dataset_in  = {"file":file,"elem_spec":elem_spec}
        index       = int(file.split('.')[-1])
        dataset     = load_datasets(dataset_in)["loaddata"]
        output_path = os.path.join(output_directory, f'dataset_{index}.tfrecord')
        data_write  = {"output_path":output_path,"dataset":dataset}
        write_tfrecord(data_in=data_write)
        print(f"Dataset {idx} written to {output_path}",flush=True)
    

def merge_data(data_in={"base_directory":"../../P125_21pi_vu/","padding":15,"shpx":1,"shpy":1,
                        "shpz":1,"data_type":"float32","datasets":[],
                        "output_path":'/tfrecord/merged_vali_data/'}):
    """
    .....................................................................................................................
    # merge_data: Function for reading the training and validation data in the tensorflow format for proper usage of 
                  Berzelius.
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        data required for selecting the fields for the training process.
        The default is {"base_directory":"../../P125_21pi_vu/","padding":15,"shpx":1,"shpy":1,"shpz":1,
                        "data_type":"float32","datasets":[],"output_path":'/tfrecord/merged_vali_data.tfrecord'}.
        Data:
            - base_directory : folder in which the data is stored
            - padding        : padding of the fields
            - shpx           : shape in the streamwise direction
            - shpy           : shape in the wall-normal direction
            - shpz           : shape in the spanwise direction
            - data_type      : type of data of the folder
            - datasets       : number of fields to read
            - output_path    : path to store the file
  
    Returns
    -------
    None
    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    base_directory = str(data_in["base_directory"])
    padding        = int(data_in["padding"])
    shpx           = int(data_in["shpx"])
    shpy           = int(data_in["shpy"])
    shpz           = int(data_in["shpz"])
    data_type      = str(data_in["data_type"])
    datasets       = np.array(data_in["datasets"],dtype='int')
    output_path    = str(data_in["output_path"])
    if data_type == "float32":
        dtype = tf.float32
    elif data_type == "float16":
        dtype = tf.float16
    else:
        print("Exit the calculation due to invalid data type.",flush=True)
        sys.exit()
    
    # -------------------------------------------------------------------------------------------------------------------
    # Define your element spec here based on the data structure
    # -------------------------------------------------------------------------------------------------------------------
    elem_spec = (tf.TensorSpec(shape=(shpy,shpz+2*padding,shpx+2*padding,3),dtype=dtype),
                 tf.TensorSpec(shape=(shpy,shpz,shpx,3),dtype=dtype))
    
    # -------------------------------------------------------------------------------------------------------------------
    # Create the tfrecords
    # -------------------------------------------------------------------------------------------------------------------
    data_tfrecords = {"output_directory":output_path,"base_directory":base_directory,"datasets":datasets,
                      "elem_spec":elem_spec}
    create_tfrecords(data_in=data_tfrecords)



