# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
main_CNN.py
-------------------------------------------------------------------------------------------------------------------------
Created on Mon Mar 18 10:12:32 2024

@author: Andres Cremades Botella

File to launch Deep model training. 
The file selects the folder containing the information of the calculation and inside the files used for configuring the
training and the calculation of the SHAP values. The user of this code requires to define the following parameters:
    - folder_def  : (str) name of the folder containing the files for configuring the case of analysis.
    - chd_str     : (str) name of the file containing the data of the channel.
    - folders_str : (str) name of the file containing the folders and files used in the problem.
    - tr_data_str : (str) name of the file containing the information required for the training.
This code has been tested on the multiworker configuration of the cluster Alvis:
    - Alvis general information: https://www.c3se.chalmers.se/about/Alvis/
    - Alvis node availability  : https://scruffy.c3se.chalmers.se/d/alvis-public/alvis-public?orgId=1
The last configuration of the calculation launcher that has been tested in the cluster is provided in the following
lines:
    #!/usr/bin/env bash
    #SBATCH --job-name=MW_A100_2
    #SBATCH -A NAISS2024-5-129 -p alvis
    #SBATCH --nodes 4
    #SBATCH --exclude=alvis[3-4]-[01-09]
    #SBATCH --ntasks-per-node=1
    #SBATCH --gpus-per-node=A100:4
    #SBATCH --cpus-per-task=64
    #SBATCH --hint=nomultithread
    #SBATCH --distribution=block:block
    #SBATCH --time=60:00:00
    #SBATCH --mail-type ALL
    #SBATCH --mail-user andrescb@kth.se
    #SBATCH --output ./logs/trainA100.out
    #SBATCH --error  ./logs/trainA100.error
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    set -x
    cd ../
    module purge
    module load  TensorFlow/2.7.1-foss-2021b-CUDA-11.4.1
    srun python main_CNN_MW.py
"""
# ----------------------------------------------------------------------------------------------------------------------
# Define the names of the files containing the definitios of the parameters
# - folder_def : folder containing the files with the definitions required in the problem
# - chd_str    : file containing the data of the channel
# - folders    : file containing the folder and file structures
# - tr_data    : file containing the data of the training
# ----------------------------------------------------------------------------------------------------------------------
folder_def  = "configuration"
chd_str     = "channel_data"
folders_str = "folders"
tr_data_str = "training_data"

# ----------------------------------------------------------------------------------------------------------------------
# Load the packages
# ----------------------------------------------------------------------------------------------------------------------
import py_bin.py_class.ann_config as ann
import os

# ----------------------------------------------------------------------------------------------------------------------
# Unlock the h5 files for avoiding problems in some clusters
# ----------------------------------------------------------------------------------------------------------------------
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# ----------------------------------------------------------------------------------------------------------------------
# Import information files
# ----------------------------------------------------------------------------------------------------------------------
exec("from "+folder_def+" import "+chd_str+" as chd")
exec("from "+folder_def+" import "+folders_str+" as folders")
exec("from "+folder_def+" import "+tr_data_str+" as tr_data")

# ----------------------------------------------------------------------------------------------------------------------
# Load the channel data to import the information regarding the channel size and the friction Reynolds number 
# and velocity
#     - L_x     : Channel size in the streamwise direction
#     - L_z     : Channel size in the spanwise direction
#     - L_y     : Half of the channel width
#     - rey     : Friction Reynolds number
#     - utau    : Friction velocity
#     - dx      : Downsampling in x
#     - dy      : Downsampling in y
#     - dz      : Downsampling in z
#     - padding : Number of nodes of the padding
# ----------------------------------------------------------------------------------------------------------------------
L_x     = chd.L_x
L_z     = chd.L_z
L_y     = chd.L_y
rey     = chd.rey
utau    = chd.utau
dx      = chd.dx
dy      = chd.dy
dz      = chd.dz
padding = chd.padding

# ----------------------------------------------------------------------------------------------------------------------
# Define the data of the model definition: data, padding, downsampling...
#     - uvw_folder      : Folder of the velocity data
#     - uvw_file        : This file does not contain the file index
#     - data_folder     : Folder for storing the data of the model
#     - umean_file      : File for the mean velocity
#     - unorm_file      : File for the normalization of the velocity
#     - uvw_folder_tf   : Folder of the velocity data with tensorflow format
#     - uvw_folderii_tf : File of the velocity data with tensorflow format
#     - ssh_flag_train  : flag for reading using ssh
#     - ssh_server      : server of the ssh connection
#     - ssh_username    : username of the server
#     - ssh_password    : password of the server
# ----------------------------------------------------------------------------------------------------------------------
uvw_folder          = folders.uvw_folder
uvw_file            = folders.uvw_file
data_folder         = folders.data_folder
umean_file          = folders.umean_file
unorm_file          = folders.unorm_file
uvw_folderii_tf     = folders.uvw_folderii_tf
ssh_flag_train      = bool(folders.ssh_flag_train)
if ssh_flag_train:
    uvw_folder_tf   = folders.uvw_folder_tf_ssh
    uvw_folder_temp = folders.uvw_folder_temp
    ssh_server      = folders.ssh_server
    ssh_username    = folders.ssh_username
    ssh_password    = folders.ssh_password
else:
    uvw_folder_tf   = folders.uvw_folder_tf
    uvw_folder_temp = '-'
    ssh_server      = '-'
    ssh_username    = '-'
    ssh_password    = '-'

# ----------------------------------------------------------------------------------------------------------------------
# Define the data for the training.
#     - ngpu            : Number of gpus
#     - learat          : Learning ratio
#     - optmom          : Momentum of the RMSprop
#     - batch_size      : Batch size
#     - field_ini       : Initial field of the training
#     - field_fin       : Final field of the training
#     - field_delta     : Distance between the fields used in the training
#     - field_mem       : Number of fields loaded in memory
#     - epoch_save      : Number of epoch to trained before saving
#     - epoch_max       : Number of maximum epochs of the training
#     - read_model      : Flag to define or read the model (False=define, True=read)
#     - model_folder    : Folder of the trained model files
#     - model_write     : Name of the trained model file
#     - model_read      : Name of the model file to read
#     - nfil            : Number of filters of the first layer of the Unet
#     - stride          : Stride of the Unet
#     - activation      : Activation function
#     - kernel          : Kernel size of the unet
#     - pooling         : Size of the poolings of the unet
#     - delta_pred      : Number of fields to advance the prediction
#     - hist_file       : File to store the training history
#     - test_size       : Percentage of data for the validation data
#     - flag_central    : Flag for choosing the segmentation distribution (True: CentralStorageStrategy,
#                                                                          False: MirroredStrategy)
#     - data_type       : Format of the data of the training
#     - multi_worker    : Flag to choose multiple worker (True: Multiple worker, False: Single worker)
#     - mean_norm       : Flag to normalize using mean and std
#     - check           : Flag to check the data is correct
#     - tfrecord_folder : Folder to the tfrecord files
#     - flag_tfrecord   : Flag to read the tfrecord file
#     - error_file      : file to store the error
#     - umax_file       : file to store the maximum and minimum velocity
#     - urmspred_file   : file to store the rms predicted by the model 
# ----------------------------------------------------------------------------------------------------------------------
ngpu            = tr_data.ngpu
learat          = tr_data.learat
optmom          = tr_data.optmom
batch_size      = tr_data.batch_size
field_ini       = tr_data.field_ini
field_fin       = tr_data.field_fin
field_delta     = tr_data.field_delta
field_mem       = tr_data.field_mem
epoch_save      = tr_data.epoch_save
epoch_max       = tr_data.epoch_max
read_model      = tr_data.read_model
model_folder    = folders.model_folder
model_write     = folders.model_write
model_read      = folders.model_read
nfil            = tr_data.nfil
stride          = tr_data.stride
activation      = tr_data.activation
kernel          = tr_data.kernel
pooling         = tr_data.pooling
delta_pred      = tr_data.delta_pred
hist_file       = folders.hist_file
test_size       = tr_data.test_size
adapt_batch     = tr_data.adapt_batch
prep_data       = tr_data.prep_data
flag_central    = tr_data.flag_central
data_type       = tr_data.data_type
multi_worker    = tr_data.multi_worker
prefetch        = tr_data.prefetch
mean_norm       = tr_data.mean_norm
check           = tr_data.check
tfrecord_folder = folders.tfrecord_folder
flag_tfrecord   = tr_data.flag_tfrecord
error_file      = folders.error_file
umax_file       = folders.umax_file
urmspred_file   = folders.urmspred_file
save_fields     = tr_data.save_fields
traintest_index = folders.traintest_index

# ----------------------------------------------------------------------------------------------------------------------
# Define dict containing the information needed for the deep model definition and call the deep learning model
# ----------------------------------------------------------------------------------------------------------------------
DL_data  = {"uvw_folder":uvw_folder,"uvw_file":uvw_file,"padding":padding,
            "dx":dx,"dy":dy,"dz":dz,"data_folder":data_folder,"umean_file":umean_file,"unorm_file":unorm_file,
            "L_x":L_x,"L_z":L_z,"L_y":L_y,"uvw_folder_tf":uvw_folder_tf,"uvw_folderii_tf":uvw_folderii_tf,
            "rey":rey,"utau":utau,"ssh_flag_train":ssh_flag_train,
            "uvw_folder_temp":folders.uvw_folder_temp,"ssh_server":folders.ssh_server,
            "ssh_username":folders.ssh_username,"ssh_password":folders.ssh_password,"error_file":error_file,
            "umax_file":umax_file,"urmspred_file":urmspred_file}
Unet = ann.deep_model(DL_data)


# ----------------------------------------------------------------------------------------------------------------------
# Define dict containing the information needed for the training, define and train the model
# ----------------------------------------------------------------------------------------------------------------------
Training_data = {"ngpu":ngpu,"learat":learat,"optmom":optmom,"batch_size":batch_size,"field_ini":field_ini,
                 "field_fin":field_fin,"field_delta":field_delta,"field_mem":field_mem,"epoch_save":epoch_save,
                 "epoch_max":epoch_max,"read_model":read_model,"model_folder":model_folder,"model_write":model_write,
                 "model_read":model_read,"nfil":nfil,"stride":stride,"activation":activation,"kernel":kernel,
                 "pooling":pooling,"delta_pred":delta_pred,"hist_file":hist_file,"test_size":test_size,
                 "adapt_batch":adapt_batch,"prep_data":prep_data,"flag_model":True,"flag_central":flag_central,
                 "data_type":data_type,"multi_worker":multi_worker,"prefetch":prefetch,"mean_norm":mean_norm,
                 "check":check,"tfrecord_folder":tfrecord_folder,"flag_tfrecord":flag_tfrecord,
                 "save_fields":save_fields,"traintest_index":traintest_index}
Unet.define_model(Training_data)

# ----------------------------------------------------------------------------------------------------------------------
# Train the model
# ----------------------------------------------------------------------------------------------------------------------
Unet.train_model()
