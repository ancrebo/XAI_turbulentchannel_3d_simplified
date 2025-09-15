# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:06:28 2024

@author:  Andres Cremades Botella

Data for the training of the model:
    - read_model   : Flag to define or read the model (False=define, True=read)
    - ngpu         : Number of gpus (None to use all the avalaible GPUs, int for a certain number)
    - learat       : Learning ratio
    - optmom       : Momentum of the RMSprop
    - batch_size   : Batch size
    - field_ini    : Initial field of the training
    - field_fin    : Final field of the training     
    - field_delta  : Distance between the fields of the training
    - field_mem    : Number of fields loaded in memory
    - test_size    : Percentage of data for the validation data
    - adapt_batch  : Flag to decide if the fields loaded in the memory need to adapt the batch size. Adapting the
                     batch is required in the case of using multiple GPU. (True: adapt the amount of fields to the
                                                                          batch size, False: use the default data 
                                                                          loaded in memory)
    - prefetch     : Number of batches to load in memory
    - epoch_save   : Number of epoch to trained before saving
    - epoch_max    : Number of maximum epochs of the training
    - nfil         : Number of filters of the first layer of the Unet
    - stride       : Stride of the Unet
    - activation    : Activation function
    - kernel        : Kernel size of the unet
    - pooling       : Size of the poolings of the unet
    - delta_pred    : Number of fields to advance the prediction
    - prep_data     : Flag to choose if the data has already been prepared for the training 
                      (True: the data is prepared in the tensorflow format, 
                       False: the data needs to be read from the flow)
    - flag_central  : Flag for choosing the segmentation distribution (True: CentralStorageStrategy,
                                                                       False: MirroredStrategy)
    - data_type     : Format of the data of the training
    - multi_worker  : Flag to choose multiple worker (True: Multiple worker, False: Single worker)
    - mean_norm     : Flag to normalize using mean and std
    - check         : Flag for checking the data
    - flag_tfrecord : Flag to read the tfrecord file
    - save_fields   : Flag for saving the fields used in the training in a file. If the training is restarted it will
                      use the same fields
"""
# ----------------------------------------------------------------------------------------------------------------------
# Do we need to create a new model or upload a previous one?
#     - read_model  : Flag to define or read the model (False=define, True=read)
# ----------------------------------------------------------------------------------------------------------------------
read_model  = False

# ----------------------------------------------------------------------------------------------------------------------
# Define the system requirements.
#     - ngpu : Number of gpus (None to use all the avalaible GPUs, int for a certain number)
# ----------------------------------------------------------------------------------------------------------------------
ngpu  = None

# ----------------------------------------------------------------------------------------------------------------------
# Options of the training
#     - learat      : Learning ratio
#     - optmom      : Momentum of the RMSprop
#     - batch_size  : Batch size
# ----------------------------------------------------------------------------------------------------------------------
learat      = 5e-5
optmom      = 0.9
batch_size  = 1

# ----------------------------------------------------------------------------------------------------------------------
# Fields used in the training
#     - field_ini   : Initial field of the training
#     - field_fin   : Final field of the training
#     - field_delta : Distance between the fields of the training
#     - field_mem   : Number of fields loaded in memory
#     - test_size   : Percentage of data for the validation data
#     - adapt_batch : Flag to decide if the fields loaded in the memory need to adapt the batch size. Adapting the
#                     batch is required in the case of using multiple GPU. (True: adapt the amount of fields to the
#                     batch size, False: use the default data loaded in memory)
#     - prefetch    : Number of batches to load in memory
# ----------------------------------------------------------------------------------------------------------------------
field_ini   = 26000
field_fin   = 26010
field_delta = 1
field_mem   = 10
test_size   = 0.2
adapt_batch = True
prefetch    = -1

# ----------------------------------------------------------------------------------------------------------------------
# Epoch of the training before saving or updating the data
#     - epoch_save  : Number of epoch to trained before saving
#     - epoch_max   : Number of maximum epochs of the training
# ----------------------------------------------------------------------------------------------------------------------
epoch_save  = 1
epoch_max   = 1

# ----------------------------------------------------------------------------------------------------------------------
# Configuration of the model. Parameters rquired for the model
#     - nfil        : Number of filters of the first layer of the Unet
#     - stride      : Stride of the Unet
#     - activation  : Activation function
#     - kernel      : Kernel size of the unet
#     - pooling     : Size of the poolings of the unet
#     - delta_pred  : Number of fields to advance the prediction
# ----------------------------------------------------------------------------------------------------------------------
nfil        = 1
stride      = 1
activation  = "relu"
kernel      = 3
pooling     = 2
delta_pred  = 1

# ----------------------------------------------------------------------------------------------------------------------
# Decide if the data has been prepared for the training or we need to read it from the flow fields and choose the 
# strategy
#     - prep_data     : Flag to choose if the data has already been prepared for the training (True: the data is 
#                       prepared in the tensorflow format, False: the data needs to be read from the flow)
#     - flag_central  : Flag for choosing the segmentation distribution (True: CentralStorageStrategy,
#                                                                        False: MirroredStrategy)
#     - data_type     : Format of the data of the training
#     - multi_worker  : Flag to choose multiple worker (True: Multiple worker, False: Single worker)
#     - mean_norm     : Flag to normalize using mean and std
#     - check         : Flag for checking the data
#     - flag_tfrecord : Flag to read the tfrecord file
#     - save_fields   : Flag for saving the fields used in the training in a file. If the training is restarted it will
#                       use the same fields
# ----------------------------------------------------------------------------------------------------------------------
prep_data     = False
flag_central  = False
data_type     = "float32"
multi_worker  = False
mean_norm     = False
check         = False
flag_tfrecord = True
save_fields   = True
