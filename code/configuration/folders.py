# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:55:21 2024

@author: Andres Cremades Botella

Folder structure for the problem: folders and files for the flow fields, structures and shaps. Parameters:
    - uvw_folder                : Folder of the velocity data
    - uvw_file                  : This file does not contain the file index
    - uvw_folder_tf             : Folder of the velocity data with tensorflow format
    - uvw_folderii_tf           : File of the velocity data with tensorflow format
    - tfrecord_folder           : Folder to the tfrecord files
    - uvw_folder_tf_ssh         : Folder of the velocity data with tensorflow format in external server
    - uvw_folder_temp           : Temporal folder for the velocity data
    - ssh_flag_train            : flag for selecting external server
    - ssh_server                : server for loading the data
    - ssh_username              : user of the server
    - ssh_password              : password of the server
    - data_folder               : Folder for storing the data of the model
    - umean_file                : File for the mean velocity
    - unorm_file                : File for the normalization of the velocity
    - urms_file                 : File to save the rms of the velocity
    - umax_file                 : File containing the maximum and minimum values of the velocity
    - hist_file                 : File to store the training history
    - error_file                : File to store the error of the predictions
    - urmspred_file             : File to save the rms predicted by the model
    - SHAPmean_file             : File to save the mean SHAP values
    - SHAPrms_file              : File to save the rms of the SHAP values
    - perc_uv_file              : File of the percolation of the uv structures
    - perc_SHAP_file            : File of the percolation of the shap structures
    - file_repetition           : File with data of the repetitions
    - file_snr                  : File with data of the SNR
    - model_folder              : Folder for storing the model
    - model_write               : Name of the model
    - model_read                : Name of a model to read
    - plot_folders              : Folder containing the figures
    - uv_folder                 : folder to store the Reynolds stress value
    - uv_file                   : file to store the Reynolds stress value
    - streak_folder             : folder of the streaks
    - streak_file               : file of the streaks
    - chong_folder              : folder of the chong vortices
    - chong_file                : file of the chong vortices
    - hunt_folder               : folder of the hunt vortices
    - hunt_file                 : file of the hunt vortices
    - SHAPq_folder              : folder to define the SHAP structures
    - SHAPq_file                : file to define the SHAP structures
    - shap_folder               : Folder to store the shap values
    - shap_file                 : File to store the shap values
    - shapseg_uv_folder         : folder to store the shap values for segmented domains using Qs
    - shapseg_uv_file           : file to store the shap values for segmented domains using Qs
    - shapseg_streak_folder     : folder to store the shap values for segmented domains using streaks
    - shapseg_streak_file       : file to store the shap values for segmented domains using streaks
    - shapseg_vortices_folder   : folder to store the shap values for segmented domains using vortices
    - shapseg_vortices_file     : file to store the shap values for segmented domains using vortices
    - uv_shap_file              : file to evaluate the coincidence of uv structures and shap structures
    - streak_shap_file          : file to evaluate the coincidence of streaks and shap structures
    - chong_shap_file           : file to evaluate the coincidence of chong vortices and shap structures
    - hunt_shap_file            : file to evaluate the coincidence of hunt vortices and shap structures
    - calc_coin_tot             : file containing the coincidence between all the structures
    - chong_uv_file             : file to evaluate the coincidence of chong and uv structures
    - streak_uv_file            : file to evaluate the coincidence of streaks and uv structures
    - streak_chong_file         : file to evaluate the coincidence of streaks and chong structures
    - uv_chong_streak_shap_file : file to evaluate the coincidence of Q, chong, streak and shap structures
    - calc_coin_tot_4types      : file containing the coincidence between all the structures using 4 types
    - traintest_index           : file to store the train and test indices
    - uv_chong_streak_file      : file with the coincidence between the Q events, streaks and Chong vortices
"""

# ----------------------------------------------------------------------------------------------------------------------
# Define the folders and files required for the problem
# ----------------------------------------------------------------------------------------------------------------------
# Data for the flow fields
#     - uvw_folder : Folder of the velocity data
#     - uvw_file   : This file does not contain the file index
# ----------------------------------------------------------------------------------------------------------------------
uvw_folder = '../data/phys/'
uvw_file   = 'P125_83pi.$INDEX$.h5.uvw'

# ----------------------------------------------------------------------------------------------------------------------
# Data for the flow fields in the tensorflow format
#     - uvw_folder_tf     : Folder of the velocity data with tensorflow format (not in use)
#     - uvw_folderii_tf   : File of the velocity data with tensorflow format (not in use)
#     - tfrecord_folder   : Folder to the tfrecord files
#     - uvw_folder_tf_ssh : Folder of the velocity data with tensorflow format in external server (not in use)
#     - uvw_folder_temp   : Temporal folder for the velocity data (not in use)
#     - ssh_flag_train    : flag for selecting external server keep False to avoid using (not in use) the not in use 
#                           was set for loading small portions of the database from a different server during training.
#                           this option is not prepared to work in TFRecords properly and is better to avoid it. 
#     - ssh_server        : server for loading the data (not in use)
#     - ssh_username      : user of the server (not in use)
#     - ssh_password      : password of the server (not in use)
# ----------------------------------------------------------------------------------------------------------------------
uvw_folder_tf     = "-"
uvw_folderii_tf   = '-'
tfrecord_folder   = '../data/tfrecord/'
uvw_folder_tf_ssh = uvw_folder_tf
uvw_folder_temp   = "-"
ssh_flag_train    = False
ssh_server        = "-"
ssh_username      = "-"
ssh_password      = "-"

# ----------------------------------------------------------------------------------------------------------------------
# Data generated by the code: statistics, training epochs...
#     - data_folder         : Folder for storing the data of the model
#     - umean_file          : File for the mean velocity
#     - unorm_file          : File for the normalization of the velocity
#     - umax_file           : File containing the maximum and minimum values of the velocity
#     - urms_file           : file to save the rms of the velocity
#     - hist_file           : File to store the training history
#     - error_file          : File to store the error in the predictions
#     - urmspred_file       : File to save the rms predicted by the model
#     - SHAPmean_file       : File to save the mean SHAP values
#     - SHAPrms_file        : File to save the rms of the SHAP values
#     - perc_uv_file        : File of the percolation of the uv structures
#     - perc_SHAP_file      : File of the percolation of the shap structures
#     - file_repetition     : File with data of the repetitions
#     - file_snr            : File with data of the SNR
# ----------------------------------------------------------------------------------------------------------------------
data_folder         = "../results/sta/"
umean_file          = "Umean.txt"
unorm_file          = "norm.txt"
umax_file           = "norm.txt"
urms_file           = "Urms.txt"
hist_file           = "hist.txt"
error_file          = "error.txt"
urmspred_file       = "Urms_pred.txt"
SHAPmean_file       = "SHAPmean.txt"
SHAPrms_file        = "SHAPrms.txt"
perc_uv_file        = "perc_uv.txt"
perc_SHAP_file      = "perc_shap.txt"
file_repetition     = "repetitions_shap.txt"
file_snr            = "repetitions_shap_snr.txt"


# ----------------------------------------------------------------------------------------------------------------------
# Data for the models
#     - model_folder : Folder for storing the model
#     - model_write  : Name of the model
#     - model_read   : Name of a model to load
# ----------------------------------------------------------------------------------------------------------------------
model_folder = "../results/models/"
model_write  = "trained_model.h5"
model_read   = "trained_model.h5"
# ----------------------------------------------------------------------------------------------------------------------
# Data for the plots:
#     - plot_folder : folder to store the plots
# ----------------------------------------------------------------------------------------------------------------------
plot_folder = "../results/plots/"

# ----------------------------------------------------------------------------------------------------------------------
# Data for the uv structures
#     - uv_folder     : folder to store the Reynolds stress value
#     - uv_file       : file to store the Reynolds stress value
#     - streak_folder : folder of the streaks
#     - streak_file   : file of the streaks
#     - chong_folder  : folder of the chong vortices
#     - chong_file    : file of the chong vortices
#     - hunt_folder   : folder of the hunt vortices
#     - hunt_file     : file of the hunt vortices
#     - SHAPq_folder  : folder to define the SHAP structures
#     - SHAPq_file    : file to define the SHAP structures
# ----------------------------------------------------------------------------------------------------------------------
uv_folder        = "../data/Q"
uv_file          = "P125_83pi.$INDEX$.Q"
streak_folder    = "../data/Streak"
streak_file      = "P125_83pi.$INDEX$.Lstreaks"
chong_folder     = "../data/Vortices"
chong_file       = "P125_83pi.$INDEX$.Chong"
hunt_folder      = "-"
hunt_file        = "-"
SHAPq_folder     = "../data/SHAPq"
SHAPq_file       = "P125_83pi_nsample$NSAMPLES$.$INDEX$.struc"

# ----------------------------------------------------------------------------------------------------------------------
# Data for the SHAP values
#     - shap_folder             : folder to store the shap values
#     - shap_file               : file to store the shap values
#     - shapseg_uv_folder       : folder to store the shap values for segmented domains using Qs
#     - shapseg_uv_file         : file to store the shap values for segmented domains using Qs
#     - shapseg_streak_folder   : folder to store the shap values for segmented domains using streaks
#     - shapseg_streak_file     : file to store the shap values for segmented domains using streaks
#     - shapseg_vortices_folder : folder to store the shap values for segmented domains using vortices
#     - shapseg_vortices_file   : file to store the shap values for segmented domains using vortices
# ----------------------------------------------------------------------------------------------------------------------
shap_folder           = "../data/SHAP"
shap_file             = "P125_83pi_nsample$NSAMPLES$.$INDEX$.h5.shap"
shapseg_uv_folder     = "../data/SHAPsegment_uv"
shapseg_uv_file       = "P125_83pi_segment_uv.$INDEX$.h5.shap"
shapseg_chong_folder  = "../data/SHAPsegment_chong"
shapseg_chong_file    = "P125_83pi_segment_chong.$INDEX$.h5.shap"
shapseg_streak_folder = "../data/SHAPsegment_streak"
shapseg_streak_file   = "P125_83pi_segment_streak.$INDEX$.h5.shap"

# ----------------------------------------------------------------------------------------------------------------------
# Coincidence folders
#     - uv_shap_file              : file to evaluate the coincidence of uv structures and shap structures
#     - streak_shap_file          : file to evaluate the coincidence of streaks and shap structures
#     - chong_shap_file           : file to evaluate the coincidence of chong vortices and shap structures
#     - hunt_shap_file            : file to evaluate the coincidence of hunt vortices and shap structures
#     - calc_coin_tot             : file containing the coincidence between all the structures
#     - chong_uv_file             : file to evaluate the coincidence of chong and uv structures
#     - streak_uv_file            : file to evaluate the coincidence of streaks and uv structures
#     - streak_chong_file         : file to evaluate the coincidence of streaks and chong structures
#     - uv_chong_streak_shap_file : file to evaluate the coincidence of Q, chong, streak and shap structures
#     - calc_coin_tot_4types      : file containing the coincidence between all the structures using 4 types
#     - traintest_index           : file to store the train and test indices
#     - uv_chong_streak_file      : file with the coincidence between the Q events, streaks and Chong vortices
# ----------------------------------------------------------------------------------------------------------------------
uv_shap_file              = "uv_shap_coin.txt"
streak_shap_file          = "streak_shap_coin.txt"
chong_shap_file           = "chong_shap_coin.txt"
hunt_shap_file            = "-"
calc_coin_tot             = "shap_uv_streak_chong_hunt_all.txt"
chong_uv_file             = "chong_uv_coin.txt"
streak_uv_file            = "streak_uv_coin.txt"
streak_chong_file         = "streak_chong_coin.txt"
uv_chong_streak_shap_file = "uv_chong_streak_shap_coin.txt"
calc_coin_tot_4types      = "shap_uv_streak_chong_all.txt"
traintest_index           = "traintest_index.txt"
uv_chong_streak_file      = "uv_chong_streak_coin.txt"
