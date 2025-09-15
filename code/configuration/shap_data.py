# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 10:06:28 2024

@author:  Andres Cremades Botella

Data for the training of the model:
    - field_ini    : Initial field of the training
    - field_fin    : Final field of the training
    - field_delta  : Separation between files
    - nsamples     : number of samples used for the calculation of the SHAP values in the Expected Gradients
    - nsamples_max : maximum number of samples loaded in memory for calculating the SHAPs
    - nrep_field   : number of repetitions of the fields for calculating the shap values. The SHAPs are averaged along
                     the repetitions
    - shap_batch   : batch size used for the gradient SHAPs
    - repeat exist : flag for repeating an existing file (True: recalculate, False: skip)
"""
# ----------------------------------------------------------------------------------------------------------------------
# Fields used in the training
#     - field_ini   : Initial field of the training
#     - field_fin   : Final field of the training
#     - field_delta : Separation between files
# ----------------------------------------------------------------------------------------------------------------------
field_ini   = 26000
field_fin   = 26001 
field_delta = 1

# ----------------------------------------------------------------------------------------------------------------------
# Select the number of samples for calculating the SHAP values
#     - nsamples     : number of samples used for the calculation of the SHAP values in the Expected Gradients
#     - nsamples_max : maximum number of samples loaded in memory for calculating the SHAPs
#     - nrep_field   : number of repetitions of the field
#     - shap_batch   : batch size used for the gradient SHAPs
#     - repeat exist : flag for repeating an existing file (True: recalculate, False: skip)
# ----------------------------------------------------------------------------------------------------------------------
nsamples     = 200
nsamples_max = 0
nrep_field   = 0
shap_batch   = 1
repeat_exist = False
