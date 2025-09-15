# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:40:22 2024

@author: Andres Cremades Botella

File containing the information about the calculation of the statistics:
    - field_ini   : initial field of the database
    - field_fin   : final field of the database
    - field_delta : separation between the fields of the database
    - save_file   : Flag to save the files of the mean, rms or normalization
    - mean_norm   : Flag to normalize using mean and std
    - file_trj    : file containing the data of Torroja
    - Hmin        : minimum percolation index
    - Hmax        : maximum percolation index
    - Hnum        : number of percolation indices
    
"""
# ----------------------------------------------------------------------------------------------------------------------
# Indices of the initial and final fields used from the database (not used for training)
#     - field_ini   : initial field of the database
#     - field_fin   : final field of the database
#     - field_delta : separation between the fields of the database
# ----------------------------------------------------------------------------------------------------------------------
field_ini   = 26000
field_fin   = 26001
field_delta = 1

# ----------------------------------------------------------------------------------------------------------------------
# Decide if save the statistics
#     - save_file : Flag to save the files of the mean, rms or normalization
#     - mean_norm : Flag to normalize using mean and std
# ----------------------------------------------------------------------------------------------------------------------
save_file = True
mean_norm = False

# ----------------------------------------------------------------------------------------------------------------------
# File of statistics from Torroja database
#     - file_trj : file containing the data of Torroja
# ----------------------------------------------------------------------------------------------------------------------
file_trj = "Re180.prof.txt"

# ----------------------------------------------------------------------------------------------------------------------
# Percolation data
#     - Hmin : minimum percolation index
#     - Hmax : maximum percolation index
#     - Hnum : number of percolation indices
# ----------------------------------------------------------------------------------------------------------------------
Hmin = 1
Hmax = 2
Hnum = 1