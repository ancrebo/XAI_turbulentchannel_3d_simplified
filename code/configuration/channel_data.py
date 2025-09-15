# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:40:22 2024

@author: Andres Cremades Botella

File containing the information about the channel:
    - L_x       : Channel size in the streamwise direction
    - L_z       : Channel size in the spanwise direction
    - L_y       : Half of the channel width
    - rey       : Friction Reynolds number
    - utau      : Friction velocity
    - dx        : Downsampling in x
    - dy        : Downsampling in y
    - dz        : Downsampling in z
    - padding   : Number of nodes of the padding
    - filvol    : volume filter in viscous units
    
"""
# ----------------------------------------------------------------------------------------------------------------------
# Import the packages
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Define the dimensions of the channel
#     - L_x : Channel size in the streamwise direction
#     - L_z : Channel size in the spanwise direction
#     - L_y : Half of the channel width
# ----------------------------------------------------------------------------------------------------------------------
L_x    = 8*np.pi
L_z    = 3*np.pi
L_y    = 1

# ----------------------------------------------------------------------------------------------------------------------
# Physical parameters
#     - rey  : Friction Reynolds number
#     - utau : Friction velocity
# ----------------------------------------------------------------------------------------------------------------------
rey    = 1.259515993997106e+02
utau   = 0.059976952095100

# ----------------------------------------------------------------------------------------------------------------------
# Downsampling and padding of the fields
#     - dx      : Downsampling in x
#     - dy      : Downsampling in y
#     - dz      : Downsampling in z
#     - padding : Number of nodes of the padding
# ----------------------------------------------------------------------------------------------------------------------
dx      = 1
dy      = 1
dz      = 1
padding = 15

# ----------------------------------------------------------------------------------------------------------------------
# Filter of the volume
#     - filvol : volume filter in viscous units
# ----------------------------------------------------------------------------------------------------------------------
filvol = 2.7e4

