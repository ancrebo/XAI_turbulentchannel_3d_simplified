# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
CNNblock_definition.py
-------------------------------------------------------------------------------------------------------------------------
Created on Thu Mar 21 12:55:17 2024

@author: Andres Cremades Botella

File for defining the blocks of the DL model. The file contains the following functions:
    Functions:
        - block    : function for defining the convolutional block: CNN+BN+Activation
        - invblock : function for defining the inverse convolutional block: CNN transpose+BN+Activation
"""

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define functions
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------


def block(data_in={"input":[],"nfil":16,"stride":1,"activ":"relu","kernel":3,"dtype":None}):
    """
    .....................................................................................................................
    # block: function for defining the convolutional block: CNN+BN+Activation.
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data for defining the block.
        The default is {"xx":[],"nfil":16,"stride":1,"activ":"relu","kernel":3,dtype:None}.
        Data:
            - input  : tensor with the input of the layer
            - nfil   : number of filters of the layer
            - stride : stride of the layer
            - activ  : activation function of the layer
            - kernel : kernel used for the convolution
            - dtype  : type of the output data, if none do not specify (float32,float16)

    Returns
    -------
    data_out : dict
        Data of the output of the block.
        Data:
            - output : tensor with the output of the layer

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Import the packages
    # -------------------------------------------------------------------------------------------------------------------
    from tensorflow.keras.layers import Conv3D, BatchNormalization, Activation
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    xx     = data_in["input"]
    nfil   = int(data_in["nfil"])
    stride = int(data_in["stride"])
    activ  = str(data_in["activ"])
    kern   = int(data_in["kernel"])
    kernel = (int(kern),int(kern),int(kern)) 
    dtype  = data_in["dtype"]
    flagt  = False
    if dtype is not None:
        if dtype != "float32" or dtype != "float16":
            dtype = "float32"
    else:
        flagt = True
    
    # -------------------------------------------------------------------------------------------------------------------
    # Define the block: Convolution+Batch Normalization+Activation
    # -------------------------------------------------------------------------------------------------------------------
    if flagt:
        xx = Conv3D(nfil, kernel_size=kernel,strides=(stride,stride,stride),padding="same")(xx)
        xx = BatchNormalization()(xx) 
        xx = Activation(activ)(xx)
    else:
        xx = Conv3D(nfil, kernel_size=kernel,strides=(stride,stride,stride),padding="same",dtype=dtype)(xx)
        xx = BatchNormalization(dtype=dtype)(xx)
        xx = Activation(activ,dtype=dtype)(xx)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Define the output
    # -------------------------------------------------------------------------------------------------------------------
    data_out = {"output":xx}
    return data_out

def invblock(data_in={"input":[],"nfil":16,"stride":1,"activ":"relu","kernel":3}):#,"outpad":(0,0,0)}):
    """
    .....................................................................................................................
    # invblock: function for defining the inverse convolutional block: CNN transpose+BN+Activation.
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        Data for defining the block.
        The default is {"input":[],"nfil":16,"stride":1,"activ":"relu","kernel":3,"outpad":(0,0,0)}.
        Data:
            - input  : tensor with the input of the layer
            - nfil   : number of filters of the layer
            - stride : stride of the layer
            - activ  : activation function of the layer
            - kernel : kernel used for the convolution
            - outpad : padding to adjust the size of the output

    Returns
    -------
    data_out : dict
        Data of the output of the block.
        Data:
            - output : tensor with the output of the layer

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read packages
    # -------------------------------------------------------------------------------------------------------------------
    from tensorflow.keras.layers import Conv3DTranspose, BatchNormalization, Activation
    
    # -------------------------------------------------------------------------------------------------------------------
    # Read data
    # -------------------------------------------------------------------------------------------------------------------
    xx     = data_in["input"]
    nfil   = int(data_in["nfil"])
    stride = int(data_in["stride"])
    activ  = str(data_in["activ"])
    kern   = int(data_in["kernel"])
    kernel = (int(kern),int(kern),int(kern)) 
    # outpad = data_in["outpad"]
    
    # -------------------------------------------------------------------------------------------------------------------
    # Define the block: Transposed Convolution+Batch Normalization+Activation Fuction
    # -------------------------------------------------------------------------------------------------------------------
    xx = Conv3DTranspose(nfil, kernel_size=kernel,strides=(stride,stride,stride),padding="valid")(xx)#,
                         # output_padding=outpad)(xx)
    xx = BatchNormalization()(xx) 
    xx = Activation(activ)(xx)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Define the output
    # -------------------------------------------------------------------------------------------------------------------
    data_out = {"output":xx}
    return data_out