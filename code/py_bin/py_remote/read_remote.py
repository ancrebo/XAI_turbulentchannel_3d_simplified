# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
urms.py
-------------------------------------------------------------------------------------------------------------------------
Created on Wed Apr 24 13:46:59 2024

@author: andre

File containing the functions used for reading the remote files in a ssh server.
Functions:
    - recursivedelete  : Function to remove the files in a folder
    - _get_r_portable  : Get files in a folder from ssh server recursively.
    - read_from_server : Function for reading the files from a server
"""

import paramiko
from paramiko import SSHClient
import os
from stat import S_ISDIR, S_ISREG
import glob


def recursivedelete(localdir):
    """
    .....................................................................................................................
    # recursivedelete: Function to remove the files in a folder
    .....................................................................................................................
    Parameters
    ----------
    localdir : list
        DESCRIPTION. directory to clean

    Returns
    -------
    None.

    """
    files_folder = glob.glob(localdir+'/*')
    for file in files_folder:
        isdir = os.path.isdir(file)
        if isdir:
            recursivedelete(file)
            os.rmdir(file)
        else:
            os.remove(file)

def _get_r_portable(sftp, remotedir, localdir):
    """
    .....................................................................................................................
    # _get_r_portable: Get files in a folder from ssh server recursively. 
                       Taken from https://stackoverflow.com/questions/6674862/recursive-directory-download-with-paramiko
    .....................................................................................................................
    Parameters
    ----------
    sftp : SFTPClient session object
        DESCRIPTION. SFTP session on the SSH server
    remotedir : str
        DESCRIPTION. Remote directory of the folder
    localdir : TYPE
        DESCRIPTION. Local directory of the folder

    Returns
    -------
    None.

    """
    for entry in sftp.listdir_attr(remotedir):
        remotepath = remotedir + "/" + entry.filename
        localpath = os.path.join(localdir, entry.filename)
        mode = entry.st_mode
        if S_ISDIR(mode):
            try:
                os.mkdir(localpath)
            except OSError:     
                pass
            _get_r_portable(sftp, remotepath, localpath)
        elif S_ISREG(mode):
            sftp.get(remotepath, localpath)

def read_from_server(data_in={"remotedir":"/P125_21pi_vu_tf_float32/P125_21pi_vu.1000",
                              "localdir":"tmpdata","server":"slogan.mech.kth.se","username":"andrescb",
                              "password":"***"}):
    """
    .....................................................................................................................
    # read_from_server: Function for reading the files from a server
    .....................................................................................................................
    Parameters
    ----------
    data_in : dict, optional
        DESCRIPTION. The default is {"remotedir":"/P125_21pi_vu_tf_float32/P125_21pi_vu.1000",
                                     "localdir":"tmpdata","server":"slogan.mech.kth.se","username":"andrescb", 
                                     "password":"***"}.
        Data:
            - remotedir : remote directory to read the files
            - localdir  : local directory to temporally store the files
            - server    : server to read files
            - username  : user of the server
            - password  : password to the user account

    Returns
    -------
    None.

    """
    # -------------------------------------------------------------------------------------------------------------------
    # Read the data
    # -------------------------------------------------------------------------------------------------------------------
    remotedir = str(data_in["remotedir"])
    localdir  = str(data_in["localdir"])
    server    = str(data_in["server"])
    username  = str(data_in["username"])
    password  = str(data_in["password"])
    
    # -------------------------------------------------------------------------------------------------------------------
    # Define the client and start the connection
    # -------------------------------------------------------------------------------------------------------------------
    client = SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server,username=username,password=password)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Open ftp and read in temporal folder
    # -------------------------------------------------------------------------------------------------------------------
    localdir.replace('\\','/')
    localdir_split = localdir.split("/")
    concatfold     = str('')
    for fold in localdir_split:
        concatfold = concatfold+fold
        try:
            os.mkdir(concatfold)
        except:
            pass
        concatfold = concatfold+'/'
    ftp_client = client.open_sftp()
    _get_r_portable(ftp_client,remotedir,localdir)
    
    # -------------------------------------------------------------------------------------------------------------------
    # Close the clients
    # -------------------------------------------------------------------------------------------------------------------
    ftp_client.close()
    client.close()

