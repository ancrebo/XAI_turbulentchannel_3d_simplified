# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------------------------------------------------
structures.py
-------------------------------------------------------------------------------------------------------------------------
Created on Wed Apr  3 12:01:42 2024

@author: Andres Cremades Botella

File to define the coherent structures. The file contains a class for the coherent structures:
    Class:
        - structures : Class of the coherent structures.
"""
# -----------------------------------------------------------------------------------------------------------------------
# Import packages for all functions
# -----------------------------------------------------------------------------------------------------------------------
import numpy as np

# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------
# Define functions
# -----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------

class structures():
    """
    .....................................................................................................................
    # structures: Class containg the information of the coherent structures. The structure has the following functions:
        * Functions:
            - __init__                      : initialization of the class
            - separate_structures           : function for obtaining the nodes of each structure
            - physicalproperties_structures : function for defining the physical properties of the coherent structures
            - detect_quadrant               : function for detecting the quadrant of the structures
            - segmentation                  : function to generate a segmentation mask according with the structures
            - structure_u1u2                : function to calculate the product of the field in the dimensions 1 and 2
                                              of the flow field for each structure
            - structure_shap                : function for calculating the total SHAP of the structure
        * Variables:
            - mat_struc            : matrix of the grid-points contained in a structure (1 if contained, 
                                                                                         0 if not contained)
            - field_1              : field in the direction 1
            - field_2              : field in the direction 2
            - flag_sign            : flag to separe the structures depending of the sign of the fields
            - uvw_folder           : folder of the flow fields
            - uvw_file             : file of the flow fields
            - dx                   : downsampling in the streamwise direction
            - dy                   : downsampling in the wall-normal direction
            - dz                   : downsampling in the spanwise direction
            - L_x                  : size of the channel in the streamwise direction
            - L_y                  : size of the channel in the wall-normal direction
            - L_z                  : size of the channel in the spanwise direction
            - rey                  : friction Reynolds number
            - utau                 : friction velocity
            - sym_quad             : flag for defining if the quadrant classification needs to be done by making a 
                                     simetry in the second direction
            - filvol               : volume for filtering the structures
            - shap_folder          : folder of the SHAP values
            - shap_file            : file of the SHAP values
            - y_h_plus             : grid of the channel in the wall-normal direction
            - grid_dx_plus         : size of the mesh elements in the streamwise direction
            - grid_dz_plus         : size of the mesh elements in the spanwise direction
            - grid_vol_plus        : volume of the grid elements
            - shpx                 : shape of the tensors in the streamwise direction
            - shpy                 : shape of the tensors in the wall-normal direction
            - shpz                 : shape of the tensors in the spanwise direction
            - nodes                : list containing the grid-points of each coherent structure
            - dim_x                : size of the structure in the streamwise direction
            - dim_y                : size of the structure in the wall-normal direction
            - dim_z                : size of the structure in the spanwise direction
            - ymin                 : minimum wall distance of the structure
            - ymax                 : maximum wall distance of the structure
            - boxvol               : volume of the box containing the structure
            - vol                  : volume of the structure
            - cg_x                 : position of the center of gravity in x
            - cg_z                 : position of the center of gravity in z
            - cg_y                 : position of the center of gravity in y
            - cg_xbox              : position of the center of gravity of the box in x
            - cg_zbox              : position of the center of gravity of the box in z
            - cg_ybox              : position of the center of gravity of the box in y
            - mat_event            : matrix of the events of each nodes
            - event                : event of each structure
            - mat_segment          : matrix containing the segmentation of the domain
            - mat_segment_filtered : matrix containing the segementation of the domain after filtering smaller
                                     structures
            - filtstr_sum          : total percentage of structures that have been filtered
    .....................................................................................................................
    """
    def __init__(self,data_in={"mat_struc":[],"field_1":[],"field_2":[],"field_3":[],"flag_sign":True,
                               "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                               "dx":1,"dy":1,"dz":1,"L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,
                               "utau":0.060523258443963,"sym_quad":True,"filvol":2.7e4}):
        """
        .................................................................................................................
        # __init__
        .................................................................................................................
        Function to initialize the structure class

        Parameters
        ----------
        data_in : dict, optional
            Data for creating the structues. 
            The default is {"mat_struc":[],"field_1":[],"field_2":[],"field_3":[],"flag_sign":True,
                            "uvw_folder":"../../P125_21pi_vu/","uvw_file":"P125_21pi_vu.$INDEX$.h5.uvw",
                            "dx":1,"dy":1,"dz":1,"L_x":2*np.pi,"L_y":1,"L_z":np.pi,"rey":125,
                            "utau":0.060523258443963,"sym_quad":True,"filvol":2.7e4,
                            "shap_folder":"../../P125_21pi_vu_SHAP_UnetXAI_gradient/",
                            "shap_file":"P125_21pi_vu.$INDEX$.h5.shap","padding":15,"data_type":"float32"}.
            Data:
                - mat_struc   : matrix of the grid-points contained in a structure (1 if contained, 0 if not contained)
                - field_1     : field in the direction 1
                - field_2     : field in the direction 2
                - field_3     : field in the direction 3
                - flag_sign   : flag to separe the structures depending of the sign of the fields
                - uvw_folder  : folder of the flow fields
                - uvw_file    : file of the flow fields
                - dx          : downsampling in the streamwise direction
                - dy          : downsampling in the wall-normal direction
                - dz          : downsampling in the spanwise direction
                - L_x         : size of the channel in the streamwise direction
                - L_y         : size of the channel in the wall-normal direction
                - L_z         : size of the channel in the spanwise direction
                - rey         : friction Reynolds number
                - utau        : friction velocity
                - sym_quad    : flag for defining if the quadrant classification needs to be done by making a simetry 
                                in the second direction
                - filvol      : volume for filtering the structures

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Import packages
        # ---------------------------------------------------------------------------------------------------------------
        from py_bin.py_class.flow_field import flow_field
        
        # ---------------------------------------------------------------------------------------------------------------
        # Read data
        # ---------------------------------------------------------------------------------------------------------------
        self.mat_struc   = np.array(data_in["mat_struc"],dtype='bool')       # matrix of the grid-points of the 
                                                                             # structures
        self.field_1     = np.array(data_in["field_1"])                      # component 1 of the field
        self.field_2     = np.array(data_in["field_2"])                      # component 2 of the field
        self.field_3     = np.array(data_in["field_3"])                      # component 3 of the field
        self.flag_sign   = bool(data_in["flag_sign"])                        # flag to separe the structures with the 
                                                                             # sign
        if self.flag_sign:
            self.sign_1  = np.array(np.sign(data_in["field_1"]),dtype='int') # sign of the field in the component 1
            self.sign_2  = np.array(np.sign(data_in["field_2"]),dtype='int') # sign of the field in the component 2
        self.uvw_folder  = str(data_in["uvw_folder"])                        # folder of the flow fields
        self.uvw_file    = str(data_in["uvw_file"])                          # file of the flow fields
        self.down_x      = int(data_in["dx"])                                # downsampling in x
        self.down_y      = int(data_in["dy"])                                # downsampling in y
        self.down_z      = int(data_in["dz"])                                # downsampling in z
        self.L_x         = float(data_in["L_x"])                             # size in x
        self.L_y         = float(data_in["L_y"])                             # size in y
        self.L_z         = float(data_in["L_z"])                             # size in z
        self.rey         = float(data_in["rey"])                             # friction reynolds number
        self.utau        = float(data_in["utau"])                            # friction velocity
        self.sym_quad    = bool(data_in["sym_quad"])                         # flag for the symmetry in quadrant 
                                                                             # selection
        self.filvol      = float(data_in["filvol"])                          # volume for filtering the structures
            
        # ---------------------------------------------------------------------------------------------------------------
        # Evaluate the properties of the channel
        # ---------------------------------------------------------------------------------------------------------------
        Data_flow = {"folder":self.uvw_folder,"file":self.uvw_file,"down_x":self.down_x,"down_y":self.down_y,
                     "down_z":self.down_z,"L_x":self.L_x,"L_y":self.L_y,"L_z":self.L_z,"rey":self.rey,"utau":self.utau}
        flowfield = flow_field(data_in=Data_flow)
        flowfield.shape_tensor()
        flowfield.flow_grid()
        
        # ---------------------------------------------------------------------------------------------------------------
        # Add the geometrical information of the channel to the structure class
        # ---------------------------------------------------------------------------------------------------------------
        self.y_h_plus      = flowfield.y_h_plus     # wall-normal dimension
        self.grid_dx_plus  = flowfield.delta_x_plus # size of the mesh elements in the streamwise direction
        self.grid_dz_plus  = flowfield.delta_z_plus # size of the mesh elements in the spanwise direction
        self.grid_vol_plus = flowfield.vol_plus     # volume of the grid elements
        self.shpx          = flowfield.shpx         # shape of the tensors in the streamwise direction
        self.shpy          = flowfield.shpy         # shape of the tensors in the wall-normal direction
        self.shpz          = flowfield.shpz         # shape of the tensors in the spanwise direction
        
    def separate_structures(self):
        """
        .................................................................................................................
        # separate_structures
        .................................................................................................................
        Function to separate the different coherent structures.

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Define the size of the matrix
        #   - ny : number of points in y direction
        #   - nz : number of points in z direction
        #   - nx : number of points in x direction 
        # ---------------------------------------------------------------------------------------------------------------
        ny,nz,nx        = self.mat_struc.shape
        mat_struc_copy  = self.mat_struc.copy()
        copy2           = mat_struc_copy.copy()
        
        # ---------------------------------------------------------------------------------------------------------------
        # Create a matrix to evaluate the connectivity-1 in all the directions
        # ---------------------------------------------------------------------------------------------------------------
        dirs = np.array([[-1,0,0],[0,-1,0],[0,0,-1],[1,0,0],[0,1,0],[0,0,1]])
        
        # ---------------------------------------------------------------------------------------------------------------
        # Calculate the connectivity
        # - list_waiting : list with the nodes that have been detected in the structure and their connectivity has to
        #                  be evaluated
        # - self.nodes   : list of the nodes contained in each structure
        # ---------------------------------------------------------------------------------------------------------------
        list_waiting    = []
        self.nodes      = []
        
        # ---------------------------------------------------------------------------------------------------------------
        # Check all the points in the domain discarding those with absence of structure
        # For the points containing the structure check all the adjacent nodes
        # Apply simetries and avoid walls
        # Repeat for the connected nodes
        # ---------------------------------------------------------------------------------------------------------------
        for ind_y in np.arange(ny):
            for ind_z in np.arange(nz):
                for ind_x in np.arange(nx):
                    if mat_struc_copy[ind_y,ind_z,ind_x] == 0:
                        continue
                    else:
                        # -----------------------------------------------------------------------------------------------
                        # Add the index of the nodes
                        #   - list_struc : list of nodes of a structure
                        # Delete the value of the point in the reference matrix (avoid calculating it twice)
                        #   - First      : flag to specify if it is the first node of a structure
                        # -----------------------------------------------------------------------------------------------
                        ind_yzx                           = [ind_y,ind_z,ind_x]
                        list_struc                        = [np.array(ind_yzx,dtype='int')]
                        mat_struc_copy[ind_y,ind_z,ind_x] = 0
                        first                             = True
                        # -----------------------------------------------------------------------------------------------
                        # If the node is the first of the structure or is connected to the structure,
                        # check the adjacent nodes
                        # -----------------------------------------------------------------------------------------------
                        while len(list_waiting)>0 or first:
                            if first:
                                # ---------------------------------------------------------------------------------------
                                # Specify that the first node of the structure has been used
                                # ---------------------------------------------------------------------------------------
                                first = False
                                if self.flag_sign:
                                    signini_1 = self.sign_1[ind_yzx[0],ind_yzx[1],ind_yzx[2]]
                                    signini_2 = self.sign_2[ind_yzx[0],ind_yzx[1],ind_yzx[2]]
                            else:
                                # ---------------------------------------------------------------------------------------
                                # Take the first node of the waiting list and remove it from the list
                                # ---------------------------------------------------------------------------------------
                                ind_yzx      = list_waiting[0]
                                list_waiting = list_waiting[1:]
                            # -------------------------------------------------------------------------------------------
                            # Define all the possible nodes in contact and apply symmetries to those nodes
                            #   - if the direction z is in the node -1 or nz switch to nz-1 or 0
                            #   - if the direction x is in the node -1 or nx switch to nx-1 or 0
                            # -------------------------------------------------------------------------------------------
                            dir_ind                     = ind_yzx+dirs
                            dir_ind[dir_ind[:,1]==-1,1] = nz-1
                            dir_ind[dir_ind[:,2]==-1,2] = nx-1
                            dir_ind[dir_ind[:,1]==nz,1] = 0
                            dir_ind[dir_ind[:,2]==nx,2] = 0
                            for dir_ii in dir_ind:
                                # ---------------------------------------------------------------------------------------
                                # Avoid the position of the walls
                                # ---------------------------------------------------------------------------------------
                                if dir_ii[0] > -1 and dir_ii[0] < ny:
                                    # -----------------------------------------------------------------------------------
                                    # If the connected point is a structure, save it in list_struc and remove it from
                                    # the reference matrix
                                    # Add the node to the waiting list to evaluate the connections from its position
                                    # -----------------------------------------------------------------------------------
                                    if mat_struc_copy[dir_ii[0],dir_ii[1],dir_ii[2]] == 1:
                                        # -------------------------------------------------------------------------------
                                        # If the structure has to be divided by the signs of the velocity check the 
                                        # signs
                                        # -------------------------------------------------------------------------------
                                        if self.flag_sign:
                                            sign1 = self.sign_1[dir_ii[0],dir_ii[1],dir_ii[2]]
                                            sign2 = self.sign_2[dir_ii[0],dir_ii[1],dir_ii[2]]
                                            if signini_1 == sign1 and signini_2 == sign2:
                                                flag_connect = True
                                            else:
                                                flag_connect = False
                                        else:
                                            flag_connect = True
                                        if flag_connect:
                                            list_struc.append(dir_ii)
                                            mat_struc_copy[dir_ii[0],dir_ii[1],dir_ii[2]]  = 0
                                            list_waiting.append(dir_ii)
                    # ---------------------------------------------------------------------------------------------------
                    # Append the list of nodes to the list of structures
                    # ---------------------------------------------------------------------------------------------------
                    self.nodes.append(np.array(list_struc).T)
        
                
            

    def physicalproperties_structures(self):
        """
        .................................................................................................................
        # physicalproperties_structures
        .................................................................................................................
        Function to calculate the physical properties of the different coherent structures.

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Create the information of the structures
        # ---------------------------------------------------------------------------------------------------------------
        self.dim_x   = np.zeros((len(self.nodes),),dtype="float") # size of the structure in the streamwise direction
        self.dim_y   = np.zeros((len(self.nodes),),dtype="float") # size of the structure in the wall-normal direction
        self.dim_z   = np.zeros((len(self.nodes),),dtype="float") # size of the structure in the spanwise direction
        self.ymin    = np.zeros((len(self.nodes),),dtype="float") # minimum wall distance of the structure
        self.ymax    = np.zeros((len(self.nodes),),dtype="float") # maximum wall distance of the structure
        self.boxvol  = np.zeros((len(self.nodes),),dtype="float") # volume of the box containing the structure
        self.vol     = np.zeros((len(self.nodes),),dtype="float") # volume of the structure
        self.cg_x    = np.zeros((len(self.nodes),),dtype="float") # position of the center of gravity in x
        self.cg_z    = np.zeros((len(self.nodes),),dtype="float") # position of the center of gravity in z
        self.cg_y    = np.zeros((len(self.nodes),),dtype="float") # position of the center of gravity in y
        self.cg_xbox = np.zeros((len(self.nodes),),dtype="float") # position of the center of gravity of the box in x
        self.cg_zbox = np.zeros((len(self.nodes),),dtype="float") # position of the center of gravity of the box in z
        self.cg_ybox = np.zeros((len(self.nodes),),dtype="float") # position of the center of gravity of the box in y
        self.inv_chn = np.zeros((len(self.nodes),),dtype="bool")  # flag for knowing if the structure is inverted
        
        # ---------------------------------------------------------------------------------------------------------------
        # Calculate the physical properties for every structure
        #     - nn           : index of the structure
        #     - struc_points : nodes of the structure
        #     - ymin         : minimum y position of the structure
        #     - ymax         : maximum y position of the structure
        #     - dim_y        : dimension in y
        #     - x_sort       : nodes in x sorted
        #     - z_sort       : nodes in z sorted
        #     - dim_x        : dimensions of the structure in the streamwise position
        #     - dim_z        : dimensions of the structure in the spanwise position
        # ---------------------------------------------------------------------------------------------------------------
        for nn  in np.arange(len(self.nodes)):
            struc_points     = self.nodes[nn].astype('int')
            ymin             = self.y_h_plus[int(np.min(struc_points[0,:]))]
            ymax             = self.y_h_plus[int(np.max(struc_points[0,:]))]
            dim_y            = np.abs(ymax-ymin)
            x_sort           = np.sort(struc_points[2,:])
            z_sort           = np.sort(struc_points[1,:])
            self.cg_xbox[nn] = np.floor(np.mean(x_sort))
            self.cg_zbox[nn] = np.floor(np.mean(z_sort))
            self.cg_ybox[nn] = np.floor(np.mean(struc_points[0,:])) 
            
            # -----------------------------------------------------------------------------------------------------------
            # Evaluate the center of gravity of the structures
            # -----------------------------------------------------------------------------------------------------------
            cg_y = 0
            for nn2 in np.arange(len(self.nodes[nn][0,:])):
                self.cg_x[nn] += self.grid_dx_plus*struc_points[2,nn2]*self.grid_vol_plus[struc_points[0,nn2],
                                                                                          struc_points[1,nn2],
                                                                                          struc_points[2,nn2]]
                self.cg_z[nn] += self.grid_dz_plus*struc_points[1,nn2]*self.grid_vol_plus[struc_points[0,nn2],
                                                                                          struc_points[1,nn2],
                                                                                          struc_points[2,nn2]]
                cg_y          += self.y_h_plus[struc_points[0,nn2]]*self.grid_vol_plus[struc_points[0,nn2],
                                                                                       struc_points[1,nn2],
                                                                                       struc_points[2,nn2]]
                self.vol[nn]  += self.grid_vol_plus[struc_points[0,nn2],struc_points[1,nn2],struc_points[2,nn2]]
            self.cg_x[nn] /= self.vol[nn]
            self.cg_z[nn] /= self.vol[nn]
            cg_y          /= self.vol[nn]
                            
            # -----------------------------------------------------------------------------------------------------------
            # Calculate the dimension of each structure in the streamwise and the spanwise dimensions
            # -----------------------------------------------------------------------------------------------------------
            dim_x = self.grid_dx_plus*(np.max(x_sort)-np.min(x_sort))
            dim_z = self.grid_dz_plus*(np.max(z_sort)-np.min(z_sort)) 
            
            # -----------------------------------------------------------------------------------------------------------
            # Check if the structure is crossing the symmetry planes x and z
            #     - flag_x : flag which determines if a structure is crossing the symmetry plane in x
            #     - flag_z : flag which determines if a structure is crossing the symmetry plane in z
            # -----------------------------------------------------------------------------------------------------------
            flag_x = bool(np.count_nonzero(x_sort==self.shpx-1)>=1 and np.count_nonzero(x_sort==0)>=1)
            flag_z = bool(np.count_nonzero(z_sort==self.shpz-1)>=1 and np.count_nonzero(z_sort==0)>=1)
            
            # -----------------------------------------------------------------------------------------------------------
            # If the structure is crossing the x plane
            # -----------------------------------------------------------------------------------------------------------
            if flag_x:
                
                # -------------------------------------------------------------------------------------------------------
                # Take the indices of the structure that are conected through the symmetry plane. 
                # 1. Take the unique indices in the x direction
                # 2. Create a vector of increasing indices with the same length as previous one
                # 3. Check if some of the indices mismatch
                # 4. Take the indices of the nodes that mismatch
                #     - x_uni     : vector of the unique indices of the nodes of the structure
                #     - com_cross : comparison of x_uni with a vector of increasing indices
                #     - ind_sym   : index of the positions of x_uni that are located on the other side of the symmetry
                # -------------------------------------------------------------------------------------------------------
                x_uni     = np.unique(x_sort)
                com_cross = x_uni==np.arange(len(x_uni))
                ind_sym   = np.where(com_cross==0)[0] 
                
                # -------------------------------------------------------------------------------------------------------
                # Check if the structure is crossing the symmetry plane and divided
                # -------------------------------------------------------------------------------------------------------
                if not len(ind_sym) == 0:
                    
                    # ---------------------------------------------------------------------------------------------------
                    # Calculate the size of the structure in the streamwise direction
                    # The minimum value is the minimum of the last part of structure and the maximum is the 
                    # maximum of the first structure.
                    #     - xmin_sym : minimum index of the part of the structure which is on the other side of the
                    #                  symmetry
                    #     - xmax_sym : maximum index of the structure once it has crossed the symmetry
                    #     - dx_sym   : size in number of nodes of the structure crossing the symmetry in the x direction
                    #     - dim_x    : size of the structructure crossing the symmetry in the x direction
                    # ---------------------------------------------------------------------------------------------------
                    xmin     = x_uni[ind_sym[0]]
                    xmax     = self.shpx+x_uni[ind_sym[0]-1]
                    dx_sym   = xmax-xmin
                    dim_x    = self.grid_dx_plus*dx_sym
                    
                    # ---------------------------------------------------------------------------------------------------
                    # Move the indices of the first part of the structure to the other side of the symmetry
                    #     - x_sort_sym  : indices of the structure once all the nodes have been moved along the 
                    #                     symmetry
                    #     - index_first : indices of the sorted nodes that are located in the first part of the 
                    #                     structure
                    # ---------------------------------------------------------------------------------------------------
                    x_sort_sym              = x_sort.copy()
                    index_first             = x_sort_sym<=x_uni[ind_sym[0]-1]
                    x_sort_sym[index_first] = x_sort_sym[index_first]+self.shpx
                    
                    # ---------------------------------------------------------------------------------------------------
                    # Calculate the center of gravity of the box of the structure in the streamwise direction with the 
                    # nodes that have been converted
                    # ---------------------------------------------------------------------------------------------------
                    self.cg_xbox[nn] = np.mod(np.floor(np.mean(x_sort_sym)),self.shpx)
                    
                    # ---------------------------------------------------------------------------------------------------
                    # Take the indices of the nodes contained in the structure in the x dimensions
                    #     - xind_nodes        : nodes of the structure transformed along the symmetry
                    #     - index_nodes_first : indices of the nodes belonging to the first part of the structure
                    # ---------------------------------------------------------------------------------------------------
                    xind_nodes                    = struc_points[2,:].copy()
                    index_nodes_first             = xind_nodes<=x_uni[ind_sym[0]-1]
                    xind_nodes[index_nodes_first] = xind_nodes[index_nodes_first]+self.shpx
                    
                    # ---------------------------------------------------------------------------------------------------
                    # Calculate the center of gravity of the structure.
                    # If the center of gravity of the structure is outside the channel move it inside
                    #     - nn2 : index of the node contained in the structure
                    #     - nn  : index of the structure
                    # ---------------------------------------------------------------------------------------------------
                    self.cg_x[nn] = 0
                    for nn2 in np.arange(len(self.nodes[nn][0,:])):
                        self.cg_x[nn] += self.grid_dx_plus*xind_nodes[nn2]*self.grid_vol_plus[struc_points[0,nn2],
                                                                                              struc_points[1,nn2],
                                                                                              struc_points[2,nn2]]
                    self.cg_x[nn] /= self.vol[nn]
                    if self.cg_x[nn] > self.grid_dx_plus*self.shpx:
                        self.cg_x[nn] -= self.grid_dx_plus*self.shpx
                        
            # -----------------------------------------------------------------------------------------------------------
            # If the structure is crossing the z plane
            # -----------------------------------------------------------------------------------------------------------
            if flag_z:
                
                # -------------------------------------------------------------------------------------------------------
                # Take the indices of the structure that are conected through the symmetry plane. 
                # 1. Take the unique indices in the z direction
                # 2. Create a vector of increasing indices with the same length as previous one
                # 3. Check if some of the indices mismatch
                # 4. Take the indices of the nodes that mismatch
                #     - z_uni     : vector of the unique indices of the nodes of the structure
                #     - com_cross : comparison of z_uni with a vector of increasing indices
                #     - ind_sym   : index of the positions of z_uni that are located on the other side of the symmetry
                # -------------------------------------------------------------------------------------------------------
                z_uni     = np.unique(z_sort)
                com_cross = z_uni==np.arange(len(z_uni))
                ind_sym   = np.where(com_cross==0)[0]  
                
                # -------------------------------------------------------------------------------------------------------
                # Check if the structure is crossing the symmetry plane and divided
                # -------------------------------------------------------------------------------------------------------
                if not len(ind_sym) == 0:
                    
                    # ---------------------------------------------------------------------------------------------------
                    # Calculate the size of the structure in the spanwise direction
                    # The minimum value is the minimum of the last part of structure and the maximum is the 
                    # maximum of the first structure.
                    #     - zmin_sym : minimum index of the part of the structure which is on the other side of the
                    #                  symmetry
                    #     - zmax_sym : maximum index of the structure once it has crossed the symmetry
                    #     - dz_sym   : size in number of nodes of the structure crossing the symmetry in the z direction
                    #     - dim_z    : size of the structructure crossing the symmetry in the z direction
                    # ---------------------------------------------------------------------------------------------------
                    zmin     = z_uni[ind_sym[0]]
                    zmax     = self.shpz+z_uni[ind_sym[0]-1]
                    dz_sym   = zmax-zmin
                    dim_z    = self.grid_dz_plus*dz_sym
                    
                    # ---------------------------------------------------------------------------------------------------
                    # Move the indices of the first part of the structure to the other side of the symmetry
                    #     - z_sort_sym  : indices of the structure once all the nodes have been moved along the
                    #                     symmetry
                    #     - index_first : indices of the sorted nodes that are located in the first part of the 
                    #                     structure
                    # ---------------------------------------------------------------------------------------------------
                    z_sort_sym              = z_sort.copy()
                    index_first             = z_sort_sym<=z_uni[ind_sym[0]-1]
                    z_sort_sym[index_first] = z_sort_sym[index_first]+self.shpz
                    
                    # ---------------------------------------------------------------------------------------------------
                    # Calculate the center of gravity of the box of the structure in the spanwise direction with the 
                    # nodes that have been converted
                    # ---------------------------------------------------------------------------------------------------
                    self.cg_zbox[nn] = np.mod(np.floor(np.mean(z_sort_sym)),self.shpz)
                    
                    # ---------------------------------------------------------------------------------------------------
                    # Take the indices of the nodes contained in the structure in the z dimensions
                    #     - zind_nodes        : nodes of the structure transformed along the symmetry
                    #     - index_nodes_first : indices of the nodes belonging to the first part of the structure
                    # ---------------------------------------------------------------------------------------------------
                    zind_nodes                    = struc_points[1,:].copy()
                    index_nodes_first             = zind_nodes<=z_uni[ind_sym[0]-1]
                    zind_nodes[index_nodes_first] = zind_nodes[index_nodes_first]+self.shpz
                    
                    # ---------------------------------------------------------------------------------------------------
                    # Calculate the center of gravity of the structure.
                    # If the center of gravity of the structure is outside the channel move it inside
                    #     - nn2 : index of the node contained in the structure
                    #     - nn  : index of the structure
                    # ---------------------------------------------------------------------------------------------------
                    self.cg_z[nn] = 0
                    for nn2 in np.arange(len(self.nodes[nn][0,:])):
                        self.cg_z[nn] += self.grid_dz_plus*zind_nodes[nn2]*self.grid_vol_plus[struc_points[0,nn2],
                                                                                    struc_points[1,nn2],
                                                                                    struc_points[2,nn2]]
                    self.cg_z[nn] /= self.vol[nn]
                    if self.cg_z[nn] > self.grid_dz_plus*self.shpz:
                        self.cg_z[nn] -= self.grid_dz_plus*self.shpz
                        
            
            # -----------------------------------------------------------------------------------------------------------
            # Select the ymin, ymax and cg_y in the correct semichannel
            # -----------------------------------------------------------------------------------------------------------
            if cg_y <= 0:
                self.ymin[nn]    = self.rey+ymin
                self.ymax[nn]    = self.rey+ymax
                self.cg_y[nn]    = self.rey+cg_y
                self.inv_chn[nn] = False
            else:
                self.ymin[nn]    = self.rey-ymax
                self.ymax[nn]    = self.rey-ymin
                self.cg_y[nn]    = self.rey-cg_y
                self.inv_chn[nn] = True
            # -----------------------------------------------------------------------------------------------------------
            # Store the dimensions of the structures and the volume of the box of each structure
            # -----------------------------------------------------------------------------------------------------------
            self.dim_x[nn]  = dim_x
            self.dim_z[nn]  = dim_z
            self.dim_y[nn]  = dim_y
            self.boxvol[nn] = dim_y*dim_x*dim_z
                                
    def detect_quadrant(self):
        """
        .................................................................................................................
        # detect_quadrant
        .................................................................................................................
        Function for calculating the quadrant of each structures and a matrix with the quadrant of each node

        Returns
        -------
        None.

        """
        # ---------------------------------------------------------------------------------------------------------------
        # Define the matrix of the events and the event of every structure
        #     - mat_event : matrix of the events of each nodes
        #     - event     : event of each structure
        # ---------------------------------------------------------------------------------------------------------------
        self.mat_event = np.zeros((self.shpy,self.shpz,self.shpx))
        self.event     = np.zeros((len(self.nodes),))
        
        # ---------------------------------------------------------------------------------------------------------------
        # Evaluate the characteristics for each structure
        #     - struc_points : indices of the nodes of a structure
        #     - voltot       : total volume occupied by each quadrant in a structure
        #     - nn           : index of the structure
        # ---------------------------------------------------------------------------------------------------------------
        for nn  in np.arange(len(self.nodes)):
            struc_points = self.nodes[nn]
            voltot       = np.zeros((4,))
            
            # -----------------------------------------------------------------------------------------------------------
            # Evaluate each node of the structure
            #     - nn_node : index of the node
            # -----------------------------------------------------------------------------------------------------------
            for nn_node in np.arange(len(struc_points[0,:])):
                
                # -------------------------------------------------------------------------------------------------------
                # Get the value of the fields in the dimensions 1 and 2 of the node
                #     - field_1 : value of the dimension 1 of the field in the node
                #     - field_2 : value of the dimension 2 of the field in the node
                # -------------------------------------------------------------------------------------------------------
                field_1 = self.field_1[struc_points[0,nn_node],struc_points[1,nn_node],struc_points[2,nn_node]]
                if self.sym_quad:
                    if self.inv_chn[nn]:
                        field_2 = -self.field_2[struc_points[0,nn_node],struc_points[1,nn_node],struc_points[2,nn_node]]
                    else:
                        field_2 = self.field_2[struc_points[0,nn_node],struc_points[1,nn_node],struc_points[2,nn_node]]
                else:
                    field_2 = self.field_2[struc_points[0,nn_node],struc_points[1,nn_node],struc_points[2,nn_node]]
                
                # -------------------------------------------------------------------------------------------------------
                # Calculate the volume of the node and add it to the corresponding index of the voltot vector
                # -------------------------------------------------------------------------------------------------------
                vol_nod = np.sqrt(field_1**2+field_2**2)*self.grid_vol_plus[struc_points[0,nn_node],
                                                                            struc_points[1,nn_node],
                                                                            struc_points[2,nn_node]]
                if field_1 > 0 and field_2 > 0:
                    voltot[0] += vol_nod
                elif field_1 < 0 and field_2 > 0:
                    voltot[1] += vol_nod
                elif field_1 < 0 and field_2 < 0:
                    voltot[2] += vol_nod
                elif field_1 > 0 and field_2 < 0:
                    voltot[3] += vol_nod
            
            # -----------------------------------------------------------------------------------------------------------
            # Choose the event with the higher associated volume
            # -----------------------------------------------------------------------------------------------------------
            max_event = np.argmax(voltot)
            if max_event == 0:
                self.event[nn] = 1
            elif max_event == 1:
                self.event[nn] = 2
            elif max_event == 2:
                self.event[nn] = 3
            elif max_event == 3:
                self.event[nn] = 4
                
            # -----------------------------------------------------------------------------------------------------------
            # Generate the matrix associating each node with its event
            #     - nn_node : index of the node
            # -----------------------------------------------------------------------------------------------------------
            for nn_node in np.arange(len(struc_points[0,:])):
                self.mat_event[struc_points[0,nn_node],struc_points[1,nn_node],struc_points[2,nn_node]] = self.event[nn]
                                               
    def segmentation(self):
        """
        .................................................................................................................
        # segmentation
        .................................................................................................................
        Function for segmenting the domain according with the structures

        Returns
        -------
        None.

        """        
        # ---------------------------------------------------------------------------------------------------------------
        # Define the segmentation of the domain with and without filtering the structures
        # ---------------------------------------------------------------------------------------------------------------
        self.mat_segment          = np.zeros((self.shpy,self.shpz,self.shpx))
        self.mat_segment_filtered = np.zeros((self.shpy,self.shpz,self.shpx))
        
        # ---------------------------------------------------------------------------------------------------------------
        # Evaluate for all the structures
        #     - point_struc : grid points of each structure
        #     - nn          : index of the structure
        #     - nn_node     : index of the point of the structure
        #     - nn2         : index of the structure after filtering the small ones
        #     - nn3         : index of the structures that have been filtered
        # ---------------------------------------------------------------------------------------------------------------
        nn2 = 0
        nn3 = 0  
        if len(self.nodes) > 0:
            for nn  in np.arange(len(self.nodes)):
                point_struc = self.nodes[nn]
                for nn_node in np.arange(len(point_struc[0,:])):
                    self.mat_segment[point_struc[0,nn_node],point_struc[1,nn_node],point_struc[2,nn_node]] = nn+1
                    if self.vol[nn] > self.filvol:
                        self.mat_segment_filtered[point_struc[0,nn_node],
                                                  point_struc[1,nn_node],
                                                  point_struc[2,nn_node]] = nn2+1
                        
                # -----------------------------------------------------------------------------------------------------------
                # Advance the index of larger and smaller structures depending on the structure
                # -----------------------------------------------------------------------------------------------------------
                if self.vol[nn] > self.filvol:
                    nn2 += 1
                else:
                    nn3 += 1
                
            # ---------------------------------------------------------------------------------------------------------------
            # Define the percentage of filtered structures
            # ---------------------------------------------------------------------------------------------------------------
            self.filtstr_sum = nn3/(nn2+nn3)
        else:
            self.filtstr_sum = 0
        print('Percentage of filtered structures: '+str(self.filtstr_sum*100)+'%',flush=True)
                                  
    def structure_u1u2(self):
        """
        .................................................................................................................
        # structure_u1u2
        .................................................................................................................
        Function to calculate the product of the field in the dimensions 1 and 2 of the flow field for each structure

        Returns
        -------
        None.

        """    
        # ---------------------------------------------------------------------------------------------------------------
        # For every structure sum all products of the fields 1 and 2 of the nodes of the structure
        #     - nn       : index of the structure
        #     - nodes_nn : indices of the nodes of the structure nn
        #     - u1u2     : product of the fields 1 and 2 in each structure
        #     - absu1u2  : absolute value of the product of the fields 1 and 2
        #     - u1u2tot  : total value of the product of the fields 1 and 2 in all the nodes
        # ---------------------------------------------------------------------------------------------------------------
        max_struc = int(np.max(self.mat_segment))
        self.u1u2 = np.zeros((max_struc,))
        absu1u2   = np.abs(np.multiply(self.field_1,self.field_2))
        u1u2tot   = np.sum(absu1u2)
        for nn  in np.arange(max_struc-1):
            nodes_nn = np.where(self.mat_segment==nn+1)
            for ii in np.arange(len(nodes_nn[0])):
                self.u1u2[nn] += absu1u2[nodes_nn[0][ii],nodes_nn[1][ii],nodes_nn[2][ii]]/u1u2tot
                
                                        
    def structure_k123(self):
        """
        .................................................................................................................
        # structure_k123
        .................................................................................................................
        Function to calculate the energy of the field in the dimensions 1, 2 and 3 of the flow field for each structure

        Returns
        -------
        None.

        """    
        # ---------------------------------------------------------------------------------------------------------------
        # For every structure sum all products of the fields 1, 2 and 3 of the nodes of the structure
        #     - nn       : index of the structure
        #     - nodes_nn : indices of the nodes of the structure nn
        #     - k123     : product energy of the fields 1, 2 and 3 in each structure
        #     - k123_tot : total value of the energy of the fields 1, 2 and 3 in all the nodes
        # ---------------------------------------------------------------------------------------------------------------
        max_struc = int(np.max(self.mat_segment))
        self.k123 = np.zeros((max_struc,))
        k123      = np.sqrt(self.field_1**2+self.field_2**2+self.field_3**2)
        k123_tot  = np.sum(k123)
        for nn  in np.arange(max_struc-1):
            nodes_nn = np.where(self.mat_segment==nn+1)
            for ii in np.arange(len(nodes_nn[0])):
                self.k123[nn] += k123[nodes_nn[0][ii],nodes_nn[1][ii],nodes_nn[2][ii]]/k123_tot
            
