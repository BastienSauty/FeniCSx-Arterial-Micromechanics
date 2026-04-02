import numpy as np
import ufl
from dolfinx import fem, plot

###########################################################################
# File: auxiliary_functions.py
# Author: B.Sauty
# Date: 30 Sept 2024
#
# Description:
#   This file contains simple functions used for creating and computing 
#   useful quantities in mechanical modelling
#
# Contents:
#   - Building the Stiffness matrix : projection_tensors, projection_tensors_func
#   - Change of Base Functions : local_basis, change_of_base_matrix, set_orientation_fields
#   - Manipulations of tensors with and without Mandel notations : 
#       - for 2nd order tensors : Tensor2Voigt, Voigt2Tensor, Voigt2Tensor_Antisym
#       - for 4th order tensors : tensors_local2global, tensors_global2local, tensordot_2_4, compress, expand
# 
# Logs
# 21 Dec 2022 : creation. Used by mech_problem_class, material_class and inclusions_class
# 30 Sept 2024 : Update, cleaning and commenting

###########################################################################
### Building the Stiffness matrix
###########################################################################

def projection_tensors():
    """ 
    Define the projection tensors J (spherical projector) and K (deviatoric projector) 
    in Mandel/Voigt notation, along with the identity matrix I.
    
    Returns:
    --------
    list of np.ndarray
        J : (6x6) spherical projection matrix.
        K : (6x6) deviatoric projection matrix.
        I : (6x6) identity matrix.
    """
    J = 1/3 * np.vstack([np.hstack([np.ones([3, 3]), np.zeros([3, 3])]), np.zeros([3, 6])])
    I = np.eye(6)
    K = I - J
    return [J, K, I]

    
def projection_tensors_func(V_stiff):
    """
    Create constant tensor functions for the spherical (J), deviatoric (K), and identity (I) 
    projections in the Mandel notation, suitable for interpolation in the FEniCSx framework.

    Args:
    -----------
    V_stiff : dolfinx.fem.FunctionSpace
        The function space for 6x6 second-order tensors representing stiffness matrices.

    Returns:
    --------
    list of dolfinx.fem.Function
        J_m : Function for the spherical projector.
        K_m : Function for the deviatoric projector.
        I_m : Function for the identity tensor.
    
    Notes:
    ------
    - The function creates constant tensor fields by assigning the same 6x6 matrix at each node.
    """
    # Get projection tensors in Mandel/Voigt notation
    J_mandel, K_mandel, I_mandel = projection_tensors() 

    # Initialize FEniCSx functions for each projection tensor
    K_m = fem.Function(V_stiff)
    J_m = fem.Function(V_stiff)
    I_m = fem.Function(V_stiff)

    # Number of nodes where the 6x6 matrices are stored
    nb_nodes = len(K_m.x.array) // 36
    shape_tensor = [nb_nodes, 36]

    # Assign deviatoric projection tensor (K) to all nodes
    K_ = np.zeros(shape_tensor)
    K_[:, :] += K_mandel.reshape(36)
    K_m.x.array[:] = K_.reshape(nb_nodes * 36)
    K_m.x.scatter_forward()

    # Assign spherical projection tensor (J) to all nodes
    J_ = np.zeros(shape_tensor)
    J_[:, :] += J_mandel.reshape(36)
    J_m.x.array[:] = J_.reshape(nb_nodes * 36)
    J_m.x.scatter_forward()

    # Assign identity tensor (I) to all nodes
    I_ = np.zeros(shape_tensor)
    I_[:, :] += I_mandel.reshape(36)
    I_m.x.array[:] = I_.reshape(nb_nodes * 36)
    I_m.x.scatter_forward()

    return [J_m, K_m, I_m]


###########################################################################
### Initialize the Deformation gradient fields at Identity
###########################################################################

def init_identity_field(field):
    """
    field must be a (3*3) function obtained through fem.Function(V_functionspace)
    """
    I = np.eye(3) # initialize at identity
    nb_elem = len(field.x.array)//9
    shape_tensor = [nb_elem, 9]
    I_ = np.zeros(shape_tensor) 
    I_[:,:] += I.reshape(9) 
    field.x.array[:] = I_.reshape(nb_elem*9)
        
    
###########################################################################
### Change of Base Functions
###########################################################################
    
def local_basis(theta, phi):
    """
    Compute the local basis vectors using the spherical coordinate system and axisymmetry around the radial direction (e_r).
    Definition and convention used can be found on : https://en.wikipedia.org/wiki/Spherical_coordinate_system
    This is based on the ISO Standard : ISO 80000-2:2019
    
    Args:
    -----------
    theta : float
        Polar angle in spherical coordinates.
    phi : float
        Azimuthal angle in spherical coordinates.

    Returns:
    --------
    tuple of np.ndarray
        e_theta : Local basis vector in the theta direction.
        e_phi : Local basis vector in the phi direction.
        e_r : Local basis vector in the radial direction.
    """
    e_r = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    e_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    return e_theta, e_phi, e_r


def change_of_base_matrix(e_1, e_2, e_3):
    """
    Construct the rotation matrix from a given local basis to the global XYZ basis.

    Args:
    -----------
    e_1, e_2, e_3 : np.ndarray
        Basis vectors expressed in the global XYZ coordinate system.

    Returns:
    --------
    ufl.as_matrix
        The rotation matrix P such that M_XYZ = P * M_123 * P.T and u_XYZ = P * u_123.
    """
    P = np.vstack([e_1, e_2, e_3])
    P = ufl.as_matrix(np.transpose(P))
    return P


def set_orientation_fields(e_theta_func, e_phi_func, e_r_func, e_theta_vec, e_phi_vec, e_r_vec):
    """
    Populate the degrees of freedom (DOFs) of the orientation field functions with constant vectors.
    
    Args:
    -----------
    e_theta_func : dolfinx.fem.Function
        Function for the e_theta orientation field.
    e_phi_func : dolfinx.fem.Function
        Function for the e_phi orientation field.
    e_r_func : dolfinx.fem.Function
        Function for the e_r orientation field.
    e_theta_vec : np.ndarray
        Constant vector to assign to e_theta_func.
    e_phi_vec : np.ndarray
        Constant vector to assign to e_phi_func.
    e_r_vec : np.ndarray
        Constant vector to assign to e_r_func.

    Notes:
    ------
    - This function modifies the input field functions in place by setting their DOFs to the provided vectors.
    - Used for initializing orientation fields in mechanical problems with cylindrical or spherical inclusions.
    """
    nb_nodes = len(e_theta_func.x.array) // len(e_theta_func)
    shape_tensor = [nb_nodes, len(e_theta_func)]

    # Set values for e_theta_func
    e_theta_tile_ = np.zeros(shape_tensor)
    e_theta_tile_[:, :] = np.tile(e_theta_vec, (nb_nodes, 1))
    e_theta_func.x.array[:] = e_theta_tile_.reshape(len(e_theta_func.x.array))
    e_theta_func.x.scatter_forward()

    # Set values for e_phi_func
    e_phi_tile_ = np.zeros(shape_tensor)
    e_phi_tile_[:, :] = np.tile(e_phi_vec, (nb_nodes, 1))
    e_phi_func.x.array[:] = e_phi_tile_.reshape(len(e_phi_func.x.array))
    e_phi_func.x.scatter_forward()

    # Set values for e_r_func
    e_r_tile_ = np.zeros(shape_tensor)
    e_r_tile_[:, :] = np.tile(e_r_vec, (nb_nodes, 1))
    e_r_func.x.array[:] = e_r_tile_.reshape(len(e_r_func.x.array))
    e_r_func.x.scatter_forward()


###########################################################################
### Conversions Between Mandel Notation and Tensor Notation
###########################################################################
    
def Tensor2Voigt(s_t):
    """
    Converts a 3x3 symmetric tensor to a 6x1 vector in Mandel/Voigt notation.
    
    Args:
        s_t: 3x3 symmetric tensor (input tensor).
    
    Returns:
        s_v: 6x1 vector in Mandel notation (normalized Voigt representation).
    """
    s_v = ufl.as_vector([s_t[0, 0], 
                         s_t[1, 1], 
                         s_t[2, 2], 
                         np.sqrt(2) / 2 * (s_t[1, 2] + s_t[2, 1]), 
                         np.sqrt(2) / 2 * (s_t[0, 2] + s_t[2, 0]), 
                         np.sqrt(2) / 2 * (s_t[0, 1] + s_t[1, 0])])
    return s_v

def Voigt2Tensor(s):
    """
    Converts a 6x1 vector in Mandel/Voigt notation to a 3x3 symmetric tensor.
    
    Args:
        s: 6x1 vector (in Mandel notation).
    
    Returns:
        s_t: 3x3 symmetric tensor.
    """
    s_t = ufl.as_matrix([[s[0],              1 / np.sqrt(2) * s[5],  1 / np.sqrt(2) * s[4]],
                         [1 / np.sqrt(2) * s[5], s[1],               1 / np.sqrt(2) * s[3]], 
                         [1 / np.sqrt(2) * s[4], 1 / np.sqrt(2) * s[3],  s[2]]]) 
    return s_t

def Voigt2Tensor_Antisym(s):
    """
    Converts a 6x1 vector into a 3x3 antisymmetric tensor. 
    The off-diagonal elements are stored using the upper triangular part of the tensor.
    When reconstructing the antiymmetric tensor, the lower triangular part is therefore 
    negative times the upper one. Actually the diagonal terms should be always zero. 
    
    Args:
        s: 6x1 vector.
    
    Returns:
        s_t: 3x3 antisymmetric tensor.
    """
    s_t = ufl.as_matrix([[s[0],               1 / np.sqrt(2) * s[5],  1 / np.sqrt(2) * s[4]], 
                         [-1 / np.sqrt(2) * s[5], s[1],               1 / np.sqrt(2) * s[3]], 
                         [-1 / np.sqrt(2) * s[4], -1 / np.sqrt(2) * s[3], s[2]]])
    return s_t

###########################################################################
### bunch of functions for rotating Fourth order tensors 
###########################################################################

def tensors_local2global(Iloc, P):
    """
    Compute the rotated 4th order tensor from the local to the global basis using the ufl indices.

    Args:
    -----------
    Iloc : ufl.as_tensor
        Ufl 4th order 3*3*3*3 tensor defined in the local basis
    P : ufl.as_matrix
        Ufl 3*3 matrix defining the change of basis matrix from local to global

    Returns:
    --------
    Iglob : ufl.as_tensor
        Rotated 3*3*3*3 tensor in the global basis 
    """
    i,j,k,l, m,n,p,q  = ufl.indices(8)
    Iglob = ufl.as_tensor(P[i,m]*P[j,n]*P[k,p]*P[l,q]* Iloc[m,n,p,q] , (i,j,k,l))
    return(Iglob)


def tensors_global2local(Iglob, P):
    """
    Compute the rotated 4th order tensor from the global to the local basis using the ufl indices.

    Args:
    -----------
    Iglob : ufl.as_tensor
        Ufl 4th order 3*3*3*3 tensor defined in the global basis
    P : ufl.as_matrix
        Ufl 3*3 matrix defining the change of basis matrix from local to global

    Returns:
    --------
    Iloc : ufl.as_tensor
        Rotated 3*3*3*3 tensor in the local basis 
    """
    PT = P.T
    i,j,k,l, m,n,p,q  = ufl.indices(8)
    Iloc = ufl.as_tensor(PT[i,m]*PT[j,n]*PT[k,p]*PT[l,q]* Iglob[m,n,p,q] , (i,j,k,l))
    return(Iloc)


def expand(I, antisym=[1,1]):
    """
    Converts a 6x6 matrix in the Mandel Notations into the full 3x3x3x3 tensor.
    
    Args:
        I: 6x6 matrix
        antisym : [1,1], [-1,1], [1,-1], [-1,-1], set the small antisymetries ;
                    if none [1, 1] : klmn = klnm , lkmn = lknm (full small symmetries)
                    if first [-1, 1] : klmn = -lkmn and klnm = klmn
                    if second [1, -1] : klmn = lkmn and klnm = -klmn
                    if both [-1, -1] : klmn = -lkmn = -klnm = lknm
    
    Returns:
        Ic : expanded Ie into fourth order tensor
    """
    
    # weight for mandel notation
    W=np.sqrt(2)
    Ie = []
    
    table_indices = [[ 0, 5, 4], 
                     [ 5, 1, 3],
                     [ 4, 3, 2]]
    
    # managing symmetries:
    f_m = antisym[0]
    f_n = antisym[1]   
    table_weights_m = [[     1,     W, W],
                       [ f_m*W,     1, W],
                       [ f_m*W, f_m*W, 1]]
    
    table_weights_n = [[     1,     W, W],
                       [ f_n*W,     1, W],
                       [ f_n*W, f_n*W, 1]]


    for i in range(3):
        Ie.append([])
        
        for j in range(3):
            Ie[i].append([])
            
            for k in range(3):
                Ie[i][j].append([])
                
                for l in range(3):
                    ind_m = table_indices[i][j]
                    ind_n = table_indices[k][l]
                    
                    w_m = table_weights_m[i][j]
                    w_n = table_weights_n[k][l]
                    
                    Ie[i][j][k].append(I[ind_m, ind_n]/(w_m * w_n))
             
    return(ufl.as_tensor(Ie))


def compress(I):
    """
    Converts a 3x3x3x3 symmetric tensor into a 6x6 Matrix in Mandel/Voigt notation.
    
    Args:
        I: 3x3x3x3 tensor with the small symmetries (left and right).
    
    Returns:
        Ic : 6x6 matrix (in Mandel notation).
    """
    W = np.sqrt(2) 
    
    Ic = np.array([ [I[0,0,0,0],   I[0,0,1,1],   I[0,0,2,2],   W*I[0,0,1,2], W*I[0,0,0,2], W*I[0,0,0,1]],
                    [I[1,1,0,0],   I[1,1,1,1],   I[1,1,2,2],   W*I[1,1,1,2], W*I[1,1,0,2], W*I[1,1,0,1]],
                    [I[2,2,0,0],   I[2,2,1,1],   I[2,2,2,2],   W*I[2,2,1,2], W*I[2,2,0,2], W*I[2,2,0,1]],
                    [W*I[1,2,0,0], W*I[1,2,1,1], W*I[1,2,2,2], 2*I[1,2,1,2], 2*I[1,2,0,2], 2*I[1,2,0,1]],
                    [W*I[0,2,0,0], W*I[0,2,1,1], W*I[0,2,2,2], 2*I[0,2,1,2], 2*I[0,2,0,2], 2*I[0,2,0,1]],
                    [W*I[0,1,0,0], W*I[0,1,1,1], W*I[0,1,2,2], 2*I[0,1,1,2], 2*I[0,1,0,2], 2*I[0,1,0,1]]])
    return(ufl.as_tensor(Ic))


def tensordot_2_4(sig,R):
    """
    Compute the simple contracted product of a second order tensor with a fourth order tensor. 
    Used when computing the Jaumann derivative at the microscopical level in Multiscale modelling
    
    Args:
        sig: 3x3 tensor
        R : 3x3x3x3 tensor
        
    Returns:
        prod = sig.R : 3x3x3x3 tensor
    """    
    i, j, k, l, m = ufl.indices(5)
    prod = ufl.as_tensor(sig[i,j]*R[j,k,l,m], (i,k,l,m))
    return(prod)


def tensordot_4_4(A,B):
    """
    Compute the double contracted product of two 4th order tensors. 
    Used for ease of code when considering rotations and keeping the localization spin tensors in full tensor notations
    
    Args:
        A : 3x3x3x3 tensor
        B : 3x3x3x3 tensor
        
    Returns:
        prod = A:B : 3x3x3x3 tensor
    """    
    i, j, k, l, m, n = ufl.indices(6)
    prod = ufl.as_tensor(A[i,j,k,l]*B[l,k,m,n], (i,j,m,n))
    return(prod)

    
def tensordot_4_2(C, d):
    """
    Compute the double contracted product of a 4th order tensors with a 2nd order tensor. 
    Args:
        C : 3x3x3x3 tensor
        d : 3x3 tensor
        
    Returns:
        prod = C:d : 3x3 tensor
    """    
    i, j, k, l = ufl.indices(4)
    prod = ufl.as_tensor(C[i,j,k,l]*d[l,k], (i,j))
    return(prod)

def tensordot_2_2(c, d):
    """
    Compute the double contracted product of a 2nd order tensors with a 2nd order tensor. 
    Args:
        c : 3x3 tensor
        d : 3x3 tensor
        
    Returns:
        prod = c:d : scalar
    """    
    i, j= ufl.indices(2)
    prod = c[i,j]*d[j,i]
    return(prod)

###########################################################################
### Old
###########################################################################


def projection_tensors_func_anisotropic(V_stiff):
    """
    BEWARE UGLY FUNCTION ; BASED ON PROJECTION TENSORS FUNC
    input : V_stiff, space function for a 6*6 2nd order tensor representing the stiffness matrixes
    output : Glt, Gtt, Cl, Clt, Ct, Ctt constant function defined on the whole space by setting their values at the nodes as the corresponding matrixes for the projection tensors
    
    Aim to provide constant tensor functions that can be interpolated in the fenicsx framework
    """
    Gltm, Gttm, Clm, Cltm, Ctm, Cttm = np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6)), np.zeros((6,6))

    Gltm[3,3] = 1 # Longitudinal transversal shear modulus 
    Gltm[4,4] = 1

    Gttm[5,5] = 1 # transversal shear modulus

    Clm[2,2] = 1 # longitudinal young modulus

    Cltm[2,0] = 1 # poisson longitudinal/transversal modulus
    Cltm[2,1] = 1
    Cltm[0,2] = 1
    Cltm[1,2] = 1
    
    Ctm[0,0] = 1 # transversal young modulus
    Ctm[1,1] = 1

    Cttm[0,1] = 1 # transversal poisson modulus
    Cttm[1,0] = 1
    
    Glt, Gtt, Cl, Clt, Ct, Ctt = fem.Function(V_stiff), fem.Function(V_stiff), fem.Function(V_stiff), fem.Function(V_stiff), fem.Function(V_stiff), fem.Function(V_stiff)
    listmatrix = [Gltm, Gttm, Clm, Cltm, Ctm, Cttm]
    listfunc = [Glt, Gtt, Cl, Clt, Ct, Ctt]
    
    nb_nodes = len(Glt.x.array)//36 # nb of nodes where the 6*6 matrix are stored
    shape_tensor = [nb_nodes, 36]
    
    for i in range(len(listfunc)):
        temp_ = np.zeros(shape_tensor) # (nbnodes, 36) matrix
        temp_[:,:] += listmatrix[i].reshape(36) # store for each nodes the (6,6) matrix reshaped into a (36,1) vector
        listfunc[i].x.array[:] = temp_.reshape(nb_nodes*36) # store in the function dofs the (nbnodes, 36) matrix reshaped into a (nb_nodes*36, 1) vector
        listfunc[i].x.scatter_forward()
    
    return(listfunc)
