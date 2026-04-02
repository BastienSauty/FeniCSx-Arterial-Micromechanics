import numpy as np
import ufl

def h_Log(z):
    hz = (1+ z)/(1-z) + 2/np.log(z)
    return(hz)

def h_GN(z):
    hz = (1-np.sqrt(z))/(1+np.sqrt(z))
    return(hz)
    
def compute_v_field(eig, obj_der):
    """
    eig_field is a n*3 table containing the eigenvalues for the n points of interest
    """
    if obj_der == 'Log':
        h = h_Log
    elif obj_der == 'GN':
        h = h_GN

    if np.linalg.norm(eig.imag) < 1e-10:
        eig = eig.real
    else :
        raise Exception(f'Eigenvalues are complex :{eig}')

    eps = np.array([eig[:,1]/eig[:,2], eig[:,2]/eig[:,0], eig[:,0]/eig[:,1]]).T

    # Case of different eigenvalues
    eig_diff = np.logical_not(np.logical_or(np.isclose(eig[:,0],eig[:,1]), np.isclose(eig[:,0],eig[:,2]),np.isclose(eig[:,1],eig[:,2]))) # All eigenvalues are different
    
    v_ = np.zeros(eig.shape)
    for kk in range(1,4):
        for jj in range(3):
            v_[eig_diff,kk-1]+= (-eig[eig_diff,jj])**(3-kk)*(h(eps[eig_diff,jj]) ) # (1+ eps[eig_diff,jj])/(1-eps[eig_diff,jj]) + 2/np.log(eps[eig_diff,jj])

        v_[eig_diff,kk-1]*=1/((eig[eig_diff,0] - eig[eig_diff,1])*(eig[eig_diff,1] - eig[eig_diff,2])*(eig[eig_diff,2] - eig[eig_diff,0]))
    
    # Case b2 != b3 == b1
    eig_13eq = np.logical_and(np.logical_not(np.isclose((eig[:,1]-eig[:,0])*(eig[:,1]-eig[:,2]),0)), np.isclose((eig[:,0]-eig[:,2]), 0)) # eigenvalues 1=3
    v_[eig_13eq, 0] = 1/(eig[eig_13eq,1] - eig[eig_13eq,2])*(h(eps[eig_13eq,0])) #    (1+ eps[eig_13eq,0])/(1-eps[eig_13eq,0]) + 2/np.log(eps[eig_13eq,0]))
    
    # Case b3 != b1 == b2
    eig_12eq = np.logical_and(np.logical_not(np.isclose((eig[:,2]-eig[:,0])*(eig[:,2]-eig[:,1]),0)), np.isclose((eig[:,0]-eig[:,1]), 0)) # eigenvalues 1=2
    v_[eig_12eq, 0] = 1/(eig[eig_12eq,2] - eig[eig_12eq,0])*(h(eps[eig_12eq,1]))#(1+ eps[eig_12eq,1])/(1-eps[eig_12eq,1]) + 2/np.log(eps[eig_12eq,1]))
    
    # Case b1 != b2 == b3
    eig_23eq = np.logical_and(np.logical_not(np.isclose((eig[:,0]-eig[:,1])*(eig[:,0]-eig[:,2]),0)), np.isclose((eig[:,1]-eig[:,2]), 0)) # eigenvalues 2=3
    v_[eig_23eq, 0] = 1/(eig[eig_23eq,0] - eig[eig_23eq,1])*(h(eps[eig_23eq,2]))#(1+ eps[eig_23eq,2])/(1-eps[eig_23eq,2]) + 2/np.log(eps[eig_23eq,2]))
    # Case b1 = b2 = b3 
    # Nothing to do, v_ = 0
    return(v_)
    
def nlog_func(b, d, v):
    """
    compute the nlog part in the objective gradient for a given b (left cauchy green) and d (increment/velocity of deformation gradient in current conf)
    v is the field of eigenvalue related coefficients
    see hdr Claire Morin eq 2.26 or ZAMM paper eq 14-16
    output : form nlog with the corresponding conditional parts
    """    
    v1part = v[0]*(ufl.dot(b,d)-ufl.dot(d,b))
    v2part = v[1]*(ufl.dot(ufl.dot(b,b), d) - ufl.dot(d, ufl.dot(b,b)) )
    v3part = v[2]*(ufl.dot(ufl.dot(ufl.dot(b,b), d), b) - ufl.dot(b, ufl.dot(d, ufl.dot(b,b)))) 
    
    nlog = v1part + v2part + v3part
    return(nlog)

