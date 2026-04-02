import numpy as np
import scipy.integrate as integrate

#  18 jan 2023
#  Functions defining the auxiliary eshelby tensor for the different inclusions, based on their shapes and the stiffness of the matrix
#  Based on the analytical solution when possible

# Cylinder and Oblate spheroid are the infinite limit of the spheroid for a shape ratio respectively null and infinite 

def eshelby_aux_cylindrical(nu_0):
    """
    input : nu_0 poisson coefficient of the matrix
    output : L 3*3*3*3 tensor auxiliary eshelby tensor for cylinder 
    """
    
    L = np.zeros((3,3,3,3))
    
    L[0,0,0,0] = (4*nu_0-5)/(8*(nu_0-1))
    L[1,1,1,1] = (4*nu_0-5)/(8*(nu_0-1))
    
    L[0,0,1,1] = -(4*nu_0-1)/(8*(nu_0-1))
    L[1,1,0,0] = -(4*nu_0-1)/(8*(nu_0-1))
    
    L[0,1,0,1] = (4*nu_0-3)/(8*(nu_0-1))
    L[1,0,0,1] = (4*nu_0-3)/(8*(nu_0-1))
    L[0,1,1,0] = (4*nu_0-3)/(8*(nu_0-1))
    L[1,0,1,0] = (4*nu_0-3)/(8*(nu_0-1))
    
    L[0,0,2,2] = -nu_0/(2*(nu_0-1))
    L[1,1,2,2] = -nu_0/(2*(nu_0-1))
    
    L[2,0,0,2] = 1/2
    L[2,1,1,2] = 1/2
    L[2,0,2,0] = 1/2
    L[2,1,2,1] = 1/2
    
    return(L)

def eshelby_aux_oblate_spheroid(nu_0):
    """
    input : nu_0 poisson coefficient of the matrix
    output : L 3*3*3*3 tensor auxiliary eshelby tensor for infinite oblate spheroids
    
    """
    
    L = np.zeros((3,3,3,3))
    
    L[2,2,2,2] = 1
    
    L[2,2,0,0] = nu_0/(1-nu_0)
    L[2,2,1,1] = nu_0/(1-nu_0)
    
    L[0,2,2,0] = 1
    L[1,2,2,1] = 1
    L[0,2,0,2] = 1
    L[1,2,1,2] = 1    
    return(L)


def eshelby_aux_spheroid(nu_0, shape_ratio):
    """
    input : nu_0 poisson coefficient of the matrix
    output : L 3*3*3*3 tensor auxiliary eshelby tensor for cylinder 
    shape_ratio = a/c in [0,1], a=b. If shape_ratio > 1 -> oblate spheroid
    """

    d = shape_ratio**2
    
    R = (1-2*nu_0)/(8*np.pi*(1-nu_0))
    Q = 3/(8*np.pi*(1-nu_0))
    
    def integ_a(v, d):
        return(1/((d+v)**2*(1+v)**(1/2)))
    
    def integ_c(v, d):
        return(1/((d+v)*(1+v)**(3/2)))
    
    def integ_aa(v, d):
        return(1/((d+v)**3*(1+v)**(1/2)))
    
    def integ_cc(v, d):
        return(1/((d+v)*(1+v)**(5/2)))
    
    def integ_ac(v, d):
        return(1/((d+v)**2*(1+v)**(3/2)))
    
    i_a = integrate.quad(lambda x: integ_a(x, d), 0, np.inf)[0]
    i_aa = integrate.quad(lambda x: integ_aa(x, d), 0, np.inf)[0]

    i_c = 2/d - i_a
    i_ac = 2/d**2 - 4*i_aa
    i_cc = 2/(3*d) - 2/3*i_ac
    
    L = np.zeros((3,3,3,3))
    
    L[0,0,0,0] = 2*np.pi*Q*d**2*i_aa + 2*np.pi*R*d*i_a
    L[1,1,1,1] = L[0,0,0,0]
    L[2,2,2,2] = 2*np.pi*Q*d*i_cc + 2*np.pi*R*d*i_c
    
    L[0,0,1,1] = 2/3*np.pi*Q*d**2*i_aa - 2*np.pi*R*d*i_a
    L[1,1,0,0] = L[0,0,1,1]
    
    L[0,0,2,2] = 2/3*np.pi*Q*d*i_ac - 2*np.pi*R*d*i_a
    L[1,1,2,2] = L[0,0,2,2]
    
    L[2,2,0,0] = 2/3*np.pi*Q*d**2*i_ac - 2*np.pi*R*d*i_c
    L[2,2,1,1] = L[2,2,0,0]
    
    L[0,1,0,1] = 2/3*np.pi*Q*d**2*i_aa + 2*np.pi*R*d*i_a
    L[1,0,0,1] = L[0,1,0,1]
    L[0,1,1,0] = L[0,1,0,1]
    L[1,0,1,0] = L[0,1,0,1]
    
    L[0,2,0,2] = 2*np.pi*Q/3*d**2*i_ac + 2*np.pi*R*d*i_c
    L[0,2,2,0] = L[0,2,0,2]
    L[1,2,2,1] = L[0,2,0,2]
    L[1,2,1,2] = L[0,2,0,2] 
    
    L[2,0,0,2] = 2*np.pi*Q/3*d*i_ac + 2*np.pi*R*d*i_a
    L[2,0,2,0] = L[2,0,0,2]
    L[2,1,2,1] = L[2,0,0,2]
    L[2,1,1,2] = L[2,0,0,2]
    return(L)