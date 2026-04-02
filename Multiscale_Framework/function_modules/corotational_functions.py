import numpy as np
import ufl


"""
logs
# 13 Juin 2024 : creation
"""

##########################################################################
# Functions and framework for Corotational framework
##########################################################################

def eigenprojections(b, b_eigvalues, V_matrix, V_vector):
    """
    Function to compute the eigenvalues and eigenprojections of the left cauchy stretch tensor
    input : 
        - b field at current state
        - b_eigvalues : eigenvalues of the b field at current state
        - V_matrix : functionspace to create matrix field for each eigenprojection
        - V_vector : functionspace to create vector field for each eigenvalue
    """
    Id = ufl.Identity(3)
    
    list_b_eigproj = []
    for j in range(3):
        b_alpha = Id
        for k in range(3):
            if k!=j:
                factor = ufl.conditional( ufl.eq(b_eigvalues[k], b_eigvalues[j]), 0, 1/(b_eigvalues[j] - b_eigvalues[k]))
                b_alpha = factor*ufl.dot(b_alpha, (b - b_eigvalues[k]*Id))
        
        list_b_eigproj.append(b_alpha)
    
    return(list_b_eigproj)


def corot_function(d, list_b_eigproj, b_eigvalues, type_obj):
    """
    Create the corotational framework to define the objective stress rate
    Omega is the corotation used in the definition of Green-Naghdi, Jaumann and log
    how to use in current config, with dtau: 
    dtau_obj = dtau + tau.Omega - Omega.tau
    For Lie derivative : 
    dtau_lie = dtau - tau.l.T -l.tau 
    hence : dtau_lie = dtau_obj - tau.d - d.tau - tau.upsilon + upsilon.tau
    """
    upsilon = 0
    tol = 1e-5
    for j in range(3):
        for k in range(3):
            if j!=k:
                if type_obj =='Log':
                    factor = ufl.conditional( ufl.eq(b_eigvalues[k], b_eigvalues[j]), 0, (b_eigvalues[k] + b_eigvalues[j])/(b_eigvalues[k] - b_eigvalues[j]) + 2/ufl.ln(b_eigvalues[j]/b_eigvalues[k]))
                elif type_obj =='Green-Naghdi':
                    factor = ufl.conditional(ufl.le(b_eigvalues[k]- b_eigvalues[j], tol), 0, (ufl.sqrt(b_eigvalues[k]) - ufl.sqrt(b_eigvalues[j]))/(ufl.sqrt(b_eigvalues[k]) + ufl.sqrt(b_eigvalues[j])))
                
                skewproj = ufl.dot(list_b_eigproj[j], ufl.dot(d, list_b_eigproj[k]))
                upsilon += factor*skewproj

    return(upsilon)