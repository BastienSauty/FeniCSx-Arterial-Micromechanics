import numpy as np
import ufl
from dolfinx import mesh, fem
import sys

# Parameter classes
from Multiscale_Framework.class_modules.parameter_class import (
    cst_scalar,
    nn_cst_young_modulus,
    inverse_matrix6_6,
    matrix_volumic_frac,
    nn_uniform_param,
)

# Diverse functions to manage tensor product and orientation
from Multiscale_Framework.function_modules.auxiliary_functions import (
    Voigt2Tensor,
    Tensor2Voigt,
    projection_tensors,
    projection_tensors_func,
    expand,
    compress,
    tensordot_4_4,
    tensordot_4_2,
    change_of_base_matrix,
    local_basis,
    set_orientation_fields,
    tensors_local2global,
    init_identity_field,
)

# Eshelby inclusion functions
from Multiscale_Framework.function_modules.eshelby_inclusions import (
    eshelby_aux_cylindrical,
    eshelby_aux_oblate_spheroid,
    eshelby_aux_spheroid,
)

"""
Classes that define the matrix and inclusions, ie the microscale level components. 
They contain all the quantities related to microscale components: parameters, mechanical states...
These classes are defined consistently. They are all callable in the same way. That is that they all contain the same attributes and methods.
They are called and used in the material classes (indifferently HD, MT...) for the consitutive behavior.
When running the simulation, the main files directly access the inclusions to measure the mechanical states as an output.
/!\ for now no changes in inclusion shapes during finite strain -> a bit inconsistent

Classes :
    Matrix : defines the isotropic matrix 
    Spherical_inclusion : simple isotropic spherical inclusion
    Active_Spherical_inclusion : spherical inclusion that is subject to inelastic strain. type 'growing sphere' or 'homeostatic sphere'.
    Spheroidal_inclusion : general spheroidal inclusion. Used for passive cells and fibers (cylinder).
    Active_Spheroidal_inclusion : spheroid with inelastic strain. Used for cells with active stress regulation. 
    Prestretched_Cylinder_inclusion : cylinder with an inelastic coupling with Cells inclusion. Used for coupling collagen inelastic strain to cells with inelastic strain.
"""


###########################################################################
### Matrix
###########################################################################

class Matrix:
    """
    Object that contains all the informations about the matrix in the multiscale model. 
    
    Parameters:
    -----------
        mat : dict
            contains the parameters that describe the matrix : young modulus and poisson ratio.
        geom : dict
            contains the geometric parameters needed : space functions, cells
    
    Attributes:
    -----------
        Too many to list. Mains are:
            Mechanical parameters: stiffness parameters, volumic fractions. The latter is managed by parameter_class.matrix_volumic_fraction
            Mechanical state : deformation gradient Fn and Kirchhoff stress tensor taun
        
    Methods: These are called in a sequential order by the material class and the mech_problem_class
    -----------
        __init__ : 
            defines the matrix object. Run basic stiffness method.
        stiffness :
            creates the stiffness and compliance matrices. 6*6 matrices in Mandel Notations
        localization_tensors : 
            create the localization tensor from the RVE-to-remote equivalence tensor. ie the tensor that estimates the strain in the matrix. Called by material_class.homogenization_scheme
        inelastic_contribution :
            impact of the inelastic contribution of other inclusion on the matrix strain rate. Called by material_class.homogenization_scheme
        microscopic_mech :
            create the expressions to compute the matrix strain and stress tensors.
        update_micro_mech :
            compute the new mechanical state based on increment of macroscopic displacement or time for inelastic contribution. Called by mech_problem_class.update_local_quantities
    """
    def __init__(self, mat, geom):
        """
        Initialize matrix material parameters, compute Lamé moduli, allocate stress fields, and assemble stiffness tensors.
        """
        self.inel = False # flag for inelastic behavior
        domain = geom['domain']
        self.E = cst_scalar(mat["young"], geom['scalar spacefunction'])
        self.nu = cst_scalar(mat["poisson"], geom['scalar spacefunction'])
        self.V_vec = geom["vector spacefunction"]
        self.V_stiff = geom["stiff spacefunction"]
        self.V_mandel = geom["mandel spacefunction"]
        self.V_mat = geom["matrix spacefunction"]
        self.cells = geom["cells"]
        self.obj_der = geom["objective derivative"] # Jaumann
        self.f = matrix_volumic_frac(geom['scalar spacefunction'], geom['cells']) # volumic fraction may or not be homogenous; special class
        
        self.mu = self.E[0]/(2*(1+self.nu[0]))
        self.lambda_ = self.E[0]*self.nu[0]/(1+self.nu[0])/(1-2*self.nu[0])
        self.k = self.E[0]/(3*(1-2*self.nu[0]))
        
        self.taun = fem.Function(self.V_mandel)
        
        self.stiffness()
    
    def stiffness(self):
        """ Define the stiffness tensor in Mandel notation (6*6 matrix) callable by self.C and its inverse the compliance tensor S. """
        J_m, K_m, I_m = projection_tensors_func(self.V_stiff)
        self.C = 2*self.mu*K_m + 3*self.k*J_m
        self.S = 1/(2*self.mu)*K_m + 1/(3*self.k)*J_m
    
    def set_parameters(self, mat):
        """
        Push new values for the matrix's physical parameters, in place, without
        rebuilding any form (no recompilation). mat is a (possibly partial) dict
        shaped like the json card's "matrix" entry, e.g. {"young": 0.06} or
        {"young": 0.06, "poisson": 0.44}.
        self.mu, self.lambda_, self.k, self.C, self.S remain valid automatically:
        they are ufl expressions built from self.E/self.nu, so they pick up the
        new values at the next assembly without needing to be rebuilt.
        """
        if "young" in mat:
            self.E.set_value(mat["young"])
        if "poisson" in mat:
            self.nu.set_value(mat["poisson"])

    def localization_tensors(self, M):
        """
        create in place A_m the localization tensor that estimates the strain in the matrix, from tensor M.
        Beware that in the case of Mori-Tanaka homogenization, it is not equal to the RVE-to-Remote tensor that is used for the other inclusions. 
        In that case see material_class.Homogenized_MT_material and the line self.matrix.localization_tensors(A_m/self.matrix.f.func[0]).
        M is a ufl 6*6 tensor.
        """
        self.A_m = M
        
    def inelastic_contribution(self, delta_i):
        """Store the form representing the inelastic strain contribution arising from the microscopic equilibrium."""
        self.delta_i = delta_i
        
    def microscopic_mech(self, l_el, l_elj, delta_t):
        """
        Builds the expressions to compute the localization of the strain rate from RVE to matrix, impact of inelastic strain rate and the resulting stress rate. 
        Parameters : 
            l_el, l_elj : respectively the velocity gradient in direction du and its incremental counterpart in direction duj
            delta_t : increment of time for inelastic components.
        New attributes created here :
            dtau_dirder, dtau_incr : stress rate in direction du, and incremental counterpart. The first one is used in the Newton Raphson linearization, the second in computing the increment.
            dtau_expr, dF_expr : fenicsx expression that computes respectively the increment of stress and the increment of deformation gradient in the matrix in the whole domain.
        """
        self.dtau = fem.Function(self.V_mandel)
        self.dF = fem.Function(self.V_mat)
        
        # Deformation gradient for the inclusion
        self.Fn = fem.Function(self.V_mat)
        init_identity_field(self.Fn)

        d_macro_dirder = ufl.sym(l_el)
        d_macro_incr = ufl.sym(l_elj)
        
        # Objective derivative
        Am = expand(self.A_m, antisym=[1,1])
        
        if self.delta_i is None:
            self.d_dirder = tensordot_4_2(Am, d_macro_dirder)
            self.d_incr =tensordot_4_2(Am, d_macro_incr)
            
        else:
            self.d_dirder = tensordot_4_2(Am, d_macro_dirder) + delta_t*self.delta_i        
            self.d_incr = tensordot_4_2(Am, d_macro_incr) + delta_t*self.delta_i
            
        self.dtau_dirder = ufl.dot(self.C, Tensor2Voigt(self.d_dirder))
        self.dtau_incr = ufl.dot(self.C, Tensor2Voigt(self.d_incr))
        
        # Stress and Strain increment expressions
        self.dtau_expr = fem.Expression(self.dtau_incr, self.V_mandel.element.interpolation_points()) # local increment of stress
        self.dF_expr = fem.Expression(ufl.dot((self.d_incr), self.Fn), self.V_mat.element.interpolation_points()) 
        
    
    def update_micro_mech(self):
        """
        update micro mechanics in matrix for an increment of duj macro displacement field : deformation gradient Fn and kirchhoff stress taun
        """
        # update stresses
        self.dtau.interpolate(self.dtau_expr, self.cells)
        self.dtau.x.scatter_forward()
        self.taun.x.array[:] += self.dtau.x.array[:] # update cauchy stresses in inclusions
        self.taun.x.scatter_forward()
        
        # update strain
        self.dF.interpolate(self.dF_expr, self.cells)
        self.dF.x.scatter_forward()
        self.Fn.x.array[:] += self.dF.x.array[:] # store total strain
        self.Fn.x.scatter_forward()
        
###########################################################################
### Sphere
###########################################################################

class Spherical_inclusion:
    """
    Object that contains all the informations about the spherical inclusion in the multiscale model. 
    
    Parameters:
    -----------
        mat : dict
            contains the parameters that describe the matrix : young modulus, poisson ratio, volumic fraction.
        geom : dict
            contains the geometric parameters needed : space functions, cells
    
    Attributes:
    -----------
        Too many to list. Mains are:
            Mechanical parameters: stiffness parameters, volumic fractions.
            Mechanical state : deformation gradient Fn and Kirchhoff stress tensor taun
            For building the localization tensors using the Eshelby theory, the object also contains the matrix stiffness parameters mu0 and k0. These cannot be changed during a simulation.
        
    Methods: These are called in a sequential order by the material class and the mech_problem_class
    -----------
        __init__ : 
            defined the spherical inclusion object. Run some basic methods : stiffness, inf_localization_tensors.
        stiffness :
            creates the stiffness and compliance matrices. 6*6 matrices in Mandel Notations
        inelastic_contribution :
            impact of the inelastic contribution of other inclusion on the spherical inclusion strain rate. Called by material_class.homogenization_scheme.
            No internal contribution here -> only in growing sphere and other inclusions objects.
        inf_localization_tensors :
            Create the infinite localization tensor defined by the theory of eshelby for a single inclusion in an infinite matrix.
        localization_tensors : 
            build the localization tensors A_i using the RVE to remote equivalence tensors (M) and the infinite locazlition tensor from the Eshelby theory (A_inf).
            Called by material_class.homogenization_scheme
        microscopic_mech :
            create the expressions to compute the sphere strain and stress tensors.
            Called by material_class.homogenization_scheme
        update_micro_mech :
            compute the new mechanical state based on increment of macroscopic displacement or time for inelastic contribution. 
            Called by mech_problem_class.update_local_quantities
    """
    
    def __init__(self, mat, geom):
        """
        Initialize inclusion material parameters, compute Lamé moduli, allocate stress fields, and assemble stiffness/localization tensors.
        """
        self.inel = False # flag for inelastic behavior
        # Inclusion properties
        domain = geom['domain']
        self.nu = fem.Constant(domain, np.float64(mat["poisson"]))
        self.f = cst_scalar(mat["volumic_fraction"], geom['scalar spacefunction']) # scalar field with constant value
        self.E = cst_scalar(mat["young"], geom['scalar spacefunction'])
        
        self.V_stiff = geom["stiff spacefunction"]
        self.V_mandel = geom["mandel spacefunction"]
        self.V_mat = geom["matrix spacefunction"]
        self.cells = geom["cells"]
        self.obj_der = geom["objective derivative"]
        
        # numeric snapshot (plain float), see material_class.py module docstring
        # - required because eshelby_aux_spheroid() runs a genuine numerical
        # integration (scipy.integrate.quad) incompatible with a symbolic value
        self.mu_0 = cst_scalar(mat["mu_0"], geom['scalar spacefunction']) # matrix values, used in localization
        self.k_0 = cst_scalar(mat["k_0"], geom['scalar spacefunction'])
        
        self.mu = self.E.func[0]/(2*(1+self.nu))
        self.lambda_ = self.E.func[0]*self.nu/(1+self.nu)/(1-2*self.nu)
        self.k = self.E.func[0]/(3*(1-2*self.nu))
        
        # define the stored stress for each inclusion to define the Jauman stress rate
        self.taun = fem.Function(self.V_mandel)

        # isotropic stiffness self.C
        self.stiffness()
        
        # form of the localization tensor used in the high dilution homogenization
        self.inf_localization_tensors()
        
        
    def stiffness(self):
        """ Define the stiffness tensor in Voigt notation callable by self.C """
        J_m, K_m, I_m = projection_tensors_func(self.V_stiff) # mandel notations
        self.I_m = I_m
        self.J_m = J_m
        self.K_m = K_m
        self.C = 2*self.mu*self.K_m + 3*self.k*self.J_m # function defined on the V_stiff functionspace
        
    def inf_localization_tensors(self):
        """
        Define the A_inf localization tensor used in the HD and MT for a spherical inclusion
        To reduce inverse computation, based on the decomposition in the projectors base 
        """
        # _isoproj = {3*k, 2*mu} -> 3*k*J_m + 2*mu*K_m with J_m the trace projector and K_m the deviatoric projector 
        Cm_isoproj = np.array([3*self.k_0[0], 2*self.mu_0[0]])
        Cp_isoproj = np.array([3*self.k, 2*self.mu])
    
        # P Hill tensor here is defined for a spherical inclusion
        Ppinv_isoproj = np.array([3*self.k_0[0]+4*self.mu_0[0], 5*self.mu_0[0]*(3*self.k_0[0]+4*self.mu_0[0])/(3*(self.k_0[0] + 2*self.mu_0[0]))])
        Cstar_isoproj = Ppinv_isoproj - Cm_isoproj

        A_isoproj = np.reciprocal(Cstar_isoproj + Cp_isoproj)*(Cstar_isoproj + Cm_isoproj) # inverse and multiplications are made element wise in the 2 component array
        self.A_inf = A_isoproj[0]*self.J_m + A_isoproj[1]*self.K_m 
        self.R_inf = 0*self.K_m
        
            
    def inelastic_contribution(self, delta_i):
        """Store the form representing the inelastic strain contribution arising from the microscopic equilibrium."""
        self.delta_i = delta_i
        
    def localization_tensors(self, M):
        """
        create in place A_i and R_i the localization tensor that estimates the strain and rotation in the inclusion from tensor M.
        M is a ufl 6*6 tensor, it is the RVE-to-Remote tensor. Called by material_class.homogenization_scheme()
        """
        self.A_i = ufl.dot(self.A_inf, M)
        self.R_i = ufl.dot(self.R_inf, M)
        
    def microscopic_mech(self, l_el, l_elj, delta_t):
        """
        Builds the expressions to compute the localization of the strain rate from RVE to inclusion, impact of inelastic strain rate and the resulting stress rate. 
        Parameters : 
            l_el, l_elj : respectively the velocity gradient in direction du and its incremental counterpart in direction duj
            delta_t : increment of time for inelastic components.
        New attributes created here :
            dtau_dirder, dtau_incr : stress rate in direction du, and incremental counterpart. The first one is used in the Newton Raphson linearization, the second in computing the increment.
            dtau_expr, dF_expr : fenicsx expression that computes respectively the increment of stress and the increment of deformation gradient in the matrix in the whole domain.
        """
        self.dtau = fem.Function(self.V_mandel)
        self.dF = fem.Function(self.V_mat)
        
        # Deformation gradient for the inclusion
        self.Fn = fem.Function(self.V_mat)
        init_identity_field(self.Fn)
        
        d_macro_dirder = ufl.sym(l_el)
        d_macro_incr = ufl.sym(l_elj)
          
        # Objective derivative 
        Ai = expand(self.A_i, antisym=[1,1])
        
        if self.delta_i is None:
            self.d_dirder = tensordot_4_2(Ai, d_macro_dirder) 
            self.d_incr = tensordot_4_2(Ai, d_macro_incr) 
            
        else:
            self.d_dirder =  tensordot_4_2(Ai, d_macro_dirder) + delta_t*self.delta_i
            self.d_incr = tensordot_4_2(Ai, d_macro_incr)  + delta_t*self.delta_i
            
        self.dtau_dirder = ufl.dot(self.C, Tensor2Voigt(self.d_dirder))
        self.dtau_incr = ufl.dot(self.C, Tensor2Voigt(self.d_incr))
        
        # Stress and Strain increment expressions
        self.dtau_expr = fem.Expression(self.dtau_incr, self.V_mandel.element.interpolation_points()) # local increment of stress
        self.F_expr = fem.Expression(ufl.dot(self.d_incr, self.Fn), self.V_mat.element.interpolation_points()) 
        
        
        
    def set_parameters(self, mat):
        """
        Push new values for this inclusion's young modulus and volumic
        fraction, in place, without rebuilding any form (no recompilation).
        mat is a (possibly partial) dict shaped like the corresponding json
        card entry. Poisson ratio is NOT covered here (kept as a plain
        fem.Constant on this class, not yet migrated to cst_scalar) - update
        it by rebuilding via setup_simulation() if it needs to change.
        """
        if "young" in mat:
            self.E.set_value(mat["young"])
        if "volumic_fraction" in mat:
            self.f.set_value(mat["volumic_fraction"])

    def update_micro_mech(self):
        """
        update micro mechanics in inclusion for an increment of duj macro displacement field : deformation gradient Fn and kirchhoff stress taun
        """
        self.dtau.interpolate(self.dtau_expr, self.cells)
        self.taun.x.array[:] += self.dtau.x.array[:] # update cauchy stresses in inclusions
        self.dtau.x.scatter_forward()
        self.taun.x.scatter_forward()
                        
        # update strain
        self.dF.interpolate(self.F_expr, self.cells)
        self.dF.x.scatter_forward()
        
        self.Fn.x.array[:] += self.dF.x.array[:] # store total strain
        self.Fn.x.scatter_forward()
        
        

###########################################################################
### Active Sphere
###########################################################################

class Active_Spherical_inclusion:
    """
    Object that contains all the informations about the Active spherical inclusion in the multiscale model. 
    This class contains some inelastic behavior that is controled in method self.inelastic_strain(). Two types : growing sphere and homeostatic sphere.
    
    Parameters:
    -----------
        mat : dict
            contains the parameters that describe the matrix : young modulus, poisson ratio, volumic fraction.
        geom : dict
            contains the geometric parameters needed : space functions, cells
    
    Attributes:
    -----------
        Too many to list. Mains are:
            Mechanical parameters: stiffness parameters, volumic fractions.
            Mechanical state : deformation gradient Fn and Kirchhoff stress tensor taun
            For building the localization tensors using the Eshelby theory, the object also contains the matrix stiffness parameters mu0 and k0. These cannot be changed during a simulation.
        
    Methods: These are called in a sequential order by the material class and the mech_problem_class
    -----------
        __init__ : 
            defined the spherical inclusion object. Run some basic methods : stiffness, inf_localization_tensors, inelastic_strain.
        stiffness :
            creates the stiffness and compliance matrices. 6*6 matrices in Mandel Notations
        inf_localization_tensors :
            Create the infinite localization tensor defined by the theory of eshelby for a single spherical inclusion in an infinite matrix.
        inelastic_strain :
            build the inelastic strain forms within the active spherical inclusion. Two types : growing sphere and homeostatic sphere.
        inelastic_contribution :
            impact of the inelastic contribution of other inclusion on the spherical inclusion strain rate. Called by material_class.homogenization_scheme.
            No internal contribution here -> only in growing sphere and other inclusions objects.
        localization_tensors : 
            build the localization tensors A_i using the RVE to remote equivalence tensors (M) and the infinite locazlition tensor from the Eshelby theory (A_inf).
            Called by material_class.homogenization_scheme
        microscopic_mech :
            creates the expressions to compute the sphere strain and stress tensors. 
            Called by material_class.homogenization_scheme
        update_micro_mech :
            compute the new mechanical state based on increment of macroscopic displacement or time for inelastic contribution. 
            Called by mech_problem_class.update_local_quantities
    """
    def __init__(self, mat, geom):
        """
        Initialize inclusion material parameters, compute Lamé moduli, allocate stress fields, assemble stiffness/localization tensors and build inelastic strain expressions.
        """
        self.inel = True # flag for inelastic behavior
        
        self.type = mat["type"]
        
        # Inclusion properties
        domain = geom['domain']
        self.nu = fem.Constant(domain, np.float64(mat["poisson"]))
        self.f = cst_scalar(mat["volumic_fraction"], geom['scalar spacefunction']) # scalar field with constant value
        self.E = cst_scalar(mat["young"], geom['scalar spacefunction'])
        
        self.V_stiff = geom["stiff spacefunction"]
        self.V_mandel = geom["mandel spacefunction"]
        self.V_mat = geom["matrix spacefunction"]
        self.cells = geom["cells"]
        self.obj_der = geom["objective derivative"]
        
        # Homeostatic behavior
        if self.type =="growing sphere":
            
            self.t = fem.Constant(domain, np.float64(0.))
            
            self.alpha = fem.Constant(domain, np.float64(mat["alpha"])) # maximum strain value
            self.t_c = fem.Constant(domain, np.float64(mat["characteristic time"])) # caracteristic time
        elif self.type =="homeostatic sphere":
            self.tau_b = fem.Constant(domain, np.float64(mat["basal stress"])) # basal hydrostatic stress
            self.t_c = fem.Constant(domain, np.float64(mat["characteristic time"])) # caracteristic time
            
        
        # numeric snapshot (plain float), see material_class.py module docstring
        # - required because eshelby_aux_spheroid() runs a genuine numerical
        # integration (scipy.integrate.quad) incompatible with a symbolic value
        self.mu_0 = cst_scalar(mat["mu_0"], geom['scalar spacefunction']) # matrix values, used in localization
        self.k_0 = cst_scalar(mat["k_0"], geom['scalar spacefunction'])
        
        self.mu = self.E.func[0]/(2*(1+self.nu))
        self.lambda_ = self.E.func[0]*self.nu/(1+self.nu)/(1-2*self.nu)
        self.k = self.E.func[0]/(3*(1-2*self.nu))
        
        
        # isotropic stiffness self.C
        self.stiffness()
        
        # form of the localization tensor used in the high dilution homogenization
        self.inf_localization_tensors()
        
        
        # Mechanical fields associated with the inclusion
        # define the stored stress for each inclusion to define the Jauman stress rate
        self.taun = fem.Function(self.V_mandel)
        # Deformation gradient for the inclusion
        self.Fn = fem.Function(self.V_mat)
        self.F_inel = fem.Function(self.V_mat)
        # fill the deformation gradient field at identity
        init_identity_field(self.Fn)
        init_identity_field(self.F_inel)
                
        self.inelastic_strain() # initialize the F_dot_inel and d_inel using the updating function with dt=0

        
        
    def stiffness(self):
        """ Define the stiffness tensor in Voigt notation callable by self.C """
        J_m, K_m, I_m = projection_tensors_func(self.V_stiff) # mandel notations
        self.I_m = I_m
        self.J_m = J_m
        self.K_m = K_m
        self.C = 2*self.mu*self.K_m + 3*self.k*self.J_m # function defined on the V_stiff functionspace
        
    def inf_localization_tensors(self):
        """
        Define the A_inf localization tensor used in the HD and MT for a spherical inclusion
        To reduce inverse computation, based on the decomposition in the projectors base 
        """
        # _isoproj = {3*k, 2*mu} -> 3*k*J_m + 2*mu*K_m with J_m the trace projector and K_m the deviatoric projector 
        Cm_isoproj = np.array([3*self.k_0[0], 2*self.mu_0[0]])
        Cp_isoproj = np.array([3*self.k, 2*self.mu])
    
        # P Hill tensor here is defined for a spherical inclusion
        Ppinv_isoproj = np.array([3*self.k_0[0]+4*self.mu_0[0], 5*self.mu_0[0]*(3*self.k_0[0]+4*self.mu_0[0])/(3*(self.k_0[0] + 2*self.mu_0[0]))])
        Cstar_isoproj = Ppinv_isoproj - Cm_isoproj
        
        # Hill tensor
        Pp_isoproj = np.reciprocal(Ppinv_isoproj)
        self.P = Pp_isoproj[0]*self.J_m + Pp_isoproj[1]*self.K_m 

        A_isoproj = np.reciprocal(Cstar_isoproj + Cp_isoproj)*(Cstar_isoproj + Cm_isoproj) # inverse and multiplications are made element wise in the 2 component array
        self.A_inf = A_isoproj[0]*self.J_m + A_isoproj[1]*self.K_m 
        self.R_inf = 0*self.K_m
        
        self.D_inf = ufl.dot(ufl.dot(self.A_inf, self.P), self.C)
        self.T_inf = 0*self.K_m
        
            
    def inelastic_strain(self):
        """
        Manage inelastic part of F. 
        This is based on Madge Martin work for growing sphere
        Adaptation of Federica Galbati's work for homeostatic sphere
        
        delta_t : fem.Constant(mech.domain,np.array([0,0], dtype=np.float64))
                    constant on the volume that controls the increment of real physical time.
        """
        if self.type == "growing sphere":
            #self.F_inel_form = (1+self.alpha*(1-np.exp(-self.t/self.t_c)))*ufl.Identity(3)
            #self.F_el = ufl.dot(self.Fn, ufl.inv(self.F_inel_form)) # since its an identity matrix, the inversion should be way easier
            self.F_el = ufl.dot(self.Fn, ufl.inv(self.F_inel))
            # self.sig_mes = self.alpha/self.t_c*ufl.exp(-self.t/self.t_c)
            self.F_dot_inel = self.alpha/self.t_c*ufl.exp(-self.t/self.t_c)*ufl.Identity(3)
            
            l_inel = ufl.dot(self.F_el, ufl.dot(self.F_dot_inel, ufl.inv(self.Fn)))
            self.d_inel = 1/2*(l_inel + l_inel.T)
            
        elif self.type == "homeostatic sphere":
            
            J = ufl.det(self.Fn)
            self.sig_mes = ufl.tr(Voigt2Tensor(self.taun)/J)
    
            self.d_inel = 1/(self.t_c*self.tau_b)*(self.sig_mes - self.tau_b)*ufl.Identity(3)
            
            F_el_inv = ufl.dot(self.F_inel, ufl.inv(self.Fn)) 
            self.F_dot_inel = ufl.dot(F_el_inv, ufl.dot(self.d_inel, self.Fn))

        
    def inelastic_contribution(self, delta_i):
        """Store the form representing the inelastic strain contribution arising from the microscopic equilibrium."""
        self.delta_i = delta_i
        
    def localization_tensors(self, M):
        """
        caller : Homogenized_material.estimation_local_tensors
        compute the localization tensor for a spherical inclusion.
        M depends on the homogenization scheme. For HD Identity. For MT it's more complex
        """
        self.A_i = ufl.dot(self.A_inf, M)
        self.R_i = ufl.dot(self.R_inf, M)
        
    def microscopic_mech(self, l_el, l_elj, delta_t):
        """
        Builds the expressions to compute the localization of the strain rate from RVE to inclusion, impact of inelastic strain rate and the resulting stress rate. 
        Parameters : 
            l_el, l_elj : respectively the velocity gradient in direction du and its incremental counterpart in direction duj
            delta_t : increment of time for inelastic components.
        New attributes created here :
            dtau_dirder, dtau_incr : stress rate in direction du, and incremental counterpart. The first one is used in the Newton Raphson linearization, the second in computing the increment.
            dtau_expr, dF_expr : fenicsx expression that computes respectively the increment of stress and the increment of deformation gradient in the matrix in the whole domain.
            dF_inel_expr : fenicsx expression that computes the increment of inelastic deformation gradient.
        """
        self.dtau = fem.Function(self.V_mandel)
        self.dF = fem.Function(self.V_mat)
        self.dF_inel = fem.Function(self.V_mat)
        
        d_macro_dirder = ufl.sym(l_el)
        d_macro_incr = ufl.sym(l_elj)
          
        # Objective derivative
        Ai = expand(self.A_i, antisym=[1,1])
        
        if self.delta_i is None:
            self.d_dirder = tensordot_4_2(Ai, d_macro_dirder)
            self.d_incr = tensordot_4_2(Ai, d_macro_incr)
        else:
            self.d_dirder = tensordot_4_2(Ai, d_macro_dirder) + delta_t*self.delta_i
            self.d_incr = tensordot_4_2(Ai, d_macro_incr) + delta_t*self.delta_i
        
        self.dtau_dirder = ufl.dot(self.C, Tensor2Voigt(self.d_dirder - delta_t*self.d_inel))
        self.dtau_incr = ufl.dot(self.C, Tensor2Voigt(self.d_incr - delta_t*self.d_inel))

        # Stress and Strain increment expressions
        self.dtau_expr = fem.Expression(self.dtau_incr, self.V_mandel.element.interpolation_points()) # local increment of stress
        self.dF_expr = fem.Expression(ufl.dot(self.d_incr, self.Fn), self.V_mat.element.interpolation_points()) 
        
        # Manage inelastic increments
        self.dF_inel_expr = fem.Expression(delta_t*self.F_dot_inel, self.V_mat.element.interpolation_points()) 
        self.delta_t = delta_t
        
        
    def set_parameters(self, mat):
        """
        Push new values for this inclusion's young modulus and volumic
        fraction, in place, without rebuilding any form (no recompilation).
        mat is a (possibly partial) dict shaped like the corresponding json
        card entry. Poisson ratio is NOT covered here (kept as a plain
        fem.Constant on this class, not yet migrated to cst_scalar) - update
        it by rebuilding via setup_simulation() if it needs to change.
        """
        if "young" in mat:
            self.E.set_value(mat["young"])
        if "volumic_fraction" in mat:
            self.f.set_value(mat["volumic_fraction"])

    def update_micro_mech(self):
        """
        update micro mechanics in inclusion for an increment of duj macro displacement field : deformation gradient Fn and kirchhoff stress taun
        """
        self.dtau.interpolate(self.dtau_expr, self.cells)
        self.taun.x.array[:] += self.dtau.x.array[:] # update cauchy stresses in inclusions
        self.dtau.x.scatter_forward()
        self.taun.x.scatter_forward()
                
        # compute increment of strain
        self.dF.interpolate(self.dF_expr, self.cells)
        self.dF.x.scatter_forward()
        self.dF_inel.interpolate(self.dF_inel_expr, self.cells)
        self.dF_inel.x.scatter_forward()
        
        # update strain deformation gradients
        self.Fn.x.array[:] += self.dF.x.array[:] # store total strain
        self.Fn.x.scatter_forward()   
        self.F_inel.x.array[:] += self.dF_inel.x.array[:] 
        self.F_inel.x.scatter_forward()   
        
        # Update local time
        if self.type =="growing sphere":
            self.t.value += self.delta_t.value
        
        
        

###########################################################################
### Cylinder and others, just depend on the shape ratio
###########################################################################

class Spheroidal_inclusion:
    """
    Object that contains all the informations about the spheroidal inclusion in the multiscale model. 
    General inclusion for spheroids, cylinders and oblate spheroids BUT only for passive mechanics.
    A bit more complex than spherical inclusion since these are not isotropic anymore.
    
    Parameters:
    -----------
        mat : dict
            contains the parameters that describe the matrix : young modulus, poisson ratio, volumic fraction, initial orientation, shape ratio.
        geom : dict
            contains the geometric parameters needed : space functions, cells
    
    Attributes:
    -----------
        Too many to list. Mains are:
            Mechanical parameters: stiffness parameters, volumic fractions. 
            Mechanical state : deformation gradient Fn and Kirchhoff stress tensor taun and orientation fields
            For building the localization tensors using the Eshelby theory, the object also contains the matrix stiffness parameters mu0 and k0. These cannot be changed during a simulation.
        
    Methods: These are called in a sequential order by the material class and the mech_problem_class
    -----------
        __init__ : 
            defined the spherical inclusion object. Run some basis methods for the inclusion: stiffness, init_orientation, set_orientation, eshelby_isomatrix, inf_localization_tensors.
        stiffness :
            creates the stiffness and compliance matrices. 6*6 matrices in Mandel Notations
        init_orientation : 
            allocate and initialize the orientation fields e_r, e_theta and e_phi based on initial angles theta and phi. 
            See Multiscale_Framework.function_modules.auxiliary_functions change_of_base_matrix
        eshelby_isomatrix : 
            builds the auxiliary tensor of eshelby for cylinder or prolate spheroid, in the inclusion LRS. Then rotate it into the RVE orientation basis.
            Builds both S_esh and R_esh for strain and rotation localization. Beware that R_esh has small antisymmetry. 
        inf_localization_tensors :
            Create the infinite localization tensor defined by the theory of eshelby for a single inclusion in an infinite matrix.
        localization_tensors : 
            build the localization tensors A_i and R_i using the RVE to remote equivalence tensors (M) and the infinite locazlition tensor from the Eshelby theory (A_inf) and (R_inf).
            Beware of small antisymmetry in R_inf.
            Called by material_class.homogenization_scheme
        inelastic_contribution :
            impact of the inelastic contribution of other inclusion on the spheroidal inclusion strain rate. Called by material_class.homogenization_scheme.
            No internal contribution here -> only in growing sphere and active spheroids  objects.
        microscopic_mech :
            create the expressions to compute the spheroidal strain and stress tensors. And orientation fields and non-constant young modulus.
            Called by material_class.homogenization_scheme
        update_micro_mech :
            compute the new mechanical state based on increment of macroscopic displacement or time for inelastic contribution. 
            Called by mech_problem_class.update_local_quantities
    """
    def __init__(self, mat, geom): # aggiungere condizone eleif oer aggiornare il odulo di young
        """
        Initialize inclusion material parameters, compute Lamé moduli, orientation fields, allocate stress fields, assemble stiffness/localization tensors.
        """
        self.inel = False # flag for inelastic behavior
        self.type = mat["type"]
        domain = geom['domain']
        self.nu = cst_scalar(mat["poisson"], geom['scalar spacefunction'])
        
        # Volumic Fraction
        if (type(mat["volumic_fraction"]) is float) or (type(mat["volumic_fraction"]) is np.float64):
            self.f = cst_scalar(mat["volumic_fraction"], geom['scalar spacefunction'])
        elif (type(mat["volumic_fraction"]) is list or type(mat["volumic_fraction"]) is np.ndarray): #--> in case of non constant volume fraction
            self.f = nn_uniform_param(mat["volumic_fraction"], geom['scalar spacefunction'], geom['cells']) 
        else:
            raise Exception("wrong data type for Volumic fraction") 

        # Young Modulus properties
        if mat["young_type"]=="Constant":
            self.E = cst_scalar(mat["young"], geom['scalar spacefunction'])
        elif mat["young_type"]=="Exponential" or mat["young_type"]=="Plateau-Ramp-Plateau":
            self.E = nn_cst_young_modulus(mat["young_type"], mat["young"], geom['scalar spacefunction'], geom['domain'], geom['cells'])
        else:
            raise Exception('wrong type for Young Modulus. Please specify "young_type": as either "Constant", "Plateau-Ramp-Plateau" or "Exponential"') 

        self.mu = self.E.func[0]/(2*(1+self.nu[0]))
        self.k = self.E.func[0]/(3*(1-2*self.nu[0]))
        
        # Reference mechanical properties (matrix)
        # numeric snapshot (plain float), see material_class.py module docstring
        self.mu_0 = cst_scalar(mat["mu_0"], geom['scalar spacefunction'])
        self.k_0 = cst_scalar(mat["k_0"], geom['scalar spacefunction'])
        
        # Cylinder orientation (kept in radians ; degrees stored too for convenience)
        self.theta = np.pi/180*mat["theta"]
        self.phi = np.pi/180*mat["phi"]
        # spheroid shape ratio
        if self.type=="prolate_spheroid":
            self.shape_ratio = cst_scalar(mat["shape_ratio"], geom['scalar spacefunction'])
        
        # Space functions 
        self.V_vec = geom["vector spacefunction"] # specifically used for cylinder and oblate spheroids, hence in the mat dictionnary
        self.V_stiff = geom["stiff spacefunction"]
        self.V_mandel = geom["mandel spacefunction"]
        self.V_scalar = geom["scalar spacefunction"]
        self.V_mat = geom["matrix spacefunction"]
        self.cells = geom["cells"]
        self.obj_der = geom["objective derivative"]
        
        # Mechanical fields associated with the inclusion
        # define the stored stress for each inclusion to define the Jauman stress rate
        self.taun = fem.Function(self.V_mandel)
        # Deformation gradient for the inclusion
        self.Fn = fem.Function(self.V_mat)
        init_identity_field(self.Fn)
        
        # Orientation of fibers -> define the 3 fields for the basis vector of fibers orientation 
        self.init_orientation()
        
        # Stiffness self.C
        self.stiffness()
        
        # Eshelby Isomatrix for fibers
        self.eshelby_isomatrix()
        
        # Create the Form of the localization tensor used in the homogenization scheme
        self.inf_localization_tensors()
        
        
    def stiffness(self):
        """ Define the stiffness tensor in Voigt notation callable by self.C """
        J_m, K_m, I_m = projection_tensors_func(self.V_stiff) # mandel notations
        self.J_m = J_m
        self.K_m = K_m
        self.I_m = I_m
        self.C = 2*self.mu*self.K_m + 3*self.k*self.J_m # function defined on the V_stiff functionspace
        

    def init_orientation(self):
        """
        called by : spheroidal_inclusion.__init__
        Create and fill with their initial values the vector functions containing the three basis vector for the local basis
        """
        self.e_theta = fem.Function(self.V_vec) 
        self.e_phi = fem.Function(self.V_vec)
        self.e_r = fem.Function(self.V_vec)
        
        # Create change of basis matrix
        self.Pass = change_of_base_matrix(self.e_theta, self.e_phi, self.e_r)    
        # set_orientation : fill the dofs
        # Compute orientation vectors from theta and phi value
        [e_theta_vec, e_phi_vec, e_r_vec] = local_basis(self.theta, self.phi) # in auxiliary function file
        # Fill the dofs to get a constant field on the volume of orientation vectors
        set_orientation_fields(self.e_theta,self.e_phi,self.e_r, e_theta_vec, e_phi_vec, e_r_vec)
    
    def eshelby_isomatrix(self):
        """
        called by : spheroidal_inclusion.__init__
        Solution of the eshelby problem for a cylinder aligned with the third vector of the local basis : [e_theta, e_phi, e_r]. Based on shape of inclusion and stiffness matrix of reference C0

        nu_0 (and shape_ratio, for the spheroid case) MUST be plain python
        floats here : eshelby_aux_spheroid() runs a genuine scipy.integrate.quad
        numerical integration, which cannot accept a symbolic UFL value - so
        we read .value (the numeric snapshot side of cst_scalar) rather than
        indexing mu_0/k_0/shape_ratio as UFL Constants ([0]). See
        material_class.py's module docstring for why mu_0/k_0 are numeric
        snapshots in the first place.
        """
        nu_0 = (3*self.k_0.value - 2*self.mu_0.value)/(2*(3*self.k_0.value + self.mu_0.value))
        if self.type == 'cylinder':
            self.L = ufl.as_tensor(eshelby_aux_cylindrical(nu_0))
        elif self.type == 'prolate_spheroid':
            self.L = ufl.as_tensor(eshelby_aux_spheroid(nu_0, self.shape_ratio.value))
            
        i, j, k, l = ufl.indices(4)
        
        self.S0 = ufl.as_tensor(1/2*(self.L[i,j,k,l] + self.L[j,i,k,l]), (i,j,k,l))
        self.R0 = ufl.as_tensor(1/2*(self.L[i,j,k,l] - self.L[j,i,k,l]), (i,j,k,l))
        
        # expanded 3*3*3*3 with adapted symmetries, rotate, then compress
        self.S_esh = compress(tensors_local2global(self.S0, self.Pass)) # small symmetrical
        self.R_esh = tensors_local2global(self.R0, self.Pass) # fisrt small antisymmetry BEWARE -> HENCE IN TENSOR NOTATIONS
        
    def inf_localization_tensors(self):
        """
        Define the A_inf localization tensor used in the HD and MT for a cylindrical inclusion
        For cylinders, cannot be based on the decomposition in the projectors base because of the anisotropic behavior. Hence directly in Mandel Notation.
        Notations used are the ones from ZAMM paper Morin2018
        """
        J,K,I = projection_tensors()
        C0 = 2*self.mu_0[0]*ufl.as_matrix(K) + 3*self.k_0[0]*ufl.as_matrix(J)
        C0inv = 1/(2*self.mu_0[0])*ufl.as_matrix(K) + 1/(3*self.k_0[0])*ufl.as_matrix(J)
        
        self.P = ufl.dot(self.S_esh, C0inv) # Symmetric Mandel notation 6*6 tensors
        # A_inf is a matrix build on its inverse expression. Needs to be called with A_inf.func
        self.A_inf = inverse_matrix6_6(ufl.as_matrix(I) + ufl.dot(self.P, self.C - C0), self.V_stiff, self.cells)
        # (Ci-Cm):A_inf is symmetric 
        self.R_inf = - tensordot_4_4(self.R_esh, expand(ufl.dot(ufl.dot(C0inv, self.A_inf.func), self.C-C0))) # TENSOR 3*3*3*3 ; first skew sym
        
    def set_orientation(self, theta=None, phi=None):
        """
        Update fiber/inclusion orientation in place (degrees), no recompilation :
        only refills the e_r/e_theta/e_phi Function fields numerically. Every
        downstream form (Eshelby tensor S_esh/R_esh, localization tensor A_inf...)
        is built symbolically from these Function objects (self.Pass), not from
        theta/phi directly, so it stays valid for any new orientation without
        being rebuilt.
        Called with no arguments, it just re-applies the currently stored
        self.theta/self.phi - used by reset_state() to restore the reference
        orientation before a fresh solve (update_micro_mech overwrites
        e_r/e_theta/e_phi incrementally with the deformed orientation while
        solving).
        """
        if theta is not None:
            self.theta = np.pi/180*theta
        if phi is not None:
            self.phi = np.pi/180*phi
        [e_theta_vec, e_phi_vec, e_r_vec] = local_basis(self.theta, self.phi)
        set_orientation_fields(self.e_theta, self.e_phi, self.e_r, e_theta_vec, e_phi_vec, e_r_vec)

    def set_parameters(self, mat):
        """
        Push new values for this inclusion's physical parameters, in place,
        without rebuilding any form (no recompilation). mat is a (possibly
        partial) dict shaped like the corresponding json card entry, e.g.
        {"young": [[0.4],[2.0],[1.1]]}, {"poisson": 0.3},
        {"volumic_fraction": 0.05}, {"theta": 12.0, "phi": 90}, {"shape_ratio": 0.2}.
        Keys not present in mat are left untouched.
        """
        if "young" in mat:
            self.E.set_value(mat["young"])
        if "poisson" in mat:
            self.nu.set_value(mat["poisson"])
        if "volumic_fraction" in mat:
            self.f.set_value(mat["volumic_fraction"])
        if ("theta" in mat) or ("phi" in mat):
            self.set_orientation(mat.get("theta"), mat.get("phi"))
        if "shape_ratio" in mat and hasattr(self, "shape_ratio"):
            self.shape_ratio.set_value(mat["shape_ratio"])

    def localization_tensors(self, M):
        """
        called by : Homogenized_material.homogenization_scheme
        compute the localization tensor for a spheroidal inclusion.
        M RVE to remote tensor. depends on the homogenization scheme. For HD Identity. For MT it's more complex
        """
        self.A_i = ufl.dot(self.A_inf.func, M)
        self.R_i = tensordot_4_4(self.R_inf, expand(M))
        
    def inelastic_contribution(self, delta_i, omega_i):
        """Store the form representing the inelastic strain contribution arising from the microscopic equilibrium.
        Output :
            self.delta_i inelastic strain
            self.omega_i inelastic rotation 
        """
        self.delta_i = delta_i
        self.omega_i = omega_i
        
        
    def microscopic_mech(self, l_el, l_elj, delta_t):
        """
        Builds the expressions to compute the localization of the strain rate from RVE to inclusion and resulting stress rate. 
        Builds the non-constant stiffness modulus expression based on the stretch measure.
        
        Parameters : 
            l_el, l_elj : respectively the velocity gradient in direction du and its incremental counterpart in direction duj
            delta_t : increment of time for inelastic components.
        New attributes created here :
            dtau_dirder, dtau_incr : stress rate in direction du, and incremental counterpart. The first one is used in the Newton Raphson linearization, the second in computing the increment.
            dtau_expr, dF_expr : fenicsx expression that computes respectively the increment of stress and the increment of deformation gradient in the matrix in the whole domain.
            e_r_expr, e_theta_expr, e_phi_expr : fenicsx expression that computes the new orientation fields.
            lambda_er_expr : fenicsx expression that computes the stretch measure used in non-constant young modulus
        """
        self.dtau = fem.Function(self.V_mandel)
        self.dF = fem.Function(self.V_mat)
        
        d_macro_dirder = ufl.sym(l_el)
        d_macro_incr = ufl.sym(l_elj)
          
        # Objective derivative
        Ri = self.R_i # no need to expand(self.R_i, antisym=[-1,1]) because it is already in tensor notation
        Ai = expand(self.A_i, antisym=[1,1])
                
        if self.delta_i is None or self.omega_i is None:
            self.d_dirder = tensordot_4_2(Ai, d_macro_dirder)  
            self.d_incr = tensordot_4_2(Ai, d_macro_incr)
            
            self.w_dirder = tensordot_4_2(Ri, d_macro_dirder)
            self.w_incr = tensordot_4_2(Ri, d_macro_incr)
        else:
            self.d_dirder = tensordot_4_2(Ai, d_macro_dirder) + delta_t*self.delta_i
            self.d_incr = tensordot_4_2(Ai, d_macro_incr) + delta_t*self.delta_i
            
            self.w_dirder = tensordot_4_2(Ri, d_macro_dirder) + delta_t*self.omega_i
            self.w_incr = tensordot_4_2(Ri, d_macro_incr) + delta_t*self.omega_i
        

        self.dtau_dirder = ufl.dot(self.C, Tensor2Voigt(self.d_dirder))
        self.dtau_incr = ufl.dot(self.C, Tensor2Voigt(self.d_incr))
        
        
        self.dtau_dirder += Tensor2Voigt(-ufl.dot(Voigt2Tensor(self.taun), self.w_dirder)+ufl.dot(self.w_dirder,Voigt2Tensor(self.taun)))
        self.dtau_incr += Tensor2Voigt(-ufl.dot(Voigt2Tensor(self.taun), self.w_incr)+ufl.dot(self.w_incr,Voigt2Tensor(self.taun))) # expression for the macro computation
        
        # Stress and Strain increment expressions
        self.dtau_expr = fem.Expression(self.dtau_incr, self.V_mandel.element.interpolation_points()) # local increment of stress
        self.dF_expr = fem.Expression(ufl.dot((self.d_incr + self.w_incr), self.Fn), self.V_mat.element.interpolation_points()) 
        
        # Orientation vectors
    
        e_r_ = ufl.dot(ufl.as_matrix(np.eye(3)) + self.d_incr + self.w_incr, self.e_r)
        e_theta_temp = ufl.dot(ufl.as_matrix(np.eye(3)) + self.d_incr + self.w_incr, self.e_theta)
    
        self.e_r_expr = fem.Expression(e_r_ / ufl.sqrt(ufl.dot(e_r_, e_r_)), self.V_vec.element.interpolation_points())
        
        e_theta_ = e_theta_temp - ufl.dot(e_theta_temp, e_r_)*e_r_
        self.e_theta_expr = fem.Expression(e_theta_ / ufl.sqrt(ufl.dot(e_theta_, e_theta_)), self.V_vec.element.interpolation_points())
        
        e_phi_ = ufl.cross(e_r_, e_theta_)
        self.e_phi_expr = fem.Expression(e_phi_ / ufl.sqrt(ufl.dot(e_phi_, e_phi_)), self.V_vec.element.interpolation_points())
        
        # temp values to store the new computed field of orientation
        # !!!! THESE ARE NECESSARY BECAUSE OF THE cross-references in the expressions
        self.e_theta_t = fem.Function(self.V_vec) 
        self.e_r_t = fem.Function(self.V_vec) 
        self.e_phi_t = fem.Function(self.V_vec) 
        
        # axial strain
        self.lambda_er = fem.Function(self.V_scalar) ##### !!!!!! doublon sur le stockage de dfib entre inclusion et nl_young_modulus #####
        axial_strain = ufl.dot(self.e_r, ufl.dot(self.Fn, self.e_r))
        self.lambda_er_expr = fem.Expression(axial_strain, self.V_scalar.element.interpolation_points())
        
        # Non constant young modulus
        if isinstance(self.E, nn_cst_young_modulus):
            self.E.init_func(axial_strain)

        
    def update_micro_mech(self):
        """
        update micro mechanics in inclusion for an increment of duj macro displacement field : deformation gradient Fn, kirchhoff stress taun 
        Also rotation -> orientation fields. Update the localization matrix A_inf since it is an inverse_6_6 matrix -> needs to be updated at each numerical step.
        For non constant stiffness parameters : update stretch measure lambda_er and young modulus E.
        """
        # update stresses
        self.dtau.interpolate(self.dtau_expr, self.cells)
        self.dtau.x.scatter_forward()
        self.taun.x.array[:] += self.dtau.x.array[:] # update cauchy stresses in inclusions
        self.taun.x.scatter_forward()
        
        # update strain
        self.dF.interpolate(self.dF_expr, self.cells)
        self.dF.x.scatter_forward()
        self.Fn.x.array[:] += self.dF.x.array[:] # store total strain
        self.Fn.x.scatter_forward()
        
        # update rotations in the temporary fields
        self.e_theta_t.interpolate(self.e_theta_expr, self.cells)
        self.e_theta_t.x.scatter_forward()
        
        self.e_r_t.interpolate(self.e_r_expr, self.cells)
        self.e_r_t.x.scatter_forward()
        
        self.e_phi_t.interpolate(self.e_phi_expr, self.cells)
        self.e_phi_t.x.scatter_forward()
        
        # transfer the new orientations within the orientation fields
        self.e_theta.x.array[:] = self.e_theta_t.x.array[:]
        self.e_theta.x.scatter_forward()
        self.e_r.x.array[:] = self.e_r_t.x.array[:]
        self.e_r.x.scatter_forward()
        self.e_phi.x.array[:] = self.e_phi_t.x.array[:]
        self.e_phi.x.scatter_forward()
        
        # update localization matrix with the new orientation
        self.A_inf.update_func()
        
        # update fiber extension
        self.lambda_er.interpolate(self.lambda_er_expr, self.cells)
        self.lambda_er.x.scatter_forward()
            
        self.E.update_func()
        
        

###########################################################################
### Cylinder and others, just depend on the shape ratio ; Active contraction
###########################################################################

class Active_Spheroidal_inclusion:
    """
    Object that contains all the informations about the active spheroidal inclusion in the multiscale model. 
    Specific case of homeostatic stress regulation in the axial direction of the spheroidal.
    
    Parameters:
    -----------
        mat : dict
            contains the parameters that describe the matrix : young modulus, poisson ratio, volumic fraction, initial orientation, shape ratio and stress regulation params.
        geom : dict
            contains the geometric parameters needed : space functions, cells
    
    Attributes:
    -----------
        Too many to list. Mains are:
            Mechanical parameters: stiffness parameters, volumic fractions. 
            Mechanical state : deformation gradient Fn and Kirchhoff stress tensor taun and orientation fields
            For building the localization tensors using the Eshelby theory, the object also contains the matrix stiffness parameters mu0 and k0. These cannot be changed during a simulation.
        
    Methods: These are called in a sequential order by the material class and the mech_problem_class
    -----------
        __init__ : 
            defined the spheroidal inclusion object. Run some basis methods for the inclusion: stiffness, init_orientation, set_orientation, eshelby_isomatrix, inf_localization_tensors.
        stiffness :
            creates the stiffness and compliance matrices. 6*6 matrices in Mandel Notations
        init_orientation : 
            allocate and initialize the orientation fields e_r, e_theta and e_phi based on initial angles theta and phi. 
            See Multiscale_Framework.function_modules.auxiliary_functions change_of_base_matrix
        eshelby_isomatrix : 
            builds the auxiliary tensor of eshelby for cylinder or prolate spheroid, in the inclusion LRS. Then rotate it into the RVE orientation basis.
            Builds both S_esh and R_esh for strain and rotation localization. Beware that R_esh has small antisymmetry. 
        inf_localization_tensors :
            Create the infinite localization tensor defined by the theory of eshelby for a single inclusion in an infinite matrix.
        inelastic_strain :
            Builds the symbolic expressions for stress regulation in the cells. 
        localization_tensors : 
            build the localization tensors A_i and R_i using the RVE to remote equivalence tensors (M) and the infinite locazlition tensor from the Eshelby theory (A_inf) and (R_inf).
            Beware of small antisymmetry in R_inf.
            Called by material_class.homogenization_scheme
        inelastic_contribution :
            impact of the inelastic contribution of other inclusion on the spheroidal inclusion strain rate. Called by material_class.homogenization_scheme.
            No internal contribution here -> only in growing sphere and active spheroids  objects.
        microscopic_mech :
            create the expressions to compute the spheroidal strain and stress tensors. And orientation fields and non-constant young modulus.
            Called by material_class.homogenization_scheme
        update_micro_mech :
            compute the new mechanical state based on increment of macroscopic displacement or time for inelastic contribution. 
            Called by mech_problem_class.update_local_quantities
    """
    def __init__(self, mat, geom): # aggiungere condizone eleif oer aggiornare il odulo di young
        """
        Initialize inclusion material parameters, compute Lamé moduli, orientation fields, allocate stress fields, assemble stiffness/localization tensors.
        Build inelastic symbolic expressions for stress regulation.
        """
        self.inel = True # flag for inelastic behavior
        self.type = mat["type"]
        domain = geom['domain']
        self.nu = fem.Constant(domain, np.float64(mat["poisson"]))
        
        # Volumic Fraction
        if (type(mat["volumic_fraction"]) is float) or (type(mat["volumic_fraction"]) is np.float64):
            self.f = cst_scalar(mat["volumic_fraction"], geom['scalar spacefunction'])
        elif (type(mat["volumic_fraction"]) is list or type(mat["volumic_fraction"]) is np.ndarray): #--> in case of non constant volume fraction
            self.f = nn_uniform_param(mat["volumic_fraction"], geom['scalar spacefunction'], geom['cells']) 
        else:
            raise Exception("wrong data type for Volumic fraction") 


        # Young Modulus properties
        if mat["young_type"]=="Constant":
            self.E = cst_scalar(mat["young"], geom['scalar spacefunction'])
        elif mat["young_type"]=="Exponential" or mat["young_type"]=="Plateau-Ramp-Plateau":
            self.E = nn_cst_young_modulus(mat["young_type"], mat["young"], geom['scalar spacefunction'], geom['domain'], geom['cells'])
        else:
            raise Exception('wrong type for Young Modulus. Please specify "young_type": as either "Constant", "Plateau-Ramp-Plateau" or "Exponential"') 
          
        
        self.mu = self.E.func[0]/(2*(1+self.nu))
        self.k = self.E.func[0]/(3*(1-2*self.nu))
        
        # Reference mechanical properties (matrix)
        # numeric snapshot (plain float), see material_class.py module docstring
        self.mu_0 = cst_scalar(mat["mu_0"], geom['scalar spacefunction'])
        self.k_0 = cst_scalar(mat["k_0"], geom['scalar spacefunction'])
        
        # Cylinder orientation
        self.theta = np.pi/180*mat["theta"]
        self.phi = np.pi/180*mat["phi"]
        # spheroid shape ratio
        self.shape_ratio = fem.Constant(domain, mat["shape_ratio"])
        
        # Space functions 
        self.V_vec = geom["vector spacefunction"] # specifically used for cylinder and oblate spheroids, hence in the mat dictionnary
        self.V_stiff = geom["stiff spacefunction"]
        self.V_mandel = geom["mandel spacefunction"]
        self.V_scalar = geom["scalar spacefunction"]
        self.V_mat = geom["matrix spacefunction"]
        self.cells = geom["cells"]
        self.obj_der = geom["objective derivative"]
        
        # homeostatic behavior
        self.t_c = fem.Constant(domain, np.float64(mat["characteristic time"]))
        self.tau_b = fem.Constant(domain, np.float64(mat["basal stress"]))
        self.l_min = fem.Constant(domain, np.float64(mat['min active strain'])) # 0.6
        self.l_max = fem.Constant(domain, np.float64(mat['max active strain'])) # 1.3
        self.alpha_thresh = fem.Constant(domain, np.float64(mat['threshold steepness'])) # 50 steepness of ththreshold for inelastic strain

        # Orientation of fibers -> define the 3 fields for the basis vector of fibers orientation 
        self.init_orientation()
        
        # Stiffness self.C
        self.stiffness()
        
        # Eshelby Isomatrix for fibers
        self.eshelby_isomatrix()
        
        # Create the Form of the localization tensor used in the homogenization scheme
        self.inf_localization_tensors()
        
        # Mechanical fields associated with the inclusion
        # define the stored stress for each inclusion to define the Jauman stress rate
        self.taun = fem.Function(self.V_mandel)
        # Deformation gradient for the inclusion
        self.Fn = fem.Function(self.V_mat)
        self.F_inel = fem.Function(self.V_mat)
        init_identity_field(self.Fn)
        init_identity_field(self.F_inel)
        
        self.inelastic_strain()
        
        
    def stiffness(self):
        """ Define the stiffness tensor in Voigt notation callable by self.C """
        J_m, K_m, I_m = projection_tensors_func(self.V_stiff) # mandel notations
        self.J_m = J_m
        self.K_m = K_m
        self.I_m = I_m
        self.C = 2*self.mu*self.K_m + 3*self.k*self.J_m # function defined on the V_stiff functionspace
        

    def init_orientation(self):
        """
        called by : spheroidal_inclusion.__init__
        Create and fill with their initial values the vector functions containing the three basis vector for the local basis
        """
        self.e_theta = fem.Function(self.V_vec) 
        self.e_phi = fem.Function(self.V_vec)
        self.e_r = fem.Function(self.V_vec)
        
        # Create change of basis matrix
        self.Pass = change_of_base_matrix(self.e_theta, self.e_phi, self.e_r)    
        
        # Compute orientation vectors from theta and phi value
        [e_theta_vec, e_phi_vec, e_r_vec] = local_basis(self.theta, self.phi) # in auxiliary function file
        # Fill the dofs to get a constant field on the volume of orientation vectors
        set_orientation_fields(self.e_theta,self.e_phi,self.e_r, e_theta_vec, e_phi_vec, e_r_vec)
        
    
    def eshelby_isomatrix(self):
        """
        called by : active_spheroidal_inclusion.__init__
        Solution of the eshelby problem for a spheroid aligned with the third vector of the local basis : [e_theta, e_phi, e_r]. Based on shape of inclusion and stiffness matrix of reference C0
        """
        # plain floats required - see Spheroidal_inclusion.eshelby_isomatrix
        nu_0 = (3*self.k_0.value - 2*self.mu_0.value)/(2*(3*self.k_0.value + self.mu_0.value))
        # self.type == 'homeostatic spheroid':
        self.L = ufl.as_tensor(eshelby_aux_spheroid(nu_0, self.shape_ratio.value))
            
        i, j, k, l = ufl.indices(4)
        
        self.S0 = ufl.as_tensor(1/2*(self.L[i,j,k,l] + self.L[j,i,k,l]), (i,j,k,l))
        self.R0 = ufl.as_tensor(1/2*(self.L[i,j,k,l] - self.L[j,i,k,l]), (i,j,k,l))
        
        # expanded 3*3*3*3 with adapted symmetries, rotate, then compress
        self.S_esh = compress(tensors_local2global(self.S0, self.Pass)) # small symmetrical
        self.R_esh = tensors_local2global(self.R0, self.Pass) # fisrt small antisymmetry BEWARE
        
    def inf_localization_tensors(self):
        """
        Define the A_inf localization tensor used in the HD and MT for a cylindrical inclusion
        For cylinders, cannot be based on the decomposition in the projectors base because of the anisotropic behavior
        Notations used are the ones from ZAMM paper
        """
        J,K,I = projection_tensors()
        C0 = 2*self.mu_0[0]*ufl.as_matrix(K) + 3*self.k_0[0]*ufl.as_matrix(J)
        C0inv = 1/(2*self.mu_0[0])*ufl.as_matrix(K) + 1/(3*self.k_0[0])*ufl.as_matrix(J)
        
        self.P = ufl.dot(self.S_esh, C0inv) # Symmetric Mandel notation 6*6 tensors
        #self.A_inf = ufl.as_matrix(inv_mat_6_6(ufl.as_matrix(I) + ufl.dot(self.P, self.C - C0)))
        self.A_inf = inverse_matrix6_6(ufl.as_matrix(I) + ufl.dot(self.P, self.C - C0), self.V_stiff, self.cells) # call with A_inf.func
        # (Ci-Cm):A_inf is symmetric 
        #self.R_inf = - ufl.dot(ufl.dot(ufl.dot(self.R_esh, C0inv), self.A_inf.func), self.C-C0)
        self.R_inf = - tensordot_4_4(self.R_esh, expand(ufl.dot(ufl.dot(C0inv, self.C-C0), self.A_inf.func))) # TENSOR 3*3*3*3 ; first skew sym
        
        
        # inelastic localization tensors
        self.D_inf = ufl.dot(ufl.dot(self.A_inf.func, self.P), self.C) # -> Mandel Notations 
        self.T_inf = tensordot_4_4(self.R_esh, expand(ufl.dot(C0inv, self.C))) + tensordot_4_4(self.R_inf, expand(ufl.dot(self.P, self.C)))
        # T_inf in tensor notations
        
    def localization_tensors(self, M):
        """
        caller : Homogenized_material.estimation_local_tensors
        compute the localization tensor for a spherical inclusion.
        M depends on the homogenization scheme. For HD Identity. For MT it's more complex
        """
        self.A_i = ufl.dot(self.A_inf.func, M)
        self.R_i = tensordot_4_4(self.R_inf, expand(M))
        
    def inelastic_strain(self):
        """
        Manage inelastic part of F. 
        Own contribution on axis contraction for cells
        Adaptation of Federica Galbati's work for homeostatic sphere
        """
        
        J = ufl.det(self.Fn)
        self.sig_mes = ufl.dot(self.e_r, ufl.dot(Voigt2Tensor(self.taun), self.e_r)) # Kirchhoff strss
        
        self.la_dot = 1/(self.t_c*self.tau_b)*(self.sig_mes - self.tau_b)
        b_in = ufl.dot(self.F_inel, self.F_inel.T)
        self.la = ufl.sqrt(ufl.dot(self.e_r, ufl.dot(b_in, self.e_r)))
        
        g_min = ufl.conditional(self.la_dot < 0 , 1/2*(1+ufl.tanh(self.alpha_thresh*(self.la - self.l_min))), 1)
        g_max = ufl.conditional(self.la_dot > 0 , 1/2*(1+ufl.tanh(self.alpha_thresh*(self.l_max - self.la))), 1)
        
        self.la_dot *= g_min*g_max
        
        M = ufl.outer(self.e_r, self.e_r)

        self.d_inel = self.la_dot*M - self.la_dot/2*(ufl.Identity(3) - M)
        
    def inelastic_contribution(self, delta_i, omega_i):
        """Store the form representing the inelastic strain contribution arising from the microscopic equilibrium.
        Output :
            self.delta_i inelastic strain
            self.omega_i inelastic rotation 
        """
        self.delta_i = delta_i
        self.omega_i = omega_i

    def microscopic_mech(self, l_el, l_elj, delta_t):
        """
        Builds the expressions to compute the localization of the strain rate from RVE to inclusion, impact of inelastic strain rate and the resulting stress rate. 
        Builds the non-constant stiffness modulus expression based on the stretch measure.
        
        Parameters : 
            l_el, l_elj : respectively the velocity gradient in direction du and its incremental counterpart in direction duj
            delta_t : increment of time for inelastic components.
        New attributes created here :
            dtau_dirder, dtau_incr : stress rate in direction du, and incremental counterpart. The first one is used in the Newton Raphson linearization, the second in computing the increment.
            dtau_expr, dF_expr : fenicsx expression that computes respectively the increment of stress and the increment of deformation gradient in the matrix in the whole domain.
            dF_inel_expr : fenicsx expression that computes the increment of inelastic deformation gradient.
            e_r_expr, e_theta_expr, e_phi_expr : fenicsx expression that computes the new orientation fields.
            lambda_er_expr : fenicsx expression that computes the stretch measure used in non-constant young modulus
        """
        self.dtau = fem.Function(self.V_mandel)
        self.dF = fem.Function(self.V_mat)
        self.dF_inel = fem.Function(self.V_mat)
        
        d_macro_dirder = ufl.sym(l_el)
        d_macro_incr = ufl.sym(l_elj)
          
        # Objective derivative
        Ri = self.R_i # no need to expand(self.R_i, antisym=[-1,1])       
        Ai = expand(self.A_i, antisym=[1,1])
        
        if self.delta_i is None or self.omega_i is None:
            self.d_dirder = tensordot_4_2(Ai, d_macro_dirder)  
            self.d_incr = tensordot_4_2(Ai, d_macro_incr)
            
            self.w_dirder = tensordot_4_2(Ri, d_macro_dirder)
            self.w_incr = tensordot_4_2(Ri, d_macro_incr)
        else:
            self.d_dirder = tensordot_4_2(Ai, d_macro_dirder) + delta_t*self.delta_i
            self.d_incr = tensordot_4_2(Ai, d_macro_incr) + delta_t*self.delta_i
            
            self.w_dirder = tensordot_4_2(Ri, d_macro_dirder) + delta_t*self.omega_i
            self.w_incr = tensordot_4_2(Ri, d_macro_incr) + delta_t*self.omega_i
        

        self.dtau_dirder = ufl.dot(self.C, Tensor2Voigt(self.d_dirder - delta_t*self.d_inel))
        self.dtau_incr = ufl.dot(self.C, Tensor2Voigt(self.d_incr - delta_t*self.d_inel))
        
        self.dtau_dirder += Tensor2Voigt(-ufl.dot(Voigt2Tensor(self.taun), self.w_dirder)+ufl.dot(self.w_dirder,Voigt2Tensor(self.taun)))
        self.dtau_incr += Tensor2Voigt(-ufl.dot(Voigt2Tensor(self.taun), self.w_incr)+ufl.dot(self.w_incr,Voigt2Tensor(self.taun))) # expression for the macro computation
        
        # Stress and Strain increment expressions
        self.dtau_expr = fem.Expression(self.dtau_incr, self.V_mandel.element.interpolation_points()) # local increment of stress
        self.dF_expr = fem.Expression(ufl.dot((self.d_incr + self.w_incr), self.Fn), self.V_mat.element.interpolation_points()) 
        
        
        # Manage inelastic increments
        l_inel = self.d_inel # regulatory mechanism is only a contraction; rotation comes from the interaction with other components
        F_el_inv = ufl.dot(self.F_inel, ufl.inv(self.Fn)) 
        self.F_dot_inel = ufl.dot(F_el_inv, ufl.dot(l_inel, self.Fn))
        self.dF_inel_expr = fem.Expression(delta_t*self.F_dot_inel, self.V_mat.element.interpolation_points()) 
        
        
        # Orientation vectors
        e_r_ = ufl.dot(ufl.as_matrix(np.eye(3)) + self.d_incr + self.w_incr, self.e_r)
        e_theta_temp = ufl.dot(ufl.as_matrix(np.eye(3)) + self.d_incr + self.w_incr, self.e_theta)
    
        self.e_r_expr = fem.Expression(e_r_ / ufl.sqrt(ufl.dot(e_r_, e_r_)), self.V_vec.element.interpolation_points())
        
        e_theta_ = e_theta_temp - ufl.dot(e_theta_temp, e_r_)*e_r_
        self.e_theta_expr = fem.Expression(e_theta_ / ufl.sqrt(ufl.dot(e_theta_, e_theta_)), self.V_vec.element.interpolation_points())
        
        e_phi_ = ufl.cross(e_r_, e_theta_)
        self.e_phi_expr = fem.Expression(e_phi_ / ufl.sqrt(ufl.dot(e_phi_, e_phi_)), self.V_vec.element.interpolation_points())
        
        # temp values to store the new computed field of orientation
        # !!!! THESE ARE NECESSARY BECAUSE OF THE cross-references in the expressions
        self.e_theta_t = fem.Function(self.V_vec) 
        self.e_r_t = fem.Function(self.V_vec) 
        self.e_phi_t = fem.Function(self.V_vec) 
        
        # axial strain
        self.lambda_er = fem.Function(self.V_scalar) ##### !!!!!! doublon sur le stockage de dfib entre inclusion et nl_young_modulus #####
        axial_strain = ufl.dot(self.e_r, ufl.dot(self.Fn, self.e_r))
        self.lambda_er_expr = fem.Expression(axial_strain, self.V_scalar.element.interpolation_points())
        
        # Non constant young modulus
        # ERROR IN MAKING THE YOUNG MODULUS DEPEND ON THE WHOLE AXIAL STRAIN AND NOT ON THE ELASTIC PART
        if isinstance(self.E, nn_cst_young_modulus):
            self.E.init_func(axial_strain)
        
    def set_parameters(self, mat):
        """
        Push new values for this inclusion's young modulus and volumic
        fraction, in place, without rebuilding any form (no recompilation).
        mat is a (possibly partial) dict shaped like the corresponding json
        card entry. Poisson ratio is NOT covered here (kept as a plain
        fem.Constant on this class, not yet migrated to cst_scalar) - update
        it by rebuilding via setup_simulation() if it needs to change.
        """
        if "young" in mat:
            self.E.set_value(mat["young"])
        if "volumic_fraction" in mat:
            self.f.set_value(mat["volumic_fraction"])

    def update_micro_mech(self):
        """
        update micro mechanics in inclusion for an increment of duj macro displacement field : deformation gradient Fn, kirchhoff stress taun and inelastic deformation gradient F_inel.
        Also rotation -> orientation fields. Update the localization matrix A_inf since it is an inverse_6_6 matrix -> needs to be updated at each numerical step.
        For non constant stiffness parameters : update stretch measure lambda_er and young modulus E.
        """
        # update stresses
        self.dtau.interpolate(self.dtau_expr, self.cells)
        self.dtau.x.scatter_forward()
        self.taun.x.array[:] += self.dtau.x.array[:] # update cauchy stresses in inclusions
        self.taun.x.scatter_forward()
        
                
        # compute increment of strain
        self.dF.interpolate(self.dF_expr, self.cells)
        self.dF.x.scatter_forward()
        self.dF_inel.interpolate(self.dF_inel_expr, self.cells)
        self.dF_inel.x.scatter_forward()
        
        # update strain deformation gradients
        self.Fn.x.array[:] += self.dF.x.array[:] # store total strain
        self.Fn.x.scatter_forward()   
        self.F_inel.x.array[:] += self.dF_inel.x.array[:] 
        self.F_inel.x.scatter_forward()   
        
        # update rotations in the temporary fields
        self.e_theta_t.interpolate(self.e_theta_expr, self.cells)
        self.e_theta_t.x.scatter_forward()
        
        self.e_r_t.interpolate(self.e_r_expr, self.cells)
        self.e_r_t.x.scatter_forward()
        
        self.e_phi_t.interpolate(self.e_phi_expr, self.cells)
        self.e_phi_t.x.scatter_forward()
        
        # transfer the new orientations within the orientation fields
        self.e_theta.x.array[:] = self.e_theta_t.x.array[:]
        self.e_theta.x.scatter_forward()
        self.e_r.x.array[:] = self.e_r_t.x.array[:]
        self.e_r.x.scatter_forward()
        self.e_phi.x.array[:] = self.e_phi_t.x.array[:]
        self.e_phi.x.scatter_forward()
        
        # update localization matrix with the new orientation
        self.A_inf.update_func()
        
        # update fiber extension
        self.lambda_er.interpolate(self.lambda_er_expr, self.cells)
        self.lambda_er.x.scatter_forward()
            
        self.E.update_func()
        

class Prestretched_Cylinder_inclusion:
    """
    Object that contains all the informations about the prestretched cylinder inclusion in the multiscale model. 
    Inelastic stretch is a first order feedback mechanisms, with fast temporal response with respect to cells.
    
    Parameters:
    -----------
        mat : dict
            contains the parameters that describe the matrix : young modulus, poisson ratio, volumic fraction, initial orientation, shape ratio and stress regulation params.
        geom : dict
            contains the geometric parameters needed : space functions, cells
    
    Attributes:
    -----------
        Too many to list. Mains are:
            Mechanical parameters: stiffness parameters, volumic fractions. 
            Mechanical state : deformation gradient Fn and Kirchhoff stress tensor taun and orientation fields
            For building the localization tensors using the Eshelby theory, the object also contains the matrix stiffness parameters mu0 and k0. These cannot be changed during a simulation.
        
    Methods: These are called in a sequential order by the material class and the mech_problem_class
    -----------
        __init__ : 
            defined the spheroidal inclusion object. Run some basis methods for the inclusion: stiffness, init_orientation, set_orientation, eshelby_isomatrix, inf_localization_tensors.
        stiffness :
            creates the stiffness and compliance matrices. 6*6 matrices in Mandel Notations
        init_orientation : 
            allocate and initialize the orientation fields e_r, e_theta and e_phi based on initial angles theta and phi. 
            See Multiscale_Framework.function_modules.auxiliary_functions change_of_base_matrix
        eshelby_isomatrix : 
            builds the auxiliary tensor of eshelby for cylinder or prolate spheroid, in the inclusion LRS. Then rotate it into the RVE orientation basis.
            Builds both S_esh and R_esh for strain and rotation localization. Beware that R_esh has small antisymmetry. 
        inf_localization_tensors :
            Create the infinite localization tensor defined by the theory of eshelby for a single inclusion in an infinite matrix.
        inelastic_strain :
            Builds the symbolic expressions for stress regulation in the prestretched cylinder. 
        localization_tensors : 
            build the localization tensors A_i and R_i using the RVE to remote equivalence tensors (M) and the infinite locazlition tensor from the Eshelby theory (A_inf) and (R_inf).
            Beware of small antisymmetry in R_inf.
            Called by material_class.homogenization_scheme
        inelastic_contribution :
            impact of the inelastic contribution of other inclusion on the spheroidal inclusion strain rate. Called by material_class.homogenization_scheme.
            No internal contribution here -> only in growing sphere and active spheroids  objects.
        microscopic_mech :
            create the expressions to compute the spheroidal strain and stress tensors. And orientation fields and non-constant young modulus.
            Called by material_class.homogenization_scheme
        update_micro_mech :
            compute the new mechanical state based on increment of macroscopic displacement or time for inelastic contribution. 
            Called by mech_problem_class.update_local_quantities
    """
    def __init__(self, mat, geom): # aggiungere condizone eleif oer aggiornare il odulo di young
        """
        Initialize inclusion material parameters, compute Lamé moduli, orientation fields, allocate stress fields, assemble stiffness/localization tensors.
        Builds the symbolic expressions for inelastic strain coupling.
        """
        self.inel = True # flag for inelastic behavior
        self.type = mat["type"]
        
        domain = geom['domain']
        self.nu = fem.Constant(domain, np.float64(mat["poisson"]))
        
        # Volumic Fraction
        if (type(mat["volumic_fraction"]) is float) or (type(mat["volumic_fraction"]) is np.float64):
            self.f = cst_scalar(mat["volumic_fraction"], geom['scalar spacefunction'])
        elif (type(mat["volumic_fraction"]) is list or type(mat["volumic_fraction"]) is np.ndarray): #--> in case of non constant volume fraction
            self.f = nn_uniform_param(mat["volumic_fraction"], geom['scalar spacefunction'], geom['cells']) 
        else:
            raise Exception("wrong data type for Volumic fraction") 

        # Young Modulus properties
        if mat["young_type"]=="Constant":
            self.E = cst_scalar(mat["young"], geom['scalar spacefunction'])
        elif mat["young_type"]=="Exponential" or mat["young_type"]=="Plateau-Ramp-Plateau":
            self.E = nn_cst_young_modulus(mat["young_type"], mat["young"], geom['scalar spacefunction'], geom['domain'], geom['cells'])
        else:
            raise Exception('wrong type for Young Modulus. Please specify "young_type": as either "Constant", "Plateau-Ramp-Plateau" or "Exponential"') 
      
        
        self.mu = self.E.func[0]/(2*(1+self.nu))
        self.k = self.E.func[0]/(3*(1-2*self.nu))
        
        # Reference mechanical properties (matrix)
        # numeric snapshot (plain float), see material_class.py module docstring
        self.mu_0 = cst_scalar(mat["mu_0"], geom['scalar spacefunction'])
        self.k_0 = cst_scalar(mat["k_0"], geom['scalar spacefunction'])
        
        # Cylinder orientation
        self.theta = np.pi/180*mat["theta"]
        self.phi = np.pi/180*mat["phi"]
        
        # Space functions 
        self.V_vec = geom["vector spacefunction"] # specifically used for cylinder and oblate spheroids, hence in the mat dictionnary
        self.V_stiff = geom["stiff spacefunction"]
        self.V_mandel = geom["mandel spacefunction"]
        self.V_scalar = geom["scalar spacefunction"]
        self.V_mat = geom["matrix spacefunction"]
        self.cells = geom["cells"]
        self.obj_der = geom["objective derivative"]
        
        # inelastic coupling with cellular activity
        self.prestretch = fem.Constant(domain, np.float64(mat["prestretch"])) # prestretch : scalar value between 0 and 1 to impose a tensile elastic prestretch (ie inelastic compression)
        self.t_carac = fem.Constant(domain, np.float64(mat["characteristic time"])) # prestretch : scalar value between 0 and 1 to impose a tensile elastic prestretch (ie inelastic compression)
        
        # Orientation of fibers -> define the 3 fields for the basis vector of fibers orientation 
        self.init_orientation()
        
        # Stiffness self.C
        self.stiffness()
        
        # Eshelby Isomatrix for fibers
        self.eshelby_isomatrix()
        
        # Create the Form of the localization tensor used in the homogenization scheme
        self.inf_localization_tensors()
        
        
        # Mechanical fields associated with the inclusion
        # define the stored stress for each inclusion to define the Jauman stress rate
        self.taun = fem.Function(self.V_mandel)
        # Deformation gradient for the inclusion
        self.Fn = fem.Function(self.V_mat)
        self.F_inel = fem.Function(self.V_mat)
        init_identity_field(self.Fn)
        init_identity_field(self.F_inel)
        
        # cell_inclusion is needed for coupling -> Collgen must be initialized AFTER active cells
        self.inelastic_strain()
        
        
    def stiffness(self):
        """ Define the stiffness tensor in Voigt notation callable by self.C """
        J_m, K_m, I_m = projection_tensors_func(self.V_stiff) # mandel notations
        self.J_m = J_m
        self.K_m = K_m
        self.I_m = I_m
        self.C = 2*self.mu*self.K_m + 3*self.k*self.J_m # function defined on the V_stiff functionspace
        

    def init_orientation(self):
        """
        called by : spheroidal_inclusion.__init__
        Create and fill with their initial values the vector functions containing the three basis vector for the local basis
        """
        self.e_theta = fem.Function(self.V_vec) 
        self.e_phi = fem.Function(self.V_vec)
        self.e_r = fem.Function(self.V_vec)
        
        # Create change of basis matrix
        self.Pass = change_of_base_matrix(self.e_theta, self.e_phi, self.e_r)    
        
        # Compute orientation vectors from theta and phi value
        [e_theta_vec, e_phi_vec, e_r_vec] = local_basis(self.theta, self.phi) # in auxiliary function file
        # Fill the dofs to get a constant field on the volume of orientation vectors
        set_orientation_fields(self.e_theta,self.e_phi,self.e_r, e_theta_vec, e_phi_vec, e_r_vec)
        
    def eshelby_isomatrix(self):
        """
        caller : Cylindrical_inclusion.__init__
        Solution of the eshelby problem for a cylinder aligned with the third vector of the local basis : [e_theta, e_phi, e_r]. Based on shape of inclusion and stiffness matrix of reference C0
        """
        # plain floats required - see Spheroidal_inclusion.eshelby_isomatrix
        nu_0 = (3*self.k_0.value - 2*self.mu_0.value)/(2*(3*self.k_0.value + self.mu_0.value))
        
        self.L = ufl.as_tensor(eshelby_aux_cylindrical(nu_0))
        i, j, k, l = ufl.indices(4)
        
        self.S0 = ufl.as_tensor(1/2*(self.L[i,j,k,l] + self.L[j,i,k,l]), (i,j,k,l))
        self.R0 = ufl.as_tensor(1/2*(self.L[i,j,k,l] - self.L[j,i,k,l]), (i,j,k,l))
        
        # expanded 3*3*3*3 with adapted symmetries, rotate, then compress
        self.S_esh = compress(tensors_local2global(self.S0, self.Pass)) # small symmetrical
        self.R_esh = tensors_local2global(self.R0, self.Pass) # fisrt small antisymmetry BEWARE
        
    def inf_localization_tensors(self):
        """
        Define the A_inf localization tensor used in the HD and MT for a cylindrical inclusion
        For cylinders, cannot be based on the decomposition in the projectors base because of the anisotropic behavior
        Notations used are the ones from ZAMM paper
        """
        J,K,I = projection_tensors()
        C0 = 2*self.mu_0[0]*ufl.as_matrix(K) + 3*self.k_0[0]*ufl.as_matrix(J)
        C0inv = 1/(2*self.mu_0[0])*ufl.as_matrix(K) + 1/(3*self.k_0[0])*ufl.as_matrix(J)
        
        self.P = ufl.dot(self.S_esh, C0inv) # Symmetric Mandel notation 6*6 tensors
        #self.A_inf = ufl.as_matrix(inv_mat_6_6(ufl.as_matrix(I) + ufl.dot(self.P, self.C - C0)))
        self.A_inf = inverse_matrix6_6(ufl.as_matrix(I) + ufl.dot(self.P, self.C - C0), self.V_stiff, self.cells) # call with A_inf.func
        # (Ci-Cm):A_inf is symmetric 
        #self.R_inf = - ufl.dot(ufl.dot(ufl.dot(self.R_esh, C0inv), self.A_inf.func), self.C-C0)
        self.R_inf = - tensordot_4_4(self.R_esh, expand(ufl.dot(ufl.dot(C0inv, self.C-C0), self.A_inf.func))) # TENSOR 3*3*3*3 ; first skew sym
        
        
        # inelastic localization tensors
        self.D_inf = ufl.dot(ufl.dot(self.A_inf.func, self.P), self.C) # -> Mandel Notations 
        self.T_inf = tensordot_4_4(self.R_esh, expand(ufl.dot(C0inv, self.C))) + tensordot_4_4(self.R_inf, expand(ufl.dot(self.P, self.C)))
        # T_inf in tensor notations
        
    def localization_tensors(self, M):
        """
        caller : Homogenized_material.estimation_local_tensors
        compute the localization tensor for a spherical inclusion.
        M depends on the homogenization scheme. For HD Identity. For MT it's more complex
        """
        self.A_i = ufl.dot(self.A_inf.func, M)
        self.R_i = tensordot_4_4(self.R_inf, expand(M))
        
    def inelastic_strain(self):
        """
        Manage inelastic part of F. 
        Prestretching is carried out as first order feedback mechanism.
        """
        
        b_in = ufl.dot(self.F_inel, self.F_inel.T)
        self.l_in = ufl.sqrt(ufl.dot(self.e_r, ufl.dot(b_in, self.e_r)))
        
        self.l_in_dot = 1/self.t_carac*(self.l_in - self.prestretch)# self.xi_coll*cell_inclusion.la_dot #*ufl.dot(self.e_r, cell_inclusion.e_r)
        
        M = ufl.outer(self.e_r, self.e_r)

        self.d_inel = self.l_in_dot*M - self.l_in_dot/2*(ufl.Identity(3) - M)
        
    def inelastic_contribution(self, delta_i, omega_i):
        """Store the form representing the inelastic strain contribution arising from the microscopic equilibrium.
        Output :
            self.delta_i inelastic strain
            self.omega_i inelastic rotation 
        """
        self.delta_i = delta_i
        self.omega_i = omega_i

    def microscopic_mech(self, l_el, l_elj, delta_t):
        """
        Builds the expressions to compute the localization of the strain rate from RVE to inclusion, impact of inelastic strain rate and the resulting stress rate. 
        Builds the non-constant stiffness modulus expression based on the stretch measure.
        
        Parameters : 
            l_el, l_elj : respectively the velocity gradient in direction du and its incremental counterpart in direction duj
            delta_t : increment of time for inelastic components.
        New attributes created here :
            dtau_dirder, dtau_incr : stress rate in direction du, and incremental counterpart. The first one is used in the Newton Raphson linearization, the second in computing the increment.
            dtau_expr, dF_expr : fenicsx expression that computes respectively the increment of stress and the increment of deformation gradient in the matrix in the whole domain.
            dF_inel_expr : fenicsx expression that computes the increment of inelastic deformation gradient.
            e_r_expr, e_theta_expr, e_phi_expr : fenicsx expression that computes the new orientation fields.
            lambda_er_expr : fenicsx expression that computes the stretch measure used in non-constant young modulus
        """
        self.dtau = fem.Function(self.V_mandel)
        self.dF = fem.Function(self.V_mat)
        self.dF_inel = fem.Function(self.V_mat)
        
        d_macro_dirder = ufl.sym(l_el)
        d_macro_incr = ufl.sym(l_elj)
          
        # Objective derivative
        Ri = self.R_i # no need to expand(self.R_i, antisym=[-1,1])       
        Ai = expand(self.A_i, antisym=[1,1])
        
        if self.delta_i is None or self.omega_i is None:
            self.d_dirder = tensordot_4_2(Ai, d_macro_dirder)  
            self.d_incr = tensordot_4_2(Ai, d_macro_incr)
            
            self.w_dirder = tensordot_4_2(Ri, d_macro_dirder)
            self.w_incr = tensordot_4_2(Ri, d_macro_incr)
        else:
            self.d_dirder = tensordot_4_2(Ai, d_macro_dirder) + delta_t*self.delta_i
            self.d_incr = tensordot_4_2(Ai, d_macro_incr) + delta_t*self.delta_i
            
            self.w_dirder = tensordot_4_2(Ri, d_macro_dirder) + delta_t*self.omega_i
            self.w_incr = tensordot_4_2(Ri, d_macro_incr) + delta_t*self.omega_i
        

        self.dtau_dirder = ufl.dot(self.C, Tensor2Voigt(self.d_dirder - delta_t*self.d_inel))
        self.dtau_incr = ufl.dot(self.C, Tensor2Voigt(self.d_incr - delta_t*self.d_inel))
        
        self.dtau_dirder += Tensor2Voigt(-ufl.dot(Voigt2Tensor(self.taun), self.w_dirder)+ufl.dot(self.w_dirder,Voigt2Tensor(self.taun)))
        self.dtau_incr += Tensor2Voigt(-ufl.dot(Voigt2Tensor(self.taun), self.w_incr)+ufl.dot(self.w_incr,Voigt2Tensor(self.taun))) # expression for the macro computation
        
        # Stress and Strain increment expressions
        self.dtau_expr = fem.Expression(self.dtau_incr, self.V_mandel.element.interpolation_points()) # local increment of stress
        self.dF_expr = fem.Expression(ufl.dot((self.d_incr + self.w_incr), self.Fn), self.V_mat.element.interpolation_points()) 
        
        
        # Manage inelastic increments
        l_inel = self.d_inel # regulatory mechanism is only a contraction; rotation comes from the interaction with other components
        F_el_inv = ufl.dot(self.F_inel, ufl.inv(self.Fn)) 
        self.F_dot_inel = ufl.dot(F_el_inv, ufl.dot(l_inel, self.Fn))
        self.dF_inel_expr = fem.Expression(delta_t*self.F_dot_inel, self.V_mat.element.interpolation_points()) 
        
        
        # Orientation vectors
        e_r_ = ufl.dot(ufl.as_matrix(np.eye(3)) + self.d_incr + self.w_incr, self.e_r)
        e_theta_temp = ufl.dot(ufl.as_matrix(np.eye(3)) + self.d_incr + self.w_incr, self.e_theta)
    
        self.e_r_expr = fem.Expression(e_r_ / ufl.sqrt(ufl.dot(e_r_, e_r_)), self.V_vec.element.interpolation_points())
        
        e_theta_ = e_theta_temp - ufl.dot(e_theta_temp, e_r_)*e_r_
        self.e_theta_expr = fem.Expression(e_theta_ / ufl.sqrt(ufl.dot(e_theta_, e_theta_)), self.V_vec.element.interpolation_points())
        
        e_phi_ = ufl.cross(e_r_, e_theta_)
        self.e_phi_expr = fem.Expression(e_phi_ / ufl.sqrt(ufl.dot(e_phi_, e_phi_)), self.V_vec.element.interpolation_points())
        
        # temp values to store the new computed field of orientation
        # !!!! THESE ARE NECESSARY BECAUSE OF THE cross-references in the expressions
        self.e_theta_t = fem.Function(self.V_vec) 
        self.e_r_t = fem.Function(self.V_vec) 
        self.e_phi_t = fem.Function(self.V_vec) 

        # Non constant young modulus
        if isinstance(self.E, nn_cst_young_modulus):
            # Elastic axial stretch
            b = ufl.dot(self.F_inel, self.F_inel.T)
            b_in = ufl.dot(self.F_inel, self.F_inel.T)
            
            axial_stretch = ufl.sqrt(ufl.dot(self.e_r, ufl.dot(b, self.e_r))) # total stretch
            inelastic_stretch =  ufl.sqrt(ufl.dot(self.e_r, ufl.dot(b_in, self.e_r)))
            elastic_stretch = axial_stretch/inelastic_stretch
            
            self.E.init_func(elastic_stretch)
        
    def set_parameters(self, mat):
        """
        Push new values for this inclusion's young modulus and volumic
        fraction, in place, without rebuilding any form (no recompilation).
        mat is a (possibly partial) dict shaped like the corresponding json
        card entry. Poisson ratio is NOT covered here (kept as a plain
        fem.Constant on this class, not yet migrated to cst_scalar) - update
        it by rebuilding via setup_simulation() if it needs to change.
        """
        if "young" in mat:
            self.E.set_value(mat["young"])
        if "volumic_fraction" in mat:
            self.f.set_value(mat["volumic_fraction"])

    def update_micro_mech(self):
        """
        update micro mechanics in inclusion for an increment of duj macro displacement field : deformation gradient Fn, kirchhoff stress taun and inelastic deformation gradient F_inel
        Also rotation -> orientation fields. Update the localization matrix A_inf since it is an inverse_6_6 matrix -> needs to be updated at each numerical step.
        For non constant stiffness parameters : update stretch measure lambda_er and young modulus E.
        """
        # update stresses
        self.dtau.interpolate(self.dtau_expr, self.cells)
        self.dtau.x.scatter_forward()
        self.taun.x.array[:] += self.dtau.x.array[:] # update cauchy stresses in inclusions
        self.taun.x.scatter_forward()
        
                
        # compute increment of strain
        self.dF.interpolate(self.dF_expr, self.cells)
        self.dF.x.scatter_forward()
        self.dF_inel.interpolate(self.dF_inel_expr, self.cells)
        self.dF_inel.x.scatter_forward()
        
        # update strain deformation gradients
        self.Fn.x.array[:] += self.dF.x.array[:] # store total strain
        self.Fn.x.scatter_forward()   
        self.F_inel.x.array[:] += self.dF_inel.x.array[:] 
        self.F_inel.x.scatter_forward()   
        
        # update rotations in the temporary fields
        self.e_theta_t.interpolate(self.e_theta_expr, self.cells)
        self.e_theta_t.x.scatter_forward()
        
        self.e_r_t.interpolate(self.e_r_expr, self.cells)
        self.e_r_t.x.scatter_forward()
        
        self.e_phi_t.interpolate(self.e_phi_expr, self.cells)
        self.e_phi_t.x.scatter_forward()
        
        # transfer the new orientations within the orientation fields
        self.e_theta.x.array[:] = self.e_theta_t.x.array[:]
        self.e_theta.x.scatter_forward()
        self.e_r.x.array[:] = self.e_r_t.x.array[:]
        self.e_r.x.scatter_forward()
        self.e_phi.x.array[:] = self.e_phi_t.x.array[:]
        self.e_phi.x.scatter_forward()
        
        # update localization matrix with the new orientation
        self.A_inf.update_func()
        
        self.E.update_func()
        