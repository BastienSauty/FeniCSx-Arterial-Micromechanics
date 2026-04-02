import numpy as np
import ufl
import sys

from petsc4py.PETSc import ScalarType


from dolfinx import mesh, fem, plot, io
import dolfinx.fem.petsc
from petsc4py import PETSc

"""
Classes to manage the whole process of modelling any geometry with a stress-rate defined constitutive law.
This stress-rate law is a material defined for each subdomain using the material_class. 
This is the main object for building the simulation object. This object is built in interaction with the main file.
Whereas the main file defines the specific geometry, load, BCs and numerical steps, these classes define the general methods and attributes associated with a simulation.

Classes :
    Mechanical_Problem_axi: class that defines an axisymmetrical problem, from a 2D (x,y) plane into a (r, theta, z) cylinder. 
    Mechanical_Problem_3D: class that defines a general 3D problem. 

Comments :
    - /!\ highly structured hexahedral or rectangular meshes MUST be used for numerical stability. The mesh is contained in the domain object !
    - The two classes are highly identical, the only differences are in the number of dimensions (2 or 3). A lot of methods are identical. 
    Every modification that is not specific to the axisymmetrical assumptions should be done to both classes.
    - For now there is no LRS, that is that the local referencing system of the materials is the general orientation. There is a huge work to do to add this
        orientation system. It would be necessary when considering an artery in the (r,theta) plane, or a 3D cylinder. The radial, circ and axial direction would change depending on the position of the cell in the geom. 
"""

##########################################################################
# Home made modules needed 
##########################################################################

# Material classes
from Multiscale_Framework.class_modules.material_class import (
    Isotropic_Elastic_material,
    Homogenized_HD_material,
    Homogenized_MT_material,
    Active_MT_material,
)

# Auxiliary functions
from Multiscale_Framework.function_modules.auxiliary_functions import (
    compress,
    expand,
    projection_tensors_func,
    Voigt2Tensor,
    Tensor2Voigt,
    local_basis,
    set_orientation_fields,
    change_of_base_matrix,
)

# Nonlinear functions
from Multiscale_Framework.function_modules.nlog_function import (
    nlog_func,
    compute_v_field,
)

##########################################################################
## Mechanical Problem defined as a class for axisymmetric problem
##########################################################################

class Mechanical_Problem_axi:
    """
    Class to define a "mechanics" object which contains the attributes and methods related to solving the mechanical equilibrium weak form, able to solve one loading step through residuals minimization.
    Axisymmetric conditions -> 2D mesh
    
    Parameters:
    -----------
        Too many parameters. See for each methods the definition of the input parameters
        
    Attributes:
    -----------
        Too many attributes to list. Mains are: 
        subdomain : dict of the subdomains, that are material_class objects. Contains all the subdomains, the macroscopic materials, the microscopic inclusions.
        un, Sn : displacement field and PKII stress fields.
        max_iter, eps_tol : number of max Newton-Raphson iteration, tolerance of the NR.
        
    Methods: even if these are methods, they HAVE to be called in a sequential order in the main files.
    -----------
        __init__ : 
            defined the mech object.
        build_space_functions : 
            input the geomtry, associate the spacefunctions for the FeniCSx fields (scalar, vector, matrix...)
        add_subdomain : 
            create the subdomain in the geometry and associates it with a material constitutive law
        build_meshtags : 
            create the meshtags ; used when computing the weak form for each subdomain. Necessary when considering subdomain with different constitutive laws (the form of their weak form is of different nature)
        grad_axi : 
            specific function for axisymmetrical formulation that transforms the gradient from 2D to 3D axisymmetrical function
        build_weak_form : 
            build the weak form, ie residual and linearized residuals in the whole geometry (specific for each subdomain through the mat.tag and meshtags)
        update_local_quantities : 
            compute the increment of every local (micro and macro) quantities for each steps of the subintegration algorithm
        compute_increment : 
            Explicit Euler subintegration algorithm for each NR step
        build_BCs : 
            builds the Boundary Conditions in the residuals and its linear increments (for Follower Neumann condition) 
        build_solver : 
            build the elements in the linear solver from the linearized form of the residuals
        solve_1_step : 
            NR algorithm for solving one loading step.
    """

    def __init__(self, name, objective_derivative, n_int):
        """
        Define the mechanical object. Name affect the outputs name
        obj_der : define objective derivative chosen ['Oldroyd', 'Jaumann', 'GN', 'Log']
        n_int : number of sub-integration 
        """
        self.name = name
        self.objective_derivative = objective_derivative #['Jaumann', 'Log'][0] # defining the objective derivative
        self.n_int = n_int
        
        # Manage outputs 
        output_folder = "outputs"
        self.mech_output_name = output_folder+"/*"+self.name
        
        # initialize dictionnary of subdomains. Will contain all the sub-materials
        self.subdomain = {}

    def build_space_functions(self, domain):
        """
        define all the needed space_functions. These are the base to define variable fields for the different quantities in the the whole geometries. In general variables are constant in the elements, and just displacements are linearly interpolated. 
        
        Input : domain : meshed geometry
        
        How to use space_functions : 
        - var_field = ufl.Function(self.V_scalar) will define a scalar field with one value per element. var_field.x.array[:] is a [n_elem] 1D array containing the values
        - vector_field = ufl.Function(self.V_vec) will define a vector field with 3 values per element. or_field.x.array[:] is a [n_elem*3] 1D array that can be used through np.reshape as [nb_elem, 3] array 
        """
        # geometry is defined outside
        self.domain = domain
        self.nb_elem = len(domain.topology.original_cell_index)        
        self.meshtags_table = np.zeros((len(self.domain.topology.original_cell_index),), dtype=np.int32) # used in build_meshtags
        
        # degree of gauss interpolation
        deg_u = 1     # Displacement linear
        deg_sig = 1   # Stress constant element wise, must use "DG" in tensorelement definition
        deg_stiff = 0 # Stiffness matrix constant
        deg_matrix = 0# Matrix field
        deg_vec = 0   # Vector field
        deg_scal = 0  # Scalar Field

        # Definition of function space for the global resolution in displacement
        self.V_u = fem.functionspace(domain, ("P", deg_u, (domain.geometry.dim, )))
        
        # Field local stiffness matrix (6,6)
        self.V_stiff = fem.functionspace(domain, ("DG", deg_stiff, (6,6)))
        # 6*6 tensor space to store stiffness matrix in Mandel notation
        
        # Field Stresses (6,1)
        self.V_mandel = fem.functionspace(domain, ("DG", deg_sig, (6,)))
        # 6*1 vector space to store 3*3 tensors in mandel notation
        
        # Matrix (3,3)
        self.V_mat = fem.functionspace(domain, ("DG", deg_matrix, (3,3)))
        
        # Scalar
        self.V_scalar = fem.functionspace(domain, ("DG", deg_scal, (1,)))
        
        # Vector (3,1)
        self.V_vec = fem.functionspace(domain, ("DG", deg_vec, (3,)))
        # 3*1 vector space to store orientation vectors and eigenvalues vector
        
    ##########################################################################
    
    def add_subdomain(self, name, card, homogenization_type='MT'):
        """
        Create the subdomain defined with the material card
        material card : a dictionnary of dictionnaries with the different components; mandatory dict are :
            # -> matrix contains reference material
            # -> geometry contains global mesh and facets of the specific material
            # -> one inclusion dict defining an inclusion for all the material except Isotropic
        """
        # Update the meshtag
        cells_sub = card["geometry"]["cells"]
        tag_sub = card["geometry"]["tag"]
        self.meshtags_table[cells_sub] = tag_sub # attribute the tag to the cell
        
        print('-------------------------', flush=True)
        print(f'Initializing Layer {name}', flush=True)
        if homogenization_type=='MT':
            self.subdomain[name] = Homogenized_MT_material(card)
        elif homogenization_type=='ActiveMT':
            self.subdomain[name] = Active_MT_material(card)
        elif homogenization_type=='HD':
            self.subdomain[name] = Homogenized_HD_material(card)
        elif homogenization_type=='Iso':
            self.subdomain[name] = Isotropic_Elastic_material(card)
        else:
            print('Domain type not in ["MT", "ActiveMT", "HD", "Iso"]', flush=True)
            
    def build_meshtags(self):
        """
        Function that check the definition of the different subdomains and define the meshtags in the fenicsx syntax
        Compare the number of cells with a material tag with the domain cells
        Than build the meshtags object using mesh.meshtags based on the previously defined meshtag_table
        """

        if np.any(self.meshtags_table==0):
            print("Subdomains are not well defined, process stoped", flush=True)
            print(f"Some elements are meshtaged 0 : {self.meshtags_table}", flush=True)
            sys.exit()
        else:
            print('-------------------------', flush=True)
            print("Subdomains are defined on the whole domain", flush=True)
            
        self.meshtags = mesh.meshtags(self.domain, self.domain.topology.dim, np.sort(self.domain.topology.original_cell_index), self.meshtags_table)

    ##########################################################################

    def grad_axi(self, u):
        """
        Function to compute the axisymmetric gradient
        """
        grad2D = ufl.grad(u)
        grad_axi = ufl.as_tensor([[grad2D[0,0], 0, grad2D[0,1]],
                                  [0, u[0]/self.r, 0],
                                  [grad2D[1,0], 0, grad2D[1,1]]])
        return(grad_axi)

    ##########################################################################
        
    def build_weak_form(self):
        """
        Build the kinematic and linearized weak form that are used later on in the NR solver
        self.obj_der is the choice for the objective derivative at the macroscopic level. 
        Use Jaumann for fast tests but Log when running exact computation
        """
        self.x = ufl.SpatialCoordinate(self.domain)
        self.r = self.x[0]
        
        # Defining Integrals
        self.metadata = {"quadrature_degree": 6}
        #self.ds = ufl.Measure('ds', domain=self.domain, metadata=metadata)
        self.dx = ufl.Measure("dx", domain=self.domain, metadata=self.metadata, subdomain_data=self.meshtags)
        
        # Manage time increment for inelastic strain
        self.delta_t = fem.Constant(self.domain, 0.) # increment of time -> affects inelastic
        self.d_el_flag = fem.Constant(self.domain, 0.) # increment of elastic disp
        
        # Displacement and Stress at current config
        self.un = fem.Function(self.V_u)
        self.Sn = fem.Function(self.V_mandel) # 6x1
        
        #---------------------------------------------------------------------#
        # Directional function for derivative of residuals
        self.du = fem.Function(self.V_u)
        self.v = ufl.TestFunction(self.V_u)
        
        # Kinematic
        self.Fn = ufl.variable(ufl.Identity(3)+self.grad_axi(self.un))
        self.J = ufl.det(self.Fn)
        self.Finv = ufl.inv(self.Fn)
        self.B = ufl.dot(self.Fn, self.Fn.T) # left Cauchy-Green

        # Green-Lagrange tensors for directional derivative of residuals
        dE = 1/2*(ufl.dot(self.Fn.T, self.grad_axi(self.v)) + ufl.dot(self.grad_axi(self.v).T, self.Fn))
        
        # Incremental kinematic for directional derivative
        l = ufl.dot(self.grad_axi(self.du), self.Finv) # Spatial velocity gradient
        d = ufl.sym(l)
        
        #---------------------------------------------------------------------#
        # Incremental functions
        self.duj = fem.Function(self.V_u) # increment of displacement 
        self.dSint = fem.Function(self.V_mandel) # increment of stress, in Mandel
        # -> dSint is the increment of stress for duj displacement
        dFj = self.grad_axi(self.duj)
      
        # Incremental kinematic for stress integration
        lj = ufl.dot(dFj, self.Finv) # Spatial velocity gradient
        dj = ufl.sym(lj)
        
        #---------------------------------------------------------------------#
        # Push forward PKII stress into Kirchhoff stress
        tau = ufl.dot(self.Fn, ufl.dot(Voigt2Tensor(self.Sn), self.Fn.T)) # in global CS
        
        # behavior law : for each material
        for mat_name, mat in self.subdomain.items():
            mat.F_res = ufl.inner(Voigt2Tensor(self.Sn), dE)*self.r*self.dx(mat.tag)
            # compute the stress directional derivative from strain flow in local RS
            mat.homogenization_scheme(l, lj, self.delta_t)
            
            # directional derivative of the material is transformed into the general RS
            dtau_obj = Voigt2Tensor(mat.dtau_obj)
            dtau_obj_incr = Voigt2Tensor(mat.dtau_obj_incr)
            
            
            # Lie derivative defined from Jaumann derivative
            if self.objective_derivative=='Jaumann':
                dtau = dtau_obj - ufl.dot(tau, d) - ufl.dot(d, tau) 
                dtauj = dtau_obj_incr - ufl.dot(tau, dj) - ufl.dot(dj, tau) # + ufl.tr(dj)*sig
          
            # Pull back increment of kirchoff stress
            dS = ufl.dot(self.Finv, ufl.dot(dtau, self.Finv.T)) # Tensor Notation

            # Increment of residuals related to constitutive law
            incr = ufl.inner(dE, dS)
            # Increment of residuals caused by transport of stress
            transport = ufl.inner(ufl.dot(self.grad_axi(self.du), Voigt2Tensor(self.Sn)), self.grad_axi(self.v)) 
            mat.dF_res = (transport+incr)*self.r*self.dx(mat.tag)
               
            # Pull back to compute PKII stress rate
            dSj = ufl.dot(self.Finv, ufl.dot(dtauj, self.Finv.T))
            mat.dSj_expr = fem.Expression(Tensor2Voigt(dSj), self.V_mandel.element.interpolation_points())
            # how to use : update self.duj, then self.dSint.interpolate(mat.dSj_expr, mat.cells) for all materials
        
        self.dF_res = sum(mat.dF_res for mat_name, mat in self.subdomain.items())
        self.F_res = sum(mat.F_res for mat_name, mat in self.subdomain.items())
        
        print('Residuals and Stress Increments are defined\n', flush=True)
        
        
    ##########################################################################
        
    def update_local_quantities(self):
        """
        compute stiffness matrix. temporary solution
        """
        for mat_name, mat in self.subdomain.items():
            mat.matrix.update_micro_mech()
            for incl in mat.inclusions.values():
                incl.update_micro_mech() # compute the increment of stress in the inclusions and new behavior
            
            if type(mat) is Homogenized_MT_material:
                # separation of the forms to save time in preprocess
                mat.H2inv.update_func()
                mat.M.update_func()
            elif type(mat) is Active_MT_material:
                mat.M.update_func()

    def compute_increment(self):
        """
        Compute the increment of all macro and micro values, from the total displacement found by the NR step, that is stored in self.du
        """
        # intermediatary steps with the same step length
        self.duj.x.array[:] = self.d_el_flag.value/self.n_int*self.du.x.array[:]
        self.delta_t.value /= self.n_int
        
        for j_int in range(self.n_int):
            # compute the small increment
            for mat_name, mat in self.subdomain.items():
                self.dSint.interpolate(mat.dSj_expr, mat.cells)
    
            self.Sn.x.array[:] += self.dSint.x.array[:]
            self.un.x.array[:] += self.duj.x.array[:]
            
            # update local quantities : stiffness and microstructure
            self.update_local_quantities()
            
            # update objective derivative values at new current state
            if self.objective_derivative=='Log':
                # compute strain eigenvalues at new Fn value
                self.B_field.interpolate(self.B_field_expr)
                fieldstrain = self.B_field.x.array.reshape(self.nb_elem,3,3)
                v_array = compute_v_field(np.linalg.eigvals(fieldstrain))
                self.v_eig_field.x.array[:]=v_array.reshape(self.nb_elem*3)
                
    ##########################################################################
    
    def product_axy(self,mat,vec):
        prod = ufl.as_vector([mat[0,0]*vec[0] + mat[0,2]*vec[1], mat[2,0]*vec[0]+mat[2,2]*vec[1]]) 
        return(prod)   
            
    def build_BCs(self, boundaries, boundary_conditions):
        """
        Building boundary conditions : code from Jorgensd tutorial
        """
        
        facet_indices, facet_markers = [], []
        self.fdim = self.domain.topology.dim - 1
        for (marker, locator) in boundaries:
            #facets = mesh.locate_entities(self.domain, self.fdim, locator)
            facets = mesh.locate_entities_boundary(self.domain, self.domain.topology.dim - 1, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        self.facet_tag = mesh.meshtags(self.domain, self.fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
        
        self.ds = ufl.Measure('ds', domain=self.domain, metadata=self.metadata, subdomain_data=self.facet_tag)
        
        def BoundaryCondition(type, marker, values):
            if type == "Dirichlet":
                axis = values[1]

                facets = self.facet_tag.find(marker) # boundary_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, boundaries[i])
                dofs = fem.locate_dofs_topological(self.V_u.sub(axis), self.fdim, facets)
                
                if values[0]=="clamped":
                    bc = fem.dirichletbc(ScalarType(0.), dofs, self.V_u.sub(axis))
                else:
                    bc = fem.dirichletbc(values[0], dofs, self.V_u.sub(axis))
                
                linearbc = 0
            elif type == "Neumann_follower":
                bc = - self.J*ufl.dot(self.product_axy(self.Finv.T, self.v), values) *self.r*self.ds(marker)

                Fndu = self.Fn+self.grad_axi(self.du)
                Jdu = ufl.det(Fndu)
                Fnduinv = ufl.inv(Fndu)
                linearbc = - Jdu*ufl.dot(self.product_axy(Fnduinv.T, self.v), values) *self.r*self.ds(marker)
            elif type =="Neumann":
                bc = - ufl.dot(self.v, values) * self.r*self.ds(marker)
                linearbc = 0           
            elif type == "Robin":
                bc = values[0] * ufl.inner(self.un-values[1], self.v)*self.r*self.ds(marker)
                linearbc = 0 # not true
            else:
                raise TypeError("Unknown boundary condition: {0:s}".format(type))
            return(bc, linearbc)

        self.bcs = []
        for condition_list in boundary_conditions:
            condition, linearbc = BoundaryCondition(condition_list[0],condition_list[1],condition_list[2])
            if condition_list[0] == "Dirichlet":
                self.bcs.append(condition)
            else:
                self.F_res += condition  
                self.dF_res += linearbc    

    
    ##########################################################################
    
    def build_solver(self):
        """
        Define the main elements for the custom Newton-Raphson using the system solver:
        J(u0).du = -F(u0) avec J = dF/du
        This is a personal adaptation of the Custom Newton Solver from Dokken
        """
        ## Defining the residuals to use in the NR solver
        self.residual = fem.form(self.F_res)
        Jac = ufl.derivative(self.dF_res, self.du)
        self.jacobian = fem.form(Jac)
        
        ## Element formatting for petsc Linear System Inversion        
        self.A = fem.petsc.create_matrix(self.jacobian)
        self.L = fem.petsc.create_vector(self.residual)

        ## Define petsc solver
        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.opts = PETSc.Options()
        prefix = f"solver_{id(self.solver)}"
        self.solver.setOptionsPrefix(prefix)
        option_prefix = self.solver.getOptionsPrefix()
        self.opts[f"{option_prefix}ksp_type"] = "preonly"
        self.opts[f"{option_prefix}pc_type"] = "lu"
        self.opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"

        self.solver.setFromOptions()
        self.solver.setOperators(self.A)
        
        
    def solve_1_step(self, delta_t):
        """
        Based on the solver, matrix A and vector L defined in self.build_solver(). 
        This function minimizes the residuals using the Newton Raphson procedure
        Convergence should be quadratic !
        delta_t : increment of time that correspond to the increase of load / inelastic strain
        """
        i = 0
        self.max_iter = 100
        eps_tol = 1e-8
        
        # Manage Inelastic Strains
        # self.delta_t is a constant field defined in build_weak_form
        self.delta_t.value = delta_t # apply the increment of time to compute the increment of inelastic strain
        self.d_el_flag.value = 0
        if self.delta_t.value != 0:
            # update all quantities -> stress, stiff, disp tot
            self.compute_increment() 
        # first step : compute the increments related to the inelastic strain. 
        self.delta_t.value = 0
        self.d_el_flag.value = 1
        # Then, search for the elastic field that respects the mechanical equilibrium
        
        while i<self.max_iter:
            # Assemble Jacobian and residual : based on Custom Newton solver from Dokken Tutorial
            with self.L.localForm() as loc_L:
                loc_L.set(0)
            self.A.zeroEntries()
            fem.petsc.assemble_matrix(self.A, self.jacobian, bcs=self.bcs)
            self.A.assemble()
            fem.petsc.assemble_vector(self.L, self.residual)
            self.L.assemble()
            self.L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.L.scale(-1)
            # Compute b - J(u_D-u_(i-1))
            fem.petsc.apply_lifting(self.L, [self.jacobian], [self.bcs], x0=[self.un.x.petsc_vec], alpha=1)
            # Set dx|_bc = u_{i-1}-u_D
            fem.petsc.set_bc(self.L, self.bcs, self.un.x.petsc_vec, 1.0)
            self.L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
            
            self.solver.solve(self.L, self.du.x.petsc_vec)
            self.du.x.scatter_forward()
            
            # update all quantities -> stress, stiff, disp tot
            self.compute_increment()
            
            # compute the value of the residuals
            correction_norm = self.L.norm()
            #print(f"Iter {i} Residuals {correction_norm}")
            i+=1
            if correction_norm < eps_tol:
                break
        
        return(i, correction_norm)
    

##########################################################################
## Mechanical Problem for a 3D problem
##########################################################################

class Mechanical_Problem_3D:
    """
    Class to define a "mechanics" object which contains the attributes and methods related to solving the mechanical equilibrium weak form, able to solve one loading step through residuals minimization.
    3D mesh.
    
    Parameters:
    -----------
        Too many parameters. See for each methods the definition of the input parameters
        
    Attributes:
    -----------
        Too many attributes to list. Mains are: 
        subdomain : dict of the subdomains, that are material_class objects. Contains all the subdomains, the macroscopic materials, the microscopic inclusions.
        un, Sn : displacement field and PKII stress fields.
        max_iter, eps_tol : number of max Newton-Raphson iteration, tolerance of the NR.
        
    Methods: even if these are methods, they HAVE to be called in a sequential order in the main files.
    -----------
        __init__ : 
            defined the mech object.
        build_space_functions : 
            input the geomtry, associate the spacefunctions for the FeniCSx fields (scalar, vector, matrix...)
        add_subdomain : 
            create the subdomain in the geometry and associates it with a material constitutive law
        build_meshtags : 
            create the meshtags ; used when computing the weak form for each subdomain. Necessary when considering subdomain with different constitutive laws (the form of their weak form is of different nature)
        build_weak_form : 
            build the weak form, ie residual and linearized residuals in the whole geometry (specific for each subdomain through the mat.tag and meshtags)
        update_local_quantities : 
            compute the increment of every local (micro and macro) quantities for each steps of the subintegration algorithm
        compute_increment : 
            Explicit Euler subintegration algorithm for each NR step
        build_BCs : 
            builds the Boundary Conditions in the residuals and its linear increments (for Follower Neumann condition) 
        build_solver : 
            build the elements in the linear solver from the linearized form of the residuals
        solve_1_step : 
            NR algorithm for solving one loading step.
    """
    
    def __init__(self, name, objective_derivative, n_int):
        """
        Define the mechanical object. Name affect the outputs name
        obj_der : define objective derivative chosen ['Oldroyd', 'Jaumann', 'GN', 'Log']
        n_int : number of sub-integration 
        """
        self.name = name
        self.objective_derivative = objective_derivative #['Jaumann', 'Log'][0] # defining the objective derivative
        self.n_int = n_int
        
        # Manage outputs 
        output_folder = "outputs"
        self.mech_output_name = output_folder+"/*"+self.name
        
        # initialize dictionnary of subdomains. Will contain all the sub-materials
        self.subdomain = {}

            
    def build_space_functions(self, domain):
        """
        define all the needed space_functions. These are the base to define variable fields for the different quantities in the the whole geometries. In general variables are constant in the elements, and just displacements are linearly interpolated. 
        
        Input : domain : meshed geometry
        obj_der
        How to use space_functions : 
        - var_field = ufl.Function(self.V_scalar) will define a scalar field with one value per element. var_field.x.array[:] is a [n_elem] 1D array containing the values
        - or_field = ufl.Function(self.V_vec) will define a vector field with 3 values per element. or_field.x.array[:] is a [n_elem*3] 1D array that can be used through np.reshape as [nb_elem, 3] array 
        """
        # geometry is defined outside
        self.domain = domain
        self.nb_elem = len(domain.topology.original_cell_index)        
        self.meshtags_table = np.zeros((len(self.domain.topology.original_cell_index),), dtype=np.int32) # used in build_meshtags
        
        # degree of gauss interpolation
        deg_u = 1     # Displacement linear
        deg_sig = 1   # Stress constant element wise, must use "DG" in tensorelement definition
        deg_stiff = 0 # Stiffness matrix constant
        deg_matrix = 0# Matrix field
        deg_vec = 0   # Vector field
        deg_scal = 0  # Scalar Field

        # Definition of function space for the global resolution in displacement
        self.V_u = fem.functionspace(domain, ("P", deg_u, (domain.geometry.dim, )))
        
        # Field local stiffness matrix (6,6)
        self.V_stiff = fem.functionspace(domain, ("DG", deg_stiff, (6,6)))
        # 6*6 tensor space to store stiffness matrix in Mandel notation
        
        # Field Stresses (6,1)
        self.V_mandel = fem.functionspace(domain, ("DG", deg_sig, (6,)))
        # 6*1 vector space to store 3*3 tensors in mandel notation
        
        # Matrix (3,3)
        self.V_mat = fem.functionspace(domain, ("DG", deg_matrix, (3,3)))
        
        # Scalar
        self.V_scalar = fem.functionspace(domain, ("DG", deg_scal, (1,)))
        
        # Vector (3,1)
        self.V_vec = fem.functionspace(domain, ("DG", deg_vec, (3,)))
        # 3*1 vector space to store orientation vectors and eigenvalues vector
        
    ##########################################################################
    
    def add_subdomain(self, name, card, homogenization_type='MT'):
        """
        Create the subdomain defined with the material card
        material card : a dictionnary of dictionnaries with the different components; mandatory dict are :
            # -> matrix contains reference material
            # -> geometry contains global mesh and facets of the specific material
            # -> one inclusion dict defining an inclusion for all the material except Isotropic
        """
        # Update the meshtag
        cells_sub = card["geometry"]["cells"]
        tag_sub = card["geometry"]["tag"]
        self.meshtags_table[cells_sub] = tag_sub # attribute the tag to the cell
        
        if homogenization_type=='MT':
            self.subdomain[name] = Homogenized_MT_material(card)
        elif homogenization_type=='ActiveMT':
            self.subdomain[name] = Active_MT_material(card)
        elif homogenization_type=='HD':
            self.subdomain[name] = Homogenized_HD_material(card)
        else:
            self.subdomain[name] = Isotropic_Elastic_material(card)
            
    def build_meshtags(self):
        """
        Function that check the definition of the different subdomains and define the meshtags in the fenicsx syntax
        Compare the number of cells with a material tag with the domain cells
        Than build the meshtags object using mesh.meshtags based on the previously defined meshtag_table
        """

        if np.any(self.meshtags_table==0):
            print("Subdomains are not well defined, process stoped", flush=True)
            sys.exit()
        else:
            print("Subdomains are defined on the whole domain", flush=True)
            
        self.meshtags = mesh.meshtags(self.domain, self.domain.topology.dim, np.sort(self.domain.topology.original_cell_index), self.meshtags_table)

    ##########################################################################
        
    def build_weak_form(self):
        """
        Build the kinematic and linearized weak form that are used later on in the NR solver
        New parameter : self.obj_der is the choice for the objective derivative at the macroscopic level. Use Jaumann for fast tests but Log when running exact computation
        """
        self.x = ufl.SpatialCoordinate(self.domain)
        
        # Defining Integrals
        self.metadata = {"quadrature_degree": 6}
        #self.ds = ufl.Measure('ds', domain=self.domain, metadata=metadata)
        self.dx = ufl.Measure("dx", domain=self.domain, metadata=self.metadata, subdomain_data=self.meshtags)
        
        # Manage time increment for inelastic strain
        self.delta_t = fem.Constant(self.domain, 0.) # increment of time -> affects inelastic
        self.d_el_flag = fem.Constant(self.domain, 0.) # increment of elastic disp
        
        # Displacement and Stress at current config
        self.un = fem.Function(self.V_u)
        self.Sn = fem.Function(self.V_mandel) # 6x1
        
        #---------------------------------------------------------------------#
        # Directional function for derivative of residuals
        self.du = fem.Function(self.V_u)
        self.v = ufl.TestFunction(self.V_u)
        
        # Kinematic
        self.Fn = ufl.variable(ufl.Identity(3)+ufl.grad(self.un))
        self.J = ufl.det(self.Fn)
        self.Finv = ufl.inv(self.Fn)
        self.B = ufl.dot(self.Fn, self.Fn.T) # left Cauchy-Green

        # Green-Lagrange tensors for directional derivative of residuals
        dE = 1/2*(ufl.dot(self.Fn.T, ufl.grad(self.v)) + ufl.dot(ufl.grad(self.v).T, self.Fn))
        
        # Incremental kinematic for directional derivative
        l = ufl.dot(ufl.grad(self.du), self.Finv) # Spatial velocity gradient
        d = ufl.sym(l)
        
        #---------------------------------------------------------------------#
        # Incremental functions
        self.duj = fem.Function(self.V_u) # increment of displacement 
        self.dSint = fem.Function(self.V_mandel) # increment of stress, in Mandel
        # -> dSint is the increment of stress for duj displacement
        dFj = ufl.grad(self.duj)
      
        # Incremental kinematic for stress integration
        lj = ufl.dot(dFj, self.Finv) # Spatial velocity gradient
        dj = ufl.sym(lj)
        
        #---------------------------------------------------------------------#
        # Push forward PKII stress into Kirchhoff stress
        tau = ufl.dot(self.Fn, ufl.dot(Voigt2Tensor(self.Sn), self.Fn.T)) # in global CS
        
        # behavior law : for each material
        for mat_name, mat in self.subdomain.items():
            mat.F_res = ufl.inner(Voigt2Tensor(self.Sn), dE)*self.dx(mat.tag)
            # compute the stress directional derivative from strain flow in local RS
            mat.homogenization_scheme(l, lj, self.delta_t)
            
            # directional derivative of the material is transformed into the general RS
            dtau_obj = Voigt2Tensor(mat.dtau_obj)
            dtau_obj_incr = Voigt2Tensor(mat.dtau_obj_incr)
            
            
            # Lie derivative defined from Jaumann derivative
            if self.objective_derivative=='Jaumann':
                dtau = dtau_obj - ufl.dot(tau, d) - ufl.dot(d, tau) 
                dtauj = dtau_obj_incr - ufl.dot(tau, dj) - ufl.dot(dj, tau) # + ufl.tr(dj)*sig
          
            # Pull back increment of kirchoff stress
            dS = ufl.dot(self.Finv, ufl.dot(dtau, self.Finv.T)) # Tensor Notation

            # Increment of residuals related to constitutive law
            incr = ufl.inner(dE, dS)
            # Increment of residuals caused by transport of stress
            transport = ufl.inner(ufl.dot(ufl.grad(self.du), Voigt2Tensor(self.Sn)), ufl.grad(self.v)) 
            mat.dF_res = (transport+incr)*self.dx(mat.tag)
               
            # Pull back to compute PKII stress rate
            dSj = ufl.dot(self.Finv, ufl.dot(dtauj, self.Finv.T))
            mat.dSj_expr = fem.Expression(Tensor2Voigt(dSj), self.V_mandel.element.interpolation_points())
            # how to use : update self.duj, then self.dSint.interpolate(mat.dSj_expr, mat.cells) for all materials
        
        self.dF_res = sum(mat.dF_res for mat_name, mat in self.subdomain.items())
        self.F_res = sum(mat.F_res for mat_name, mat in self.subdomain.items())
        
        print('Residuals and Stress Increments are defined\n', flush=True)
        
    ##########################################################################
        

    def update_local_quantities(self):
        """
        compute stiffness matrix. temporary solution
        """
        for mat_name, mat in self.subdomain.items():
            mat.matrix.update_micro_mech()
            for incl in mat.inclusions.values():
                incl.update_micro_mech() # compute the increment of stress in the inclusions and new behavior
            
            if type(mat) is Homogenized_MT_material:
                # separation of the forms to save time in preprocess
                mat.H2inv.update_func()
                mat.M.update_func()
            elif type(mat) is Active_MT_material:
                mat.M.update_func()


    def compute_increment(self):
        """
        Compute the increment of all macro and micro values, from the total displacement found by the NR step, that is stored in self.du
        """
        
        # intermediatary steps with the same step length
        self.duj.x.array[:] = self.d_el_flag.value/self.n_int*self.du.x.array[:]
        self.delta_t.value /= self.n_int
        
        for j_int in range(self.n_int):
            # compute the small increment
            for mat_name, mat in self.subdomain.items():
                self.dSint.interpolate(mat.dSj_expr, mat.cells)
    
            self.Sn.x.array[:] += self.dSint.x.array[:]
            self.un.x.array[:] += self.duj.x.array[:]
            
            # update local quantities : stiffness and microstructure
            self.update_local_quantities()
            
            # update objective derivative values at new current state
            if self.objective_derivative=='Log':
                # compute strain eigenvalues at new Fn value
                self.B_field.interpolate(self.B_field_expr)
                fieldstrain = self.B_field.x.array.reshape(self.nb_elem,3,3)
                v_array = compute_v_field(np.linalg.eigvals(fieldstrain))
                self.v_eig_field.x.array[:]=v_array.reshape(self.nb_elem*3)
                
    ##########################################################################
            
    def build_BCs(self, boundaries, boundary_conditions):
        """
        Building boundary conditions : code from Jorgensd tutorial
        """
        
        facet_indices, facet_markers = [], []
        self.fdim = self.domain.topology.dim - 1
        for (marker, locator) in boundaries:
            #facets = mesh.locate_entities(self.domain, self.fdim, locator)
            facets = mesh.locate_entities_boundary(self.domain, self.domain.topology.dim - 1, locator)
            facet_indices.append(facets)
            facet_markers.append(np.full_like(facets, marker))
        facet_indices = np.hstack(facet_indices).astype(np.int32)
        facet_markers = np.hstack(facet_markers).astype(np.int32)
        sorted_facets = np.argsort(facet_indices)
        self.facet_tag = mesh.meshtags(self.domain, self.fdim, facet_indices[sorted_facets], facet_markers[sorted_facets])
        
        self.ds = ufl.Measure('ds', domain=self.domain, metadata=self.metadata, subdomain_data=self.facet_tag)
        
        def BoundaryCondition(type, marker, values):
            if type == "Dirichlet":
                axis = values[1]

                facets = self.facet_tag.find(marker) # boundary_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, boundaries[i])
                dofs = fem.locate_dofs_topological(self.V_u.sub(axis), self.fdim, facets)
                
                if values[0]=="clamped":
                    bc = fem.dirichletbc(ScalarType(0.), dofs, self.V_u.sub(axis))
                else:
                    bc = fem.dirichletbc(values[0], dofs, self.V_u.sub(axis))
                
                linearbc = 0
            elif type == "Neumann_follower":
                bc = - self.J*ufl.dot(self.product_axy(self.Finv.T, self.v), values) *self.ds(marker)

                Fndu = self.Fn+ ufl.grad(self.du)
                Jdu = ufl.det(Fndu)
                Fnduinv = ufl.inv(Fndu)
                linearbc = - Jdu*ufl.dot(ufl.dot(Fnduinv.T, self.v), values) *self.ds(marker)
            elif type =="Neumann":
                bc = - ufl.dot(self.v, values) * self.ds(marker)
                linearbc = 0           
            elif type == "Robin":
                bc = values[0] * ufl.inner(self.un-values[1], self.v)*self.ds(marker)
                linearbc = 0 # not true
            else:
                raise TypeError("Unknown boundary condition: {0:s}".format(type))
            return(bc, linearbc)

        self.bcs = []
        for condition_list in boundary_conditions:
            condition, linearbc = BoundaryCondition(condition_list[0],condition_list[1],condition_list[2])
            if condition_list[0] == "Dirichlet":
                self.bcs.append(condition)
            else:
                self.F_res += condition  
                self.dF_res += linearbc    

    
    ##########################################################################
    
    def build_solver(self):
        """
        Define the main elements for the custom Newton-Raphson using the system solver:
        J(u0).du = -F(u0) avec J = dF/du
        This is a personal adaptation of the Custom Newton Solver from Dokken
        """
        ## Defining the residuals to use in the NR solver
        self.residual = fem.form(self.F_res)
        Jac = ufl.derivative(self.dF_res, self.du)
        self.jacobian = fem.form(Jac)
        
        ## Element formatting for petsc Linear System Inversion        
        self.A = fem.petsc.create_matrix(self.jacobian)
        self.L = fem.petsc.create_vector(self.residual)

        ## Define petsc solver
        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setType(PETSc.KSP.Type.PREONLY)
        self.solver.getPC().setType(PETSc.PC.Type.LU)
        self.opts = PETSc.Options()
        prefix = f"solver_{id(self.solver)}"
        self.solver.setOptionsPrefix(prefix)
        option_prefix = self.solver.getOptionsPrefix()
        self.opts[f"{option_prefix}ksp_type"] = "preonly"
        self.opts[f"{option_prefix}pc_type"] = "lu"
        self.opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"

        self.solver.setFromOptions()
        self.solver.setOperators(self.A)
        
        
    def solve_1_step(self, delta_t):
        """
        Based on the solver, matrix A and vector L defined in self.build_solver(). 
        This function minimizes the residuals using the Newton Raphson procedure
        Convergence should be quadratic !
        delta_t : increment of time that correspond to the increase of load / inelastic strain
        """
        i = 0
        self.max_iter = 100
        eps_tol = 1e-8
        
        # Manage Inelastic Strains
        # self.delta_t is a constant field defined in build_weak_form
        self.delta_t.value = delta_t # apply the increment of time to compute the increment of inelastic strain
        self.d_el_flag.value = 0
        if self.delta_t.value != 0:
            # update all quantities -> stress, stiff, disp tot
            self.compute_increment() 
        # first step : compute the increments related to the inelastic strain. 
        self.delta_t.value = 0
        self.d_el_flag.value = 1
        # Then, search for the elastic field that respects the mechanical equilibrium
        
        while i<self.max_iter:
            # Assemble Jacobian and residual : based on Custom Newton solver from Dokken Tutorial
            with self.L.localForm() as loc_L:
                loc_L.set(0)
            self.A.zeroEntries()
            fem.petsc.assemble_matrix(self.A, self.jacobian, bcs=self.bcs)
            self.A.assemble()
            fem.petsc.assemble_vector(self.L, self.residual)
            self.L.assemble()
            self.L.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            self.L.scale(-1)
            # Compute b - J(u_D-u_(i-1))
            fem.petsc.apply_lifting(self.L, [self.jacobian], [self.bcs], x0=[self.un.x.petsc_vec], alpha=1)
            # Set dx|_bc = u_{i-1}-u_D
            fem.petsc.set_bc(self.L, self.bcs, self.un.x.petsc_vec, 1.0)
            self.L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
            
            self.solver.solve(self.L, self.du.x.petsc_vec)
            self.du.x.scatter_forward()
            
            # update all quantities -> stress, stiff, disp tot
            self.compute_increment()
            
            #correction_norm = self.du.vector.norm(0)
            correction_norm = self.L.norm()
            #print(f"Iter {i} Residuals {correction_norm}")
            i+=1
            if correction_norm < eps_tol:
                break
        
            
        return(i, correction_norm)
    