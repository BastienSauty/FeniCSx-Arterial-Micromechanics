import numpy as np
import ufl
from dolfinx import mesh, fem
from scipy.interpolate import interp1d
import sys


"""
A set of classes that are used to define some parameters as fields in a given geometry. 
These parameters' fields are accessed through self.func[0] (or equivalent direct
indexing self[0] for cst_scalar) in the other objects of the framework, that is
mostly in the inclusions_class file. Every physical, constant-over-the-domain
parameter (poisson ratio, volumic fraction, a Constant young modulus, reference
moduli, shape ratio, active-behavior constants...) goes through cst_scalar, so
the whole framework shares one calling convention regardless of whether the
parameter is spatially uniform (cst_scalar), spatially varying (nn_uniform_param),
or history/state-dependent (nn_cst_young_modulus).
Never let a raw python scalar leak directly into a symbolic form - use cst_scalar
instead, at least - as that can mess with the finite element code in fenicsx.
/!\ Stiffness of the matrix is defined as fem.Constant for ease of manipulation. Only the inclusions stiffnesses are defined with cst_scalar and nn_cst_young_modulus /!\
    
Classes :
    inverse_matrix6_6 : manage the definition and inversion of a 6*6 matrix in fenicsx. Usefull for the Mori-Tanaka homogenization
    cst_scalar : unified class for any scalar physical parameter, constant over the domain (poisson ratio, volumic fraction, a Constant young modulus, reference moduli, shape ratio...)
    nn_cst_young_modulus : define different types of non-constant young modulus that depends on a scalar value (a scalar measure of the stretch)
    nn_uniform_param : specific class to define a parameter that changes over the geometry of the artery
    matrix_volumic_frac : manage the volumic fraction of the matrix. Especially useful when the other volumic fractions change in the geometry, leading to a non constant volumic frac in the matrix.

Every class in this module implements a set_value() method (parameter-appropriate signature)
that updates the parameter's existing fem.Constant / fem.Function objects in place. None of
them rebuild any UFL form or fem.Expression, so calling set_value() never triggers an FFCx
JIT recompilation - the whole point of these classes is to let a value change (e.g. during
calibration) independently of the compiled code.
"""


class inverse_matrix6_6:
    """
    Class to manage the inversion of matrix 6*6
    Used in Cylinders and spheroid inclusions, Mori Tanaka homogenization
    Don't forget to update it at each numerical step.
    
    Parameters:
    -----------
        expr : expression of the 6*6 matrix to inverse. Used through its interpolation onto the funcinv function.
        V_stiff : functionspace for 6*6 matrix. Discontinuous, constant per element : fem.functionspace(domain, ("DG", 0, (6,6)))
        cells : restrict the matrix to a set of cells (for multiple materials, see mech_problem_class subdomains)
        
    Attributes:
    -----------
        funcinv : field that contains the value of the 6*6 matrix to inverse in each gaussian point
        func : field that contains the value of the inversed matrix. This is the field to use as the inverse of the expression that we input.
        
    Methods:
    -----------
        update_func : compute the value of the funcinv field. Inverse the matrix at each point. Store the inversed matrix in self.func.
    """
    
    def __init__(self, expr, V_stiff, cells):
        self.cells = cells
        self.funcinv = fem.Function(V_stiff) # funcinv contains the interpolation of the expression
        self.func = fem.Function(V_stiff) # contains the results of the inversion of funcinv dofs
        self.expr = fem.Expression(expr, V_stiff.element.interpolation_points())
        self.nb_elem = len(self.func.x.array)//36
    
    def update_func(self):
        """
        Update field func through first the update of the funcinv based on the interpolation of expr, then takes the numerical array, inverse for each element the 6*6 matrix, fill the dofs of func
        """
        # interpolate the expression for each element
        self.funcinv.interpolate(self.expr, self.cells) 
        self.funcinv.x.scatter_forward()
         
        # inverse each 6*6 matrix only in the cells of the layer/material 
        field_funcinv = self.funcinv.x.array.reshape(self.nb_elem,6,6)
        field_func_cells = np.linalg.inv(field_funcinv[self.cells, :,:]) 
        # fill the total field
        field_func = np.zeros(field_funcinv.shape)
        field_func[self.cells, :, :] = field_func_cells
        # store in the dolfinx field
        self.func.x.array[:] = field_func.reshape(self.nb_elem*6*6)
        self.func.x.scatter_forward()


#-----------------------------------------------------------------------------#
#   Scalar Parameters - constant over the domain, value freely updatable
#-----------------------------------------------------------------------------#

class cst_scalar:
    """
    Unified class for ANY scalar physical parameter that is constant over its
    whole domain but whose value can be changed at any time : poisson ratio,
    volumic fraction, a Constant-type young modulus, reference matrix moduli
    (mu_0, k_0), shape ratio, active-behavior constants (alpha, characteristic
    time, basal stress...). Replaces the former cst_param (which was
    fem.Function-based) and the ad-hoc bare fem.Constant that used to be
    scattered across inclusions_class.py, so every "single value, constant
    over the domain" parameter in the framework now goes through the exact
    same class and the exact same calling convention.

    Backed by a single fem.Constant of shape (1,) (self.const), so updating
    its value never touches mesh/DOF machinery the way a fem.Function would
    (no interpolation, no MPI scatter) - genuinely cheaper than the
    fem.Function-based cst_param it replaces. It stays a drop-in replacement
    for cst_param though : cst_scalar[0] (equivalently cst_scalar.func[0],
    .func is just an alias returning self) reads exactly like the field-based
    parameter classes below (nn_cst_young_modulus, nn_uniform_param), so it
    can be freely combined/summed with them in the same UFL expression (e.g.
    summing volumic fractions across inclusions of different kinds, some
    spatially uniform and some not).

    Parameters:
    -----------
        value : scalar value
        V_scalar : functionspace this parameter conceptually lives on. Only
            used to recover the mesh (V_scalar.mesh) ; kept as a constructor
            argument for drop-in compatibility with every former
            cst_param(value, V_scalar) call site.

    Attributes:
    -----------
        const : the underlying shape-(1,) fem.Constant. Index with [0] (or
            equivalently .func[0]) to get a scalar UFL expression.

    Methods:
    -----------
        set_value : update the runtime value in place (no recompilation)
        update_func : no-op, kept for interface consistency with the other parameter classes
    """
    def __init__(self, value, V_scalar):
        self.V_scalar = V_scalar
        domain = V_scalar.mesh
        self.value = np.float64(value)
        self.const = fem.Constant(domain, np.array([self.value], dtype=np.float64))

    @property
    def func(self):
        """Alias so cst_scalar.func[0] reads exactly like cst_param used to, and
        like the other field-based parameter classes below (nn_cst_young_modulus,
        nn_uniform_param)."""
        return self.const

    def __getitem__(self, idx):
        """cst_scalar[0] is equivalent to cst_scalar.func[0]."""
        return self.const[idx]

    def set_value(self, value):
        """Update the constant's runtime value in place. No form/expression rebuilt."""
        self.value = np.float64(value)
        self.const.value = np.array([self.value], dtype=np.float64)

    def update_func(self):
        """no-op ; kept so cst_scalar exposes the same interface as the other parameter classes"""
        pass


#-----------------------------------------------------------------------------#
#   Young Modulus - One Class for different cases
#-----------------------------------------------------------------------------#

class nn_cst_young_modulus:
    """
    A unified class to manage a changing young modulus. The evolution of the young modulus is local, controlled by a scalar_form that is interpolated at each step 
    in the whole domain. Type flag to distinguish the different cases of non constant param.
    The object is first initialized with its general attributes, but with a zero value func field.
    The func needs to be initialized by calling init_fund, with the controlling measure that is scalar_form.
    
    Parameters:
    -----------
        param_type : str flag that controls the choice of non constant parameter that we consider : either 'Plateau-Ramp-Plateau' or 'Exponential'
        value : list of parameters in the law of the non constant young modulus.
        V_scalar : functionspace for constant scalar value. Discontinuous, constant per element : fem.functionspace(domain, ("DG", deg_scal, (1,)))
        domain, cells : geometry and subgeometry to restrict the computation in a single material (see subdomains in mech_problem_class)
        
    Attributes:
    -----------
        func : field that contains the changing value of the parameter. This is a scalar field that is function of the scalar_form given in init_func input.
            access in the construction of the fenicsx forms and expression using nn_cst_young_modulus.func[0] 
        param_expr : expression that is computed in each cells of the domain and interpolated onto func.
        
        
    Methods:
    -----------
        init_func :
            initialize the param.func and its expression as a function of a scalar measure of stretch, given by scalar_form.
            
        update_func :
            compute the parameter field for the given values of the scalar_form input.
    """
    
    def __init__(self, param_type, value, V_scalar, domain, cells):
        #(self, param_type, value, geom):
        self.V_scalar = V_scalar # geom['scalar spacefunction']
        self.domain = domain # geom['domain']
        self.param_type = param_type
        self.cells = cells # geom['cells']
        self.value = value 
        self.func = fem.Function(self.V_scalar)
        
    def init_func(self, scalar_form):
        """
        scalar_form is the expression controling the evolution of the parameter
        in the most common case it's basically the axial elongation of the fiber (cylinder)
        and the young modulus is just stiffening the fiber as it is stretched
        
        This is where we build the different type of young modulus we implement and distinguish through the self.param_type
        """
        self.scalar_form = scalar_form
        
        if self.param_type=='Plateau-Ramp-Plateau':
            # syntax based on a y : x -> y(x) simple scalar function
            # plateau y0 for x<x0 ; y1 for x > x1 ; piecewise linear otherwise
            # could be expanded for multiple components
            self.y0_const = fem.Constant(self.domain, np.float64(self.value[0][0]))
            self.y1_const = fem.Constant(self.domain, np.float64(self.value[0][1]))
            self.x0_const = fem.Constant(self.domain, np.float64(self.value[1][0]))
            self.x1_const = fem.Constant(self.domain, np.float64(self.value[1][1]))
            
            param_expr = ufl.conditional(self.scalar_form>self.x1_const, self.y1_const, 
                                         ufl.conditional(self.scalar_form<self.x0_const, self.y0_const, 
                                                        self.y0_const+(self.y1_const-self.y0_const)/(self.x1_const-self.x0_const)*(self.scalar_form-self.x0_const)))
            
            self.param_expr = fem.Expression(param_expr, self.V_scalar.element.interpolation_points())
        elif self.param_type=='Exponential':
            # multi component exponential ; same syntax is used for every component ;
            # component are summed ; number is obtained fromlength of param list
            # E(x) = sum_i e_i * exp(k_i * (x - l_i))
            e = self.value[0]
            k = self.value[1]
            l = self.value[2]
            # Store as fem.Constant so the coefficients live outside the symbolic
            # form: their runtime value can change (e.g. during calibration)
            # without altering the UFL form signature used as the FFCx JIT cache key.
            self.e_const = [fem.Constant(self.domain, np.float64(ei)) for ei in e]
            self.k_const = [fem.Constant(self.domain, np.float64(ki)) for ki in k]
            self.l_const = [fem.Constant(self.domain, np.float64(li)) for li in l]
            param_form = 0
            for i in range(len(e)):
                param_form += self.e_const[i]*ufl.exp(self.k_const[i]*(self.scalar_form-self.l_const[i]))
            
            self.param_expr = fem.Expression(param_form, self.V_scalar.element.interpolation_points())
            
        self.update_func()
        
    def update_func(self):
        """
        Compute the scalar field of the non constant parameter
        """
        self.func.interpolate(self.param_expr, self.cells)
        self.func.x.scatter_forward()

    def set_value(self, value):
        """
        Update the law's coefficients in place (no recompilation). Must be called
        AFTER init_func() has run once (i.e. after the owning inclusion's
        microscopic_mech() has been called, typically inside build_weak_form()) since
        it only updates the fem.Constant objects already created by init_func -
        it does not rebuild self.param_expr.

        value must have the same structure as at construction :
            - 'Exponential' : [[e0,e1,...], [k0,k1,...], [l0,l1,...]]
            - 'Plateau-Ramp-Plateau' : [[y0,y1], [x0,x1]]
        """
        self.value = value
        if self.param_type == 'Exponential':
            e, k, l = value
            for i in range(len(e)):
                self.e_const[i].value = np.float64(e[i])
                self.k_const[i].value = np.float64(k[i])
                self.l_const[i].value = np.float64(l[i])
        elif self.param_type == 'Plateau-Ramp-Plateau':
            self.y0_const.value = np.float64(value[0][0])
            self.y1_const.value = np.float64(value[0][1])
            self.x0_const.value = np.float64(value[1][0])
            self.x1_const.value = np.float64(value[1][1])
        self.update_func()

#-----------------------------------------------------------------------------#
#   Volumic Fractions
#-----------------------------------------------------------------------------#

class nn_uniform_param:
    """
    General class to manage a parameter that is non uniform over the thickness of the artery.
    Its value depends on a 1d spline. The field is interpolated onto the values of the spline.
    The control points of the spline are given in the input value.
    
    Parameters:
    -----------
        value : a two element list. Each Element is a list that correspond to the control points of the spline
        V_scalar : functionspace for constant scalar value. Discontinuous, constant per element : fem.functionspace(domain, ("DG", deg_scal, (1,)))
        cells : geometry and subgeometry to restrict the computation in a single material (see subdomains in mech_problem_class)
        
    Attributes:
    -----------
        func : field that contains the changing value of the parameter. This is a scalar field that is function of the scalar_form given in init_func input.
            access in the construction of the fenicsx forms and expression using nn_uniform_param.func[0] 
    """
    
    def __init__(self, value, V_scalar, cells):
        self.V_scalar = V_scalar # geom['scalar spacefunction']
        self.cells = cells # geom['cells']
        self.value = value
        
        self.func = fem.Function(self.V_scalar)
        # Manage the spline : value[0] is the x axis ; value[1] is the y axis : 
            # y the value of the function we want to interpolate at the position x 
        spline_func = interp1d(self.value[0], self.value[1], kind='linear')
        
        # Here we use the fenicsx option to interpolate a field on a lambda function with the x input
        # x is a Dummy variable provided by the fem.Function.interpolate method and it's actually the position
        # so here, x[0] is just the position on the horizontal axix
        self.func.interpolate(lambda x: spline_func(x[0]), self.cells)
        self.func.x.scatter_forward()

    def set_value(self, value):
        """
        Update the spline control points in place (no recompilation) : value is a
        two element list [x_control_points, y_control_points], same shape as at
        construction. Re-interpolation with a python callable does not involve
        FFCx / the JIT compiler.
        """
        self.value = value
        spline_func = interp1d(self.value[0], self.value[1], kind='linear')
        self.func.interpolate(lambda x: spline_func(x[0]), self.cells)
        self.func.x.scatter_forward()


class matrix_volumic_frac:
    """
    create and fill the volumic fraction field using fem expression
    Object first created with a null value field
    create_field method is called once every inclusions have been defined so we can define the matrix volumic fraction
    
    Parameters:
    -----------
        V_scalar : functionspace for constant scalar value. Discontinuous, constant per element : fem.functionspace(domain, ("DG", deg_scal, (1,)))
        cells : geometry and subgeometry to restrict the computation in a single material (see subdomains in mech_problem_class)
        
    Attributes:
    -----------
        func : field that contains the volumic fraction of the matrix.
            access in the construction of the fenicsx forms and expression using matrix_volumic_frac.func[0] 
    
    Methods:
    -----------
        create_field : input : inclusion_form a form that contains the total value of volumic fractions of the other inclusions
            create and compute the volumic fraction of the matrix as a complement to one. 
            
    """
    def __init__(self, V_scalar, cells):
        self.V_scalar = V_scalar #geom['scalar spacefunction']
        self.cells = cells #geom['cells']
        self.func = fem.Function(self.V_scalar)
        
    def create_field(self, inclusion_form):
        """
        inclusion_form is the form that sums the incl.f.func
        """
        self.vol_frac_expr = fem.Expression(1-inclusion_form[0], self.V_scalar.element.interpolation_points())
        self.func.interpolate(self.vol_frac_expr, self.cells)

    def refresh(self):
        """
        Re-interpolate the matrix volumic fraction using the expression already
        built by create_field(). Call this after any inclusion's volumic fraction
        has been changed via set_value(), since it is defined as a complement to
        the (unchanged) symbolic sum of inclusion fractions - no recompilation.
        """
        self.func.interpolate(self.vol_frac_expr, self.cells)
        self.func.x.scatter_forward()
        
