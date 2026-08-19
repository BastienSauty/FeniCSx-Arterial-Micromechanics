import numpy as np
import sys
import ufl
import time
# Import classes from inclusions
from Multiscale_Framework.class_modules.inclusions_class import (
    Matrix,
    Spherical_inclusion,
    Spheroidal_inclusion,
    Active_Spherical_inclusion,
    Active_Spheroidal_inclusion,
    Prestretched_Cylinder_inclusion,
)

# Parameters
from Multiscale_Framework.class_modules.parameter_class import cst_scalar, inverse_matrix6_6

# Auxiliary functions
from Multiscale_Framework.function_modules.auxiliary_functions import (
    expand,
    tensordot_4_4,
    tensordot_4_2,
)

"""
Classes that define the material used in the mech_problem_class for each subdomain.
These materials define the microscopic constituents (throught the inclusions_class) and their homogenized -- macroscopic -- response.
When working on developping the macroscopic framework (geometry, BCs...) you should use the Isotropic_Elastic_Material (simple grade-zero hypoelastic material).
For the multiscale schemes (HD, MT, activeMT), see inclusions_class for the definition of the lower-scale components.
These classes are defined consistently. They are all callable in the same way. That is that they all contain the same attributes and methods.

Classes :
    Isotropic_Elastic_material : simple homogenous grade-zero hypoelastic material
    Homogenized_HD_material : High dilution scheme. 
    Homogenized_MT_material : Mori-Tanaka scheme (symmetrized) with passive elastic response of its microscopic constituents.
    Active_MT_material : unsymmetrized Mori-Tanaka scheme WITH inelastic behavior. These inelastic behaviors are defined for each active inclusions (see inclusions_class).

Note on mu_0/k_0 (the reference matrix moduli every inclusion needs for its
Eshelby/localization tensors) : these MUST be plain python floats, not a UFL
expression. Each inclusion's Eshelby tensor is built (once, at construction,
in inclusions_class.py's eshelby_isomatrix()) via eshelby_aux_spheroid(), which
runs a genuine scipy.integrate.quad numerical integration - fundamentally
incompatible with a symbolic/live value, regardless of any UFL Constant
indexing trick. So mu_0/k_0 are computed HERE as numeric snapshots (see
_numeric_reference_moduli below) from the matrix's young modulus/poisson
ratio, and stored in each inclusion as a cst_scalar - this is also physically
standard for Eshelby-based homogenization : the comparison medium is fixed
for the life of the mechanical problem, not something that evolves.
/!\ Consequence : if the matrix's own young modulus/poisson ratio is changed
later (e.g. Matrix.set_parameters() during calibration), the matrix's OWN
constitutive law (self.matrix.C, built from self.matrix.E/nu directly) stays
correct automatically (it's genuinely symbolic), but each inclusion's
mu_0/k_0 and its already-built Eshelby tensors (S_esh, R_esh, A_inf) do NOT
follow - they were fixed at construction time. update_parameters() below
deliberately does not try to "refresh" them (that would only update the raw
mu_0/k_0 value while leaving S_esh/R_esh inconsistent with it, which is worse
than doing nothing). If the matrix's own properties need to be a genuine
calibration/sensitivity target, rebuild the subdomain via setup_simulation()
for that case instead of reusing a persistent context.
"""


def _numeric_reference_moduli(matrix):
    """
    Compute the matrix's reference shear/bulk moduli (mu_0, k_0) as PLAIN
    PYTHON FLOATS from its current young modulus / poisson ratio (read from
    matrix.E.value / matrix.nu.value - the numeric side of cst_scalar,
    distinct from matrix.E[0]/matrix.nu[0] which stay symbolic UFL). See the
    module docstring above for why these must be real numbers rather than a
    live UFL expression - eshelby_aux_spheroid() runs a numerical integration
    that cannot accept a symbolic value.
    """
    E0 = float(matrix.E.value)
    nu0 = float(matrix.nu.value)
    mu0 = E0/(2*(1+nu0))
    k0 = E0/(3*(1-2*nu0))
    return mu0, k0


##########################################################################
## Different materials definition
##########################################################################

class Isotropic_Elastic_material:
    """
    Class defining the material law of an elastic isotropic material
    Simple grade-zero hypoelastic material.
    
    Parameters:
    -----------
        cells : restrict the material to a set of cells (used for multiple materials, see mech_problem_class subdomains)
        tag : int identifier of the subdomain
        material : dictionnary that contains the geometry and the matrix
        
    Attributes:
    -----------
        matrix : matrix object (inclusions_class) with a volumic fraction of 1 and a constant stiffness matrix.
        dtau_obj : objective macroscopic stress rate / directional derivative alongside direction du
        dtau_obj_incr : increment of macroscopic stress -> subincrement of displacement alongside direction duj = du/n_int
            -> see mech_problem_class.compute_increment and solve_1_step methods to understand the difference between dtau_obj and dtau_obj_incr
        
    Methods:
    -----------
        homogenization_scheme : l_el, l_elj, delta_t = velocity gradient, incremental velocity gradient, increment of time (for inelastic contribution)
            create the homogenization law. Here simple case with only the stress increment of the matrix.
    """
    
    def __init__(self, material):
        """
        Initialize isotropic grade-zero hypoelastic material with only a matrix.
        material is a dictionnary containing at least 2 dictionnaries : geometry, matrix.
        """
        ## From geometry dict : associate facets, dofs for the stiffness tensor
        self.cells = material["geometry"]["cells"]
        self.tag = material["geometry"]["tag"]
        
        ### Define matrix
        self.matrix = Matrix(material["matrix"], material["geometry"])
        self.inclusions = {}
        self.matrix.f = cst_scalar(1, material["geometry"]["scalar spacefunction"])
        
        ## Inelastic contribution for consistency with all material classes.
        self.matrix.inelastic_contribution(None) #ufl.as_tensor(np.zeros((3,3))))

    def update_parameters(self, material):
        """
        Push new matrix parameters, in place, no recompilation. material is a
        (possibly partial) dict shaped like this subdomain's json card, e.g.
        {"matrix": {"young": 0.06}}. No inclusions to update - this material
        is 100% matrix by construction.
        """
        if "matrix" in material:
            self.matrix.set_parameters(material["matrix"])

    def homogenization_scheme(self, l_el, l_elj, delta_t):
        """
        Create the constitutive law arising from mulsticale homogenization.
        l_el : spatial velocity gradient in the LRS (Local Referencing System -> rotated in the element orientation)
        l_elj : incremental value of l_el
        delta_t : increment of time 
        These are all FeniCSx fields that are used as "forms" or "Expressions" -> they are not numerical values per se.
        """
        
        inv_A_esh = ufl.as_matrix(np.eye(6)) # A_inf = A_i
        
        self.matrix.localization_tensors(inv_A_esh)
        
        self.matrix.microscopic_mech(l_el, l_elj, delta_t) 
        
        self.dtau_obj = self.matrix.dtau_dirder
        self.dtau_obj_incr = self.matrix.dtau_incr
    
        

###########################################################################
### High Dilution Homogenization
###########################################################################    
    
class Homogenized_HD_material:
    """
    Class defining the material law of a homogenized material using High Dilution homogenization 
    This is the direct implementation of the Eshelby solution -> hypothesis of inclusion in infinite matrix; LOW volumic fractions
    High-Dilution scheme:
        - Top Bottom : strain localization thanks to the solution of Eshelby
        - Bottom Up : weighted average stress rates in the inclusions
    
    Parameters:
    -----------
        cells : restrict the material to a set of cells (used for multiple materials, see mech_problem_class subdomains)
        tag : int identifier of the subdomain
        material : dictionnary that contains the geometry, the matrix and the inclusions. inclusions_keys lists the subdict in material that define each inclusion.
        
    Attributes:
    -----------
        matrix : matrix object (inclusions_class) with a volumic fraction of 1 and a constant stiffness matrix.
        inclusions : dict that contains all the inclusions objects. 
        dtau_obj, dtau_obj_incr : objective macroscopic stress rate / directional derivative alongside direction du and duj respectively
        
    Methods:
    -----------
        homogenization_scheme : l_el, l_elj, delta_t = velocity gradient, incremental velocity gradient, increment of time (for inelastic contribution)
            create the homogenization law. High-Dilution schemes.
    """
    
    def __init__(self, material):
        """
        Initialize material with a list of constituents, associated volumic fraction and mechanical behaviour, initial orientation
        material is a dictionnary containing at least 2 dictionnaries : geometry, matrix and at least one inclusion
        """
        ## From geometry dict : get facets and tag
        self.cells = material["geometry"]["cells"]
        self.tag = material["geometry"]["tag"]
        
        ### Define matrix
        self.matrix = Matrix(material["matrix"], material["geometry"])
        
        ### Define list of inclusions : list of object
        self.inclusions = {}
        self.inclusions_keys = list(material.keys()) # names of inclusions to call in the material dict
        self.inclusions_keys.remove("geometry")
        self.inclusions_keys.remove("matrix")
        
        # Total volumic fraction of inclusions :
        f_incl_form = 0.0
        
        mu0_ref, k0_ref = _numeric_reference_moduli(self.matrix)
        for name_incl in self.inclusions_keys:
            # reference values of the matrix for stiffnesses - plain numeric
            # floats, see _numeric_reference_moduli and the module docstring
            material[name_incl]["mu_0"] = mu0_ref
            material[name_incl]["k_0"] = k0_ref
            if material[name_incl]["type"] == "sphere":
                # Spheric inclusion
                # create material using the spherical inclusion dictionnary
                self.inclusions[name_incl] = Spherical_inclusion(material[name_incl], material["geometry"])    
                self.inclusions[name_incl].inelastic_contribution(None)
                
                f_incl_form += self.inclusions[name_incl].f.func
                
            elif material[name_incl]["type"] == "cylinder" or material[name_incl]["type"] == "prolate_spheroid":
                # cylindrical inclusion
                # create material using the cylinder inclusion dictionnary
                self.inclusions[name_incl] = Spheroidal_inclusion(material[name_incl], material["geometry"])   
                self.inclusions[name_incl].inelastic_contribution(None, None)  
                
                f_incl_form += self.inclusions[name_incl].f.func       
            
            else:
                type_=material[name_incl]["type"]
                print(f"non valid type : {type_} component with name {name_incl} ignored", flush=True)
            
            print(f"Inclusion {name_incl} initialized\n", flush=True)
        
        # Volumic fraction is defined from volumic fraction of inclusions
        self.matrix.f.create_field(f_incl_form)
        
        if any(x < 0 for x in self.matrix.f.func.x.array[:]):
            print("Matrix has at least a null or negative volumic fraction, code terminated", flush=True)
            sys.exit()
        else:
            print("Matrix Volumic fraction Correctly defined", flush=True)
        
        self.matrix.inelastic_contribution(None)
        
        
    def update_parameters(self, material):
        """
        Push a new (possibly partial) set of physical parameters - shaped like
        the json card used to build this material - into the existing matrix
        and inclusion objects, in place. No UFL form or fem.Expression is
        rebuilt, so this never triggers an FFCx JIT recompilation.
        """
        if "matrix" in material:
            self.matrix.set_parameters(material["matrix"])
        for name_incl in self.inclusions_keys:
            if name_incl in material:
                self.inclusions[name_incl].set_parameters(material[name_incl])
        # NOTE: deliberately NOT refreshing mu_0/k_0 here even if
        # material["matrix"] changed young/poisson above - they (and the
        # Eshelby tensors built from them) are numeric snapshots fixed at
        # construction time, see the module docstring for why. Calibrating
        # the matrix's own young modulus/poisson ratio requires rebuilding
        # this subdomain via setup_simulation() instead of update_parameters().
        self.matrix.f.refresh()

    def homogenization_scheme(self, l_el, l_elj, delta_t):
        """
        Create the constitutive law arising from mulsticale homogenization
        l_el : spatial velocity gradient in the LRS 
        l_elj : spatial velocity grad for increment of disp
        delta_t : increment of time
        Implementation of the High Dilution scheme to form the expression for the macroscopic objective stress rate
        """
        
        inv_A_esh = ufl.as_matrix(np.eye(6)) # A_inf = A_i
        
        self.matrix.localization_tensors(inv_A_esh)
        
        self.matrix.microscopic_mech(l_el, l_elj, delta_t) 
        
        self.dtau_obj = self.matrix.f.func[0]*self.matrix.dtau_dirder
        self.dtau_obj_incr = self.matrix.f.func[0]*self.matrix.dtau_incr
    
        for incl in self.inclusions.values():
            incl.localization_tensors(inv_A_esh) # A_inf = A_i
            
            incl.microscopic_mech(l_el, l_elj, delta_t) 
            # directional derivative
            self.dtau_obj += incl.f.func[0]*(incl.dtau_dirder)
            
            # incremental
            self.dtau_obj_incr += incl.f.func[0]*incl.dtau_incr
            
            
###########################################################################
### Mori-Tanaka Homogenization with Segura2023 symmetrization Scheme
###########################################################################        
        

class Homogenized_MT_material:
    """
    Class defining the material law of a homogenized material using Mori-Tanaka homogenization 
    This is the symmetrized implementation of the Mori-Tanaka homogenization; estimate an average strain in the matrix that surrounds every inclusions
    Symmetrized Mori-Tanaka scheme:
        - Top Bottom : strain localization thanks to the Mori-Tanaka strain localization. See Segura2023, Morin2018
        - Bottom Up : weighted average stress rates in the inclusions
    
    Parameters:
    -----------
        cells : restrict the material to a set of cells (used for multiple materials, see mech_problem_class subdomains)
        tag : int identifier of the subdomain
        material : dictionnary that contains the geometry, the matrix and the inclusions. inclusions_keys lists the subdict in material that define each inclusion.
        
    Attributes:
    -----------
        matrix : matrix object (inclusions_class) with a volumic fraction of 1 and a constant stiffness matrix.
        inclusions : dict that contains all the inclusions objects. 
        dtau_obj, dtau_obj_incr : objective macroscopic stress rate / directional derivative alongside direction du and duj respectively
        
    Methods:
    -----------
        homogenization_scheme : l_el, l_elj, delta_t = velocity gradient, incremental velocity gradient, increment of time (for inelastic contribution)
            create the homogenization law. Symmetrized Mori-Tanaka schemes.
    """
    def __init__(self, material):
        """
        Initialize material with a list of constituents, associated volumic fraction and mechanical behaviour, initial orientation
        material is a dictionnary containing at least 2 dictionnaries : geometry, matrix and at least one inclusion
        """
        ## From geometry dict : associate facets, dofs for the stiffness tensor
        self.cells = material["geometry"]["cells"]
        self.V_stiff = material["geometry"]["stiff spacefunction"] # used to form the expressions at the correct interpolation points
        self.tag = material["geometry"]["tag"]
        
        ### Define matrix
        self.matrix = Matrix(material["matrix"], material["geometry"])
        
        ### Define list of inclusions : list of object
        self.inclusions = {}
        self.inclusions_keys = list(material.keys()) # names of inclusions to call in the material dict
        self.inclusions_keys.remove("geometry")
        self.inclusions_keys.remove("matrix")
        
        # Total volumic fraction of inclusions :
        f_incl_form = 0.0
        
        mu0_ref, k0_ref = _numeric_reference_moduli(self.matrix)
        for name_incl in self.inclusions_keys:
            # reference values of the matrix for stiffnesses - plain numeric
            # floats, see _numeric_reference_moduli and the module docstring
            material[name_incl]["mu_0"] = mu0_ref
            material[name_incl]["k_0"] = k0_ref

            if material[name_incl]["type"] == "sphere":
                # spheroid inclusion
                # create material using the spherical inclusion dictionnary
                self.inclusions[name_incl] = Spherical_inclusion(material[name_incl], material["geometry"]) 
                self.inclusions[name_incl].inelastic_contribution(None)         
            
                f_incl_form += self.inclusions[name_incl].f.func    
                
            elif material[name_incl]["type"] == "cylinder" or material[name_incl]["type"] == "prolate_spheroid":
                # cylindrical inclusion
                # create material using the cylinder inclusion dictionnary
                self.inclusions[name_incl] = Spheroidal_inclusion(material[name_incl], material["geometry"])   
                self.inclusions[name_incl].inelastic_contribution(None, None)         
            
                f_incl_form += self.inclusions[name_incl].f.func    
            else:
                type_=material[name_incl]["type"]
                print(f"non valid type : {type_} component with name {name_incl} ignored", flush=True)
            
            print(f"Inclusion {name_incl} initialized\n", flush=True)
        
        # Volumic fraction is defined from volumic fraction of inclusions
        self.matrix.f.create_field(f_incl_form)
        
        if any(x < 0 for x in self.matrix.f.func.x.array[:]):
            print("Matrix has at least a null or negative volumic fraction, code terminated", flush=True)
            sys.exit()
        else:
            print("Matrix Volumic fraction Correctly defined", flush=True)
        
        self.matrix.inelastic_contribution(None)
        
        
    def update_parameters(self, material):
        """
        Push a new (possibly partial) set of physical parameters - shaped like the
        json card used to build this material - into the existing matrix and
        inclusion objects, in place. No UFL form or fem.Expression is rebuilt
        anywhere in this call, so it never triggers an FFCx JIT recompilation ;
        it is meant to be called every iteration of a calibration / sensitivity
        loop, right before re-solving the mechanical problem.

        material : dict shaped like the "adventitia"/"media" card, e.g. only the
            entries you want to change need to be present :
                {"matrix": {"young": 0.06},
                 "collagen_0": {"young": [[0.4],[2.0],[1.1]], "theta": -78.3}}
        """
        if "matrix" in material:
            self.matrix.set_parameters(material["matrix"])

        for name_incl in self.inclusions_keys:
            if name_incl in material:
                self.inclusions[name_incl].set_parameters(material[name_incl])

        # NOTE: deliberately NOT refreshing mu_0/k_0 here even if
        # material["matrix"] changed young/poisson above - they (and the
        # Eshelby tensors built from them) are numeric snapshots fixed at
        # construction time, see the module docstring for why. Calibrating
        # the matrix's own young modulus/poisson ratio requires rebuilding
        # this subdomain via setup_simulation() instead of update_parameters().

        # inclusion volumic fractions may have changed above -> refresh the
        # matrix's complementary volumic fraction field (cheap re-interpolation
        # of the expression already built in create_field(), no rebuild)
        self.matrix.f.refresh()

    def homogenization_scheme(self, l_el, l_elj, delta_t):
        """
        Create the constitutive law arising from mulsticale homogenization
        l_el : spatial velocity gradient in the LRS 
        l_elj : spatial velocity grad for increment of disp
        delta_t : increment of time
        Implementation of the Mori-Tanaka scheme to form the expression for the macroscopic objective stress rate
        """
        #---------------------------------------------------------------------#
        # RVE-to-remote strain conversion tensor M 
        Minv = self.matrix.f.func[0]*ufl.as_matrix(np.eye(6))
        
        self.H1 = ufl.as_matrix(np.zeros((6,6)))
        self.H2 = ufl.as_matrix(np.zeros((6,6)))
        
        for incl in self.inclusions.values():
            if type(incl) is Spherical_inclusion:
                self.H1 += incl.f.func[0]*incl.A_inf
                self.H2 += incl.f.func[0]*ufl.dot(incl.C-self.matrix.C, incl.A_inf)
            else:
                self.H1 += incl.f.func[0]*incl.A_inf.func
                self.H2 += incl.f.func[0]*ufl.dot(incl.C-self.matrix.C, incl.A_inf.func)
        
        self.H2inv = inverse_matrix6_6(self.H2, self.V_stiff, self.cells) # H2inv is the inverse of the H2 matrix
        
        Minv += 1/2*self.H1 + 1/2*ufl.dot(ufl.dot(self.H2inv.func, ufl.transpose(self.H1)), self.H2)
        
        self.M = inverse_matrix6_6(Minv, self.V_stiff, self.cells) # M is the inverse of Minv, accessible as M.func
                
        print("M tensor initialized\n", flush=True)
        #---------------------------------------------------------------------#
        # Localization
        A_m = ufl.as_matrix(np.eye(6))
        for incl in self.inclusions.values():
            incl.localization_tensors(self.M.func)  # this step creates incl.A_i = A_inf * M the localization tensor
            A_m -= incl.f.func[0]*incl.A_i
        
        
        self.matrix.localization_tensors(A_m/self.matrix.f.func[0])
        
        self.matrix.microscopic_mech(l_el, l_elj, delta_t) 
        
        #---------------------------------------------------------------------#
        # Now compute elastic increments 
        # Homogenization and stress increment at micro level
        # compute stress and strain rates in the matrix and inclusions
        self.dtau_obj = self.matrix.f.func[0]*self.matrix.dtau_dirder
        self.dtau_obj_incr = self.matrix.f.func[0]*self.matrix.dtau_incr
        
        for incl in self.inclusions.values():
            
            incl.microscopic_mech(l_el, l_elj, delta_t) 
            
            # directional derivative
            self.dtau_obj += incl.f.func[0]*(incl.dtau_dirder)
            
            # incremental
            self.dtau_obj_incr += incl.f.func[0]*incl.dtau_incr
    
        print("Mori-Tanaka Homogenization finished\n", flush=True)
        

###########################################################################
### Active Mori-Tanaka Homogenization Unsymmetrized
###########################################################################        
        

class Active_MT_material:
    """
    Class defining the material law of a homogenized material using unsymmetrized Mori-Tanaka homogenization with inelastic strains in the inclusions
    See HDR Morin 2023 -> define the strain localization tensors using inelastic strains.
    Active Mori-Tanaka scheme:
        - Top Bottom : strain localization thanks to the Mori-Tanaka strain localization. Adding the impact of the inelastic strains
        - Bottom Up : weighted average stress rates in the inclusions
    
    Parameters:
    -----------
        cells : restrict the material to a set of cells (used for multiple materials, see mech_problem_class subdomains)
        tag : int identifier of the subdomain
        material : dictionnary that contains the geometry, the matrix and the inclusions. inclusions_keys lists the subdict in material that define each inclusion.
        
    Attributes:
    -----------
        matrix : matrix object (inclusions_class) with a volumic fraction of 1 and a constant stiffness matrix.
        inclusions : dict that contains all the inclusions objects. 
        dtau_obj, dtau_obj_incr : objective macroscopic stress rate / directional derivative alongside direction du and duj respectively
        
    Methods:
    -----------
        homogenization_scheme : l_el, l_elj, delta_t = velocity gradient, incremental velocity gradient, increment of time (for inelastic contribution)
            create the homogenization law. Active Mori-Tanaka schemes.
    """
    
    def __init__(self, material):
        """
        Initialize material with a list of constituents, associated volumic fraction and mechanical behaviour, initial orientation
        material is a dictionnary containing at least 2 dictionnaries : geometry, matrix and at least one inclusion
        """
        ## From geometry dict : associate facets, dofs for the stiffness tensor
        self.cells = material["geometry"]["cells"]
        self.V_stiff = material["geometry"]["stiff spacefunction"] # used to form the expressions at the correct interpolation points
        self.tag = material["geometry"]["tag"]
        
        ### Define matrix
        self.matrix = Matrix(material["matrix"], material["geometry"])
        
        ### Define list of inclusions : list of object
        self.inclusions = {}
        self.inclusions_keys = list(material.keys()) # names of inclusions, not really useful
        self.inclusions_keys.remove("geometry")
        self.inclusions_keys.remove("matrix")
        self.inclusions_keys = sorted(self.inclusions_keys, key=lambda k: (0 if k == "cells" else 1, k)) # manage cell first
        
        # Total volumic fraction of inclusions :
        f_incl_form = 0.0
        
        mu0_ref, k0_ref = _numeric_reference_moduli(self.matrix)
        for name_incl in self.inclusions_keys:
            # reference values of the matrix for stiffnesses - plain numeric
            # floats, see _numeric_reference_moduli and the module docstring
            material[name_incl]["mu_0"] = mu0_ref
            material[name_incl]["k_0"] = k0_ref

            if material[name_incl]["type"] == "sphere":
                # Passive sphere inclusion
                # create material using the spherical inclusion dictionnary
                self.inclusions[name_incl] = Spherical_inclusion(material[name_incl], material["geometry"]) 
                
                f_incl_form += self.inclusions[name_incl].f.func   
                
            elif material[name_incl]["type"] == "cylinder" or material[name_incl]["type"] == "prolate_spheroid":
                # Passive cylindrical inclusion
                self.inclusions[name_incl] = Spheroidal_inclusion(material[name_incl], material["geometry"])   
                f_incl_form += self.inclusions[name_incl].f.func   
                
            elif material[name_incl]["type"] == "growing sphere" or material[name_incl]["type"] == "homeostatic sphere":
                # Active sphere with homeostatic stress regulation
                self.inclusions[name_incl] = Active_Spherical_inclusion(material[name_incl], material["geometry"]) 
                f_incl_form += self.inclusions[name_incl].f.func   
                
            elif material[name_incl]["type"] == "homeostatic spheroid":
                # Active sphere with homeostatic stress regulation
                self.inclusions[name_incl] = Active_Spheroidal_inclusion(material[name_incl], material["geometry"]) 
                f_incl_form += self.inclusions[name_incl].f.func   
                cell_inclusion = name_incl
            
            elif material[name_incl]["type"] == 'prestretched cylinder':
                # Cylinder with inelastic stretch coupled to cell active stretch
                self.inclusions[name_incl] = Prestretched_Cylinder_inclusion(material[name_incl], material["geometry"],  self.inclusions[cell_inclusion]) 
                f_incl_form += self.inclusions[name_incl].f.func   
            
            else:
                type_=material[name_incl]["type"]
                print(f"non valid type : {type_} component with name {name_incl} ignored", flush=True)
            
            print(f"Inclusion {name_incl} initialized\n", flush=True)
        
        # Volumic fraction is defined from volumic fraction of inclusions
        self.matrix.f.create_field(f_incl_form)
        
        if any(x < 0 for x in self.matrix.f.func.x.array[:]):
            print("Matrix has at least a null or negative volumic fraction, code terminated", flush=True)
            sys.exit()
        else:
            print("Matrix Volumic fraction Correctly defined", flush=True)
        
        self.matrix.inelastic_contribution(None)
        
        
    def update_parameters(self, material):
        """
        Push a new (possibly partial) set of physical parameters - shaped like
        the json card used to build this material - into the existing matrix
        and inclusion objects, in place. No UFL form or fem.Expression is
        rebuilt, so this never triggers an FFCx JIT recompilation.
        """
        if "matrix" in material:
            self.matrix.set_parameters(material["matrix"])
        for name_incl in self.inclusions_keys:
            if name_incl in material:
                self.inclusions[name_incl].set_parameters(material[name_incl])
        # NOTE: deliberately NOT refreshing mu_0/k_0 here even if
        # material["matrix"] changed young/poisson above - they (and the
        # Eshelby tensors built from them) are numeric snapshots fixed at
        # construction time, see the module docstring for why. Calibrating
        # the matrix's own young modulus/poisson ratio requires rebuilding
        # this subdomain via setup_simulation() instead of update_parameters().
        self.matrix.f.refresh()

    def homogenization_scheme(self, l_el, l_elj, delta_t):
        """
        Unsymmetrized Mori Tanaka implementation with active behavior. See Ch V and APP A.
        Based on the material point model
        Bottom Up part -> Homogenization
        delta_t : increment of time : only inelastic strain is affected by the increment of time 
        """

        #---------------------------------------------------------------------#
        # RVE-to-remote strain conversion tensor M 
        Minv = self.matrix.f.func[0]*ufl.as_matrix(np.eye(6))
        
        for incl in self.inclusions.values():
            if type(incl) is Spherical_inclusion or type(incl) is Active_Spherical_inclusion:
                Minv += incl.f.func[0]*incl.A_inf
            else:
                Minv += incl.f.func[0]*incl.A_inf.func
            
            
        self.M = inverse_matrix6_6(Minv, self.V_stiff, self.cells) # M is the inverse of Minv, accessible as M.func
        
        print("M tensor initialized\n", flush=True)   
        
        #---------------------------------------------------------------------#
        # Localization
        A_m = ufl.as_matrix(np.eye(6))
        for incl in self.inclusions.values():
            incl.localization_tensors(self.M.func)  # this step creates incl.A_i = A_inf * M the localization tensor
            A_m -= incl.f.func[0]*incl.A_i
        
        self.matrix.localization_tensors(A_m/self.matrix.f.func[0])
        
        #---------------------------------------------------------------------#
        # Inelastic contribution
        # update inelastic contribution here
        # two nested loop for inclusions
        for incl_key_i in self.inclusions.keys():
            delta_i = ufl.as_tensor(np.zeros((3,3)))
            omega_i = ufl.as_tensor(np.zeros((3,3)))
            incl_i = self.inclusions[incl_key_i]
            if incl_i.inel:
                # First Manage the inelastic strain from self                
                Dii = ufl.dot(incl_i.I_m - incl_i.f.func[0]*incl_i.A_i, incl_i.D_inf) #A_inf_temp), incl_i.P), incl_i.C) #(I-f_i*A_i):A_i_inf:P_i:C_i
                
                # BEWARE OF MANDEL NOTATION / TENSOR NOTATION
                delta_i += tensordot_4_2(expand(Dii), incl_i.d_inel)
                
                if type(incl_i) is Active_Spheroidal_inclusion or type(incl_i) is Prestretched_Cylinder_inclusion:
                    Tii = tensordot_4_4(incl_i.T_inf - incl_i.f.func[0]*tensordot_4_4(incl_i.R_inf,expand(self.M.func)), expand(incl_i.D_inf))
                    omega_i += tensordot_4_2(Tii, incl_i.d_inel)
            # strain rate coming from the equilibrium of the inelastic strain rates in the different inclusions. Initialized with self contribution
            
            # Then include crossed interactions. 
            for incl_key_j in self.inclusions.keys():
                if incl_key_j!=incl_key_i and self.inclusions[incl_key_j].inel:
                    incl_j = self.inclusions[incl_key_j]
                        
                    Dij = -incl_j.f.func[0]*ufl.dot(incl_i.A_i, incl_j.D_inf) # compute Dij, with i!=j
                    # BEWARE OF MANDEL NOTATION / TENSOR NOTATION
                    delta_i += tensordot_4_2(expand(Dij), incl_j.d_inel)
                    
                    if type(incl_i) is Spheroidal_inclusion or type(incl_i) is Active_Spheroidal_inclusion or type(incl_i) is Prestretched_Cylinder_inclusion: # R_inf != 0 -> manage rotations
                        Tij = - incl_j.f.func[0]*tensordot_4_4(incl_i.R_inf, expand(ufl.dot(self.M.func, incl_j.D_inf)))
                        omega_i += tensordot_4_2(Tij, incl_j.d_inel)
                        

            # !!!! Dim*delta_m = 0. For now, NO free strain in matrix. SHOULD ADD MATRIX CONTIRUBTION HERE
            if type(incl_i) is Spheroidal_inclusion or type(incl_i) is Active_Spheroidal_inclusion or type(incl_i) is Prestretched_Cylinder_inclusion:
                incl_i.inelastic_contribution(delta_i, omega_i)
            else:
                incl_i.inelastic_contribution(delta_i)
            
        # Matrix inelastic strain contribution of other components
        delta_m = ufl.as_tensor(np.zeros((3,3)))
        for incl_key_i in self.inclusions.keys():
            if self.inclusions[incl_key_i].inel:
                incl_i = self.inclusions[incl_key_i] 
                
                Dmi = -incl_i.f.func[0]*ufl.dot(self.matrix.A_m, incl_i.D_inf) # compute Dmi
                delta_m = tensordot_4_2(expand(Dmi), incl_i.d_inel)

        # !!!! Dim*delta_m = 0. For now, NO free strain in matrix. SHOULD ADD MATRIX CONTIRUBTION HERE
        self.matrix.inelastic_contribution(delta_m)
        
        #---------------------------------------------------------------------#
        # Now compute elastic and inelastic increments 
        # Homogenization and stress increment at micro level
        # compute stress and strain rates in the matrix and inclusions
        
        self.matrix.microscopic_mech(l_el, l_elj, delta_t) 
        
        self.dtau_obj = self.matrix.f.func[0]*self.matrix.dtau_dirder
        self.dtau_obj_incr = self.matrix.f.func[0]*self.matrix.dtau_incr
        
        for incl in self.inclusions.values():
            
            incl.microscopic_mech(l_el, l_elj, delta_t) 
            
            # directional derivative
            self.dtau_obj += incl.f.func[0]*incl.dtau_dirder
            
            # incremental
            self.dtau_obj_incr += incl.f.func[0]*incl.dtau_incr
    
        print("Active Mori-Tanaka Homogenization finished\n", flush=True)
        
