This is the folder for the final code of the thesis :

2D Axisymmetrical cylinder with rectangle mesh under internal pressure with axial tension -> main_cylinder_25_05_02.py
    -> based on a axisymmetrical formulation of the mechanical equilibrium : class Mechanical_Problem_axi in the class_modules.mech_problem_class module
3D cube with hexahedron following the Eichinger boundary conditions -> main_eichinger_25_05_27.py
    -> based on a classical 3D formulation of the mechanical equilibrium : class Mechanical_Problem_3D ; can be generalized to more complex 3D mesh as was done in hypoelastic codes

Based on the latest development for solving hypoelastic problem using a Mori-Tanaka homogenization with an active behavior:
- Jaumann derivative (no shear stress)
- Forward Euler with subiteration integration
- Active behavior : inelastic strain rate -> ActiveMT class from class_modules.material_class module . Beware it does not consider matrix inelastic strain. And it is not symmetrized.
- Other configuration can be chosen to consider passive multiscale behavior: isotropic material, high dilution, Mori-Tanaka symmetrized or not symmetrized.

Useful for : 
- passive calibration of the multiscale model
- active regulation of cells

Developed 2nd May 2025

HOW TO USE : 
using fenicsx and dolfinx 0.9.0.0

in the main file run the function run_simulation(name, folder_name, simu_card, layer_card)
    - name is the name of the output files
    - folder_name is the name of the output folder (inside folder output)
    - simu_card gathers the global configuration of the simulation like geometry, value for boundary conditions, time discretization...
    - layer_card gathers the properties of the heterogenous material with the inclusions we want to consider in the homogenization.
    
Examples of simu_card.json and layer_card.json are defined within the folder json_card
The simulation can be run as such by starting the main_.py file, providing : name, foldername, simu_card, layer_card -> strings that will be parsed into importing the corresponding json cards
    In a terminal, run
        python main_cylinder_25_05_02.py test folder_test simu_card layer_card

The simulation can be run from an outside python code, like for sensitivity analysis by importing the module.
    In an external python code, run : 
        from main_cylinder_25_05_02.py import run_simulation
        run_simulation(name, folder_name, simu_card, layer_card)
        Here simu_card and layer_card are loaded json files
        
HOW TO TWEAK :
Now the hard part is to define what we want to get from the code. The basis structure of the code is to consider several layer conceptual objects:
The class_modules.mech_problem_class defines the objects used for defining the Finite Element Framework that solves the mechanical equilibrium.
It depends on the constitutive law for the material, which is defined in the modules class_modules.material_class. This is where the Mori-Tanaka homogenization is defined and this is the bridge between inclusion and homogenized behavior.
This material class depends on the description of the inclusions and its matrix, which are defined in class_modules.inclusion_class. This is where the matrix and all the inclusions are defined, like the sphere, spheroids, active sphere with active behavior, active spheroids.
Finally these inclusions depends on some parameters, like the volumic fraction, the Young Modulus, etc. In the case on complex parameters, like for a stiffening fibers, the parameters are defined in the module class_modules.parameter_class


When we want to exploit the code, what we want is to be able to change the input/output of the model, and maybe change some constitutive laws to see how it affects the overall behavior. 
In the first case ; changing the input/output :
    Everything needs to be done in the main file. The main files, as provided, have a structure that interacts only with the mech_problem object. If needs be to change the main, the process of interactions should be kept the same. 
    However one might want to change the boundary conditions -> change the boundary_conditions list in accordance to the specific syntax :
        ["Dirichlet", surface number, applied condition : (displacement or "clamped" , direction)]
        ["Neumann or Neumann_follower", surface number, applied condition : vector of applied pressure]
      This is just an adaptation of the mixed boundary condition approach proposed by Dokkens in its fenicsx tutorial  
    
    Changing the output quantities, the quantities of interest, needs to be done within the main file. 
    O/I interface should be improved with respect to that to have a cleaner approach to storing outputs.
    There are two types of outputs : 
        fields that are stored within the xdmf file. An export fem.Function needs to be created, interpolated at each steps and exported within the file. See what was done for stress tensor
        
        scalar values that are stored within a Result object : using class_modules.result_class import Results
            for these values, a list of output needs to be provided describing all the output quantities : for example list_outputs = ['time','ux', 'uy', 'uz', 'fxx', 'fyy', 'Fel_cell', 'Finel_cell', 'la']
            for each of these values a specific fenicsx form can be computed on the whole domain or just on specific points. Then at each steps, the results need to be stored.
            an example of defining a scalar result would be the average displacement ux on the surface 2.
                this quantity is defined by the following forms : 
                    ux_form = fem.form(ufl.dot(mech.un, ufl.as_vector([1,0,0]))*mech.ds(2))  
                    surf = fem.assemble_scalar(fem.form(1.0*mech.ds(2)))
                Then the result object is created using the list of outputs names :
                    result = Results(name, folder_name, list_outputs, n_NR)
                Then at every steps, the average displacement on the surface 2 is computed and stored like this : 
                    result.outputs['ux'][n]=fem.assemble_scalar(ux_form)/surf
    
    One might want to have some information on the inclusions behavior. The field of interest needs to be called by :
        mech.subdomain['layer_name'].inclusions['inclusion_name'].quantity.
        
        for example when considering cells that contracts we want to have the inelastic field associated with those cells. It is callable by :
            mech.subdomain['layer'].inclusions['cellX'].F_inel
        then this quantity can be either interpolated onto a fem.Function for an xdmf export OR it can be manipulated to extract some values. For example if we want to have the average value of Jacobian of the inelastic strain, we build the following integral on the mech.dx volume:
            f_inel_form = fem.form(1/3*ufl.tr(mech.subdomain['layer'].inclusions['cellX'].F_inel)*mech.dx) 
        
        The latter form is assembled and stored in the result object like : 
            result.outputs['Finel_cell'][n] =fem.assemble_scalar(f_inel_form)/volume
        volume, because it characterizes the initial volume in the reference configuration, it does not change, and was defined as 
            volume = fem.assemble_scalar(fem.form(1.0*mech.dx)) same for surface just above
            
Changing the parameters classes : 
    this is a bit more tricky, but mainly just follow the structure of the parameter class already developed :
        __init__ which creates all the needed quantities and objects specific for the parameter
        update_func is a requirements, it updates the value of the FE field in a given configuration
        it might requires some external informations regarding the expression of the parameter, which would be initialized through init_func(form)
        
        the parameter is an object, and its field is called in the fenicsx calculus using parameter.func
        
        For exemple, when considering an evolving Young Modulus in a cell inclusion. 
            In the inclusion class is an object self.E which is the young modulus, initialized by incl.E = twovalues_param(value, geom) (geom being a dictionary containing some FE informations, value is a two component list)
            The object E is an attributes of the inclusion object. It itself is a twovalues_param object. It contain a func field, which is used by the inclusion as the scalar field of the Young Modulus.
            The expression for making the field vary is created by the inclusion class by calling the method : incl.E.init_func(scalar_form) with scalar form the function that drives the variation of the yound modulus.
            Then for each steps, the inclusion is updated by the mechanical problem using the inclusion method incl.update_micro_mech. In this method, the young modulus object is updated through incl.E.update_func()
            
            overall three steps : 
                - initialization of the parameter object.
                - initialization of the expression for controlling the behavior of the parameter
                - updating at each steps in the inclusion method update_micro_mech. 
                
