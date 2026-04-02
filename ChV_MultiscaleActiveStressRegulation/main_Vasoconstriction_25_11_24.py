#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  27 2025

@author: bastien.sauty

Main file for multiscale model implemented for an axisymmetrical cylinder under
internal pressure and axial tension

Adding the Vasoconstriction effects - Stress regulation by the cells
And Inelastic prestretch in the collagen fibers. 
successive load steps managed by outside control -> load_class

This main file particular role is to define correctly the outputs. Here the
quantities of interest are through-the-thickness quantities, to assess the 
impact of cylindrical geometry on Vcn.
"""

import json, sys, time, os
import numpy as np
import matplotlib.pyplot as plt

# Fenicsx modules in the main
from dolfinx import mesh, fem, io
from mpi4py import MPI
import ufl

# Homemade Libraries
from Multiscale_Framework.class_modules.mech_problem_class import Mechanical_Problem_axi
from Multiscale_Framework.class_modules.result_class import Results
from Multiscale_Framework.class_modules.load_class import Artery_load

from Multiscale_Framework.function_modules.auxiliary_functions import Tensor2Voigt, Voigt2Tensor


def run_simulation(name, folder_name, simu_card, adventitia_card, media_card, dry_run=False):
    """
    Run the tension-inflation test on axisymmetric cylinder
    Vasoconstriction - active stress regulation
    Collagen pre-stretch

    Parameters
    ----------
    simu_card : json card containing the general simulation parameters
            - Geometry 
            - Pressure, axial tension, NR_steps
            - Name, integration_scheme, objective_derivative, n_int
    adventitia_card : json card containing the material properties for the adventitia layer
    media_card : json card containing the material properties for the media layer

    Returns
    -------
    Result object.

    """
    t_simu0 = time.time()
    ### Unpacking simulation parameters
    # General 
    objective_derivative = simu_card["objective_derivative"]
    n_int = simu_card["n_int"]
    load_phase = simu_card['load_phase']

    # Loads
    step_load = Artery_load(load_phase)
    n_steps = len(step_load.list_dt)

    # geom
    ri = simu_card['ri']
    re = simu_card['re']    
    lz = simu_card['lz']
    ri_adv = simu_card['ri_adv']

    nr = simu_card['nr']
    nz = simu_card['nz']   
    
    # Correct the inner radius of the adv from the number of elements in the media
    nmed = int(nr*(ri_adv-ri)/(re-ri))
    interval = (re-ri)/nr
    ri_adv = ri+nmed*interval
    print(f"Interval radius of the adventitia corrected to {ri_adv}")
    
    #-------------------------------------------------------------------------#
    ### Initiate Mech object
    mech = Mechanical_Problem_axi(name, objective_derivative, n_int)
    
    #-------------------------------------------------------------------------#
    ### Geometry : create Mesh
    domain = mesh.create_rectangle(MPI.COMM_WORLD,[[ri, 0], [re, lz]], [nr, nz], mesh.CellType.quadrilateral)

    # Initialize dolfinx space functions
    mech.build_space_functions(domain)
    
    #-------------------------------------------------------------------------#
    ### Material properties
    # Find cells where material is applied
    def Omega_adv(x):
        return (x[0] >= ri_adv) | (np.isclose(x[0], ri_adv))
    
    # array of indices of the corresponding cells
    tag_adv = 1 # associate the tag 1 to the layer
    cells_adv = mesh.locate_entities(mech.domain, mech.domain.topology.dim, Omega_adv) 
    
    # Fill layer card with fenicsx functions
    adventitia_card["geometry"] = {"type": "geometry",
                               "cells": cells_adv,
                               "tag": tag_adv,
                               "domain": mech.domain,
                               "stiff spacefunction": mech.V_stiff,
                               "mandel spacefunction": mech.V_mandel,
                               "scalar spacefunction": mech.V_scalar,
                               "vector spacefunction": mech.V_vec,
                               "matrix spacefunction": mech.V_mat,
                               "objective derivative": mech.objective_derivative} # str, list, FunctionSpace
    
    # Manage subdomain
    mech.add_subdomain("adventitia", adventitia_card, 'MT')
    
    # Find cells where material is applied
    def Omega_media(x):
        return (x[0] <= ri_adv) | (np.isclose(x[0], ri_adv))
    
    # array of indices of the corresponding cells
    tag_media = 2 # associate the tag 1 to the layer
    cells_media = mesh.locate_entities(mech.domain, mech.domain.topology.dim, Omega_media) 
    
    # Fill layer card with fenicsx functions
    media_card["geometry"] = {"type": "geometry",
                               "cells": cells_media,
                               "tag": tag_media,
                               "domain": mech.domain,
                               "stiff spacefunction": mech.V_stiff,
                               "mandel spacefunction": mech.V_mandel,
                               "scalar spacefunction": mech.V_scalar,
                               "vector spacefunction": mech.V_vec,
                               "matrix spacefunction": mech.V_mat,
                               "objective derivative": mech.objective_derivative} # str, list, FunctionSpace
    
    # Manage subdomain
    mech.add_subdomain("media", media_card, 'ActiveMT')
    
    # Build the meshtags object to manage different materials
    mech.build_meshtags()

    #-------------------------------------------------------------------------#
    ### Building Weak form
    # Linearized Residuals for Custom Newton-Raphson solver
    mech.build_weak_form()
    # Important void run with null disp to initiate stiffness matrix and various fields (like young modulus)
    mech.update_local_quantities()
    
    #-------------------------------------------------------------------------#
    ### Define the boundaries and associated boundary conditions
    
    # build boundary facet_tags
    boundaries = [(1, lambda x: np.isclose(x[0], ri)),
                  (2, lambda x: np.isclose(x[0], re)),
                  (3, lambda x: np.isclose(x[1], 0)),
                  (4, lambda x: np.isclose(x[1], lz))]
    
    T_press = fem.Constant(mech.domain,np.array([0,0], dtype=np.float64)) 

    disp_z = fem.Constant(mech.domain,0.) # tensile

    boundary_conditions = [["Dirichlet", 3, ("clamped", 1)],
                           ["Dirichlet", 4, (disp_z, 1)],
                           ["Neumann_follower", 1, T_press]]

    
    mech.build_BCs(boundaries, boundary_conditions)
    
    #-------------------------------------------------------------------------#
    ### Build Solver
    mech.build_solver()

    #-------------------------------------------------------------------------#
    ### Define Quantity of interest
    # volumes for averaging quantities
    volume_media = fem.assemble_scalar(fem.form(mech.r*mech.dx(tag_media)))
    volume_adv = fem.assemble_scalar(fem.form(mech.r*mech.dx(tag_adv)))
    
    # Displacement field : linear piecewise to get values at border
    V_u_exp = fem.functionspace(domain, ("P", 1, (1, )))
    ur = fem.Function(V_u_exp)
    ur_expr = fem.Expression(ufl.dot(ufl.as_vector([1,0]),mech.un), V_u_exp.element.interpolation_points())
    
    # Average quantities in the whole layers : circ and axial stresses:
    tau_tissue = ufl.dot(mech.Fn, ufl.dot(Voigt2Tensor(mech.Sn), mech.Fn.T))
    S_yy_form = ufl.dot(ufl.as_vector([0,1,0]), ufl.dot(tau_tissue, ufl.as_vector([0,1,0]))) 
    S_zz_form = ufl.dot(ufl.as_vector([0,0,1]), ufl.dot(tau_tissue, ufl.as_vector([0,0,1])))
    
    S_yy_avg_media_expr = fem.form(S_yy_form*mech.r*mech.dx(tag_media)) # average stresses in the media and adventitia, used with assemble_scalar
    S_yy_avg_adv_expr = fem.form(S_yy_form*mech.r*mech.dx(tag_adv))
    S_zz_avg_media_expr = fem.form(S_zz_form*mech.r*mech.dx(tag_media))
    S_zz_avg_adv_expr = fem.form(S_zz_form*mech.r*mech.dx(tag_adv))
    
    
    # Extract values along one radial line -> position z= middle lower element 
    # get radial position at the centroids of elements
    r_pos = fem.Function(mech.V_scalar)
    r_pos_expr = fem.Expression(mech.x[0], mech.V_scalar.element.interpolation_points()) # 
    r_pos.interpolate(r_pos_expr)
    
    bottom_cells_media = mesh.locate_entities(mech.domain, mech.domain.topology.dim, lambda x: (x[1] <= lz/nz) & (x[0] <= ri_adv))
    r_pos_bottom_media = r_pos.x.array[bottom_cells_media]
    
    # Local quantities :
    # Circumferential stress
    S_yy = fem.Function(mech.V_scalar) # only used for storing the value along the bottom line
    S_yy_expr = fem.Expression(S_yy_form, mech.V_scalar.element.interpolation_points())
    S_yy.interpolate(S_yy_expr)
    S_zz = fem.Function(mech.V_scalar) # only used for storing the value along the bottom line
    S_zz_expr = fem.Expression(S_zz_form, mech.V_scalar.element.interpolation_points())
    S_zz.interpolate(S_zz_expr)
    
    
    # Get stress tensors  in the matrix, cell and collagen in media
    keys = mech.subdomain['media'].inclusions.keys()
    collagen_media_keys = [k for k in keys if k.startswith("collagen")]
    
    s_tensor_matrix= mech.subdomain['media'].matrix.taun
    s_tensor_cell = mech.subdomain['media'].inclusions['cells'].taun
    f_collagen_media = sum(mech.subdomain['media'].inclusions[key].f.func[0] for key in collagen_media_keys )
    s_tensor_collagen_media = sum(mech.subdomain['media'].inclusions[key].f.func[0] * mech.subdomain['media'].inclusions[key].taun for key in collagen_media_keys )/f_collagen_media
    
    # axial and circ stresses at micro scale
    s_yy_matrix = fem.Function(mech.V_scalar)
    s_yy_cell = fem.Function(mech.V_scalar)
    s_yy_collagen = fem.Function(mech.V_scalar)
    s_zz_matrix = fem.Function(mech.V_scalar)
    s_zz_cell = fem.Function(mech.V_scalar)
    s_zz_collagen = fem.Function(mech.V_scalar)
    
    s_yy_matrix_expr = fem.Expression(ufl.dot(s_tensor_matrix, ufl.as_vector([0,1,0,0,0,0])), mech.V_scalar.element.interpolation_points())
    s_yy_cell_expr = fem.Expression(ufl.dot(s_tensor_cell, ufl.as_vector([0,1,0,0,0,0])), mech.V_scalar.element.interpolation_points())
    s_yy_collagen_expr = fem.Expression(ufl.dot(s_tensor_collagen_media, ufl.as_vector([0,1,0,0,0,0])), mech.V_scalar.element.interpolation_points())
    
    s_zz_matrix_expr = fem.Expression(ufl.dot(s_tensor_matrix, ufl.as_vector([0,0,1,0,0,0])), mech.V_scalar.element.interpolation_points())
    s_zz_cell_expr = fem.Expression(ufl.dot(s_tensor_cell, ufl.as_vector([0,0,1,0,0,0])), mech.V_scalar.element.interpolation_points())
    s_zz_collagen_expr = fem.Expression(ufl.dot(s_tensor_collagen_media, ufl.as_vector([0,0,1,0,0,0])), mech.V_scalar.element.interpolation_points())
    
    s_yy_matrix.interpolate(s_yy_matrix_expr)
    s_yy_cell.interpolate(s_yy_cell_expr)
    s_yy_collagen.interpolate(s_yy_collagen_expr)
    
    s_zz_matrix.interpolate(s_zz_matrix_expr)
    s_zz_cell.interpolate(s_zz_cell_expr)
    s_zz_collagen.interpolate(s_zz_collagen_expr)
    
    # elastic and inelastic stretches in the cell
    incl_cell = mech.subdomain['media'].inclusions['cells']

    b = ufl.dot(incl_cell.Fn, incl_cell.Fn.T)
    b_in = ufl.dot(incl_cell.F_inel, incl_cell.F_inel.T)
    axial_stretch = ufl.sqrt(ufl.dot(incl_cell.e_r, ufl.dot(b, incl_cell.e_r))) # total stretch
    inelastic_stretch =  ufl.sqrt(ufl.dot(incl_cell.e_r, ufl.dot(b_in, incl_cell.e_r)))
    
    lambda_cell_expr = fem.Expression(axial_stretch, mech.V_scalar.element.interpolation_points())
    lambda_cell_in_expr = fem.Expression(inelastic_stretch, mech.V_scalar.element.interpolation_points())

    lambda_cell = fem.Function(mech.V_scalar)
    lambda_cell_in = fem.Function(mech.V_scalar)
    
    lambda_cell.interpolate(lambda_cell_expr)
    lambda_cell_in.interpolate(lambda_cell_in_expr)
        
    # aggregated measures of cell quantities
    # average stretch and stress
    cell_s_yy_mes = ufl.dot(s_tensor_cell, ufl.as_vector([0,1,0,0,0,0]))
    
    s_yy_cell_avg_expr = fem.form(cell_s_yy_mes*mech.r*mech.dx(tag_media))
    lambda_cell_in_avg_expr = fem.form(inelastic_stretch*mech.r*mech.dx(tag_media))

    
    V_scalar_lin = fem.functionspace(mech.domain, ("CG", 1, (1,)))
    
    cell_s_yy_mes_field = fem.Function(V_scalar_lin) # Need to interpolate cell_s_yy_mes onto this field
    cell_l_in_mes_field = fem.Function(V_scalar_lin) # Need to interpolate inelastic_stretch onto this field
    
    lambda_cell_in_expr_lin = fem.Expression(inelastic_stretch, V_scalar_lin.element.interpolation_points())
    s_yy_cell_expr_lin = fem.Expression(ufl.dot(s_tensor_cell, ufl.as_vector([0,1,0,0,0,0])), V_scalar_lin.element.interpolation_points())
    
    cell_s_yy_grad2D = ufl.grad(cell_s_yy_mes_field)
    cell_l_in_grad2D = ufl.grad(cell_l_in_mes_field)
    
    s_yy_cell_grad_avg_expr = fem.form(cell_s_yy_grad2D[0,0]*mech.r*mech.dx(tag_media))
    lambda_cell_in_grad_avg_expr = fem.form(cell_l_in_grad2D[0,0]*mech.r*mech.dx(tag_media))
    
    #-------------------------------------------------------------------------#
    ### Results Object
    list_outputs_avg = ['time','ri_d', 're_d', 'press', 'area', 
                        'S_yy_avg_adv', 'S_yy_avg_media', 'S_zz_avg_adv', 'S_zz_avg_media',
                        's_yy_cell_avg', 'lambda_cell_in_avg', 's_yy_cell_grad_avg', 'lambda_cell_in_grad_avg']
    
    dict_outputs = {}
    for key in list_outputs_avg:
        dict_outputs[key] = {'points':None}
    
    list_outputs_local = ['S_yy', 'S_zz',
                        's_yy_matrix','s_yy_cell', 's_yy_collagen',
                        's_zz_matrix','s_zz_cell', 's_zz_collagen',
                        'lambda_cell', 'lambda_cell_in']
    
    for key in list_outputs_local:
        dict_outputs[key] = {'points':r_pos_bottom_media}
    
    result = Results(name, folder_name, dict_outputs, n_steps)
    
    result.outputs['time'][0]=0
    result.outputs['ri_d'][0]=ri+np.max(ur.x.array[:])
    result.outputs['re_d'][0]=re+np.min(ur.x.array[:])
    result.outputs['press'][0]=T_press.value[0]
    result.outputs['area'][0] = np.pi*(result.outputs['re_d'][0]**2 - result.outputs['ri_d'][0]**2)
    
    result.outputs["S_yy_avg_adv"][0] = fem.assemble_scalar(S_yy_avg_adv_expr)/volume_adv
    result.outputs["S_yy_avg_media"][0] = fem.assemble_scalar(S_yy_avg_media_expr)/volume_media
    result.outputs["S_zz_avg_adv"][0] = fem.assemble_scalar(S_zz_avg_adv_expr)/volume_adv
    result.outputs["S_zz_avg_media"][0] = fem.assemble_scalar(S_zz_avg_media_expr)/volume_media
    
    result.outputs["s_yy_cell_avg"][0] = fem.assemble_scalar(s_yy_cell_avg_expr)/volume_media
    result.outputs["lambda_cell_in_avg"][0] = fem.assemble_scalar(lambda_cell_in_avg_expr)/volume_media
    
    cell_s_yy_mes_field.interpolate(s_yy_cell_expr_lin)
    cell_l_in_mes_field.interpolate(lambda_cell_in_expr_lin)
    
    result.outputs["s_yy_cell_grad_avg"][0] = fem.assemble_scalar(s_yy_cell_grad_avg_expr)/volume_media
    result.outputs["lambda_cell_in_grad_avg"][0] = fem.assemble_scalar(lambda_cell_in_grad_avg_expr)/volume_media
    
        
    # double export of xdmf results directly in result object for simplicity of extraction
    result.outputs["S_yy"][0, :] = S_yy.x.array[bottom_cells_media]
    result.outputs["S_zz"][0, :] = S_zz.x.array[bottom_cells_media]
    
    result.outputs["s_yy_matrix"][0, :] = s_yy_matrix.x.array[bottom_cells_media]
    result.outputs["s_zz_matrix"][0, :] = s_zz_matrix.x.array[bottom_cells_media]
    result.outputs["s_yy_cell"][0, :] = s_yy_cell.x.array[bottom_cells_media]
    result.outputs["s_zz_cell"][0, :] = s_zz_cell.x.array[bottom_cells_media]
    result.outputs["s_yy_collagen"][0, :] = s_yy_collagen.x.array[bottom_cells_media]
    result.outputs["s_zz_collagen"][0, :] = s_zz_collagen.x.array[bottom_cells_media]
    
    result.outputs["lambda_cell"][0, :] = lambda_cell.x.array[bottom_cells_media]
    result.outputs["lambda_cell_in"][0, :] = lambda_cell_in.x.array[bottom_cells_media]
    
    if dry_run:
        print("    [JIT] Kernels compiled and cached. Exiting warm-up.")
        return None # Exit early
    
    #-------------------------------------------------------------------------#
    ### Run simulation
    for n in range(1,n_steps+1):
        ### Apply increment of Boundary Conditions -> object step_load of type Vcn_load
        T_press.value[0] += step_load.list_dP[n-1]
        disp_z.value += step_load.list_duz[n-1]
        delta_t = step_load.list_dt[n-1]
        
        ### Solve for one step
        try:
            num_its, conv = mech.solve_1_step(delta_t)
        except:
            print('Step crashed ; simulation stopped')
            return(False, mech)
        
        if num_its==mech.max_iter:
            print('Step not converged ; simulation stopped')
            return(False, mech)

        ### Compute specific values and fields
        ur.interpolate(ur_expr)
        S_yy.interpolate(S_yy_expr)
        S_zz.interpolate(S_zz_expr)
                
        s_yy_matrix.interpolate(s_yy_matrix_expr)
        s_yy_cell.interpolate(s_yy_cell_expr)
        s_yy_collagen.interpolate(s_yy_collagen_expr)
        
        s_zz_matrix.interpolate(s_zz_matrix_expr)
        s_zz_cell.interpolate(s_zz_cell_expr)
        s_zz_collagen.interpolate(s_zz_collagen_expr)
        
        lambda_cell.interpolate(lambda_cell_expr)
        lambda_cell_in.interpolate(lambda_cell_in_expr)
        
        ### Store results : macro and aggregated
        result.outputs['time'][n]=step_load.list_t[n]
        result.outputs['ri_d'][n]=ri+np.max(ur.x.array[:])
        result.outputs['re_d'][n]=re+np.min(ur.x.array[:])
        result.outputs['press'][n]=T_press.value[0]
        result.outputs['area'][n] = np.pi*(result.outputs['re_d'][n]**2 - result.outputs['ri_d'][n]**2)
        
        result.outputs["S_yy_avg_adv"][n] = fem.assemble_scalar(S_yy_avg_adv_expr)/volume_adv
        result.outputs["S_yy_avg_media"][n] = fem.assemble_scalar(S_yy_avg_media_expr)/volume_media
        result.outputs["S_zz_avg_adv"][n] = fem.assemble_scalar(S_zz_avg_adv_expr)/volume_adv
        result.outputs["S_zz_avg_media"][n] = fem.assemble_scalar(S_zz_avg_media_expr)/volume_media
            
        result.outputs["s_yy_cell_avg"][n] = fem.assemble_scalar(s_yy_cell_avg_expr)/volume_media
        result.outputs["lambda_cell_in_avg"][n] = fem.assemble_scalar(lambda_cell_in_avg_expr)/volume_media
            
        cell_s_yy_mes_field.interpolate(s_yy_cell_expr_lin)
        cell_l_in_mes_field.interpolate(lambda_cell_in_expr_lin)
        
        result.outputs["s_yy_cell_grad_avg"][n] = fem.assemble_scalar(s_yy_cell_grad_avg_expr)/volume_media
        result.outputs["lambda_cell_in_grad_avg"][n] = fem.assemble_scalar(lambda_cell_in_grad_avg_expr)/volume_media
        
        # Local quantities : export only certain cells
        result.outputs["S_yy"][n, :] = S_yy.x.array[bottom_cells_media]
        result.outputs["S_zz"][n, :] = S_zz.x.array[bottom_cells_media]
        
        result.outputs["s_yy_matrix"][n, :] = s_yy_matrix.x.array[bottom_cells_media]
        result.outputs["s_zz_matrix"][n, :] = s_zz_matrix.x.array[bottom_cells_media]
        result.outputs["s_yy_cell"][n, :] = s_yy_cell.x.array[bottom_cells_media]
        result.outputs["s_zz_cell"][n, :] = s_zz_cell.x.array[bottom_cells_media]
        result.outputs["s_yy_collagen"][n, :] = s_yy_collagen.x.array[bottom_cells_media]
        result.outputs["s_zz_collagen"][n, :] = s_zz_collagen.x.array[bottom_cells_media]
        
        result.outputs["lambda_cell"][n, :] = lambda_cell.x.array[bottom_cells_media]
        result.outputs["lambda_cell_in"][n, :] = lambda_cell_in.x.array[bottom_cells_media]

        print(f"Simu step {n}, Number of iterations {num_its}, Press {result.outputs['press'][n]} MPa, RI {result.outputs['ri_d'][n]}, area {result.outputs['area'][n]}, Residuals {conv}")
    if n == n_steps:
        result.runtime = time.time() - t_simu0
        result.export()
        print(f'Runtime for simu {name} is {result.runtime:.4f} sec')  

        return(result)
    else:
        return(False)
    

def load_JSON(card_name):
    print('Opening card : '+card_name)
    try:
        with open(card_name, 'r') as file:
            card = json.load(file)  # Load JSON as a dictionary
            return(card)
    except FileNotFoundError:
        print(f"Error: File not found: {card_name}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON: {e}")

if __name__ == '__main__':

    name = 'test'
    folder_name = 'Vasoconstriction'
    simu_card_name = 'json_cards/simu_card_Vcn.json'
    media_card_name ='json_cards/media_card_Vcn.json'
    adventitia_card_name ='json_cards/adventitia_card_Vcn.json'
    # Load material card        
    simu_card = load_JSON(simu_card_name)
    adventitia_card = load_JSON(adventitia_card_name)
    media_card = load_JSON(media_card_name)
        
    if not os.path.exists("./outputs/"+folder_name):
        os.makedirs("./outputs/"+folder_name)
    
    result = run_simulation(name, folder_name, simu_card, adventitia_card, media_card)
    
    plt.figure()  
    time_list = result.outputs['time'][:]

    ri_d = result.outputs['ri_d'][:]
    re_d = result.outputs['re_d'][:]
    area =  result.outputs['area'][:]
    plt.plot(time_list, ri_d, label='$R_i$')
    plt.plot(time_list, re_d, label='$R_e$')
    plt.grid()
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(time_list, area)
    plt.grid()
    plt.show()

