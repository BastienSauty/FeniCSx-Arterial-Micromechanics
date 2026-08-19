#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:30:20 2025

@author: bastien.sauty

Main file for multiscale model implemented for an axisymmetrical cylinder under
internal pressure and axial tension

### LOGS:
    - 22 Oct 2025 : 
            - remove sig_er_avg output; stress_collagen, stress_cell, stress_hom
            - creation of S_yy, S_zz outputs : 
                'S_yy_avg_adv', 'S_yy_avg_media', 'S_zz_avg_adv', 'S_zz_avg_media', 
                's_yy_matrix_avg','s_yy_cell_avg','s_yy_collagen_media_avg','s_yy_collagen_adv_avg',
                's_zz_matrix_avg','s_zz_cell_avg','s_zz_collagen_media_avg','s_zz_collagen_adv_avg'
    - 21 Jui 2026 :
            - add F_zz_form and F_zz as output : compute the total reaction force of the boundary 4 where dirichlet BC is applied
"""

import json, sys, time, os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Fenicsx modules in the main
from dolfinx import mesh, fem, io
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import ufl

# Homemade Libraries
from Multiscale_Framework.class_modules.mech_problem_class import Mechanical_Problem_axi
from Multiscale_Framework.class_modules.result_class import Results
from Multiscale_Framework.class_modules.load_class import Artery_load

from Multiscale_Framework.function_modules.auxiliary_functions import (
    Tensor2Voigt,
    Voigt2Tensor,
)


def run_simulation(name, folder_name, simu_card, adventitia_card, media_card):
    """
    Run the tension-inflation test on axisymmetric cylinder

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
    n_NR = simu_card["n_NR"]
    load_phase = simu_card['load_phase']
    XDMF_export = simu_card['XDMF_export']
    
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
    print(f"Internal radius of the adventitia corrected to {ri_adv}", flush=True)
    
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
    mech.add_subdomain("media", media_card, 'MT')
    
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

    # Angle for the collagen fibers in the Adventitia
    keys = mech.subdomain['adventitia'].inclusions.keys()
    collagen_adv_keys = [k for k in keys if k.startswith("collagen")]
    collagen_adv_expr = {}
    for key in collagen_adv_keys:
        
        eer = mech.subdomain["adventitia"].inclusions[key].e_r
        eet = mech.subdomain["adventitia"].inclusions[key].e_theta

        # costp = ufl.dot(eet, ufl.as_vector([0,-1,0]))
        costp = ufl.dot(eer, ufl.as_vector([0,0,1]))
        # sintp = ufl.dot(eer, ufl.as_vector([0,1,0])) # on the assumption that fiber stay circumferential
        sintp = ufl.dot(eet, ufl.as_vector([0,0,-1]))
        
        angle = ufl.atan2(sintp, costp)
        abs_angle = ufl.conditional(angle >= 0, angle, -angle)
        folded_angle = ufl.conditional(
            abs_angle <= np.pi/2,
            angle,
            angle - np.pi * ufl.sign(angle)) 
        
        collagen_adv_theta = 180/np.pi*folded_angle*mech.r*mech.dx(tag_adv)
        collagen_adv_young = mech.subdomain["adventitia"].inclusions[key].E.func[0]*mech.r*mech.dx(tag_adv)
        collagen_adv_stretch = mech.subdomain["adventitia"].inclusions[key].lambda_er[0]*mech.r*mech.dx(tag_adv)
        
        collagen_adv_expr[key] = {'theta':fem.form(collagen_adv_theta), 
                              'young': fem.form(collagen_adv_young), 
                              'stretch': fem.form(collagen_adv_stretch)}
        
    collagen_adv_keys_theta = [key + '_adv_theta' for key in collagen_adv_keys]
    collagen_adv_keys_stretch = [key + '_adv_stretch' for key in collagen_adv_keys]
    collagen_adv_keys_young = [key + '_adv_young' for key in collagen_adv_keys]
    
    # Angle for the collagen fibers in the Media
    keys = mech.subdomain['media'].inclusions.keys()
    collagen_media_keys = [k for k in keys if k.startswith("collagen")]
    collagen_media_expr = {}
    for key in collagen_media_keys:
        
        eer = mech.subdomain["media"].inclusions[key].e_r
        eet = mech.subdomain["media"].inclusions[key].e_theta

        # costp = ufl.dot(eet, ufl.as_vector([0,-1,0]))
        costp = ufl.dot(eer, ufl.as_vector([0,0,1]))
        # sintp = ufl.dot(eer, ufl.as_vector([0,1,0])) # on the assumption that fiber stay circumferential
        sintp = ufl.dot(eet, ufl.as_vector([0,0,-1]))
        
        angle = ufl.atan2(sintp, costp)
        abs_angle = ufl.conditional(angle >= 0, angle, -angle)
        folded_angle = ufl.conditional(
            abs_angle <= np.pi/2,
            angle,
            angle - np.pi * ufl.sign(angle)) 
        
        collagen_media_theta = 180/np.pi*folded_angle*mech.r*mech.dx(tag_media)
        collagen_media_young = mech.subdomain["media"].inclusions[key].E.func[0]*mech.r*mech.dx(tag_media)
        collagen_media_stretch = mech.subdomain["media"].inclusions[key].lambda_er[0]*mech.r*mech.dx(tag_media)
        
        collagen_media_expr[key] = {'theta':fem.form(collagen_media_theta), 
                              'young': fem.form(collagen_media_young), 
                              'stretch': fem.form(collagen_media_stretch)}
        
    collagen_media_keys_theta = [key + '_media_theta' for key in collagen_media_keys]
    collagen_media_keys_stretch = [key + '_media_stretch' for key in collagen_media_keys]
    collagen_media_keys_young = [key + '_media_young' for key in collagen_media_keys]
    
    # Extract values along one line 
    # get radial position at the centroids of elements
    r_pos = fem.Function(mech.V_scalar)
    r_pos_expr = fem.Expression(mech.x[0], mech.V_scalar.element.interpolation_points()) # 
    r_pos.interpolate(r_pos_expr)
    
    bottom_line_cells = mesh.locate_entities(mech.domain, mech.domain.topology.dim, lambda x: (x[1] <= lz/nz) & (x[0] <= ri_adv))
    r_pos_bottom = r_pos.x.array[bottom_line_cells]
    
    # local quantity saved in result object
    S_yy = fem.Function(mech.V_scalar) # only used for storing the value along the bottom line
    tau_tissue = ufl.dot(mech.Fn, ufl.dot(Voigt2Tensor(mech.Sn), mech.Fn.T))
    S_yy_form = ufl.dot(ufl.as_vector([0,1,0]), ufl.dot(tau_tissue, ufl.as_vector([0,1,0]))) 
    S_yy_expr = fem.Expression(S_yy_form, mech.V_scalar.element.interpolation_points())
    S_yy.interpolate(S_yy_expr)
    S_zz_form = ufl.dot(ufl.as_vector([0,0,1]), ufl.dot(tau_tissue, ufl.as_vector([0,0,1])))
    
    S_yy_avg_media_expr = fem.form(S_yy_form*mech.r*mech.dx(tag_media)) # average stresses in the media and adventitia, used with assemble_scalar
    S_yy_avg_adv_expr = fem.form(S_yy_form*mech.r*mech.dx(tag_adv))
    S_zz_avg_media_expr = fem.form(S_zz_form*mech.r*mech.dx(tag_media))
    S_zz_avg_adv_expr = fem.form(S_zz_form*mech.r*mech.dx(tag_adv))
    
    # reaction force at the boundary where dirichlet BC is applied
    F_zz_form = fem.form(2*np.pi*S_zz_form*mech.r*mech.ds(4))

    # Get stress tensors  in the matrix, cell and collagen in media and adv
    # f_matrix = mech.subdomain['media'].matrix.f.func[0]
    s_tensor_matrix= mech.subdomain['media'].matrix.taun
    # f_cell = mech.subdomain['media'].inclusions['cells'].f.func[0]
    s_tensor_cell = mech.subdomain['media'].inclusions['cells'].taun
    f_collagen_media = sum(mech.subdomain['media'].inclusions[key].f.func[0] for key in collagen_media_keys )
    s_tensor_collagen_media = sum(mech.subdomain['media'].inclusions[key].f.func[0] * mech.subdomain['media'].inclusions[key].taun for key in collagen_media_keys )/f_collagen_media
    
    f_collagen_adv = sum(mech.subdomain['adventitia'].inclusions[key].f.func[0] for key in collagen_adv_keys )
    s_tensor_collagen_adv = sum(mech.subdomain['adventitia'].inclusions[key].f.func[0] * mech.subdomain['adventitia'].inclusions[key].taun for key in collagen_adv_keys )/f_collagen_adv
    
    # axial and circ stresses
    s_yy_matrix = ufl.dot(s_tensor_matrix, ufl.as_vector([0,1,0,0,0,0]))
    s_yy_cell = ufl.dot(s_tensor_cell, ufl.as_vector([0,1,0,0,0,0]))
    s_yy_collagen_media = ufl.dot(s_tensor_collagen_media, ufl.as_vector([0,1,0,0,0,0]))
    s_yy_collagen_adv = ufl.dot(s_tensor_collagen_adv, ufl.as_vector([0,1,0,0,0,0]))
    
    s_zz_matrix = ufl.dot(s_tensor_matrix, ufl.as_vector([0,0,1,0,0,0]))
    s_zz_cell = ufl.dot(s_tensor_cell, ufl.as_vector([0,0,1,0,0,0]))
    s_zz_collagen_media = ufl.dot(s_tensor_collagen_media, ufl.as_vector([0,0,1,0,0,0]))
    s_zz_collagen_adv = ufl.dot(s_tensor_collagen_adv, ufl.as_vector([0,0,1,0,0,0]))

    # create forms that are then computed as integral through assemble_scalar
    s_yy_matrix_avg_form = fem.form(s_yy_matrix*mech.r*mech.dx(tag_media))
    s_yy_cell_avg_form = fem.form(s_yy_cell*mech.r*mech.dx(tag_media))
    s_yy_collagen_media_avg_form = fem.form(s_yy_collagen_media*mech.r*mech.dx(tag_media))
    s_yy_collagen_adv_avg_form = fem.form(s_yy_collagen_adv*mech.r*mech.dx(tag_adv))
    
    s_zz_matrix_avg_form = fem.form(s_zz_matrix*mech.r*mech.dx(tag_media))
    s_zz_cell_avg_form = fem.form(s_zz_cell*mech.r*mech.dx(tag_media))
    s_zz_collagen_media_avg_form = fem.form(s_zz_collagen_media*mech.r*mech.dx(tag_media))
    s_zz_collagen_adv_avg_form = fem.form(s_zz_collagen_adv*mech.r*mech.dx(tag_adv))    
    
    # Export cell mechanics -> serves for xdmf and local result object
    cell_incl = mech.subdomain['media'].inclusions['cells']
    # axial stress
    sig_er_export = fem.Function(mech.V_scalar)
    sig_er_form = ufl.dot(cell_incl.e_r, ufl.dot(Voigt2Tensor(cell_incl.taun), cell_incl.e_r))
    sig_er_export_expr = fem.Expression(sig_er_form, mech.V_scalar.element.interpolation_points())

    sig_er_export.interpolate(sig_er_export_expr, mech.subdomain['media'].cells)
    
    # Initialize exports in XDMF
    if XDMF_export: 
        V_export = fem.functionspace(mech.domain, ("P", 1, (mech.domain.geometry.dim, )))
        u_export = fem.Function(V_export)
    
        V_sig_export = fem.functionspace(mech.domain, ("DG", 0, (6,)))
        Sn_export = fem.Function(V_sig_export)
        En_export = fem.Function(V_sig_export) # Green-Lagrange Strain !! because it is symmetric and can be stored in Mandel notations
        En_export_expr = fem.Expression(Tensor2Voigt(1/2*(ufl.dot(mech.Fn, mech.Fn.T) - ufl.Identity(3))) , V_sig_export.element.interpolation_points())
    
    
        file_VERIF_init = f"./outputs/{folder_name}/{name}_mesh_export.xdmf" #outputs/"+folder_name+"/"+name+"mesh_export.xdmf"
        
        with io.XDMFFile(mech.domain.comm, file_VERIF_init, "w") as xdmfci:
            xdmfci.write_mesh(mech.domain)
            xdmfci.write_function(u_export)
        xdmfci.close()
    
        ### XDMF Results files
        file_domain = f"./outputs/{folder_name}/{name}_fields.xdmf" #"./outputs/"+folder_name+"/"+name+"fields.xdmf"
        xdmf = io.XDMFFile(mech.domain.comm, file_domain, "w")
        xdmf.write_mesh(mech.domain)
    
        Sn_export.name="PKII"
        En_export.name="Green-Lagrange"
        u_export.name='disp'
        
        sig_er_export.name='cell_axial_stress'
        
        xdmf.write_function(u_export,0)
        xdmf.write_function(Sn_export, 0)
        xdmf.write_function(En_export, 0)
        xdmf.write_function(sig_er_export, 0)
    
        print ("Export XDMF init : PKII and Green-Lagrange Strain", flush=True)
        
    #-------------------------------------------------------------------------#
    ### Results Object
    list_outputs_avg = ['time','ri_d', 're_d', 'press', 'area', 'F_zz',
                        'S_yy_avg_adv', 'S_yy_avg_media', 'S_zz_avg_adv', 'S_zz_avg_media', 
                        's_yy_matrix_avg','s_yy_cell_avg','s_yy_collagen_media_avg','s_yy_collagen_adv_avg',
                        's_zz_matrix_avg','s_zz_cell_avg','s_zz_collagen_media_avg','s_zz_collagen_adv_avg'] 
    list_outputs_avg+= collagen_adv_keys_theta + collagen_adv_keys_stretch + collagen_adv_keys_young
    list_outputs_avg+= collagen_media_keys_theta + collagen_media_keys_stretch + collagen_media_keys_young
    dict_outputs = {}
    for key in list_outputs_avg:
        dict_outputs[key] = {'points':None}
    
    list_outputs_local = ['S_yy', 'sig_er']
    for key in list_outputs_local:
        dict_outputs[key] = {'points':r_pos_bottom}
        
    result = Results(name, folder_name, dict_outputs, n_steps)
    
    result.outputs['time'][0]=0
    result.outputs['ri_d'][0]=ri+np.max(ur.x.array[:])
    result.outputs['re_d'][0]=re+np.min(ur.x.array[:])
    result.outputs['press'][0]=T_press.value[0]
    result.outputs['area'][0] = np.pi*(result.outputs['re_d'][0]**2 - result.outputs['ri_d'][0]**2)
    result.outputs['F_zz'][0] = fem.assemble_scalar(F_zz_form)
    
    for key in collagen_adv_keys:
        result.outputs[key+'_adv_theta'][0] = fem.assemble_scalar(collagen_adv_expr[key]['theta'])/volume_adv
        result.outputs[key+'_adv_stretch'][0] = fem.assemble_scalar(collagen_adv_expr[key]['stretch'])/volume_adv
        result.outputs[key+'_adv_young'][0] = fem.assemble_scalar(collagen_adv_expr[key]['young'])/volume_adv
    
    for key in collagen_media_keys:
        result.outputs[key+'_media_theta'][0] = fem.assemble_scalar(collagen_media_expr[key]['theta'])/volume_media
        result.outputs[key+'_media_stretch'][0] = fem.assemble_scalar(collagen_media_expr[key]['stretch'])/volume_media
        result.outputs[key+'_media_young'][0] = fem.assemble_scalar(collagen_media_expr[key]['young'])/volume_media
    
    
    result.outputs["s_yy_matrix_avg"][0] = fem.assemble_scalar(s_yy_matrix_avg_form)/volume_media
    result.outputs["s_yy_cell_avg"][0] = fem.assemble_scalar(s_yy_cell_avg_form)/volume_media
    result.outputs["s_yy_collagen_media_avg"][0] = fem.assemble_scalar(s_yy_collagen_media_avg_form)/volume_media
    result.outputs["s_yy_collagen_adv_avg"][0] = fem.assemble_scalar(s_yy_collagen_adv_avg_form)/volume_adv
    
    result.outputs["s_zz_matrix_avg"][0] = fem.assemble_scalar(s_zz_matrix_avg_form)/volume_media
    result.outputs["s_zz_cell_avg"][0] = fem.assemble_scalar(s_zz_cell_avg_form)/volume_media
    result.outputs["s_zz_collagen_media_avg"][0] = fem.assemble_scalar(s_zz_collagen_media_avg_form)/volume_media
    result.outputs["s_zz_collagen_adv_avg"][0] = fem.assemble_scalar(s_zz_collagen_adv_avg_form)/volume_adv
    
    
    result.outputs["S_yy_avg_adv"][0] = fem.assemble_scalar(S_yy_avg_adv_expr)/volume_adv
    result.outputs["S_yy_avg_media"][0] = fem.assemble_scalar(S_yy_avg_adv_expr)/volume_media
    result.outputs["S_zz_avg_adv"][0] = fem.assemble_scalar(S_zz_avg_adv_expr)/volume_adv
    result.outputs["S_zz_avg_media"][0] = fem.assemble_scalar(S_zz_avg_adv_expr)/volume_media

    # double export of xdmf results directly in result object for simplicity of extraction
    result.outputs["S_yy"][0, :] = S_yy.x.array[bottom_line_cells]
    result.outputs["sig_er"][0, :] = sig_er_export.x.array[bottom_line_cells]
    
    #-------------------------------------------------------------------------#
    ### Run simulation
    for n in range(1,n_NR+1):
        ### Apply increment of Boundary Conditions -> object step_load of type Artery_load
        T_press.value[0] += step_load.list_dP[n-1]
        disp_z.value += step_load.list_duz[n-1]
        delta_t = step_load.list_dt[n-1]
        
        ### Solve for one step
        try:
            num_its, conv = mech.solve_1_step(delta_t)
        except:
            print('Step crashed ; simulation stopped', flush=True)
            if XDMF_export:
                xdmf.close()
            
            result.runtime = False
            result.export()
            return(False, mech)
        
        if num_its==mech.max_iter:
            print('Step not converged ; simulation stopped', flush=True)
            if XDMF_export:
                xdmf.close()
            
            result.runtime = False
            result.export()
            return(False, mech)

        ### Compute specific values and fields
        ur.interpolate(ur_expr)
        S_yy.interpolate(S_yy_expr)
        sig_er_export.interpolate(sig_er_export_expr)
    
        ### Store results
        result.outputs['time'][n]=n
        result.outputs['ri_d'][n]=ri+np.max(ur.x.array[:])
        result.outputs['re_d'][n]=re+np.min(ur.x.array[:])
        result.outputs['press'][n]=T_press.value[0]
        result.outputs['area'][n] = np.pi*(result.outputs['re_d'][n]**2 - result.outputs['ri_d'][n]**2)
        result.outputs['F_zz'][n] = fem.assemble_scalar(F_zz_form)
        
        
        for key in collagen_adv_keys:
            result.outputs[key+'_adv_theta'][n] = fem.assemble_scalar(collagen_adv_expr[key]['theta'])/volume_adv
            result.outputs[key+'_adv_stretch'][n] = fem.assemble_scalar(collagen_adv_expr[key]['stretch'])/volume_adv
            result.outputs[key+'_adv_young'][n] = fem.assemble_scalar(collagen_adv_expr[key]['young'])/volume_adv
        
        for key in collagen_media_keys:
            result.outputs[key+'_media_theta'][n] = fem.assemble_scalar(collagen_media_expr[key]['theta'])/volume_media
            result.outputs[key+'_media_stretch'][n] = fem.assemble_scalar(collagen_media_expr[key]['stretch'])/volume_media
            result.outputs[key+'_media_young'][n] = fem.assemble_scalar(collagen_media_expr[key]['young'])/volume_media


        result.outputs["S_yy_avg_adv"][n] = fem.assemble_scalar(S_yy_avg_adv_expr)/volume_adv
        result.outputs["S_yy_avg_media"][n] = fem.assemble_scalar(S_yy_avg_media_expr)/volume_media
        result.outputs["S_zz_avg_adv"][n] = fem.assemble_scalar(S_zz_avg_adv_expr)/volume_adv
        result.outputs["S_zz_avg_media"][n] = fem.assemble_scalar(S_zz_avg_media_expr)/volume_media
        
        result.outputs["s_yy_matrix_avg"][n] = fem.assemble_scalar(s_yy_matrix_avg_form)/volume_media
        result.outputs["s_yy_cell_avg"][n] = fem.assemble_scalar(s_yy_cell_avg_form)/volume_media
        result.outputs["s_yy_collagen_media_avg"][n] = fem.assemble_scalar(s_yy_collagen_media_avg_form)/volume_media
        result.outputs["s_yy_collagen_adv_avg"][n] = fem.assemble_scalar(s_yy_collagen_adv_avg_form)/volume_adv
        
        result.outputs["s_zz_matrix_avg"][n] = fem.assemble_scalar(s_zz_matrix_avg_form)/volume_media
        result.outputs["s_zz_cell_avg"][n] = fem.assemble_scalar(s_zz_cell_avg_form)/volume_media
        result.outputs["s_zz_collagen_media_avg"][n] = fem.assemble_scalar(s_zz_collagen_media_avg_form)/volume_media
        result.outputs["s_zz_collagen_adv_avg"][n] = fem.assemble_scalar(s_zz_collagen_adv_avg_form)/volume_adv
        
        
        result.outputs["S_yy"][n, :] = S_yy.x.array[bottom_line_cells]
        result.outputs["sig_er"][n, :] = sig_er_export.x.array[bottom_line_cells]
        ### XDMF Export

        if XDMF_export:
            u_export.interpolate(mech.un)
            Sn_export.interpolate(mech.Sn)
            En_export.interpolate(En_export_expr)
            xdmf.write_function(u_export, n)
            xdmf.write_function(Sn_export, n)
            xdmf.write_function(En_export, n)
            xdmf.write_function(sig_er_export, n)

        print(f"Time step {n}, Number of iterations {num_its}, Press {result.outputs['press'][n]} MPa, RI {result.outputs['ri_d'][n]}, area {result.outputs['area'][n]}, Residuals {conv}", flush=True)
    if n == n_NR:
        result.runtime = time.time() - t_simu0
        result.export()
        print(f'Runtime for simu {name} is {result.runtime:.4f} sec', flush=True)  

        return(result, mech)
    else:
        return(False, mech)
    

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
    argv = sys.argv[1:]
    
    if len(argv) >= 3 and argv[0] == "python" and argv[1] == "-m":
        argv = argv[3:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("foldername")
    parser.add_argument("simu_card")
    parser.add_argument("media_card")
    parser.add_argument("adventitia_card")
    
    args = parser.parse_args(argv)
    
    name = args.name
    folder_name = args.foldername
    simu_card_name = f"json_cards/{args.simu_card}.json"
    media_card_name = f"json_cards/{args.media_card}.json"
    adventitia_card_name = f"json_cards/{args.adventitia_card}.json"
    
    # Load material card        
    simu_card = load_JSON(simu_card_name)
    adventitia_card = load_JSON(adventitia_card_name)
    media_card = load_JSON(media_card_name)
        
    if not os.path.exists("./outputs/"+folder_name):
        os.makedirs("./outputs/"+folder_name)
    
    result, mech = run_simulation(name, folder_name, simu_card, adventitia_card, media_card)
    
    plt.figure()  
    time_list = result.outputs['time'][:]

    ri_d = result.outputs['ri_d'][:]
    re_d = result.outputs['re_d'][:]
    Press = result.outputs['press'][:]
    F_zz = result.outputs['F_zz'][:]
    area =  result.outputs['area'][:]
    plt.plot(time_list, ri_d, label=f'$R_i$')
    plt.plot(time_list, re_d, label=f'$R_e$')
    plt.grid()
    plt.legend()
    plt.savefig(f'images_output/{folder_name}/pressure_radius.pdf')
    plt.show()
    
    plt.figure()
    plt.plot(time_list, Press)
    plt.grid()
    plt.savefig(f'images_output/{folder_name}/pressure_time.pdf')
    plt.show()
    
    plt.figure()
    plt.plot(time_list, F_zz)
    plt.grid()
    plt.savefig(f'images_output/{folder_name}/axial_force_time.pdf')
    plt.show()

