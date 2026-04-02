#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 12:30:20 2025

@author: bastien.sauty

Main file for multiscale model implemented for an axisymmetrical cylinder under
internal pressure and axial tension

Adding the Vasoconstriction effects - Stress regulation by the cells
"""

import json, sys, time, os
import numpy as np
import matplotlib.pyplot as plt

# Fenicsx modules in the main
from dolfinx import mesh, fem, io
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import ufl

# Homemade Libraries
from class_modules.mech_problem_class import Mechanical_Problem_axi
from class_modules.result_class import Results

from function_modules.auxiliary_functions import Tensor2Voigt, tensordot_2_2, Voigt2Tensor




def run_simulation(name, folder_name, simu_card, adventitia_card, media_card):
    """
    Run the tension-inflation test on axisymmetric cylinder
    Vasoconstriction - active stress regulation

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
    int_scheme = simu_card["int_scheme"]
    n_int = simu_card["n_int"]
    n_NR = simu_card["n_NR"]
    XDMF_export = simu_card['XDMF_export']

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
    
    # loads
    P_max = simu_card["P_max"]*0.000133322 # from mmHg to MPa
    uz_tot = float(simu_card["uz_tot"]) # has to be a float
    
    #-------------------------------------------------------------------------#
    ### Initiate Mech object
    mech = Mechanical_Problem_axi(name, objective_derivative, int_scheme, n_int)
    
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
    # Displacement field : linear piecewise to get values at border
    
    V_u_exp = fem.functionspace(domain, ("P", 1, (1, )))
    ur = fem.Function(V_u_exp)
    ur_expr = fem.Expression(ufl.dot(ufl.as_vector([1,0]),mech.un), V_u_exp.element.interpolation_points())

    # Angle for the collagen fibers in the Adventitia
    volume_adv = fem.assemble_scalar(fem.form(mech.r*mech.dx(tag_adv)))
    keys = mech.subdomain['adventitia'].inclusions.keys()
    collagen_keys = [k for k in keys if k.startswith("collagen")]
    collagen_expr = {}
    for key in collagen_keys:
        
        eer = mech.subdomain["adventitia"].inclusions[key].e_r
        eet = - (mech.subdomain["adventitia"].inclusions[key].e_theta) 

        costp = ufl.dot(eet, ufl.as_vector([0,-1,0]))
        sintp = ufl.dot(eer, ufl.as_vector([0,1,0]))
        
        angle = ufl.conditional(ufl.Or(costp>1e-2, costp<-1e-2), ufl.atan(sintp/costp), ufl.conditional(sintp>0, np.pi/2, -np.pi/2))
        collagen_theta = 180/np.pi*angle*mech.r*mech.dx(tag_adv)
        collagen_young = mech.subdomain["adventitia"].inclusions[key].E.func[0]*mech.r*mech.dx(tag_adv)
        collagen_stretch = mech.subdomain["adventitia"].inclusions[key].lambda_er[0]*mech.r*mech.dx(tag_adv)
        
        collagen_expr[key] = {'theta':fem.form(collagen_theta), 
                              'young': fem.form(collagen_young), 
                              'stretch': fem.form(collagen_stretch)}
        
    collagen_keys_theta = [key + '_theta' for key in collagen_keys]
    collagen_keys_stretch = [key + '_stretch' for key in collagen_keys]
    collagen_keys_young = [key + '_young' for key in collagen_keys]
    
    # Extract values along one line 
    # get radial position at the centroids of elements
    r_pos = fem.Function(mech.V_scalar)
    r_pos_expr = fem.Expression(mech.x[0], mech.V_scalar.element.interpolation_points()) # 
    r_pos.interpolate(r_pos_expr)
    
    bottom_line_cells = mesh.locate_entities(mech.domain, mech.domain.topology.dim, lambda x: (x[1] <= lz/nz) & (x[0] <= ri_adv))
    r_pos_bottom = r_pos.x.array[bottom_line_cells]
    
    # Initialize exports in XDMF
    if XDMF_export: 
        V_export = fem.functionspace(mech.domain, ("P", 1, (mech.domain.geometry.dim, )))
        u_export = fem.Function(V_export)
    
        V_sig_export = fem.functionspace(mech.domain, ("DG", 0, (6,)))
        Sn_export = fem.Function(V_sig_export)
        En_export = fem.Function(V_sig_export) # Green-Lagrange Strain !! because it is symmetric and can be stored in Mandel notations
        En_export_expr = fem.Expression(Tensor2Voigt(1/2*(ufl.dot(mech.Fn, mech.Fn.T) - ufl.Identity(3))) , V_sig_export.element.interpolation_points())
    
        # Export cell mechanics
        cell_incl = mech.subdomain['media'].inclusions['cells']
        # inelastic stretch
        la_export = fem.Function(mech.V_scalar)
        b_inel = ufl.dot(cell_incl.F_inel, ufl.transpose(cell_incl.F_inel))
        la_form = ufl.sqrt(ufl.dot(cell_incl.e_r, ufl.dot(b_inel, cell_incl.e_r)))
        la_export_expr = fem.Expression(la_form, mech.V_scalar.element.interpolation_points())
        # axial stress
        sig_er_export = fem.Function(mech.V_scalar)
        sig_er_form = ufl.dot(cell_incl.e_r, ufl.dot(Voigt2Tensor(cell_incl.taun), cell_incl.e_r))
        sig_er_export_expr = fem.Expression(sig_er_form, mech.V_scalar.element.interpolation_points())

        S_yy = fem.Function(mech.V_scalar)
        S_yy_expr = fem.Expression(ufl.dot(mech.Sn, ufl.as_vector([0,1,0,0,0,0])), mech.V_scalar.element.interpolation_points())
        S_yy.interpolate(S_yy_expr)
        la_export.interpolate(la_export_expr, mech.subdomain['media'].cells)
        sig_er_export.interpolate(sig_er_export_expr, mech.subdomain['media'].cells)
    
        file_VERIF_init = "./outputs/"+folder_name+"/"+name+"mesh_export.xdmf"
        
        with io.XDMFFile(mech.domain.comm, file_VERIF_init, "w") as xdmfci:
            xdmfci.write_mesh(mech.domain)
            xdmfci.write_function(u_export)
        xdmfci.close()
    
        ### XDMF Results files
        file_domain = "./outputs/"+folder_name+"/"+name+"fields.xdmf"
        xdmf = io.XDMFFile(mech.domain.comm, file_domain, "w")
        xdmf.write_mesh(mech.domain)
    
        Sn_export.name="PKII"
        En_export.name="Green-Lagrange"
        u_export.name='disp'
        
        la_export.name='cell_active_stretch'
        sig_er_export.name='cell_axial_stress'
        
        xdmf.write_function(u_export,0)
        xdmf.write_function(Sn_export, 0)
        xdmf.write_function(En_export, 0)
        xdmf.write_function(la_export, 0)
        xdmf.write_function(sig_er_export, 0)
    
        print ("Export XDMF init : PKII and Green-Lagrange Strain")
        
    #-------------------------------------------------------------------------#
    ### Results Object
    list_outputs_avg = ['time','ri_d', 're_d', 'press', 'area'] + collagen_keys_theta + collagen_keys_stretch + collagen_keys_young
    dict_outputs = {}
    for key in list_outputs_avg:
        dict_outputs[key] = {'points':None}
    
    list_outputs_local = ['S_yy', 'sig_er', 'lambda_cell']
    for key in list_outputs_local:
        dict_outputs[key] = {'points':r_pos_bottom}
    
    result = Results(name, folder_name, dict_outputs, n_NR)
    
    result.outputs['time'][0]=0
    result.outputs['ri_d'][0]=ri+np.max(ur.x.array[:])
    result.outputs['re_d'][0]=re+np.min(ur.x.array[:])
    result.outputs['press'][0]=T_press.value[0]
    result.outputs['area'][0] = np.pi*(result.outputs['re_d'][0]**2 - result.outputs['ri_d'][0]**2)
    for key in collagen_keys:
        result.outputs[key+'_theta'][0] = fem.assemble_scalar(collagen_expr[key]['theta'])/volume_adv
        result.outputs[key+'_stretch'][0] = fem.assemble_scalar(collagen_expr[key]['stretch'])/volume_adv
        result.outputs[key+'_young'][0] = fem.assemble_scalar(collagen_expr[key]['young'])/volume_adv
        
    # double export of xdmf results directly in result object for simplicity of extraction
    result.outputs["S_yy"][0, :] = S_yy.x.array[bottom_line_cells]
    result.outputs["sig_er"][0, :] = sig_er_export.x.array[bottom_line_cells]
    result.outputs["lambda_cell"][0, :] = la_export.x.array[bottom_line_cells]

    #-------------------------------------------------------------------------#
    ### Manage Time and Load -> build functions of time
    n_uz = int(n_NR/3)
    n_p = int(n_NR/3) #int(n_NR+1-n_uz)
    n_t = int(n_NR+2-n_uz-n_p)
    list_P = np.concatenate((np.zeros(n_uz-1), np.linspace(0, P_max, n_p), P_max*np.ones(n_t)))
    list_uz = np.concatenate((np.linspace(0, uz_tot, n_uz), uz_tot*np.ones(n_p+n_t-1)))
    list_time = np.concatenate((np.zeros(n_p+n_uz-1), np.linspace(0, 30, n_t)))
    
    list_dP = list_P[1:]-list_P[0:-1]
    list_duz = list_uz[1:]-list_uz[0:-1]
    list_dt = list_time[1:]-list_time[0:-1]

    #-------------------------------------------------------------------------#
    ### Run simulation
    for n in range(1,n_NR+1):
        ### Apply increment of Boundary Conditions
        T_press.value[0] += list_dP[n-1]
        disp_z.value += list_duz[n-1]
        delta_t = list_dt[n-1]
        
        ### Solve for one step
        try:
            num_its, conv = mech.solve_1_step(delta_t)
        except:
            print('Step crashed ; simulation stopped')
            if XDMF_export:
                xdmf.close()
            return(False, mech)
        
        if num_its==mech.max_iter:
            print('Step not converged ; simulation stopped')
            if XDMF_export:
                xdmf.close()
            return(False, mech)

        ### Compute specific values and fields
        ur.interpolate(ur_expr)
        S_yy.interpolate(S_yy_expr)
        la_export.interpolate(la_export_expr, mech.subdomain['media'].cells)
        sig_er_export.interpolate(sig_er_export_expr, mech.subdomain['media'].cells)
    
        ### Store results
        result.outputs['time'][n]=list_time[n]
        result.outputs['ri_d'][n]=ri+np.max(ur.x.array[:])
        result.outputs['re_d'][n]=re+np.min(ur.x.array[:])
        result.outputs['press'][n]=T_press.value[0]
        result.outputs['area'][n] = np.pi*(result.outputs['re_d'][n]**2 - result.outputs['ri_d'][n]**2)
        
        for key in collagen_keys:
            result.outputs[key+'_theta'][n] = fem.assemble_scalar(collagen_expr[key]['theta'])/volume_adv
            result.outputs[key+'_stretch'][n] = fem.assemble_scalar(collagen_expr[key]['stretch'])/volume_adv
            result.outputs[key+'_young'][n] = fem.assemble_scalar(collagen_expr[key]['young'])/volume_adv
            
        result.outputs["S_yy"][n, :] = S_yy.x.array[bottom_line_cells]
        result.outputs["sig_er"][n, :] = sig_er_export.x.array[bottom_line_cells]
        result.outputs["lambda_cell"][n, :] = la_export.x.array[bottom_line_cells]

        ### XDMF Export

        if XDMF_export:
            u_export.interpolate(mech.un)
            Sn_export.interpolate(mech.Sn)
            En_export.interpolate(En_export_expr)
            
            la_export.interpolate(la_export_expr, mech.subdomain['media'].cells)
            sig_er_export.interpolate(sig_er_export_expr, mech.subdomain['media'].cells)
            
            xdmf.write_function(u_export, n)
            xdmf.write_function(Sn_export, n)
            xdmf.write_function(En_export, n)
            xdmf.write_function(la_export, n)
            xdmf.write_function(sig_er_export, n)

        print(f"Simu step {n}, Number of iterations {num_its}, Press {result.outputs['press'][n]} MPa, RI {result.outputs['ri_d'][n]}, area {result.outputs['area'][n]}, Residuals {conv}")
    if n == n_NR:
        result.runtime = time.time() - t_simu0
        result.export()
        print(f'Runtime for simu {name} is {result.runtime:.4f} sec')  

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
    if len(sys.argv)!=6:
        print(f"Specify : name, foldername, simu_card, media_card, adventitia_card")
    else:
        name = sys.argv[1]
        folder_name = sys.argv[2]
        simu_card_name = 'json_cards/'+sys.argv[3]+'.json'
        media_card_name ='json_cards/'+sys.argv[4]+'.json'
        adventitia_card_name ='json_cards/'+sys.argv[5]+'.json'
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
        area =  result.outputs['area'][:]
        plt.plot(time_list, ri_d, label=f'$R_i$')
        plt.plot(time_list, re_d, label=f'$R_e$')
        plt.grid()
        plt.legend()
        plt.show()
        
        plt.figure()
        plt.plot(time_list, area)
        plt.grid()
        plt.show()

