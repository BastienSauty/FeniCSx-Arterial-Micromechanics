#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 12:13:46 2025
Updated July 

@author: bastien.sauty

Section 3.2 : impact of the fiber discretization process on the model
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle
import json

import multiprocessing
from functools import partial

import matplotlib as mpl

from matplotlib.lines import Line2D

mpl.rcParams['text.usetex'] = True

plt.rcParams.update({
    "text.usetex": True,          # use LaTeX for all text
    "font.family": "serif",       # LaTeX default serif font
    "font.serif": ["Computer Modern"],  # optional, match LaTeX
    "axes.labelsize": 10,         # adjust label size
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from Multiscale_Framework.class_modules.load_class import Artery_load

from Multiscale_Framework.function_modules.discretization_collagen import discretizing_distribution, plot_discrete_vs_continuous, project_discrete_to_grid_centered_bins, RMSE_L2, normalized_moment_error


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
        

def load_exp_data(namefile):
    data = np.load(namefile)
    exp_pdf = data['avg_density'] # experimental continuous PDF
    angles = data['angles'] # continuous angle space
    exp_pdf /= np.trapz(exp_pdf, angles) # normalization
    return([angles, exp_pdf])

#-----------------------------------------------------------------------------#
# Specific sensitivity analysis : number of collagen families
#-----------------------------------------------------------------------------#

def pipeline_SA(N, name, folder_name, simu_card, media_card, adventitia_card):
    """
    process to run the simulation for a given number N of discrete families
    """
    
    namefile = f"{name}_{N}_scal.pkl"
    
    
    # Set up unique cache folder for this simulation
    name_simu = f"{name}_{N}"
    cache_path = os.path.expanduser(f"~/.cache/fenics/{name_simu}/")
    os.makedirs(cache_path, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = cache_path
    
    # Import everything AFTER setting environment variables
    from dolfinx import fem, mesh, io
    import ufl
    from petsc4py import PETSc

    # Limit PETSc threading
    opts = PETSc.Options()
    opts["mat_mkl_cpardiso_omp_num_threads"] = 1  # If using MKL Pardiso
    opts["mat_mumps_icntl_14"] = 40  # Example: MUMPS memory allocation control
    opts["num_threads"] = 1
    
    from ChIV_ModellingPassiveArterialTissue.main_ArterialTissue_25_06_04 import run_simulation
    
    # Discretization
    filename_exp_collagen_distrib = 'ChIV_ModellingPassiveArterialTissue/passive_calibration/avg_density_Low.npz'
    theta_coll_adv, weights_coll_adv = discretizing_distribution(filename_exp_collagen_distrib, N, f'{name}_{N}_init', folder_name, plot=True, verbose=False) # function in discretization_collagen.py
    f_tot = 0.5
    weights_coll_adv *= f_tot
    
    
    try:
        file = open(f'./outputs/{folder_name}/{namefile}', 'rb')

        result = pickle.load(file) # load and store pkl file
        file.close()
    except:
        name_simu = name+'_'+str(N)
        print(f'Running simulation {name_simu}')
        
        # Build Adventitia card
        for n in range(N):
            key = 'collagen_'+str(n)
            adventitia_card[key] = {"type": "cylinder",
                                    "young" : media_card['collagen0']['young'],
                                    "young_type":"Exponential",
                                    "poisson" : 0.34,
                                    "volumic_fraction": float(weights_coll_adv[n]),
                                    "theta": float(theta_coll_adv[n]),
                                    "phi": 90}
        
        # Run simulation
        result, mech = run_simulation(name_simu, folder_name, simu_card, adventitia_card, media_card)

    return([theta_coll_adv, weights_coll_adv, result])



def error_to_exp(angles_exp, pdf_exp, angles_model, weights_model, name, folder_name, keyword, plot=False, verbose=True):
    """
    Compute the errors between the experimental distribution and model distribution obtained by the simulation
    """
    # sort the discrete distribution by ascending order
    sort_ind = np.argsort(angles_model)
    angles_model = angles_model[sort_ind]
    weights_model = weights_model[sort_ind]
    
    
    projection_model = project_discrete_to_grid_centered_bins(angles_model, weights_model, angles_exp)
    if plot:
        plot_discrete_vs_continuous(angles_exp, pdf_exp, angles_model, weights_model, projection_model, f'{name}_{len(weights_model)}_{keyword}', folder_name, savefig=True, verbose=verbose)
    
    
    error_L2 = RMSE_L2(angles_exp, pdf_exp, projection_model)
    error_moments = normalized_moment_error(angles_model, weights_model, projection_model, angles_exp, max_order=4)
    error_total = normalized_moment_error(angles_model, weights_model, pdf_exp, angles_exp, max_order=4)
    return([error_L2, error_moments, error_total])


def extract_angles(result, press_value):
    """
    extract the orientation angles of the collagen fibers
    """
    keys = result.outputs.keys()
    theta_keys = [k for k in keys if k.startswith('collagen_') and k.endswith('_theta')]
    pressure_list = result.outputs['press']
        
    target_pressure_mpa = press_value * 0.000133322
    
    # Find index of closest value
    index = np.argmin(np.abs(pressure_list - target_pressure_mpa))
    
    angles_model = [result.outputs[k][index] for k in theta_keys]
    
    return(np.array(angles_model))



if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')#, force=True) 
    except:
        print("context already set")

    # Simulation parameters and directory    
    #-----------------------------------------------------------------------------#
    # Load general parameters
    #-----------------------------------------------------------------------------#

    folder_name = 'ChIV.3.2.Fiberdiscretization/'
    name = 'fiberdiscretization_SA_vf'
    # namefile = name +'_scal.pkl'

    simu_card_name = 'json_cards/simu_card_calib.json'
    media_card_name ='json_cards/media_card_calib.json'
    # Load material card        
    simu_card = load_JSON(simu_card_name)
    media_card = load_JSON(media_card_name)
        
    simu_card['XDMF_export'] = 0
    load_phase = simu_card['load_phase']
    step_load = Artery_load(load_phase)
    # Not loading the adventitia because we build it from scratch depending on the number of fiber families
    # adventitia_card_name ='json_cards/adventitia_card.json'
    # adventitia_card = load_JSON(adventitia_card_name)
    
    # One parameter sensitivity -> Fiber waviness
    N_discrete_list = np.array([4,5,6,7,8,9, 10, 13,15,18,20, 23, 25])

    young_matrix = media_card['matrix']['young']
    adventitia_card = {"matrix" : {
                "type": "matrix",
                "young" : young_matrix,
                "poisson" : 0.45}}

    

    pipeline_partial = partial(
                                pipeline_SA,
                                name=name,
                                folder_name=folder_name,
                                simu_card=simu_card,
                                media_card=media_card,
                                adventitia_card=adventitia_card
                            )
    
    with mp.Pool(processes=4) as pool:
        all_results = pool.map(pipeline_partial, N_discrete_list)
                
    
    #-----------------------------------------------------------------------------#
    # Compute error for kinematic
    #-----------------------------------------------------------------------------## 
    
    errors_representation = np.zeros((len(N_discrete_list), 4))
    errors_projection = np.zeros((len(N_discrete_list), 4))
    errors_discretization = np.zeros((len(N_discrete_list), 4))
    # load dias and sys values
    exp_pdf_Low = 'ChIV_ModellingPassiveArterialTissue/passive_calibration/avg_density_Low.npz'
    exp_pdf_Dias = 'ChIV_ModellingPassiveArterialTissue/passive_calibration/avg_density_Dias.npz'
    exp_pdf_Sys = 'ChIV_ModellingPassiveArterialTissue/passive_calibration/avg_density_Sys.npz'
    
    angles_Low, exp_pdf_Low = load_exp_data(exp_pdf_Low)
    angles_Dias, exp_pdf_Dias = load_exp_data(exp_pdf_Dias)
    angles_Sys, exp_pdf_Sys = load_exp_data(exp_pdf_Sys)
    
    for i, N in enumerate(N_discrete_list):
        angles_model_Init, weights_model, result = all_results[i]
        weights_model /= 0.5
        angles_model_Low = extract_angles(result, 10.)
        angles_model_Dias = extract_angles(result, 80.)
        angles_model_Sys = extract_angles(result, 120.)
        
        [error_L2_Init, error_moments_Init, error_total_Init] = error_to_exp(angles_Low, exp_pdf_Low, angles_model_Init, weights_model, name, folder_name, keyword='Init', plot=True, verbose=True)
        [error_L2_Low, error_moments_Low, error_total_Low] = error_to_exp(angles_Low, exp_pdf_Low, angles_model_Low, weights_model, name, folder_name, keyword='Low', plot=True, verbose=True)
        [error_L2_Dias, error_moments_Dias, error_total_Dias] = error_to_exp(angles_Dias, exp_pdf_Dias, angles_model_Dias, weights_model, name, folder_name, keyword='Dias', plot=True, verbose=True)
        [error_L2_Sys, error_moments_Sys, error_total_Sys] = error_to_exp(angles_Sys, exp_pdf_Sys, angles_model_Sys, weights_model, name, folder_name, keyword='Sys', plot=True, verbose=True)
        
        errors_representation[i, :] = [error_L2_Init, error_L2_Low, error_L2_Dias, error_L2_Sys]
        errors_projection[i, :] = [error_moments_Init, error_moments_Low, error_moments_Dias, error_moments_Sys]
        errors_discretization[i, :] = [error_total_Init, error_total_Low, error_total_Dias, error_total_Sys]
        
    #%%
    #-----------------------------------------------------------------------------#
    # Plot results
    #-----------------------------------------------------------------------------#
    press_list = 7500.62*step_load.list_P
    lambdaz_list = 1+step_load.list_uz/simu_card['lz']

    # --- Compute phase lengths ---
    n_phase1 = step_load.index_phase[0][1]          
    n_phase2 = len(press_list) - step_load.index_phase[1][0]  
    indices_1 = slice(step_load.index_phase[0][0], step_load.index_phase[0][1])
    indices_2 = slice(step_load.index_phase[1][0], step_load.index_phase[1][1]+1)
    
    #-----------------------------------------------------------------------------#
    # Pressure-Radius
    #-----------------------------------------------------------------------------#
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(5,3),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={'width_ratios': [n_phase1, n_phase2], 'wspace': 0.05}  # tighten space
    )
                
    # Example colormap (continuous and smooth)
    cmap = plt.cm.get_cmap('viridis', len(N_discrete_list))
    colors = [cmap(i / (len(N_discrete_list) - 1)) for i in range(len(N_discrete_list))]
    # Plot all curves
    for i, N in enumerate(N_discrete_list):
        color = colors[i]
        _, _, result = all_results[i]
    
        ri_d = result.outputs['ri_d'][:]
        re_d = result.outputs['re_d'][:]
    
        # Solid for Ri, dashed for Re — same color for both
        ax1.plot(lambdaz_list[indices_1], ri_d[indices_1], color=color, linestyle='-', linewidth=1.5)
        ax1.plot(lambdaz_list[indices_1], re_d[indices_1], color=color, linestyle='--', linewidth=1.5)
    
        ax2.plot(press_list[indices_2], ri_d[indices_2], color=color, linestyle='-', linewidth=1.5)
        ax2.plot(press_list[indices_2], re_d[indices_2], color=color, linestyle='--', linewidth=1.5)
    
    ax1.set_xlim([np.min(lambdaz_list), np.max(lambdaz_list)])
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel("Radius [mm]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.7))
    ax2.set_xlim([np.min(press_list), np.max(press_list)])
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Hide left y-axis ticks/labels on the right subplot (sharey=True keeps the scale)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)
            
    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'$R_i$ (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'$R_e$ (dashed)')
    ]
    
    # Create and store the style legend first (so it doesn’t get overwritten)
    style_legend_box = ax2.legend(
        handles=style_legend,
        loc='lower right',
        fontsize=8,
        frameon=True,
        framealpha=0.85,
        facecolor='white',
        edgecolor='gray',
        fancybox=True,
        borderpad=0.4,
        title_fontsize=9
    )
    
    # Add the style legend manually as an artist
    ax2.add_artist(style_legend_box)
    
    # --- 2️⃣ Main legend (N values) OUTSIDE ax2 ---
    legend_elements = [
        Line2D([0], [0], color=cmap(i / (len(N_discrete_list) - 1)), lw=2,
               label=rf'N={N_discrete_list[i]}')
        for i in range(len(N_discrete_list))
    ]
    
    ax2.legend(
        handles=legend_elements, 
        title='N values',
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    
    plt.savefig(f'images_output/{folder_name}/{name}_pressure_radius.pdf', bbox_inches='tight')
    plt.show()
    
    #-------------------------------------------------------------------------#
    # Rotation Errors
    #-------------------------------------------------------------------------#
    #%% 
    plt.figure(figsize=(4,3))
    plt.semilogy(N_discrete_list, errors_representation[:,0], label='Unloaded configuration', marker='+', linestyle='-', linewidth=1, alpha=0.8)
    # plt.semilogy(N_discrete_list, errors_representation[:,1], label='Low pressure', marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.semilogy(N_discrete_list, errors_representation[:,2], label='Diastolic pressure', marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.semilogy(N_discrete_list, errors_representation[:,3], label='Systolic pressure', marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.xlabel('Number of families')
    plt.ylabel('L2 Error')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.ylim([1e-3, 1e-1])
    plt.legend(
        loc='lower right',
        frameon=True,
        framealpha=0.8,
        fancybox=True,
        ncol=2,               # two columns
        fontsize=8,
        handlelength=2.5,
        columnspacing=0.8
    )
    plt.tight_layout()
    plt.savefig(f'images_output/{folder_name}/{name}_L2error_Rotation.pdf')
    plt.show()
    
    plt.figure(figsize=(4,3))
    plt.semilogy(N_discrete_list, errors_discretization[:,0], label='Unloaded configuration', marker='+', linestyle='-', linewidth=1, alpha=0.8)
    # plt.semilogy(N_discrete_list, errors_discretization[:,1], label='Low pressure', marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.semilogy(N_discrete_list, errors_discretization[:,2], label='Diastolic pressure', marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.semilogy(N_discrete_list, errors_discretization[:,3], label='Systolic pressure', marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.xlabel('Number of families')
    plt.ylabel('Total Error -- Moment metric')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.ylim([2e-3, 1e1])
    plt.legend(
        loc='lower right',
        frameon=True,
        framealpha=0.8,
        fancybox=True,
        ncol=2,               # two columns
        fontsize=8,
        handlelength=2.5,
        columnspacing=0.8
    )
    plt.tight_layout()
    
    plt.savefig(f'images_output/{folder_name}/{name}_Momenterror_Rotation.pdf')
    plt.show()
    
    #-------------------------------------------------------------------------#
    # Plot last radius value at max pressure
    #-------------------------------------------------------------------------#
    #%%
    list_lastradii = np.zeros(N_discrete_list.shape)
    for j, N in enumerate(N_discrete_list):
        _, _, result = all_results[j]
        list_lastradii[j] = result.outputs['ri_d'][-1]
        
    plt.figure()
    plt.plot(N_discrete_list, list_lastradii, marker='+')
    plt.grid()
    plt.xlabel('Number of families')
    plt.ylabel(fr'$R_i$ at $P=140$mmHg [mm]')
    plt.show()