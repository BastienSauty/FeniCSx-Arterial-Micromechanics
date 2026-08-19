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

mpl.rcParams['text.usetex'] = True

plt.rcParams.update({
    "text.usetex": True,
    # Inject Latin Modern and T1 font encoding into the LaTeX preamble
    "text.latex.preamble": r"\usepackage[T1]{fontenc} \usepackage{lmodern}",
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"], # Matches your thesis main text
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from Multiscale_Framework.function_modules.discretization_collagen import load_and_discretize_distribution, plot_discrete_vs_continuous, project_discrete_to_grid_centered_bins, RMSE_L2, normalized_moment_error

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

def pipeline_SA(N, dta_sample, name, folder_name, simu_card, media_card, adventitia_card, force_run_simu=True):
    """
    process to run the simulation for a given number N of discrete families
    dta_sample : id of the sample (between DTA1 and DTA7)
    """
    
    namefile = f"{dta_sample}_{name}_{N}_scal.pkl"
    
    # Set up unique cache folder for this simulation
    name_simu = f"{dta_sample}_{name}_{N}"
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
    filename_exp_collagen_distrib = f'ChIV_ModellingPassiveArterialTissue/passive_calibration/sample_data/{dta_sample}_density_Low.npz'
    theta_coll_adv, weights_coll_adv = load_and_discretize_distribution(filename_exp_collagen_distrib, N, f'{dta_sample}_{name}_{N}_init', folder_name, plot=False, verbose=False) # function in discretization_collagen.py
    
    f_tot = 0.5
    weights_coll_adv *= f_tot
    
    try:
        file = open(f'./outputs/{folder_name}/{namefile}', 'rb')

        result = pickle.load(file) # load and store pkl file
        file.close()
    except:
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
        if force_run_simu:
            result, mech = run_simulation(name_simu, folder_name, simu_card, adventitia_card, media_card)
        else:
            result = False

    return([theta_coll_adv, weights_coll_adv, result])

#-----------------------------------------------------------------------------#
# Some functions to compute somethings on the angles
#-----------------------------------------------------------------------------#

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
        plot_discrete_vs_continuous(angles_exp, pdf_exp, angles_model, weights_model, projection_model, f'{name}_{keyword}', folder_name, savefig=True, verbose=verbose)
    
    
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

#-----------------------------------------------------------------------------#
# Class to store the results
#-----------------------------------------------------------------------------#

class sample_result:
    """
    simple class to store the error results and plot those sample wise
    """
    def __init__(self, dta_sample, N_discrete_list, name, folder_name):
        self.dta_sample = dta_sample
        self.N_discrete_list = N_discrete_list
        self.errors_representation = np.zeros((len(N_discrete_list), 4))
        self.errors_extrapolation = np.zeros((len(N_discrete_list), 4))
        self.errors_discretization = np.zeros((len(N_discrete_list), 4))
        self.folder_name = folder_name
        self.name = name
        
    def plot_errors(self, verbose=False):
        #%%
        plt.figure(figsize=(4,3))
        plt.semilogy(self.N_discrete_list, self.errors_representation[:,0], label='Init pressure', marker='+', linestyle='-', linewidth=1, alpha=0.8)
        plt.semilogy(self.N_discrete_list, self.errors_representation[:,1], label='Low pressure', marker='+', linestyle='-', linewidth=1, alpha=0.8)
        plt.semilogy(self.N_discrete_list, self.errors_representation[:,2], label='Diastolic pressure', marker='+', linestyle='-', linewidth=1, alpha=0.8)
        plt.semilogy(self.N_discrete_list, self.errors_representation[:,3], label='Systolic pressure', marker='+', linestyle='-', linewidth=1, alpha=0.8)
        plt.xlabel('Number of families')
        plt.ylabel('L2 Error')
        plt.grid(True, linestyle='--', alpha=0.3)
        #plt.ylim([1e-3, 2e-1])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'images_output/{self.folder_name}/{self.name}_{dta_sample}_L2error_Rotation.pdf')
        
        if verbose:
            plt.show()
        else:
            plt.close()
        
        plt.figure(figsize=(4,3))
        plt.semilogy(self.N_discrete_list, self.errors_discretization[:,0], label='Init pressure', marker='+', linestyle='-', linewidth=1, alpha=0.8)
        plt.semilogy(self.N_discrete_list, self.errors_discretization[:,1], label='Low pressure', marker='+', linestyle='-', linewidth=1, alpha=0.8)
        plt.semilogy(self.N_discrete_list, self.errors_discretization[:,2], label='Diastolic pressure', marker='+', linestyle='-', linewidth=1, alpha=0.8)
        plt.semilogy(self.N_discrete_list, self.errors_discretization[:,3], label='Systolic pressure', marker='+', linestyle='-', linewidth=1, alpha=0.8)
        plt.xlabel('Number of families')
        plt.ylabel('Total Error - Moment metric')
        plt.grid(True, linestyle='--', alpha=0.3)
        #plt.ylim([1e-2, 2e2])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'images_output/{self.folder_name}/{self.name}_{dta_sample}_Momenterror_Rotation.pdf')
        
        if verbose:
            plt.show()
        else:
            plt.close()


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
    folder_name = 'ChIV.3.2.2.Fiberdiscretization_SampleWise/'
    name = 'fiberdiscretization_SW_vf'
    # namefile = name +'_scal.pkl'

    simu_card_name = 'json_cards/simu_card_calib.json'
    media_card_name ='json_cards/media_card_calib.json'
    
    # Load material card        
    simu_card = load_JSON(simu_card_name)
    media_card = load_JSON(media_card_name)
    simu_card['XDMF_export'] = 0
    # Not loading the adventitia because we build it from scratch depending on the number of fiber families
    # adventitia_card_name ='json_cards/adventitia_card.json'
    # adventitia_card = load_JSON(adventitia_card_name)
    
    # One parameter sensitivity -> number of fiber families
    N_discrete_list = np.array([4,5,6,7,8,9, 10, 13,15,18,20, 23, 25])

    young_matrix = media_card['matrix']['young']
    adventitia_card = {"matrix" : {
                "type": "matrix",
                "young" : young_matrix,
                "poisson" : 0.45}}

    sample_result_dict = {} # contain all the results

    for id_DTA in range(1,8):
        dta_sample = f'DTA{id_DTA}'
        sample_result_dict[dta_sample] = sample_result(dta_sample, N_discrete_list, name, folder_name)
        
        pipeline_partial = partial(
                                    pipeline_SA,
                                    dta_sample=dta_sample,
                                    name=name,
                                    folder_name=folder_name,
                                    simu_card=simu_card,
                                    media_card=media_card,
                                    adventitia_card=adventitia_card,
                                    force_run_simu=True
                                )
        
        with mp.Pool(processes=5) as pool:
            all_results = pool.map(pipeline_partial, N_discrete_list)
            
        # no plot of pressure radius behavior
        #-----------------------------------------------------------------------------#
        # Compute error for kinematic
        #-----------------------------------------------------------------------------## 
        # load dias and sys values        
        exp_pdf_Low = f'ChIV_ModellingPassiveArterialTissue/passive_calibration/sample_data/{dta_sample}_density_Low.npz'
        exp_pdf_Dias = f'ChIV_ModellingPassiveArterialTissue/passive_calibration/sample_data/{dta_sample}_density_Dias.npz'
        exp_pdf_Sys = f'ChIV_ModellingPassiveArterialTissue/passive_calibration/sample_data/{dta_sample}_density_Sys.npz'
        
        angles_Low, exp_pdf_Low = load_exp_data(exp_pdf_Low)
        angles_Dias, exp_pdf_Dias = load_exp_data(exp_pdf_Dias)
        angles_Sys, exp_pdf_Sys = load_exp_data(exp_pdf_Sys)
        
        for i, N in enumerate(N_discrete_list):
            angles_model_Init, weights_model, result = all_results[i]
            weights_model /= 0.5
            
            if result and result.runtime:
                angles_model_Low = extract_angles(result, 10.)
                angles_model_Dias = extract_angles(result, 80.)
                angles_model_Sys = extract_angles(result, 120.)
                
                [error_L2_Init, error_extrapolation_Init, error_total_Init] = error_to_exp(angles_Low, exp_pdf_Low, angles_model_Init, weights_model, f'{dta_sample}_{name}_{N}', folder_name, keyword='Init', plot=True, verbose=False)
                [error_L2_Low, error_extrapolation_Low, error_total_Low] = error_to_exp(angles_Low, exp_pdf_Low, angles_model_Low, weights_model,f'{dta_sample}_{name}_{N}', folder_name,  keyword='Low', plot=True, verbose=False)
                [error_L2_Dias, error_extrapolation_Dias, error_total_Dias] = error_to_exp(angles_Dias, exp_pdf_Dias, angles_model_Dias, weights_model, f'{dta_sample}_{name}_{N}', folder_name, keyword='Dias', plot=True, verbose=False)
                [error_L2_Sys, error_extrapolation_Sys, error_total_Sys] = error_to_exp(angles_Sys, exp_pdf_Sys, angles_model_Sys, weights_model, f'{dta_sample}_{name}_{N}', folder_name, keyword='Sys', plot=True, verbose=False)
                
                sample_result_dict[dta_sample].errors_representation[i, :] = [error_L2_Init, error_L2_Low, error_L2_Dias, error_L2_Sys]
                sample_result_dict[dta_sample].errors_extrapolation[i, :] = [error_extrapolation_Init, error_extrapolation_Low, error_extrapolation_Dias, error_extrapolation_Sys]
                sample_result_dict[dta_sample].errors_discretization[i, :] = [error_total_Init, error_total_Low, error_total_Dias, error_total_Sys]
            
        sample_result_dict[dta_sample].plot_errors(verbose=True)
    
   
    #%%
    #-----------------------------------------------------------------------------#
    # Analysis on the whole bunch of data
    #-----------------------------------------------------------------------------#
    def extract_mean_std(error_keyword, sample_result_dict):
        # Assuming all sample_result objects have the same N_discrete_list length
        n_discrete = len(next(iter(sample_result_dict.values())).N_discrete_list) # get the length of the array of number of families  array([ 4,  5,  6,  7,  8,  9, 10, 13, 15, 18, 20, 23, 25])
        error_dim = 4  # Since shape is (n_discrete, 3)
        
        # Create lists to collect valid error vectors
        all_valid_errors = [[] for _ in range(n_discrete)]  # One list per discretization level
        
        # Collect non-zero rows
        for result in sample_result_dict.values():
            for i in range(n_discrete):
                if error_keyword=='L2error':
                    error_vec = result.errors_representation[i, :]
                elif error_keyword=='extrapolationerror':
                    error_vec = result.errors_extrapolation[i, :]
                elif error_keyword=='Totalerror':
                    error_vec = result.errors_discretization[i, :]
                    
                if not np.all(error_vec == 0):  # Check it's not [0, 0, 0]
                    all_valid_errors[i].append(error_vec)
        
        # Convert to arrays and compute stats
        mean_errors = np.zeros((n_discrete, error_dim))
        std_errors = np.zeros((n_discrete, error_dim))
        nb_sample = np.zeros((n_discrete,)) # number of valid sample for each discretization (N families)
        
        for i in range(n_discrete):
            if all_valid_errors[i]:  # Only compute if we have valid entries
                stacked = np.stack(all_valid_errors[i], axis=0)  # Shape: (num_valid_samples, 3)
                mean_errors[i, :] = np.mean(stacked, axis=0)
                std_errors[i, :] = np.std(stacked, axis=0)
                nb_sample[i] = stacked.shape[0]
                
        return(mean_errors, std_errors, nb_sample)
    
    from scipy.stats import t
    
    def plot_totalsample_errors(error_keyword, mean_errors, std_errors, nb_sample):
        plt.figure(figsize=(4, 3))
        
        # Labels for each pressure component
        labels = ['Unloaded configuration', 'Low pressure', 'Diastolic pressure', 'Systolic pressure']
        colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:green' ]  # Optional: color control
        
        # Assume mean, std, and nb_sample are already defined
        alpha = 0.05  # for 95% confidence
        df = nb_sample - 1  # degrees of freedom
        
        # Compute t-critical value dynamically
        t_crit = t.ppf(1 - alpha/2, df)
        
        for i in range(4):  # For each error component
            if i!=1:
                mean = mean_errors[:, i]
                std = std_errors[:, i]
                lower = mean - t_crit*std/np.sqrt(nb_sample) #std/np.sqrt(7*(1-0.95))
                upper = mean + t_crit*std/np.sqrt(nb_sample) # std/np.sqrt(7*(1-0.95))
            
                plt.semilogy(N_discrete_list, mean, label=labels[i],
                             marker='+', linestyle='-', linewidth=1, alpha=0.8, color=colors[i])
                
                plt.fill_between(N_discrete_list, lower, upper, 
                                 color=colors[i], alpha=0.2)  # Shaded region
        
        plt.xlabel('Number of families')
        plt.ylabel(rf'{error_keyword} - mean $\pm$ confidence 95\%')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.ylim([1e-3, 1e-1])
        # plt.ylim([1e-2, 2e2])    
        plt.legend(
                loc='lower right',
                frameon=True,
                framealpha=0.8,
                fontsize=8,
                fancybox=True,
                ncol=2,               # two columns
                handlelength=2.5,
                columnspacing=0.8
            )
        plt.tight_layout()
        plt.savefig(f'images_output/{folder_name}/{name}_mean_{error_keyword}_Rotation.pdf')
        plt.show()
    
    
    mean_L2error, std_L2error, nb_sample_L2 = extract_mean_std('L2error', sample_result_dict)
    plot_totalsample_errors('L2 Error', mean_L2error, std_L2error, nb_sample_L2)
    mean_totalerror, std_totalerror, nb_sample_total = extract_mean_std('Totalerror', sample_result_dict)
    plot_totalsample_errors('Total error', mean_totalerror, std_totalerror, nb_sample_total)
    