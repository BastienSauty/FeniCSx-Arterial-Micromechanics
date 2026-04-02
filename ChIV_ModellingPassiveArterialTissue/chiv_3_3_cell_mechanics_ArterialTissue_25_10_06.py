#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 12:13:46 2025
Updated July 

@author: bastien.sauty

Section 3.3 : impact of the parameters mechs on cellular mechanics
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
        
# Define a dataclass to manage the inputs to the plotter functions 
from dataclasses import dataclass
from typing import Any

@dataclass
class PlotConfig:
    lambdaz_list: np.ndarray
    press_list: np.ndarray
    f_cells: float
    f_coll_media: float
    n_phase1: int
    n_phase2: int
    indices_1: slice
    indices_2: slice
    
#-----------------------------------------------------------------------------#
# Specific sensitivity analysis : Matrix stiffness, collagen stiffness ?
#-----------------------------------------------------------------------------#

def pipeline_SA(p, name, folder_name, simu_card, media_card, adventitia_card, collagen_keys_media, collagen_keys_adventitia):
    """
    process to run the simulation for a given vector of parameters p
    p = [E_m, E_cells, E_coll, k_coll, lambda0_coll]
    - E_m : young modulus of the matrix
    - E_cells : young modulus of the cells
    - E_coll : young modulus of the collagen
    - k_coll : nonlinearity param collagen
    - lambda0_coll : "prestretch" in the exponential of the young modulus collagen
    """
    
    E_m, E_c, E_coll, k_coll, lambda_coll = p
    
    namefile = f"{name}_{E_m}_{E_c}_{E_coll}_{k_coll}_{lambda_coll}_scal.pkl"
    
    # Change E values
    media_card["matrix"]['young'] = p[0]
    adventitia_card["matrix"]['young'] = p[0]
    
    media_card["cells"]['young'] = p[1]
    

    for key_media in collagen_keys_media:
        media_card[key_media]['young'][0][0] = p[2]
        media_card[key_media]['young'][1][0] = p[3]
        media_card[key_media]['young'][2][0] = p[4]
    
    for key_adv in collagen_keys_adventitia:
        adventitia_card[key_adv]['young'][0][0] = p[2]
        adventitia_card[key_adv]['young'][1][0] = p[3]
        adventitia_card[key_adv]['young'][2][0] = p[4]
        
    # Set up unique cache folder for this simulation
    name_simu = f"{name}_{E_m}_{E_c}_{E_coll}_{k_coll}_{lambda_coll}"
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
    
    try:
        file = open(f'./outputs/{folder_name}/{namefile}', 'rb')
        result = pickle.load(file) # load and store pkl file
        file.close()
            
    except FileNotFoundError:
        print(f"[!] File not found: ./outputs/{folder_name}/{namefile}")
        print(f"Running simulation {name_simu}")
        
        # Run simulation
        result, mech = run_simulation(name_simu, folder_name, simu_card, adventitia_card, media_card)
    except Exception as e:
        print(f"[!] Error while loading result file:")
        print(f"    File: ./outputs/{folder_name}/{namefile}")
        print(f"    Error: {e}")

    return(result)

#-----------------------------------------------------------------------------#
# Plotting function
#-----------------------------------------------------------------------------#

def plot_pressure_radius(folder_name: str, name: str, param_name:str, list_result: list, param_list:list, config: PlotConfig):
    """
    Plotting function to plot the pressure radius curve for the different results obtained during the sensitivity analysis
    param_name : string that is the name of the parameter studied in this sensitivity analysis, ie E_m
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(5,3),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={'width_ratios': [config.n_phase1, config.n_phase2], 'wspace': 0.05}  # tighten space
    )
    
    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]
    
    for i, result in enumerate(list_result):
        color = colors[i]
        ri_d = result.outputs['ri_d'][:]
        re_d = result.outputs['re_d'][:]
        # axial stretch
        ax1.plot(lambdaz_list[config.indices_1], ri_d[config.indices_1], color=color, linestyle='-', linewidth=1.5)
        ax1.plot(lambdaz_list[config.indices_1], re_d[config.indices_1], color=color, linestyle='--', linewidth=1.5)
    
        ax2.plot(press_list[config.indices_2], ri_d[config.indices_2], color=color, linestyle='-', linewidth=1.5)
        ax2.plot(press_list[config.indices_2], re_d[config.indices_2], color=color, linestyle='--', linewidth=1.5)
        
    # Phase 1: Axial stretch
    # ---------------------------------------------------------------
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel("Radius [mm]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    # Phase 2: Inflation
    # ---------------------------------------------------------------    
    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.3))
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
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2,
               label=rf'{param_list[i]}')
        for i in range(len(list_result))
    ]
    
    ax2.legend(
        handles=legend_elements, 
        title=rf'${param_name}$ [MPa]',
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param_name}_pressure_radius.pdf')
    plt.show()


def plot_layer_stress(folder_name: str, name: str, param_name:str, list_result: list, param_list:list, stress_direction:str, config: PlotConfig):
    """
    Plotting function to plot the pressure radius curve for the different results obtained during the sensitivity analysis
    param_name : string that is the name of the parameter studied in this sensitivity analysis, ie E_m
    stress_direction : Circumferential or Axial
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(5,3),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={'width_ratios': [config.n_phase1, config.n_phase2], 'wspace': 0.05}  # tighten space
    )
    
    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]
    
    for i, result in enumerate(list_result):
        color = colors[i]
        if stress_direction=='Circumferential':
            S_yy_adv = result.outputs['S_yy_avg_adv'][:]
            S_yy_media = result.outputs['S_yy_avg_media'][:]
        elif stress_direction=='Axial':
            S_yy_adv = result.outputs['S_zz_avg_adv'][:]
            S_yy_media = result.outputs['S_zz_avg_media'][:]
        
        # axial stretch
        ax1.plot(lambdaz_list[config.indices_1], S_yy_media[config.indices_1], color=color, linestyle='-', linewidth=1.5)
        ax1.plot(lambdaz_list[config.indices_1], S_yy_adv[config.indices_1], color=color, linestyle='--', linewidth=1.5)
    
        ax2.plot(press_list[config.indices_2], S_yy_media[config.indices_2], color=color, linestyle='-', linewidth=1.5)
        ax2.plot(press_list[config.indices_2], S_yy_adv[config.indices_2], color=color, linestyle='--', linewidth=1.5)
        
    # Phase 1: Axial stretch
    # ---------------------------------------------------------------
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel(f"{stress_direction} stress [MPa]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    # Phase 2: Inflation
    # ---------------------------------------------------------------    
    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.3))
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Hide left y-axis ticks/labels on the right subplot (sharey=True keeps the scale)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)
    
    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'Media (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'Adventitia (dashed)')
    ]
    # Create and store the style legend first (so it doesn’t get overwritten)
    style_legend_box = ax2.legend(
        handles=style_legend,
        loc='best',
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
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2,
               label=rf'{param_list[i]}')
        for i in range(len(list_result))
    ]
    
    ax2.legend(
        handles=legend_elements, 
        title=rf'${param_name}$ [MPa]',
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param_name}_layer_{stress_direction}_stress.pdf')
    plt.show()
    
def plot_layer_distribution(folder_name: str, name: str, param_name:str, list_result: list, param_list:list, config: PlotConfig):
    """
    Plotting ratio of stresses between layers
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(5,3),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={'width_ratios': [config.n_phase1, config.n_phase2], 'wspace': 0.05}  # tighten space
    )
    
    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]
    
    for i, result in enumerate(list_result):
        color = colors[i]
    
        S_yy_adv = result.outputs['S_yy_avg_adv'][:]
        S_yy_media = result.outputs['S_yy_avg_media'][:]
        S_zz_adv = result.outputs['S_zz_avg_adv'][:]
        S_zz_media = result.outputs['S_zz_avg_media'][:]
    
        S_zz_props = S_zz_media/S_zz_adv
        S_yy_props = S_yy_media/S_yy_adv
        # axial stretch
        ax1.plot(lambdaz_list[config.indices_1], S_yy_props[config.indices_1], color=color, linestyle='-', linewidth=1.5)
        ax1.plot(lambdaz_list[config.indices_1], S_zz_props[config.indices_1], color=color, linestyle='--', linewidth=1.5)
    
        ax2.plot(press_list[config.indices_2], S_yy_props[config.indices_2], color=color, linestyle='-', linewidth=1.5)
        ax2.plot(press_list[config.indices_2], S_zz_props[config.indices_2], color=color, linestyle='--', linewidth=1.5)
        
    # Phase 1: Axial stretch
    # ---------------------------------------------------------------
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel(f"Ratio of the stress component in the media over the adventitia")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    ax1.set_ylim([-1,2])
    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    # Phase 2: Inflation
    # ---------------------------------------------------------------    
    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.3))
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Hide left y-axis ticks/labels on the right subplot (sharey=True keeps the scale)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)
    
    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'Circumferential (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'Axial (dashed)')
    ]
    # Create and store the style legend first (so it doesn’t get overwritten)
    style_legend_box = ax2.legend(
        handles=style_legend,
        loc='best',
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
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2,
               label=rf'{param_list[i]}')
        for i in range(len(list_result))
    ]
    
    ax2.legend(
        handles=legend_elements, 
        title=rf'${param_name}$ [MPa]',
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param_name}_layer_distribution_stress.pdf')
    plt.show()
    

def plot_circ_stress(folder_name: str, name: str, param_name:str, list_result: list, param_list:list, config: PlotConfig):
    """
    Plotting function to plot the circumferential stress curve for the different results obtained during the sensitivity analysis
    param_name : string that is the name of the parameter studied in this sensitivity analysis, ie E_m
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(5,3),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={'width_ratios': [config.n_phase1, config.n_phase2], 'wspace': 0.05}  # tighten space
    )
    
    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]
    
    for i, result in enumerate(list_result):
        color = colors[i]
        # Stresses
        
        stress_matrix = result.outputs['s_yy_matrix_avg']
        stress_cell = result.outputs['s_yy_cell_avg']
        stress_collagen = result.outputs['s_yy_collagen_media_avg']
        stress_hom = result.outputs['S_yy_avg_media']
        
        # axial stretch
        ax1.plot(lambdaz_list[config.indices_1], stress_matrix[config.indices_1], color=color, linestyle=':', linewidth=1.5)
        ax1.plot(lambdaz_list[config.indices_1], stress_cell[config.indices_1], color=color, linestyle='-', linewidth=1.5)
        ax1.plot(lambdaz_list[config.indices_1], stress_collagen[config.indices_1], color=color, linestyle='--', linewidth=1.5)
        ax1.plot(lambdaz_list[config.indices_1], stress_hom[config.indices_1], color=color, linestyle='-.', linewidth=1.5)
                
                
        ax2.plot(press_list[config.indices_2], stress_matrix[config.indices_2], color=color, linestyle=':', linewidth=1.5)
        ax2.plot(press_list[config.indices_2], stress_cell[config.indices_2], color=color, linestyle='-', linewidth=1.5)
        ax2.plot(press_list[config.indices_2], stress_collagen[config.indices_2], color=color, linestyle='--', linewidth=1.5)
        ax2.plot(press_list[config.indices_2], stress_hom[config.indices_2], color=color, linestyle='-.', linewidth=1.5)
        
        
    # Phase 1: Axial stretch
    # ---------------------------------------------------------------
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel("Circumferential stress [MPa]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    ax2.set_yscale('symlog', linthresh=1e-3)  # 'linthresh' defines the linear region around 0

    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    # Phase 2: Inflation
    # ---------------------------------------------------------------    
    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.3))
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Hide left y-axis ticks/labels on the right subplot (sharey=True keeps the scale)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)
    
    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle=':', label=r'$\sigma_{matrix}$ (dot)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'$\sigma_{smc}$ (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'$\sigma_{coll}$ (dashed)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='-.', label=r'$\sigma_{tissue}$ (dash-dot)')
    ]
    # Create and store the style legend first (so it doesn’t get overwritten)
    style_legend_box = ax2.legend(
        handles=style_legend,
        loc='best',
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
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2,
               label=rf'{param_list[i]}')
        for i in range(len(list_result))
    ]
    
    ax2.legend(
        handles=legend_elements, 
        title=rf'${param_name}$ [MPa]',
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param_name}_stress_distribution.pdf')
    plt.show()

def plot_normalized_stress(folder_name: str, name: str, param_name:str, list_result: list, param_list:list, stress_direction:str, config: PlotConfig):
    """
    Plotting function to plot the normalized stress curve for the different results obtained during the sensitivity analysis
    param_name : string that is the name of the parameter studied in this sensitivity analysis, ie E_m
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(5,3),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={'width_ratios': [config.n_phase1, config.n_phase2], 'wspace': 0.05}  # tighten space
    )
    
    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]
    
    f_matrix = 1 - config.f_cells - config.f_coll_media
    
    for i, result in enumerate(list_result):
        color = colors[i]
        # Proportions
        if stress_direction=='Axial':
            stress_hom = result.outputs['S_zz_avg_media']
            stress_matrix_prop = f_matrix*result.outputs['s_zz_matrix_avg']/stress_hom
            stress_cell_prop = config.f_cells*result.outputs['s_zz_cell_avg']/stress_hom
            stress_collagen_prop = config.f_coll_media*result.outputs['s_zz_collagen_media_avg']/stress_hom
            stress_hom_prop = stress_hom/stress_hom
            
            # axial stretch
            ax1.plot(lambdaz_list[config.indices_1], stress_cell_prop[config.indices_1], color=color, linestyle='-', linewidth=1.5)
            ax1.plot(lambdaz_list[config.indices_1], stress_collagen_prop[config.indices_1], color=color, linestyle='--', linewidth=1.5)
            ax1.plot(lambdaz_list[config.indices_1], stress_matrix_prop[config.indices_1], color=color, linestyle=':', linewidth=1.5)
                    
        elif stress_direction=='Circumferential':
            stress_hom = result.outputs['S_yy_avg_media']
            stress_matrix_prop = f_matrix*result.outputs['s_yy_matrix_avg']/stress_hom
            stress_cell_prop = config.f_cells*result.outputs['s_yy_cell_avg']/stress_hom
            stress_collagen_prop = config.f_coll_media*result.outputs['s_yy_collagen_media_avg']/stress_hom
            stress_hom_prop = stress_hom/stress_hom
    
                
        ax2.plot(press_list[config.indices_2], stress_cell_prop[config.indices_2], color=color, linestyle='-', linewidth=1.5)
        ax2.plot(press_list[config.indices_2], stress_collagen_prop[config.indices_2], color=color, linestyle='--', linewidth=1.5)
        ax2.plot(press_list[config.indices_2], stress_matrix_prop[config.indices_2], color=color, linestyle=':', linewidth=1.5)
        
                
        
    # Phase 1: Axial stretch
    # ---------------------------------------------------------------
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel(f"Normalized {stress_direction} stress")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    ax1.set_ylim([0,1])
    if stress_direction=='Circumferential':
        ax1.fill_between(
            lambdaz_list[indices_1],
            0, 1,
            color='gray',
            alpha=0.3,
            hatch='//'
        )
    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    # Phase 2: Inflation
    # ---------------------------------------------------------------    
    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.3))
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Hide left y-axis ticks/labels on the right subplot (sharey=True keeps the scale)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)
    
    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'$\sigma_{smc}$ (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'$\sigma_{coll}$ (dashed)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle=':', label=r'$\sigma_{matrix}$ (dot)')
    ]
    # Create and store the style legend first (so it doesn’t get overwritten)
    style_legend_box = ax2.legend(
        handles=style_legend,
        loc='best',
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
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2,
               label=rf'{param_list[i]}')
        for i in range(len(list_result))
    ]
    
    ax2.legend(
        handles=legend_elements, 
        title=rf'${param_name}$ [MPa]',
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param_name}_stress_{stress_direction}_distribution.pdf')
    plt.show()
    
    

def plot_fiber_orientation(folder_name: str, name: str, param_name:str, list_result: list, param_list:list, config: PlotConfig):
    """
    Plotting function to plot the center fiber angle for the different results obtained during the sensitivity analysis
    param_name : string that is the name of the parameter studied in this sensitivity analysis, ie E_m
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(5,3),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={'width_ratios': [config.n_phase1, config.n_phase2], 'wspace': 0.05}  # tighten space
    )
    
    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]
    
    for i, result in enumerate(list_result):
        color = colors[i]
        # Proportions
        theta_coll = result.outputs['collagen_4_adv_theta']
        
        # axial stretch
        ax1.plot(lambdaz_list[config.indices_1], theta_coll[config.indices_1], color=color, linestyle='-', linewidth=1.5)
        ax1.plot(lambdaz_list[config.indices_1], theta_coll[config.indices_1], color=color, linestyle='--', linewidth=1.5)
                
        ax2.plot(press_list[config.indices_2], theta_coll[config.indices_2], color=color, linestyle='-', linewidth=1.5)
        ax2.plot(press_list[config.indices_2], theta_coll[config.indices_2], color=color, linestyle='--', linewidth=1.5)
        # ax2.plot(press_list[config.indices_2], stress_hom_prop[config.indices_2], label=r'Tissue')
        
        
    # Phase 1: Axial stretch
    # ---------------------------------------------------------------
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel("Fiber Angle")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    # Phase 2: Inflation
    # ---------------------------------------------------------------    
    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.3))
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Hide left y-axis ticks/labels on the right subplot (sharey=True keeps the scale)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)
    
    # --- 2️⃣ Main legend (N values) OUTSIDE ax2 ---
    legend_elements = [
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2,
               label=rf'{param_list[i]}')
        for i in range(len(list_result))
    ]
    
    ax2.legend(
        handles=legend_elements, 
        title=rf'${param_name}$ [MPa]',
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param_name}_fiber_angle.pdf')
    plt.show()
    
    
#%%


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

    folder_name = 'ChIV.3.3.cell_mechanics/'
    name = 'cell_mech_vf'

    simu_card_name = 'json_cards/simu_card_calib.json'
    media_card_name ='json_cards/media_card_calib.json'
    adventitia_card_name ='json_cards/adventitia_card_calib.json'
    # Load material card        
    simu_card = load_JSON(simu_card_name)
    media_card = load_JSON(media_card_name)
    adventitia_card = load_JSON(adventitia_card_name)
    
    keys = media_card.keys()
    collagen_keys_media = [k for k in keys if k.startswith("collagen")]
    keys = adventitia_card.keys()
    collagen_keys_adventitia = [k for k in keys if k.startswith("collagen")]
        
    simu_card['XDMF_export'] = 0
    load_phase = simu_card['load_phase']
    step_load = Artery_load(load_phase)    

    pipeline_partial = partial(
                                pipeline_SA,
                                name=name,
                                folder_name=folder_name,
                                simu_card=simu_card,
                                media_card=media_card,
                                adventitia_card=adventitia_card,
                                collagen_keys_media=collagen_keys_media,
                                collagen_keys_adventitia=collagen_keys_adventitia                                
                            )
    
    E_m0, E_c0, E_coll0, k_coll0, lambda_coll0 = [0.05, 0.01, 0.67, 3.64, 1.1]  # E_m, E_cells, E_coll, k_coll, lambda0_coll
    E_m_list = [0.025, 0.04, 0.05, 0.06, 0.08]
    p_list = [[E_m, E_c0, E_coll0, k_coll0, lambda_coll0] for E_m in E_m_list] # first change only Em
    id_Em = slice(0, len(E_m_list))
    
    E_c_list = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    p_list += [[E_m0, E_c, E_coll0, k_coll0, lambda_coll0] for E_c in E_c_list] # then change only Ec
    id_Ec = slice(len(E_m_list), len(p_list))
    
    E_coll_list = [0.5, 0.67, 0.75, 1.0]
    p_list += [[E_m0, E_c0, E_coll, k_coll0, lambda_coll0] for E_coll in E_coll_list] # then change only Ec
    id_Ecoll = slice(len(p_list)-len(E_coll_list), len(p_list))
    
    
    with mp.Pool(processes=5) as pool:
        all_results = pool.map(pipeline_partial, p_list)
    
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
    
    f_cells = media_card['cells']['volumic_fraction']
    f_coll_media = 4*media_card['collagen0']['volumic_fraction']
        
    config = PlotConfig(lambdaz_list=lambdaz_list,
                        press_list=press_list,
                        f_cells=f_cells,
                        f_coll_media=f_coll_media,
                        n_phase1=n_phase1,
                        n_phase2=n_phase2,
                        indices_1=indices_1,
                        indices_2=indices_2)
    
    # ---------------------------------------------------------------
    # Pressure-Radius
    # ---------------------------------------------------------------
    plot_pressure_radius(folder_name, name, 'E_m', all_results[id_Em], E_m_list, config)
    plot_pressure_radius(folder_name, name, 'E_{smc}', all_results[id_Ec], E_c_list, config)
    plot_pressure_radius(folder_name, name, 'E_{coll}', all_results[id_Ecoll], E_coll_list, config)
    
    #%%
    # ---------------------------------------------------------------
    # Layer stress
    # ---------------------------------------------------------------
    plot_layer_stress(folder_name, name, 'E_m', all_results[id_Em], E_m_list, 'Circumferential', config)
    plot_layer_stress(folder_name, name, 'E_{smc}', all_results[id_Ec], E_c_list, 'Circumferential', config)
    plot_layer_stress(folder_name, name, 'E_{coll}', all_results[id_Ecoll], E_coll_list, 'Circumferential',config)
    
    # axial stress
    plot_layer_stress(folder_name, name, 'E_m', all_results[id_Em], E_m_list, 'Axial', config)
    plot_layer_stress(folder_name, name, 'E_{smc}', all_results[id_Ec], E_c_list, 'Axial', config)
    plot_layer_stress(folder_name, name, 'E_{coll}', all_results[id_Ecoll], E_coll_list, 'Axial',config)
    
    #%%
    # # Distribution
    # plot_layer_distribution(folder_name, name, 'E_m', all_results[id_Em], E_m_list, config)
    # plot_layer_distribution(folder_name, name, 'E_{smc}', all_results[id_Ec], E_c_list, config)
    # plot_layer_distribution(folder_name, name, 'E_{coll}', all_results[id_Ecoll], E_coll_list, config)
    
    #%%
    # ---------------------------------------------------------------
    # stress distribtuion
    # ---------------------------------------------------------------
    plot_circ_stress(folder_name, name, 'E_m', all_results[id_Em], E_m_list, config)
    plot_circ_stress(folder_name, name, 'E_{smc}', all_results[id_Ec], E_c_list, config)
    plot_circ_stress(folder_name, name, 'E_{coll}', all_results[id_Ecoll], E_coll_list, config)
    
    #%%
    # ---------------------------------------------------------------
    # normalized stress
    # ---------------------------------------------------------------
    plot_normalized_stress(folder_name, name, 'E_m', all_results[id_Em], E_m_list, 'Axial', config)
    plot_normalized_stress(folder_name, name, 'E_{smc}', all_results[id_Ec], E_c_list, 'Axial', config)
    plot_normalized_stress(folder_name, name, 'E_{coll}', all_results[id_Ecoll], E_coll_list, 'Axial', config)
    
    plot_normalized_stress(folder_name, name, 'E_m', all_results[id_Em], E_m_list, 'Circumferential', config)
    plot_normalized_stress(folder_name, name, 'E_{smc}', all_results[id_Ec], E_c_list, 'Circumferential', config)
    plot_normalized_stress(folder_name, name, 'E_{coll}', all_results[id_Ecoll], E_coll_list, 'Circumferential', config)
    
    #%%
    # ---------------------------------------------------------------
    # fiber angle
    # ---------------------------------------------------------------
    plot_fiber_orientation(folder_name, name, 'E_m', all_results[id_Em], E_m_list, config)
    plot_fiber_orientation(folder_name, name, 'E_{smc}', all_results[id_Ec], E_c_list, config)
    plot_fiber_orientation(folder_name, name, 'E_{coll}', all_results[id_Ecoll], E_coll_list, config)
    
