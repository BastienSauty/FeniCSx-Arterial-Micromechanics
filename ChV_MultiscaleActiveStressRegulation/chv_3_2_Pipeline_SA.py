#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 15:47:37 2025

@author: bastien.sauty

Define the pipeline for the sensitivity analysis of the vasoconstrictive model. 
For one set of parameters p :
    - first simulation with passive behavior to obtain tau_b_ref, ie the average stress inside the cells
    - run a sensitivity analysis with varying tau_b in [0.1 tau_b_ref, 10 tau_b_ref]
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
import os
import copy

import sys
import subprocess
import tempfile
import shutil


import functools
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed


import matplotlib as mpl
import matplotlib.cm as cm

from dataclasses import dataclass

@dataclass(frozen=True)
class ParameterSpec:
    key: str            # internal / filename-safe name (e.g. "tau")
    label: str          # LaTeX label (e.g. r"\tau")
    unit: str | None    # unit string (e.g. "MPa", "s", None)


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

from Multiscale_Framework.class_modules.load_class import Artery_load
from ChV_MultiscaleActiveStressRegulation.chv_3_2_simulation_worker import pipeline_Vcn_result


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
        
def cleanup_jit_cache(cache_root="./jit_cache"):
    """
    Removes the entire jit_cache directory and recreates it empty.
    """
    if os.path.exists(cache_root):
        print(f"--> [CLEANUP] Removing all temporary JIT folders in {cache_root}...")
        try:
            shutil.rmtree(cache_root)
            os.makedirs(cache_root, exist_ok=True)
            print("    [OK] jit_cache cleared.")
        except Exception as e:
            print(f"    [ERROR] Could not fully clear jit_cache: {e}")


#%% 
# Plotting function 

def plot_error_stress_cell_basal_from_pipelines(folder_name: str, name: str, param: ParameterSpec, pipelines: list, param_list: list, delta: float = 0.05):
    """
    Balanced version: Major grid only for clarity, with vertical dashed lines 
    restored to clearly define the boundaries of the shaded zones.
    """
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    
    n = len(pipelines)
    cmap = plt.cm.get_cmap('viridis', n)
    colors = [cmap(i / (n - 1)) if n > 1 else cmap(0.5) for i in range(n)]
    
    for p_id, pipeline in enumerate(pipelines):
        color = colors[p_id]
        tau = pipeline.tau_list
        err = pipeline.error_cell_stress_basal
        
        # 1. Shaded Functioning Range (zorder 1 - background)
        if hasattr(pipeline, "tau_range") and pipeline.tau_range is not None:
            t_left, t_right = pipeline.tau_range
            ax.axvspan(t_left, t_right, color=color, alpha=0.07, zorder=1)
            
            # 2. Re-introducing Vertical Boundary Lines (zorder 2)
            # We use a very thin dashed line to mark the limits clearly
            ax.axvline(t_left, color=color, linestyle='--', linewidth=0.8, alpha=0.3, zorder=2)
            ax.axvline(t_right, color=color, linestyle='--', linewidth=0.8, alpha=0.3, zorder=2)

        # 3. Main Error Curve (zorder 4 - top)
        ax.semilogy(
            tau, err,
            linewidth=1.8,
            color=color,
            alpha=0.9,
            zorder=4
        )

    # 4. Tolerance Line (zorder 3)
    ax.axhline(delta, color='black', linestyle=':', linewidth=1.2, alpha=0.8, zorder=3)

    # 5. Axes Formatting
    ax.set_ylabel(r"Error to target cell stress $[-]$")
    ax.set_xlabel(r"$\tau$ [MPa]")
    
    # Aesthetic matching your reference layout
    ax.set_facecolor((0.83, 0.83, 0.83, 0.4)) 
    
    # FIX: Major grid only (prevents minor-log horizontal line chaos)
    ax.grid(True, which='major', linestyle=":", linewidth=0.6, color='black', alpha=0.2)
    
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_ylim(bottom=1e-5, top=10) # Set a fixed bottom to keep 'V' shapes clean

    # 6. Dual Legend System
    # Internal: Tolerance
    tolerance_handle = [Line2D([0], [0], color='black', linestyle=':', linewidth=1.2, label=r'$\pm 5\%$ tolerance')]
    tol_leg = ax.legend(
        handles=tolerance_handle, 
        loc='lower left', 
        fontsize=8, 
        frameon=True, 
        framealpha=0.9, 
        facecolor='white', 
        edgecolor='gray'
    )
    ax.add_artist(tol_leg)

    # External: Parameter values
    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=2, label=rf'{param_list[i]}')
        for i in range(len(pipelines))
    ]
    
    ax.legend(
        handles=legend_elements, 
        title=rf"${param.label}$ [{param.unit}]",
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )

    save_path = f'images_output/{folder_name}/{name}_{param.key}_error_stress_cell_basal_final.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

#%%

def plot_tau_range_vs_parameter(folder_name: str, name: str, param: ParameterSpec, pipelines: list, param_list: list):
    """
    Plots the functioning range [tau_min, tau_max] with LaTeX formatting 
    and internal legend box.
    """
    # 1. Data Processing
    extracted_data = []
    for i, pipeline in enumerate(pipelines):
        if hasattr(pipeline, "tau_range") and pipeline.tau_range is not None:
            extracted_data.append((
                param_list[i], 
                pipeline.tau_range[0], 
                pipeline.tau_range[1], 
                pipeline.tau_b_ref
            ))
    
    extracted_data.sort(key=lambda x: x[0])
    p_vals = np.array([x[0] for x in extracted_data])
    t_mins = np.array([x[1] for x in extracted_data])
    t_maxs = np.array([x[2] for x in extracted_data])
    t_refs = np.array([x[3] for x in extracted_data])

    # 2. Plot Setup (4x3 matches your inelastic stretch reference)
    fig, ax1 = plt.subplots(figsize=(4, 3), constrained_layout=True)
    
    # 3. Plotting
    # Shading
    ax1.fill_between(p_vals, t_mins, t_maxs, color='gray', alpha=0.15, zorder=1)
    
    # Reference Stress - Diamond markers
    ax1.plot(p_vals, t_refs, 'k-D', linewidth=1.8, markersize=4, 
             label=r'$\tau_{b}^{ref}$', zorder=4)
    
    # Boundary Lines - Circle and Square markers
    ax1.plot(p_vals, t_mins, color='tab:blue', linestyle='--', marker='o', 
             linewidth=1.5, markersize=3.5, label=r'$\tau_{b}^{min}$', zorder=3)
             
    ax1.plot(p_vals, t_maxs, color='tab:orange', linestyle='--', marker='s', 
             linewidth=1.5, markersize=3.5, label=r'$\tau_{b}^{max}$', zorder=2)

    # 4. Formatting
    ax1.autoscale(enable=True, axis='x', tight=True)
    
    # X-axis label with LaTeX underscore for E_c if applicable
    # (Assuming param.label contains 'E_c', it will render as E_c in Computer Modern)
    ax1.set_xlabel(rf"${param.label}$ [{param.unit}]")
    ax1.set_ylabel(r"$\tau$ [MPa]")
    
    # Aesthetics: Darker grid and gray background
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4)) 
    ax1.grid(True, which='both', linestyle=":", linewidth=0.6, color='black', alpha=0.3)
    
    # 5. Legend (Styled Box in lower right)
    ax1.legend(
        loc='best',
        frameon=True,
        framealpha=0.9,
        facecolor='white',
        edgecolor='gray',
        fancybox=True,
        borderpad=0.5,
        fontsize=9
    )

    # 6. Save and Show
    save_path = f'images_output/{folder_name}/{name}_{param.key}_tau_sensitivity.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
#%%
def plot_radius_range_sensitivity(folder_name: str, name: str, param: ParameterSpec, pipelines: list, param_list: list):
    """
    Plots the sensitivity of the basal radius functioning range [r_min, r_max]
    and the specific radius at the exact tau_b_ref.
    Matches the LaTeX formatting and layout of the tau_range plot.
    """
    # 1. Data Processing
    extracted_data = []
    for i, pipeline in enumerate(pipelines):
        if hasattr(pipeline, "r_range") and pipeline.r_range is not None:
            extracted_data.append((
                param_list[i], 
                pipeline.r_range[0],   # radius at tau_min
                pipeline.r_range[1],   # radius at tau_max
                pipeline.r_b_ref       # radius at exact tau_b_ref
            ))
    
    # Sort data by the parameter (x-axis)
    extracted_data.sort(key=lambda x: x[0])
    
    p_vals = np.array([x[0] for x in extracted_data])
    r_mins = np.array([x[1] for x in extracted_data])
    r_maxs = np.array([x[2] for x in extracted_data])
    r_refs = np.array([x[3] for x in extracted_data])
    
    # 2. Plot Setup (4x3 to match the inelastic stretch and tau plots)
    fig, ax1 = plt.subplots(figsize=(4, 3), constrained_layout=True)
    
    # 3. Plotting
    # Shading the range (Safe Zone)
    ax1.fill_between(p_vals, r_mins, r_maxs, color='gray', alpha=0.15, zorder=1)
    
    # Reference Radius - Solid Black with Diamond markers
    ax1.plot(p_vals, r_refs, 'k-D', linewidth=1.8, markersize=4, 
             label=r'$r(\tau_{b}^{ref})$', zorder=4)
    
    # Boundary Lines - Dashed with Circle and Square markers
    # We use blue/orange to keep consistency with the tau plots
    ax1.plot(p_vals, r_mins, color='tab:blue', linestyle='--', marker='o', 
             linewidth=1.2, markersize=3.5, label=r'$r(\tau_{b}^{min})$', zorder=3)
             
    ax1.plot(p_vals, r_maxs, color='tab:orange', linestyle='--', marker='s', 
             linewidth=1.2, markersize=3.5, label=r'$r(\tau_{b}^{max})$', zorder=2)

    # 4. Formatting
    ax1.autoscale(enable=True, axis='x', tight=True)
    
    # Axis labels with LaTeX
    ax1.set_xlabel(rf"${param.label}$ [{param.unit}]")
    ax1.set_ylabel(r"Basal Radius $r_i$ [mm]")
    
    # Aesthetics: Gray background and dark visible grid
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4)) 
    ax1.grid(True, which='both', linestyle=":", linewidth=0.6, color='black', alpha=0.3)
    
    # 5. Legend (Styled Box in lower right, inside the plot)
    ax1.legend(
        loc='best',
        frameon=True,
        framealpha=0.9,
        facecolor='white',
        edgecolor='gray',
        fancybox=True,
        borderpad=0.5,
        fontsize=9
    )

    # 6. Save and Show
    save_path = f'images_output/{folder_name}/{name}_{param.key}_radius_sensitivity.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
def plot_lambda_range_vs_parameter(folder_name, name, param, pipelines, param_list):
    """
    Plots the functioning range of inelastic cell stretch [lambda_min, lambda_max].
    """
    extracted_data = []
    for i, pipeline in enumerate(pipelines):
        if hasattr(pipeline, "lambda_range") and pipeline.lambda_range is not None:
            extracted_data.append((
                param_list[i], 
                pipeline.lambda_range[0], 
                pipeline.lambda_range[1], 
                pipeline.lambda_cell_in_ref
            ))
    
    extracted_data.sort(key=lambda x: x[0])
    p_vals = np.array([x[0] for x in extracted_data])
    l_mins = np.array([x[1] for x in extracted_data])
    l_maxs = np.array([x[2] for x in extracted_data])
    l_refs = np.array([x[3] for x in extracted_data])

    fig, ax1 = plt.subplots(figsize=(4, 3), constrained_layout=True)
    
    # Shading and Plotting
    ax1.fill_between(p_vals, l_mins, l_maxs, color='gray', alpha=0.15)
    ax1.plot(p_vals, l_refs, 'k-D', linewidth=1.8, markersize=4, label=r'$\lambda_{in}^{ref}$')
    ax1.plot(p_vals, l_mins, color='tab:blue', linestyle='--', marker='o', markersize=3.5, label=r'$\lambda_{in}^{min}$')
    ax1.plot(p_vals, l_maxs, color='tab:orange', linestyle='--', marker='s', markersize=3.5, label=r'$\lambda_{in}^{max}$')

    # Formatting
    ax1.set_xlabel(rf"${param.label}$ [{param.unit}]")
    ax1.set_ylabel(r"Inelastic Stretch $\lambda_{in}$ [-]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4)) 
    ax1.grid(True, which='both', linestyle=":", linewidth=0.6, color='black', alpha=0.3)
    ax1.legend(loc='best', frameon=True, fontsize=9)

    plt.savefig(f'images_output/{folder_name}/{name}_{param.key}_lambda_sensitivity.pdf')
    plt.show()
    
#%%
def plot_stress_components_vs_parameter(folder_name, name, param, pipelines, param_list):
    """
    Plots the functioning range [min, max] and reference values for 
    Total Media, Collagen, Matrix, and Cell stress components.
    """
    extracted_data = []
    for i, pipeline in enumerate(pipelines):
        if hasattr(pipeline, "S_yy_media_range"):
            extracted_data.append({
                'p_val': param_list[i],
                'total': (pipeline.S_yy_media_range, pipeline.S_yy_media_ref),
                'coll':  (pipeline.s_yy_collagen_range, pipeline.s_yy_collagen[i]),
                'mat':   (pipeline.s_yy_matrix_range, pipeline.s_yy_matrix[i]),
                'cell':  (pipeline.s_yy_cell_range, pipeline.s_yy_cell[i])
            })

    # Sort by parameter value
    extracted_data.sort(key=lambda x: x['p_val'])
    p_vals = np.array([x['p_val'] for x in extracted_data])

    fig, ax1 = plt.subplots(figsize=(4, 3), constrained_layout=True)

    # Define components to plot: (key, label, color)
    components = [
        ('total', 'Total',    'black'),
        ('coll',  'Collagen', 'tab:green'),
        ('mat',   'Matrix',   'tab:red'),
        ('cell',  'Cell',     'tab:purple')
    ]

    for key, label, color in components:
        # Extract arrays for this component
        mins = np.array([x[key][0][0] for x in extracted_data])
        maxs = np.array([x[key][0][1] for x in extracted_data])
        refs = np.array([x[key][1] for x in extracted_data])

        # Shaded Area
        ax1.fill_between(p_vals, mins, maxs, color=color, alpha=0.1, zorder=2)
        
        # Min/Max Dashed Lines
        ax1.plot(p_vals, mins, color=color, linestyle='--', linewidth=0.8, alpha=0.7, zorder=3)
        ax1.plot(p_vals, maxs, color=color, linestyle='--', linewidth=0.8, alpha=0.7, zorder=3)
        
        # Reference Line (Solid)
        ax1.plot(p_vals, refs, color=color, linestyle='-', linewidth=2, label=rf'$S_{{yy}}^{{{label}}}$', zorder=4)

    # Formatting
    ax1.set_xlabel(rf"${param.label}$ [{param.unit}]")
    ax1.set_ylabel(r"Circumferential Stress $\sigma_{\theta\theta}$ [MPa]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4)) 
    ax1.grid(True, which='both', linestyle=":", linewidth=0.6, color='black', alpha=0.3)
    
    # Legend
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=True, fontsize=9, edgecolor='gray')

    # Save and Show
    save_path = f'images_output/{folder_name}/{name}_{param.key}_all_stresses_ranges.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
#%%
def plot_radial_profile_single(folder_name, name, param, pipelines, param_list, index_basal, mode='ref'):
    """
    Plots a single radial distribution of lambda_in.
    Modes: 'min' (tau_b_min), 'ref' (tau_b_ref), 'max' (tau_b_max)
    """
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    
    # 1. Setup Colors and Sorting
    colors = cm.viridis(np.linspace(0, 1, len(pipelines)))
    combined = sorted(zip(param_list, pipelines, colors), key=lambda x: x[0])
    
    sorted_params = [x[0] for x in combined]
    sorted_pipelines = [x[1] for x in combined]
    sorted_colors = [x[2] for x in combined]

    # 2. Map mode to title and index logic
    mode_map = {
        'min': {'title': r'Min Constriction ($\tau_{b}^{min}$)', 'linestyle': ':', 'suffix': 'taumin'},
        'ref': {'title': r'Baseline ($\tau_{b}^{ref}$)', 'linestyle': '-', 'suffix': 'tauref'},
        'max': {'title': r'Max Constriction ($\tau_{b}^{max}$)', 'linestyle': '--', 'suffix': 'taumax'}
    }
    
    current_mode = mode_map[mode]

    # 3. Plotting Loop
    for i, pipeline in enumerate(sorted_pipelines):
        # Determine the correct simulation index based on the requested mode
        if mode == 'min':
            sim_idx = pipeline.id_range[0]
        elif mode == 'max':
            sim_idx = pipeline.id_range[-1]
        else: # 'ref'
            sim_idx = np.argmin(np.abs(pipeline.tau_list - pipeline.tau_b_ref))
        
        result = pipeline.SA_results[sim_idx]
        r_pos = result.dict_outputs['S_yy']['points']
        
        # Normalize radius: 0 = r_i, 1 = r_i,adv
        r_norm = (r_pos - r_pos[0]) / (r_pos[-1] - r_pos[0])
        l_in_profile = result.outputs['lambda_cell_in'][index_basal, :]
        
        ax.plot(r_norm, l_in_profile, 
                color=sorted_colors[i], 
                linestyle=current_mode['linestyle'], 
                linewidth=2.0, 
                zorder=3)

    # 4. Formatting
    ax.set_title(current_mode['title'], fontsize=11)
    ax.set_xlabel(r"Normalized Radius ($r_i \to r_{i,adv}$)")
    ax.set_ylabel(r"Inelastic Stretch $\lambda_{in}$ [-]")
    
    ax.margins(y=0.15)
    ax.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax.grid(True, linestyle=":", color='black', alpha=0.3)

    # 5. External Legend
    legend_elements = [
        Line2D([0], [0], color=sorted_colors[i], lw=2.5, label=f'{sorted_params[i]:.2g}')
        for i in range(len(sorted_params))
    ]
    
    ax.legend(
        handles=legend_elements, 
        title=rf"${param.label}$ [{param.unit}]",
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        fontsize=8
    )

    # 6. Save
    save_path = f'images_output/{folder_name}/{name}_{param.key}_profile_{current_mode["suffix"]}.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close(fig) # Close to avoid memory issues in loops
    
#%%
def plot_cell_passive_stress(folder_name, name, param, pipelines, param_list, index_basal):
    """
    Plots a single radial distribution of lambda_in.
    Modes: 'min' (tau_b_min), 'ref' (tau_b_ref), 'max' (tau_b_max)
    """
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    
    # 1. Setup Colors and Sorting
    colors = cm.viridis(np.linspace(0, 1, len(pipelines)))
    combined = sorted(zip(param_list, pipelines, colors), key=lambda x: x[0])
    
    sorted_params = [x[0] for x in combined]
    sorted_pipelines = [x[1] for x in combined]
    sorted_colors = [x[2] for x in combined]


    # 3. Plotting Loop
    for i, pipeline in enumerate(sorted_pipelines):
        # Determine the correct simulation index based on the requested mode
        if mode == 'min':
            sim_idx = pipeline.id_range[0]
        elif mode == 'max':
            sim_idx = pipeline.id_range[-1]
        else: # 'ref'
            sim_idx = np.argmin(np.abs(pipeline.tau_list - pipeline.tau_b_ref))
        
        result = pipeline.SA_results[sim_idx]
        r_pos = result.dict_outputs['S_yy']['points']
        
        # Normalize radius: 0 = r_i, 1 = r_i,adv
        r_norm = (r_pos - r_pos[0]) / (r_pos[-1] - r_pos[0])
        l_in_profile = result.outputs['s_yy_cell'][index_basal, :]
        
        ax.plot(r_norm, l_in_profile, 
                color=sorted_colors[i], 
                linestyle='-', 
                linewidth=2.0, 
                zorder=3)

    # 4. Formatting
    ax.set_title('Relaxed state')
    ax.set_xlabel(r"Normalized Radius ($r_i \to r_{i,adv}$)")
    ax.set_ylabel(r"Cell stress [MPa]")
    
    ax.margins(y=0.15)
    ax.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax.grid(True, linestyle=":", color='black', alpha=0.3)

    # 5. External Legend
    legend_elements = [
        Line2D([0], [0], color=sorted_colors[i], lw=2.5, label=f'{sorted_params[i]:.2g}')
        for i in range(len(sorted_params))
    ]
    
    ax.legend(
        handles=legend_elements, 
        title=rf"${param.label}$ [{param.unit}]",
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        fontsize=8
    )

    # 6. Save
    save_path = f'images_output/{folder_name}/{name}_{param.key}_cell_stress_relaxed.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close(fig) # Close to avoid memory issues in loops
        

#%%
def plot_stress_distrib(folder_name, name, param, pipelines, param_list, index_basal):
    # 1. Remove constrained_layout=True to prevent conflict with bbox_to_anchor
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    
    
    # ... [Setup Colors and Sorting - Same as before] ...
    colors = cm.viridis(np.linspace(0, 1, len(pipelines)))
    combined = sorted(zip(param_list, pipelines, colors), key=lambda x: x[0])
    sorted_params = [x[0] for x in combined]
    sorted_pipelines = [x[1] for x in combined]
    sorted_colors = [x[2] for x in combined]

    style_map = {
        's_yy_cell': {'ls': '-', 'label': 'Cell Stress'},
        'S_yy': {'ls': '--', 'label': 'Total Stress'},
        's_yy_collagen': {'ls': ':', 'label': 'Collagen Stress'}
    }

    # 2. Plotting Loop
    for i, pipeline in enumerate(sorted_pipelines):
        sim_idx = np.argmin(np.abs(pipeline.tau_list - pipeline.tau_b_ref))
        result = pipeline.SA_results[sim_idx]
        r_pos = result.dict_outputs['S_yy']['points']
        r_norm = (r_pos - r_pos[0]) / (r_pos[-1] - r_pos[0])
        
        for key, style in style_map.items():
            profile = result.outputs[key][index_basal, :]
            ax.plot(r_norm, profile, color=sorted_colors[i], 
                    linestyle=style['ls'], linewidth=1.8, zorder=3)

    # 3. Formatting
    ax.set_title(r"Baseline ($\tau_{b}^{ref}$)", fontsize=11)
    ax.set_xlabel(r"Normalized Radius ($r_i \to r_{i,adv}$)")
    ax.set_ylabel(r"Circumferential Stress [MPa]")
    ax.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax.grid(True, linestyle=":", color='black', alpha=0.3)

    # 4. Inner legend
    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'Cell stress (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'Media stress (dashed)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle=':', label=r'Collagen stress (dotted)')
    ]
    # Create and store the style legend first (so it doesn’t get overwritten)
    style_legend_box = ax.legend(
        handles=style_legend,
        loc='upper right',
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
    ax.add_artist(style_legend_box)
    
    # 5. External Legend
    legend_elements = [
        Line2D([0], [0], color=sorted_colors[i], lw=2.5, label=f'{sorted_params[i]:.2g}')
        for i in range(len(sorted_params))
    ]
    
    ax.legend(
        handles=legend_elements, 
        title=rf"${param.label}$ [{param.unit}]",
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        fontsize=8
    )

    # 6. Save - bbox_inches='tight' is critical here!
    save_path = f'images_output/{folder_name}/{name}_{param.key}_stress_distrib.pdf'
    plt.savefig(save_path, bbox_inches='tight') # This expands the "camera" to include the outer legend
    plt.show()
    plt.close(fig)

#%%
# SA process call the simulation worker for parallel computing

def cmd_worker_wrapper(p, name, folder_name, passive_simu_card, simu_card, 
                       media_card, adventitia_card, index_passive, 
                       index_relaxed, index_basal):
    """
    Launcher that creates a physical subprocess command to ensure 0% shared memory.
    Returns the reconstructed pipeline_result object to the main process.
    """
    
    E_m, E_c, f_c = p
    # Deepcopy to avoid modifying the original cards if this is in a loop
    local_media_card = copy.deepcopy(media_card)
    local_media_card["matrix"]['young'] = E_m
    local_media_card["cells"]['young'] = E_c
    local_media_card["cells"]['volumic_fraction'] = f_c
    
    name_simu_base = f"{name}_{E_m}_{E_c}_{f_c}"
    result_path = os.path.join("./outputs", folder_name, f"{name_simu_base}_FINAL_PIPELINE.pkl")
    
    # 2. Create the temporary file for the subprocess
    fd, tmp_path = tempfile.mkstemp(suffix=".pkl")
    try:
        with os.fdopen(fd, 'wb') as tmp:
            args = {
                "p_vals": p, 
                "name": name, 
                "folder_name": folder_name,
                "passive_simu_card": passive_simu_card, 
                "simu_card": simu_card,
                "media_card": local_media_card, # <--- Use the updated local copy
                "adventitia_card": adventitia_card,
                "index_passive": index_passive
            }
            pickle.dump(args, tmp)

        # 2. Execute the simulation_worker.py
        # We use sys.executable to ensure we use the same Python environment/Conda env
        print(f"--> [LAUNCHING SUBPROCESS] for {name_simu_base}")
        subprocess.run([sys.executable, "ChV_MultiscaleActiveStressRegulation/chv_3_2_simulation_worker.py", tmp_path], check=True)
        
        # 3. Load the final result saved by the worker
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Worker finished but {result_path} was not found.")
            
        with open(result_path, "rb") as f:
            pipeline_result = pickle.load(f)
        
        # 4. Post-process (These update the object in the main process memory)
        print(f"--> [RECOVERED] {name_simu_base}. Processing results...")
        pipeline_result.extract_results(index_relaxed, index_basal)
        pipeline_result.compute_functioning_range(delta=0.05)
        
        return pipeline_result

    except subprocess.CalledProcessError as e:
        print(f"--> [CRITICAL ERROR] Subprocess failed for {name_simu_base}: {e}")
        return None # Or raise, depending on how you want the main loop to handle failure
        
    finally:
        # Cleanup the temporary arguments file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    
    os.environ["FENICS_JIT_TIMEOUT"] = "600"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["CC"] = "gcc"
    
    # ---------------------------------------------------------------
    # Simulation parameters and directory 
    # ---------------------------------------------------------------

    folder_name = 'ChV.3.2.SensitivityAnalysis'
    name = 'Vcn'

    passive_simu_card_name = 'json_cards/passive_simu_card_VcnSA.json'
    simu_card_name = 'json_cards/simu_card_VcnSA.json'
    media_card_name ='json_cards/media_card_Vcn.json'
    adventitia_card_name ='json_cards/adventitia_card_Vcn.json'
    # Load material card        
    passive_simu_card = load_JSON(passive_simu_card_name)
    simu_card = load_JSON(simu_card_name)
    media_card = load_JSON(media_card_name)
    adventitia_card = load_JSON(adventitia_card_name)
    
    simu_card['XDMF_export'] = 0
    passive_simu_card['XDMF_export'] = 0
    
    # Get the index of 100 mmHg in the passive simulation -> to extract the tau_b_ref
    passive_load_phase = passive_simu_card['load_phase']
    passive_step_load = Artery_load(passive_load_phase)    
    passive_press_list = 7500.62*passive_step_load.list_P
    
    def find_index_press(press_list, value):
        list_indices = []
        for i, p in enumerate(press_list):
            if np.isclose(p, value):
                list_indices.append(i)
        return(list_indices)
    
    press_100 = find_index_press(passive_press_list, 100)
    index_passive = press_100[0]
    
    # same but for the active phase
    load_phase = simu_card['load_phase']
    step_load = Artery_load(load_phase)    
    press_list = 7500.62*step_load.list_P
    press_100 = find_index_press(press_list, 100)
    index_relaxed = press_100[0]
    index_basal = press_100[-2]
    
    #-----------------------------------------------------------------------------#
    # Define the Sensitivity analysis : set of parameters that change
    #-----------------------------------------------------------------------------#
    E_m0, E_c0, f_c0 = [0.05, 0.01, 0.24] # initial values for the SA
    
    E_m_list = [0.025, 0.05, 0.075, 0.1] # vector of changing values for matrix stiffness
    p_list = [[E_m, E_c0, f_c0] for E_m in E_m_list] # first change only Em
    
    E_m_param = ParameterSpec(
                            key="Em",
                            label=r"E_m",
                            unit="MPa"
                        )
    #-----------------------------------------------------------------------------#
    # Run the SA : E_m
    #-----------------------------------------------------------------------------#
    
    # 2. Prepare the Worker Function
    # We use partial to "freeze" the arguments that don't change (cards, folder, etc.)
    # This makes the executor call much cleaner.
    run_worker = functools.partial(
        cmd_worker_wrapper,
        name=name,
        folder_name=folder_name,
        passive_simu_card=passive_simu_card,
        simu_card=simu_card,
        media_card=media_card,
        adventitia_card=adventitia_card,
        index_passive=index_passive,
        index_relaxed=index_relaxed,
        index_basal=index_basal
    )

    # 3. Execution Phase
    print(f"--> [START] Launching Sensitivity Analysis for {E_m_param.key}")
    E_m_SA_results = []

    # max_workers=3 as requested. Each worker will manage 1 subprocess at a time.
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Map futures to their specific parameter set
        future_to_p = {executor.submit(run_worker, p): p for p in p_list}
        
        results_map = {} # Temporary dictionary to hold results keyed by their p-tuple
        for future in as_completed(future_to_p):
            p_tuple = tuple(future_to_p[future]) # Convert list to tuple for hashing
            try:
                pipeline_result = future.result()
                if pipeline_result is not None:
                    results_map[p_tuple] = pipeline_result
                    print(f"   [SUCCESS] Parameters {p_tuple} processed.")
            except Exception as exc:
                print(f"   [FAILURE] Parameters {p_tuple} generated an exception: {exc}")

    # --- STRICT SORTING ---
    # Reconstruct the list in the EXACT order of p_list
    E_m_SA_results = []
    for p in p_list:
        p_tuple = tuple(p)
        if p_tuple in results_map:
            E_m_SA_results.append(results_map[p_tuple])
        else:
            print(f"   [WARNING] No result found for {p_tuple}, skipping in final list.")

    # 1. Filter the parameter values to match successful simulations
    successful_E_m = [p[0] for p in p_list if tuple(p) in results_map]
    
    #%% 2. Plot Error Curves (expects the list of pipeline objects)
    plot_error_stress_cell_basal_from_pipelines(folder_name, name, E_m_param, E_m_SA_results, successful_E_m, delta=0.05)
    
    #%% 3. Plot Tau Range (sensitivity of the functioning range)
    plot_tau_range_vs_parameter( folder_name, name, E_m_param, E_m_SA_results, successful_E_m)
    
    #%% 4. Plot Radius Sensitivity
    plot_radius_range_sensitivity(folder_name, name, E_m_param, E_m_SA_results, successful_E_m)
    
    #%% 5. Plot inelastic strain sensitivity
    # plot_lambda_range_vs_parameter(folder_name, name, E_m_param, E_m_SA_results, successful_E_m)
    #%%
    # plot_stress_components_vs_parameter(folder_name, name, E_m_param, E_m_SA_results, successful_E_m)
    # plot_lambda_grad_range_sensitivity(folder_name, name, E_m_param, E_m_SA_results, successful_E_m)
    #%%
    for mode in ['min', 'ref', 'max']:
        plot_radial_profile_single(folder_name, name, E_m_param, E_m_SA_results, successful_E_m, index_basal, mode=mode)
        
    #%%
    plot_cell_passive_stress(folder_name, name, E_m_param, E_m_SA_results, successful_E_m, index_relaxed)
    plot_stress_distrib(folder_name, name, E_m_param, E_m_SA_results, successful_E_m, index_basal)
    #%% 
    #-----------------------------------------------------------------------------#
    # Run the SA : E_c
    #-----------------------------------------------------------------------------#
    
    E_m0, E_c0, f_c0 = [0.05, 0.01, 0.24] 
    
    E_c_list = [0.008, 0.01, 0.013, 0.017, 0.02, 0.03, 0.05] 
    p_c_list = [[E_m0, E_c, f_c0] for E_c in E_c_list] 
    
    E_c_param = ParameterSpec(
        key="Ec",
        label=r"E_c",
        unit="MPa"
    )
    
    # 1. Execution Phase (Parallel Subprocesses)
    print(f"--> [START] Launching Sensitivity Analysis for {E_c_param.key}")
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Submit all p_c_list tasks to the pool
        future_to_p = {executor.submit(run_worker, p): p for p in p_c_list}
        
        results_map_c = {} 
        for future in as_completed(future_to_p):
            p_tuple = tuple(future_to_p[future])
            try:
                pipeline_result = future.result()
                if pipeline_result is not None:
                    results_map_c[p_tuple] = pipeline_result
                    print(f"   [SUCCESS] Parameters {p_tuple} processed.")
            except Exception as exc:
                print(f"   [FAILURE] Parameters {p_tuple} generated an exception: {exc}")
    
    # 2. Strict Sorting & Filtering
    # Reconstruct the list in the exact order of E_c_list
    E_c_SA_results = []
    successful_E_c = []
    
    for p in p_c_list:
        p_tuple = tuple(p)
        if p_tuple in results_map_c:
            E_c_SA_results.append(results_map_c[p_tuple])
            successful_E_c.append(p[1]) # Extract the E_c value (index 1 in p)
        else:
            print(f"   [WARNING] No result found for {p_tuple}, skipping.")
    
    # 3. Plotting Phase
    # We use successful_E_c to ensure x-axis and legends match the available data
    #%%
    plot_error_stress_cell_basal_from_pipelines(folder_name, name, E_c_param, E_c_SA_results, successful_E_c, delta=0.05)
    
    #%%
    plot_tau_range_vs_parameter(folder_name, name, E_c_param, E_c_SA_results, successful_E_c)
    
    #%%
    plot_radius_range_sensitivity(folder_name, name, E_c_param, E_c_SA_results, successful_E_c)
    #%% 5. Plot inelastic strain sensitivity
    # plot_lambda_range_vs_parameter(folder_name, name, E_c_param, E_c_SA_results, successful_E_c)
    #%%
    # plot_stress_components_vs_parameter(folder_name, name, E_c_param, E_c_SA_results, successful_E_c)
    
    # plot_lambda_grad_range_sensitivity(folder_name, name, E_c_param, E_c_SA_results, successful_E_c)
    #%%
    for mode in ['min', 'ref', 'max']:
        plot_radial_profile_single(folder_name, name, E_c_param, E_c_SA_results, successful_E_c, index_basal, mode=mode)
    #%%
    plot_cell_passive_stress(folder_name, name, E_c_param, E_c_SA_results, successful_E_c, index_relaxed)
    plot_stress_distrib(folder_name, name, E_c_param, E_c_SA_results, successful_E_c, index_basal) 
        
    #%% 
    #-----------------------------------------------------------------------------#
    # Run the SA : f_c
    #-----------------------------------------------------------------------------#
    
    E_m0, E_c0 = [0.05, 0.01] 
    
    f_c_list = [0.10, 0.20, 0.24, 0.30, 0.40] 
    p_c_list = [[E_m0, E_c0, f_c] for f_c in f_c_list] 
    
    f_c_param = ParameterSpec(
        key="fc",
        label=r"f_c",
        unit="--"
    )
    
    # 1. Execution Phase (Parallel Subprocesses)
    print(f"--> [START] Launching Sensitivity Analysis for {f_c_param.key}")
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Submit all p_c_list tasks to the pool
        future_to_p = {executor.submit(run_worker, p): p for p in p_c_list}
        
        results_map_c = {} 
        for future in as_completed(future_to_p):
            p_tuple = tuple(future_to_p[future])
            try:
                pipeline_result = future.result()
                if pipeline_result is not None:
                    results_map_c[p_tuple] = pipeline_result
                    print(f"   [SUCCESS] Parameters {p_tuple} processed.")
            except Exception as exc:
                print(f"   [FAILURE] Parameters {p_tuple} generated an exception: {exc}")
    
    # 2. Strict Sorting & Filtering
    # Reconstruct the list in the exact order of E_c_list
    f_c_SA_results = []
    successful_f_c = []
    
    for p in p_c_list:
        p_tuple = tuple(p)
        if p_tuple in results_map_c:
            f_c_SA_results.append(results_map_c[p_tuple])
            successful_f_c.append(p[2]) # Extract the E_c value (index 1 in p)
        else:
            print(f"   [WARNING] No result found for {p_tuple}, skipping.")
    
    # 3. Plotting Phase
    # We use successful_f_c to ensure x-axis and legends match the available data
    #%%
    plot_error_stress_cell_basal_from_pipelines(folder_name, name, f_c_param, f_c_SA_results, successful_f_c, delta=0.05)
    
    #%%
    plot_tau_range_vs_parameter(folder_name, name, f_c_param, f_c_SA_results, successful_f_c)
    
    #%%
    plot_radius_range_sensitivity(folder_name, name, f_c_param, f_c_SA_results, successful_f_c)
    #%% 5. Plot inelastic strain sensitivity
    # plot_lambda_range_vs_parameter(folder_name, name, E_c_param, E_c_SA_results, successful_E_c)
    #%%
    # plot_stress_components_vs_parameter(folder_name, name, E_c_param, E_c_SA_results, successful_E_c)
    
    # plot_lambda_grad_range_sensitivity(folder_name, name, E_c_param, E_c_SA_results, successful_E_c)
    #%%
    for mode in ['min', 'ref', 'max']:
        plot_radial_profile_single(folder_name, name, f_c_param, f_c_SA_results, successful_f_c, index_basal, mode=mode)    
    #%%
    plot_cell_passive_stress(folder_name, name, f_c_param, f_c_SA_results, successful_f_c, index_relaxed)
    plot_stress_distrib(folder_name, name, f_c_param, f_c_SA_results, successful_f_c, index_basal)
    
    #%%
    # Final Step: Disk housekeeping
    cleanup_jit_cache("./jit_cache")
    
    print("--> [FINISH] Sensitivity Analysis Pipeline Complete.")