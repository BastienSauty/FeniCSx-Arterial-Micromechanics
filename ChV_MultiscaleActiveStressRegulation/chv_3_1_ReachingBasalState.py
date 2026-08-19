#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 15:37:46 2025

@author: bastien.sauty

Ch V : section 3.1.1 : results, vasoconstriction through active stress regulation
influence of parameters on the multiscale behavior of the tissue
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
from matplotlib.patches import ConnectionPatch

from matplotlib.lines import Line2D

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

@dataclass(frozen=True)
class ParameterSpec:
    key: str            # internal / filename-safe name (e.g. "tau")
    label: str          # LaTeX label (e.g. r"\tau")
    unit: str | None    # unit string (e.g. "MPa", "s", None)


@dataclass
class PlotConfig:
    lambdaz_list: np.ndarray
    press_list: np.ndarray
    time_list: np.ndarray
    id_beforeVcn: slice
    id_Vcn: slice
    id_afterVcn: slice
    index_relaxed:int
    index_basal:int
    
    
        
#-----------------------------------------------------------------------------#
# Pipeline for running simulations with specific set of parameters
#-----------------------------------------------------------------------------#

def pipeline_Vcn(p, name, folder_name, simu_card, media_card, adventitia_card):
    """
    process to run the simulation for a given vector of parameters p
    p = []
    
    """
    
    E_m, E_c, tau = p
    
    namefile = f"{name}_{E_m}_{E_c}_{tau}_scal.pkl"
    
    # Change E values
    media_card["matrix"]['young'] = p[0]
    
    media_card["cells"]['young'] = p[1]
    
    media_card["cells"]['basal stress'] = p[2]
    
        
    # Set up unique cache folder for this simulation
    name_simu = f"{name}_{E_m}_{E_c}_{tau}"
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
    
    from ChV_MultiscaleActiveStressRegulation.main_Vasoconstriction_25_11_24 import run_simulation
    
    try:
        file = open(f'./outputs/{folder_name}/{namefile}', 'rb')
        result = pickle.load(file) # load and store pkl file
        file.close()
            
    except FileNotFoundError:
        print(f"[!] File not found: ./outputs/{folder_name}/{namefile}")
        print(f"Running simulation {name_simu}")
        
        # Run simulation
        result = run_simulation(name_simu, folder_name, simu_card, adventitia_card, media_card)
    except Exception as e:
        print(f"[!] Error while loading result file:")
        print(f"    File: ./outputs/{folder_name}/{namefile}")
        print(f"    Error: {e}")

    return(result)

#%%
#-----------------------------------------------------------------------------#
# Post-proc functions
#-----------------------------------------------------------------------------#

def axisymmetric_relative_error(r, f, f_target):
    """
    Axisymmetric L2 relative error to a uniform target field.

    Parameters
    ----------
    r : array_like
        Radial positions (size N)
    f : array_like
        Field values at radial positions (size N)
    f_target : float
        Target uniform value

    Returns
    -------
    error : float
        Relative L2 error
    """

    r = np.asarray(r)
    f = np.asarray(f)

    # Radial spacing (non-uniform safe)
    dr = np.gradient(r)

    # Weighted squared error
    num = np.sum((f - f_target)**2 * r * dr)

    # Weighted norm of target field
    den = np.sum((f_target**2) * r * dr)
    
    return np.sqrt(num / den)

def robustness_interval_interpolated(tau, error, delta=0.05):
    """
    Compute interpolated tau-interval where error < delta.

    Parameters
    ----------
    tau : array_like
        Control parameter values (monotonic).
    error : array_like
        Relative error values (>0).
    delta : float
        Tolerance threshold (e.g. 0.05 for 5%).

    Returns
    -------
    width : float
        Width of acceptable tau interval.
    interval : tuple or None
        (tau_min, tau_max) if interval exists, else None.
    """

    tau = np.asarray(tau)
    error = np.asarray(error)

    # Work in log-space for numerical stability (since you plot semilogy)
    log_err = np.log(error)
    log_delta = np.log(delta)

    crossings = []

    for i in range(len(tau) - 1):
        y0, y1 = log_err[i], log_err[i + 1]

        if (y0 - log_delta) * (y1 - log_delta) < 0:
            # Linear interpolation in tau
            t = (log_delta - y0) / (y1 - y0)
            tau_cross = tau[i] + t * (tau[i + 1] - tau[i])
            crossings.append(tau_cross)

    if len(crossings) < 2:
        return 0.0, None

    tau_min, tau_max = crossings[0], crossings[-1]
    return tau_max - tau_min, (tau_min, tau_max)

#%%
#-----------------------------------------------------------------------------#
# Plotting function
#-----------------------------------------------------------------------------#
def plot_successive_load(folder_name: str, name: str, param: ParameterSpec, list_result: list, param_list: list, step_load: Artery_load):
    # Load Control Data
    time_list = step_load.list_t
    time_list /= time_list[-1]
    press_list = 7500.62 * step_load.list_P
    uz_list = np.ones(step_load.list_uz.shape) + 10 * step_load.list_uz

    # Reduced height (from 8 to 5) to make the text appear larger relative to the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6,5), sharex='all')
    axes = [ax1, ax2, ax3]

    # Plotting with original logic/colors
    ax1.plot(uz_list, color='tab:blue') # Default Blue
    ax1.set_ylabel('Axial strain', fontsize=11)

    ax2.plot(press_list, color='tab:orange') # Default Orange
    ax2.set_ylabel('Pressure [mmHg]', fontsize=11)

    ax3.plot(time_list, color='tab:green') # Default Green
    ax3.set_ylabel('Normalized time', fontsize=11)
    ax3.set_xlabel('Step number', fontsize=11)

    # Global font size adjustment for tick labels
    for ax in axes:
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_zorder(1)

    # Shading and Phase Numbering
    for i, (start, end) in enumerate(step_load.index_phase):
        # Shading: subtle alternating background
        bg_color = "white" if i % 2 == 0 else "#f0f0f0"
        for ax in axes:
            ax.axvspan(start, end, color=bg_color, zorder=0)

        # Circled Phase Number
        mid_point = (start + end) / 2
        ax1.text(mid_point, 1.12, f'{i+1}', transform=ax1.get_xaxis_transform(),
                 ha='center', va='center', fontweight='bold', size=10,
                 bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", lw=1))

        # Vertical dotted boundaries
        con = ConnectionPatch(
            (start, ax1.get_ylim()[1]), (start, ax3.get_ylim()[0]),
            "data", "data", axesA=ax1, axesB=ax3,
            color="black", ls="dotted", lw=1, alpha=0.4
        )
        fig.add_artist(con)

    # Final boundary line
    last_end = step_load.index_phase[-1][1]
    con_final = ConnectionPatch(
        (last_end, ax1.get_ylim()[1]), (last_end, ax3.get_ylim()[0]),
        "data", "data", axesA=ax1, axesB=ax3,
        color="black", ls="dotted", lw=1, alpha=0.4
    )
    fig.add_artist(con_final)

    fig.tight_layout()
    # Ensure space at top for the circles
    fig.subplots_adjust(top=0.85)
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param.key}_successive_load.pdf')
    plt.show()
def plot_pressure_radius(folder_name: str, name: str, param:ParameterSpec, list_result: list, param_list:list, config: PlotConfig):
    """
    Plotting function to plot the pressure radius curve for the different results obtained during the sensitivity analysis
    param_name : string that is the name of the parameter studied in this sensitivity analysis, ie E_m
    """
    fig, ax1 = plt.subplots(
        1, 1,
        figsize=(4,3),
        sharey=True,
        constrained_layout=True
    )
    
    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]
    
    for i, result in enumerate(list_result):
        color = colors[i]
        ri_d = result.outputs['ri_d'][:]
        re_d = result.outputs['re_d'][:]
        
        ax1.plot(ri_d[config.id_beforeVcn], press_list[config.id_beforeVcn], color=color, linestyle='-', linewidth=1.5)
        ax1.plot(ri_d[config.id_afterVcn], press_list[config.id_afterVcn], color=color, linestyle='--', linewidth=1.5)
        
    # Phase 1: Axial stretch
    # ---------------------------------------------------------------
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel(r"Pressure [mmHg]")
    ax1.set_xlabel("Radius [mm]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    
    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'Relaxed (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'Basal (dashed)')
    ]
    # Create and store the style legend first (so it doesn’t get overwritten)
    style_legend_box = ax1.legend(
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
    ax1.add_artist(style_legend_box)
    
    # --- 2️⃣ Main legend (N values) OUTSIDE ax2 ---
    legend_elements = [
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2,
               label=rf'{param_list[i]}')
        for i in range(len(list_result))
    ]
    
    ax1.legend(
        handles=legend_elements, 
        title=rf"${param.label}$ [{param.unit}]",
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param.key}_pressure_radius.pdf')
    plt.show()


def plot_artery_radius_temporal(folder_name: str, name: str, param:ParameterSpec, list_result: list, param_list:list, config: PlotConfig):
    """
    plot the radius of the artery at 100 mmHg during constriction
    """
    fig, ax1 = plt.subplots(
        1, 1,
        figsize=(4,3),
        sharey=True,
        constrained_layout=True
    )
    
    index_temp = config.id_Vcn
    time_list = config.time_list
    
    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]
    
    for i, result in enumerate(list_result):
        color = colors[i]
        
        s_cell_relaxed = result.outputs['ri_d'][index_temp]

        ax1.plot(time_list[index_temp], s_cell_relaxed, color=color, linestyle='-', linewidth=1.5)
        
        
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel("Internal radius [mm]")
    ax1.set_xlabel("Time [hr]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    # --- 2️⃣ Main legend (N values) OUTSIDE ax2 ---
    legend_elements = [
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2,
               label=rf'{param_list[i]}')
        for i in range(len(list_result))
    ]
    
    ax1.legend(
        handles=legend_elements, 
        title=rf"${param.label}$ [{param.unit}]",
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param.key}_artery_radius_temporal.pdf')
    plt.show()
    

def plot_tissue_stress(folder_name: str, name: str, param:ParameterSpec, list_result: list, param_list:list, config: PlotConfig):
    """
    plot the circumferential stress in the tunica media as a function of the radial position.
    """
    fig, ax1 = plt.subplots(
        1, 1,
        figsize=(4,3),
        sharey=True,
        constrained_layout=True
    )
    
    index_relaxed = config.index_relaxed
    index_basal = config.index_basal
    
    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]
    
    for i, result in enumerate(list_result):
        color = colors[i]
            
        r_pos = result.dict_outputs['S_yy']['points']
        
        S_yy_relaxed = result.outputs['S_yy'][index_relaxed, :]
        S_yy_basal = result.outputs['S_yy'][index_basal, :]

        
        ax1.plot(r_pos, S_yy_relaxed, color=color, linestyle='-', linewidth=1.5)
        ax1.plot(r_pos, S_yy_basal, color=color, linestyle='--', linewidth=1.5)
        
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel(r"Stress [MPa]")
    ax1.set_xlabel("Radial position [mm]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    
    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'Relaxed (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'Basal (dashed)')
    ]
    # Create and store the style legend first (so it doesn’t get overwritten)
    style_legend_box = ax1.legend(
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
    ax1.add_artist(style_legend_box)
    
    # --- 2️⃣ Main legend (N values) OUTSIDE ax2 ---
    legend_elements = [
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2,
               label=rf'{param_list[i]}')
        for i in range(len(list_result))
    ]
    
    ax1.legend(
        handles=legend_elements, 
        title=rf"${param.label}$ [{param.unit}]",
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param.key}_tissue_local_stress.pdf')
    plt.show()



def plot_cell_stress(folder_name: str, name: str, param:ParameterSpec, list_result: list, param_list:list, config: PlotConfig):
    """
    plot the stress in the cell inclusion as function of the radial position. 
    This is the stress in the axis er of the cell.
    """
    fig, ax1 = plt.subplots(
        1, 1,
        figsize=(4,3),
        sharey=True,
        constrained_layout=True
    )
    
    index_relaxed = config.index_relaxed
    index_basal = config.index_basal
    
    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]
    
    for i, result in enumerate(list_result):
        color = colors[i]
            
        r_pos = result.dict_outputs['S_yy']['points']
        
        s_cell_basal = result.outputs['s_yy_cell'][index_basal, :]

        ax1.plot(r_pos, s_cell_basal, color=color, linestyle='--', linewidth=1.5)
        
    
    s_cell_relaxed = result.outputs['s_yy_cell'][index_relaxed, :]  
    ax1.plot(r_pos, s_cell_relaxed, color='k', linestyle='-', linewidth=1.5)
        
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel(r"Cell Stress [MPa]")
    ax1.set_xlabel("Radial position [mm]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    
    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'Relaxed (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'Basal (dashed)')
    ]
    # Create and store the style legend first (so it doesn’t get overwritten)
    style_legend_box = ax1.legend(
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
    ax1.add_artist(style_legend_box)
    
    # --- 2️⃣ Main legend (N values) OUTSIDE ax2 ---
    legend_elements = [
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2,
               label=rf'{param_list[i]}')
        for i in range(len(list_result))
    ]
    
    ax1.legend(
        handles=legend_elements, 
        title=rf"${param.label}$ [{param.unit}]",
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param.key}_cell_local_stress.pdf')
    plt.show()
    


def plot_cell_stress_temporal(folder_name: str, name: str, param:ParameterSpec, list_result: list, param_list:list, config: PlotConfig):
    """
    plot the stress in the cell inclusion as function of the radial position. 
    This is the stress in the axis er of the cell.
    """
    fig, ax1 = plt.subplots(
        1, 1,
        figsize=(4,3),
        sharey=True,
        constrained_layout=True
    )
    
    index_temp = config.id_Vcn
    time_list = config.time_list
    
    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]
    
    for i, result in enumerate(list_result):
        color = colors[i]
        
        s_cell_relaxed = result.outputs['s_yy_cell_avg'][index_temp]

        ax1.plot(time_list[index_temp], s_cell_relaxed, color=color, linestyle='-', linewidth=1.5)
        
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel(r"Cell Stress [MPa]")
    ax1.set_xlabel("Time [hr]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    # --- 2️⃣ Main legend (N values) OUTSIDE ax2 ---
    legend_elements = [
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2,
               label=rf'{param_list[i]}')
        for i in range(len(list_result))
    ]
    
    ax1.legend(
        handles=legend_elements, 
        title=rf"${param.label}$ [{param.unit}]",
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param.key}_cell_local_stress_temporal.pdf')
    plt.show()
    
def plot_inelastic_stretch(folder_name: str, name: str, param:ParameterSpec, list_result: list, param_list:list, config: PlotConfig):
    """
    plot the inelastic axial stretch in the cell inclusion as function of the radial position. 
    """
    fig, ax1 = plt.subplots(
        1, 1,
        figsize=(4,3),
        sharey=True,
        constrained_layout=True
    )
    
    index_relaxed = config.index_relaxed
    index_basal = config.index_basal
    
    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]
    
    for i, result in enumerate(list_result):
        color = colors[i]
            
        r_pos = result.dict_outputs['S_yy']['points']
        
        s_cell_basal = result.outputs['lambda_cell_in'][index_basal, :]

        
        ax1.plot(r_pos, s_cell_basal, color=color, linestyle='--', linewidth=1.5)
    
    s_cell_relaxed = result.outputs['lambda_cell_in'][index_relaxed, :]  
    ax1.plot(r_pos, s_cell_relaxed, color='k', linestyle='-', linewidth=1.5)
        
    
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel(r"Cell Inelastic stretch [-]")
    ax1.set_xlabel("Radial position [mm]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    
    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'Relaxed (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'Basal (dashed)')
    ]
    # Create and store the style legend first (so it doesn’t get overwritten)
    style_legend_box = ax1.legend(
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
    ax1.add_artist(style_legend_box)
    
    # --- 2️⃣ Main legend (N values) OUTSIDE ax2 ---
    legend_elements = [
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2,
               label=rf'{param_list[i]}')
        for i in range(len(list_result))
    ]
    
    ax1.legend(
        handles=legend_elements, 
        title=rf"${param.label}$ [{param.unit}]",
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=9,
        title_fontsize=10
    )
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param.key}_cell_local_in_stretch.pdf')
    plt.show()
    
#%%
def plot_avg_stretch(folder_name: str, name: str, param:ParameterSpec, list_result: list, param_list:list, config: PlotConfig):
    """
    plot the stress in the cell inclusion as function of the radial position. 
    This is the stress in the axis er of the cell.
    """
    fig, ax1 = plt.subplots(
        1, 1,
        figsize=(4,3),
        sharey=True,
        constrained_layout=True
    )
    
    index_relaxed = config.index_relaxed
    index_basal = config.index_basal
    
    lambda_in_relaxed = []
    lambda_in_basal = []
    
    for i, result in enumerate(list_result):
        lambda_in_relaxed.append(result.outputs['lambda_cell_in_avg'][index_relaxed])
        lambda_in_basal.append(result.outputs['lambda_cell_in_avg'][index_basal])
        

    ax1.plot(param_list, lambda_in_relaxed, linestyle='-', linewidth=1.5, color='tab:blue')
    ax1.plot(param_list, lambda_in_basal, linestyle='--', linewidth=1.5, color='tab:blue')
        
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel(r"Cell Inelastic stretch [-]")
    ax1.set_xlabel(rf"${param.label}$ [{param.unit}]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'Relaxed (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'Basal (dashed)')
    ]
    # Create and store the style legend first (so it doesn’t get overwritten)
    style_legend_box = ax1.legend(
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
    ax1.add_artist(style_legend_box)
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param.key}_cell_avg_inelastic_stretch.pdf')
    plt.show()
    
def plot_avg_stress(folder_name: str, name: str, param:ParameterSpec, list_result: list, param_list:list, config: PlotConfig):
    """
    plot the stress in the cell inclusion as function of the radial position. 
    This is the stress in the axis er of the cell.
    """
    fig, ax1 = plt.subplots(
        1, 1,
        figsize=(4,3),
        sharey=True,
        constrained_layout=True
    )
    
    index_relaxed = config.index_relaxed
    index_basal = config.index_basal
    
    lambda_in_relaxed = []
    lambda_in_basal = []
    
    for i, result in enumerate(list_result):
        lambda_in_relaxed.append(result.outputs['s_yy_cell_avg'][index_relaxed])
        lambda_in_basal.append(result.outputs['s_yy_cell_avg'][index_basal])

    ax1.plot(param_list, lambda_in_relaxed, linestyle='-', linewidth=1.5, color='tab:blue')
    ax1.plot(param_list, lambda_in_basal, linestyle='--', linewidth=1.5, color='tab:blue')
        
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel(r"Cell stress [MPa]")
    ax1.set_xlabel(rf"${param.label}$ [{param.unit}]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'Relaxed (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'Basal (dashed)')
    ]
    # Create and store the style legend first (so it doesn’t get overwritten)
    style_legend_box = ax1.legend(
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
    ax1.add_artist(style_legend_box)
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param.key}_cell_avg_inelastic_stress.pdf')
    plt.show()
    
    


def plot_error_stress_cell(folder_name: str, name: str, param:ParameterSpec, list_result: list, param_list:list, config: PlotConfig):
    """
    plot the stress in the cell inclusion as function of the radial position. 
    This is the stress in the axis er of the cell.
    """
    fig, ax1 = plt.subplots(
        1, 1,
        figsize=(4,3),
        sharey=True,
        constrained_layout=True
    )
    
    index_relaxed = config.index_relaxed
    index_basal = config.index_basal
    
    error_cell_stress_relaxed = []
    error_cell_stress_basal = []
    
    for i, result in enumerate(list_result):

        
        r_pos = result.dict_outputs['S_yy']['points']
        s_cell_relaxed = result.outputs['s_yy_cell'][index_relaxed, :]
        s_cell_basal = result.outputs['s_yy_cell'][index_basal, :]
        
        error_cell_stress_relaxed.append(axisymmetric_relative_error(r_pos, s_cell_relaxed, param_list[i]))
        error_cell_stress_basal.append(axisymmetric_relative_error(r_pos, s_cell_basal, param_list[i]))
    
    
    dr = np.gradient(r_pos)
    tau_b_ref = np.sum(s_cell_relaxed * r_pos * dr)/np.sum(r_pos * dr)

    ax1.semilogy(param_list, error_cell_stress_relaxed, linestyle='-', linewidth=1.5, color='tab:blue')
    ax1.semilogy(param_list, error_cell_stress_basal, linestyle='--', linewidth=1.5, color='tab:blue')
        
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_ylabel(r"Error to target cell stress [-]")
    ax1.set_xlabel(rf"${param.label}$ [{param.unit}]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    # tolerance
    delta = 0.05  # 5% tolerance

    width_basal, interval_basal = robustness_interval_interpolated(param_list, error_cell_stress_basal, delta)
    
    # draw horizontal line with dots and add legend
    ax1.axhline(delta, color='k', linestyle=':', linewidth=1.0, alpha=0.8, label=r'$\pm 5\%$ tolerance')
    tolerance_handle = Line2D([0], [0], color='k', linestyle=':', linewidth=1.0, label=r'$\pm 5\%$ tolerance')
    tolerance_legend = ax1.legend(handles=[tolerance_handle], loc='lower left', fontsize=8,frameon=True, framealpha=0.85, facecolor='white', edgecolor='gray')

    ax1.add_artist(tolerance_legend)
    
    if interval_basal is not None:
        # draw vertical lines for interval
        ax1.axvline(interval_basal[0], color='tab:blue', linestyle='--', alpha=0.4)
        ax1.axvline(interval_basal[1], color='tab:blue', linestyle='--', alpha=0.4)
        # shaded background
        ax1.axvspan(interval_basal[0], interval_basal[1], color='tab:blue', alpha=0.12, zorder=0)
        
        tau_left, tau_right = interval_basal
        y_arrow = delta * 20  # above tolerance line
    
        # draw horizontal double line
        ax1.annotate("", xy=(tau_left, y_arrow), xytext=(tau_right, y_arrow), arrowprops=dict(arrowstyle="<->", color="tab:blue", linewidth=1.2))
        # write text for delta tau
        ax1.text(0.5 * (tau_left + tau_right), y_arrow * 1.1, rf"$\Delta\tau={width_basal:.3f}$", ha="center", va="bottom", fontsize=8, color="tab:blue")
        ax1.text(0.5 * (tau_left + tau_right), y_arrow * 0.8, rf"$\tau_b^{{ref}}={tau_b_ref:.3f}$", ha="center", va="top", fontsize=8, color="tab:blue")

    # Show y-axis ticks and labels only on the left subplot
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'Relaxed (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'Basal (dashed)')
    ]
    # Create and store the style legend first (so it doesn’t get overwritten)
    style_legend_box = ax1.legend(
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
    ax1.add_artist(style_legend_box)
    
    plt.savefig(f'images_output/{folder_name}/{name}_{param.key}_error_stress_cell.pdf')
    plt.show()
    

#%% Main part

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

    folder_name = 'ChV.3.1.ReachingBasalState'
    name = 'cell_vcn'

    simu_card_name = 'json_cards/simu_card_Vcn.json'
    media_card_name ='json_cards/media_card_Vcn.json'
    adventitia_card_name ='json_cards/adventitia_card_Vcn.json'
    # Load material card        
    simu_card = load_JSON(simu_card_name)
    media_card = load_JSON(media_card_name)
    adventitia_card = load_JSON(adventitia_card_name)
    
    simu_card['XDMF_export'] = 0
    load_phase = simu_card['load_phase']
    step_load = Artery_load(load_phase)    

    pipeline_partial = partial(
                                pipeline_Vcn,
                                name=name,
                                folder_name=folder_name,
                                simu_card=simu_card,
                                media_card=media_card,
                                adventitia_card=adventitia_card                              
                            )
    
    E_m0, E_c0, tau0 = [0.05, 0.01, 0.1] 
    
    # E_m_list = [0.025, 0.04, 0.05, 0.06, 0.08]
    # p_list = [[E_m, E_c0, tau0] for E_m in E_m_list] # first change only Em
    # id_Em = slice(0, len(E_m_list))
    
    # E_c_list = [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 1.0]
    # p_list += [[E_m0, E_c, tau0] for E_c in E_c_list] # then change only Ec
    # id_Ec = slice(len(E_m_list), len(p_list))
    
    tau_list = [0.005, 0.01, 0.0115, 0.013, 0.014, 0.015, 0.016, 0.017, 0.0185, 0.02, 0.025, 0.03] #[0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
    p_list = [[E_m0, E_c0, tau] for tau in tau_list] # then change only Ec
    id_Tau = slice(len(p_list)-len(tau_list), len(p_list))
    
    
    with mp.Pool(processes=5) as pool:
        all_results = pool.map(pipeline_partial, p_list)
        
        
    #%%
    #-----------------------------------------------------------------------------#
    # Plot results
    #-----------------------------------------------------------------------------#
    press_list = 7500.62*step_load.list_P
    lambdaz_list = 1+step_load.list_uz/simu_card['lz']
    time_list = step_load.list_t
    
    # indices for selecting the correct results
    id_beforeVcn = step_load.index_phase[1] # pressure loading phase before vasoconstriction
    id_afterVcn = step_load.index_phase[-1] # pressure loading phase after vasoconstriction
    id_Vcn = step_load.index_phase[3] # pressure loading phase after vasoconstriction
    
    id_beforeVcn = np.arange(id_beforeVcn[0],id_beforeVcn[1]+1)
    id_afterVcn = np.arange(id_afterVcn[0],id_afterVcn[1]+1)
    id_Vcn = np.arange(id_Vcn[0],id_Vcn[1]+1)
    
        
    def find_index_press(press_list, value):
        list_indices = []
        for i, p in enumerate(press_list):
            if np.isclose(p, value):
                list_indices.append(i)
        return(list_indices)
    
    
    press_100 = find_index_press(press_list, 100)
    index_relaxed = press_100[0]
    index_basal = press_100[-2]

    
    config = PlotConfig(lambdaz_list=lambdaz_list,
                        press_list=press_list,
                        time_list=time_list,
                        id_beforeVcn=id_beforeVcn,
                        id_afterVcn=id_afterVcn,
                        id_Vcn=id_Vcn,
                        index_basal=index_basal,
                        index_relaxed=index_relaxed)
    
    tau_param = ParameterSpec(
                            key="tau",
                            label=r"\tau",
                            unit="MPa"
                        )
    
    #%%
    # Successive load
    plot_successive_load(folder_name, name, tau_param, all_results[id_Tau], tau_list, step_load)

    #%%
    # ---------------------------------------------------------------
    # Pressure-Radius
    # ---------------------------------------------------------------
    plot_pressure_radius(folder_name, name, tau_param, all_results[id_Tau], tau_list, config)
    
    plot_artery_radius_temporal(folder_name, name, tau_param, all_results[id_Tau], tau_list, config)
    
    #%%
    # ---------------------------------------------------------------
    # Tissue Circ stress
    # # ---------------------------------------------------------------
    plot_tissue_stress(folder_name, name, tau_param, all_results[id_Tau], tau_list, config)
    
    #%%
    # ---------------------------------------------------------------
    # Cellular stress
    # # ---------------------------------------------------------------
    plot_cell_stress(folder_name, name, tau_param, all_results[id_Tau], tau_list, config)
    
    plot_avg_stress(folder_name, name, tau_param, all_results[id_Tau], tau_list, config)
    
    plot_cell_stress_temporal(folder_name, name, tau_param, all_results[id_Tau], tau_list, config)
    #%%
    plot_error_stress_cell(folder_name, name, tau_param, all_results[id_Tau], tau_list, config)
    
    #%%
    # ---------------------------------------------------------------
    # Cellular inelastic stretch
    # # ---------------------------------------------------------------
    plot_inelastic_stretch(folder_name, name, tau_param, all_results[id_Tau], tau_list, config)
    
    plot_avg_stretch(folder_name, name, tau_param, all_results[id_Tau], tau_list, config)