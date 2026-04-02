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
import time
import pickle
import json
import os

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib as mpl
from functools import partial

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

        
def export_cards_for_worker(name_simu, passive_simu_card, simu_card, media_card, adventitia_card):
    """
    Export the cards for a given parameter set to JSON files that
    can be read by the worker subprocess.
    Returns the paths of the exported files.
    """
    os.makedirs("./json_cards/tmp_cards", exist_ok=True)

    passive_simu_card_file = f"./json_cards/tmp_cards/{name_simu}_passive_simu_card.json"
    simu_card_file = f"./json_cards/tmp_cards/{name_simu}_simu_card.json"
    media_card_file = f"./json_cards/tmp_cards/{name_simu}_media_card.json"
    adventitia_card_file = f"./json_cards/tmp_cards/{name_simu}_adventitia_card.json"

    with open(passive_simu_card_file, "w") as f:
        json.dump(passive_simu_card, f, indent=2)
    with open(simu_card_file, "w") as f:
        json.dump(simu_card, f, indent=2)
    with open(media_card_file, "w") as f:
        json.dump(media_card, f, indent=2)
    with open(adventitia_card_file, "w") as f:
        json.dump(adventitia_card, f, indent=2)

    return passive_simu_card_file, simu_card_file, media_card_file, adventitia_card_file


def axisymmetric_relative_error(radius, function, function_target_scalar_value):
    """
    Axisymmetric L2 relative error to a uniform target field.
    """

    r = np.asarray(radius)
    f = np.asarray(function)
    dr = np.gradient(r) # Radial spacing (non-uniform safe)

    # Weighted squared error
    num = np.sum((f - function_target_scalar_value)**2 * r * dr)
    den = np.sum((function_target_scalar_value**2) * r * dr)
    
    return np.sqrt(num / den)

def axisymmetric_average(radius, function):
    dr = np.gradient(radius)
    return(np.sum(function * radius * dr)/np.sum(radius * dr))

class pipeline_Vcn_result:
    """ 
    Class that manages the results for the pipeline in the sensitivity analysis in ChV Vasoconstriction.
    """
    def __init__(self, passive_result, SA_results, tau_b_ref, tau_list):
        self.passive_result = passive_result
        self.SA_results = SA_results
        self.tau_b_ref = tau_b_ref
        self.tau_list = np.array(tau_list)
    
    def extract_results(self, index_relaxed, index_basal):
        self.relaxed_radius = np.zeros(self.tau_list.shape)
        self.basal_radius = np.zeros(self.tau_list.shape)
        self.error_cell_stress_basal = np.zeros(self.tau_list.shape)
        
        for i, result in enumerate(self.SA_results):
            self.relaxed_radius[i] = result.outputs['ri_d'][index_relaxed]
            self.basal_radius[i] = result.outputs['ri_d'][index_basal]
            
            r_pos = result.dict_outputs['S_yy']['points']
            s_cell_basal = result.outputs['s_yy_cell'][index_basal, :]
            
            self.error_cell_stress_basal[i] = axisymmetric_relative_error(r_pos, s_cell_basal, self.tau_list[i])
        
    def compute_functioning_range(self, delta):
        """
        min, ref, max -> for tau, radius and stretch
        """
        self.id_range = np.argwhere(self.error_cell_stress_basal < delta)
        
        self.tau_range = [self.tau_list[int(self.id_range[0][0])], self.tau_list[int(self.id_range[-1][0])]]
        self.r_range = [self.basal_radius[int(self.id_range[0][0])], self.basal_radius[int(self.id_range[-1][0])]]
        # --- Find exact index for tau_b_ref ---
        # Finding where tau_list matches the reference value
        idx_ref = np.where(np.isclose(self.tau_list, self.tau_b_ref))[0][0]
        self.r_b_ref = self.basal_radius[idx_ref]
        
#%% 
# Plotting function

def plot_error_stress_cell_basal_from_pipelines(folder_name: str, name: str, param: ParameterSpec, pipelines: list, param_list:list, delta: float = 0.05):
    """
    Plot basal cell stress error as a function of tau
    for multiple pipeline_Vcn_result objects.
    """

    fig, ax = plt.subplots(
        figsize=(4, 3),
        constrained_layout=True
    )
    
    n = len(pipelines)
    cmap = plt.cm.get_cmap('viridis', n)
    
    if n == 1:
        colors = [cmap(0.5)]  # pick the middle of the colormap
    else:
        colors = [cmap(i / (n - 1)) for i in range(n)]
    
    for p_id, pipeline in enumerate(pipelines):
        color = colors[p_id % len(colors)]

        tau = pipeline.tau_list
        err = pipeline.error_cell_stress_basal
        tau_b_ref = pipeline.tau_b_ref

        # --- main curve ---
        ax.semilogy(
            tau, err,
            linewidth=1.8,
            color=color,
            alpha=0.95
        )

        # --- tolerance line intersection ---

        if hasattr(pipeline, "tau_range") and pipeline.tau_range is not None:
            tau_left, tau_right = pipeline.tau_range
            width = tau_right - tau_left

            # vertical bounds
            ax.axvline(tau_left, color=color, linestyle='--', linewidth=1.2, alpha=0.5)
            ax.axvline(tau_right, color=color, linestyle='--', linewidth=1.2, alpha=0.5)

            # shaded functioning range
            ax.axvspan(tau_left, tau_right, color=color, alpha=0.12, zorder=0)

            # horizontal arrow
            y_arrow = delta * 20
            ax.annotate("",xy=(tau_left, y_arrow),xytext=(tau_right, y_arrow),arrowprops=dict(arrowstyle="<->",color=color,linewidth=1.4,alpha=0.9))

            ax.text(0.5 * (tau_left + tau_right),y_arrow * 1.15,rf"$\Delta\tau={width:.3f}$",ha="center",va="bottom",fontsize=8,color=color)
            ax.text(0.5 * (tau_left + tau_right), y_arrow * 0.8, rf"$\tau_b^{{ref}}={tau_b_ref:.3f}$", ha="center", va="top", fontsize=8, color=color)

    # --- tolerance line ---
    ax.axhline(delta,color='k',linestyle=':',linewidth=1.0,alpha=0.8)
    tolerance_handle = Line2D([0], [0], color='k', linestyle=':', linewidth=1.0, label=r'$\pm 5\%$ tolerance')
    tolerance_legend = ax.legend(handles=[tolerance_handle], loc='lower left', fontsize=8,frameon=True, framealpha=0.85, facecolor='white', edgecolor='gray')

    ax.add_artist(tolerance_legend)

    # --- axes formatting ---
    ax.set_ylabel(r"Error to target cell stress [-]")
    # ax.set_xlabel(rf"${param.label}$ [{param.unit}]")
    ax.set_xlabel(rf"$\tau$ [MPa]")

    ax.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax.grid(axis='both', linestyle=":", linewidth=0.5)
    ax.autoscale(enable=True, axis='x', tight=True)
    
    
    # --- 2️⃣ Main legend (N values) OUTSIDE ax2 ---
    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=2, # <--- Added [i] to pick the specific color
               label=rf'{param_list[i]}')
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
    

    plt.savefig(f'images_output/{folder_name}/{name}_{param.key}_error_stress_cell_basal.pdf')
    plt.show()

#%%
def plot_tau_range_vs_parameter(folder_name: str, name: str, param: ParameterSpec, pipelines: list, param_list: list):
    """
    Plots the functioning range [tau_left, tau_right] as a function of 
    the input parameter values.
    """
    
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    
    # 1. Extract data points
    extracted_data = []
    for i, pipeline in enumerate(pipelines):
        # Ensure the pipeline has already computed its range
        if hasattr(pipeline, "tau_range") and pipeline.tau_range is not None:
            p_val = param_list[i]
            t_left = pipeline.tau_range[0]
            t_right = pipeline.tau_range[1]
            t_ref = pipeline.tau_b_ref # Direct access as per your class definition
            
            extracted_data.append((p_val, t_left, t_right, t_ref))
    
    # 2. Sort by the parameter value (x-axis) to prevent line tangling
    extracted_data.sort(key=lambda x: x[0])
    
    p_vals = np.array([x[0] for x in extracted_data])
    t_lefts = np.array([x[1] for x in extracted_data])
    t_rights = np.array([x[2] for x in extracted_data])
    t_refs = np.array([x[3] for x in extracted_data])
    
    # 3. Plotting
    # Safe Zone Shading
    ax.fill_between(p_vals, t_lefts, t_rights, color='gray', alpha=0.15, label='Functioning Range')
    
    # Boundary Lines
    ax.plot(p_vals, t_lefts, 'o--', color='#1f77b4', label=r'$\tau_{left}$', linewidth=1.0, markersize=4)
    ax.plot(p_vals, t_rights, 's--', color='#d62728', label=r'$\tau_{right}$', linewidth=1.0, markersize=4)
    
    # Reference Stress (The target value)
    ax.plot(p_vals, t_refs, 'D-', color='black', label=r'$\tau_{b}^{ref}$', linewidth=1.8, markersize=5)
    
    # 4. Formatting
    ax.set_xlabel(rf"{param.label} [{param.unit}]")
    ax.set_ylabel(r"$\tau$ [MPa]")
    ax.set_title("Tau Range Sensitivity Analysis", fontsize=10, fontweight='bold')
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_facecolor((0.83, 0.83, 0.83, 0.2))
    
    # Place legend outside to avoid obscuring the curves
    ax.legend(loc='center left',frameon=False, fontsize=9)
    
    # 5. Save and Show
    save_path = f'images_output/{folder_name}/{name}_{param.key}_tau_sensitivity.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
#%%

def plot_radius_range_sensitivity(folder_name: str, name: str, param: ParameterSpec, pipelines: list, param_list: list):
    """
    Plots the sensitivity of the basal radius functioning range 
    and the specific radius at the exact tau_b_ref.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    
    extracted_data = []
    for i, pipeline in enumerate(pipelines):
        if hasattr(pipeline, "r_range") and pipeline.r_range is not None:
            p_val = param_list[i]
            r_left = pipeline.r_range[0]   # radius at tau_left
            r_right = pipeline.r_range[1]  # radius at tau_right
            r_ref = pipeline.r_b_ref       # radius at exact tau_b_ref
            
            extracted_data.append((p_val, r_left, r_right, r_ref))
    
    # Sort data by the parameter (x-axis)
    extracted_data.sort(key=lambda x: x[0])
    
    p_vals = np.array([x[0] for x in extracted_data])
    r_lefts = np.array([x[1] for x in extracted_data])
    r_rights = np.array([x[2] for x in extracted_data])
    r_refs = np.array([x[3] for x in extracted_data])
    
    # --- Plotting ---
    # Shaded Functioning Range
    ax.fill_between(p_vals, r_lefts, r_rights, color='forestgreen', alpha=0.12, label='Safe Zone (Radius)')
    
    # Limits (Dashed)
    ax.plot(p_vals, r_lefts, 'o--', color='#2ca02c', label=r'$r(\tau_{left})$', linewidth=1.0, markersize=4)
    ax.plot(p_vals, r_rights, 's--', color='#d62728', label=r'$r(\tau_{right})$', linewidth=1.0, markersize=4)
    
    # Reference Radius (Solid Black)
    ax.plot(p_vals, r_refs, 'D-', color='black', label=r'$r(\tau_{b}^{ref})$', linewidth=1.8, markersize=5)
    
    # Formatting
    ax.set_xlabel(rf"{param.label} [{param.unit}]")
    ax.set_ylabel(r"Basal Radius $r_i$ [mm]")
    ax.set_title("Basal Radius sensitivity to parameter", fontsize=10, fontweight='bold')
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_facecolor((0.83, 0.83, 0.83, 0.2))
    
    # Legend outside
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    
    # Save
    save_path = f'images_output/{folder_name}/{name}_{param.key}_radius_sensitivity_full.pdf'
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
#%%
# Whole SA process

def pipeline_SA(p, name, folder_name, passive_simu_card, simu_card, media_card, adventitia_card, index_passive):
    """
    For one vector of parameters p, run the set of simulation that explores the behavior of the model.
    That is : one passive simulation to establish tau_b_ref followed by a set of active ones with changing tau
    """
    
    # ---------------------------------------------------------------
    # Input parameters
    # ---------------------------------------------------------------
    E_m, E_c = p
    
    media_card["matrix"]['young'] = E_m
    media_card["cells"]['young'] = E_c
    name_simu = f"{name}_{E_m}_{E_c}"
    
    # Export the cards for this parameter set
    passive_simu_card_tmpfile, simu_card_tmpfile, media_card_tmpfile, adventitia_card_tmpfile = export_cards_for_worker(name_simu, passive_simu_card, simu_card, media_card, adventitia_card)
    
    # ---------------------------------------------------------------
    # Passive simulation to establish tau_b_ref the average cell stress
    # ---------------------------------------------------------------
    
    namefile = f"{name_simu}_passive_scal.pkl"
    filepath = f"./outputs/{folder_name}/{namefile}"
    try:
        with open(filepath, "rb") as f:
            passive_result = pickle.load(f)
            print(f"Loaded {namefile} from disk.")
    except FileNotFoundError:
        print(f"File not found: {namefile}. Running simulation...")
            
        cmd = [
            sys.executable,
            "./ChV_MultiscaleActiveStressRegulation/chv_3_2_subprocessworker.py",
            "None",
            name_simu,
            folder_name,
            passive_simu_card_tmpfile,
            media_card_tmpfile,
            adventitia_card_tmpfile
        ]
        
        # Run synchronously
        subprocess.run(cmd)
            
        with open(filepath, "rb") as f:
            passive_result = pickle.load(f)
        
    # Compute the tau_b_ref out of the results 
    r_pos = passive_result.dict_outputs['S_yy']['points']
    dr = np.gradient(r_pos)
    s_cell_relaxed = passive_result.outputs['s_yy_cell'][index_passive, :] # Index of the step where pressure is 100 mmHg
    tau_b_ref = np.sum(s_cell_relaxed * r_pos * dr)/np.sum(r_pos * dr)
    
    # ---------------------------------------------------------------
    # Set of active simulation for changing tau_b
    # ---------------------------------------------------------------
    tau_min, tau_max = 0.1*tau_b_ref, 3*tau_b_ref
    # center_clustered_tangent_grid
    N, strength = 15, 1.2 # 15 points for tau, 1.2 stretching of tangent function
    s = np.linspace(-1, 1, N)

    t = np.tan(strength * s) # Tangent stretching (avoid singularity)
    t /= np.max(np.abs(t))  # normalize to [-1, 1]

    # Map to physical range
    half_range = min(tau_b_ref - tau_min, tau_max - tau_b_ref)
    tau_list = tau_b_ref + t * half_range
    
    # Subprocess to manage parallelization

    def run_tau_simu(tau_b):
        cmd = [
            sys.executable,
            "./ChV_MultiscaleActiveStressRegulation/chv_3_2_subprocessworker.py",
            str(tau_b),
            name_simu,
            folder_name,
            simu_card_tmpfile,
            media_card_tmpfile,
            adventitia_card_tmpfile
        ]
        # Use .run() here because the Executor handles the "parallel" part
        return subprocess.run(cmd)
    
    max_parallel = 3
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        list(executor.map(run_tau_simu, tau_list))
            
    # Now load all results into SA_results
    SA_results = []
    for tau_b in tau_list:
        namefile = f"{name_simu}_{tau_b:.6f}_scal.pkl"
        with open(f"./outputs/{folder_name}/{namefile}", "rb") as f:
            SA_results.append(pickle.load(f))
            
    # ---------------------------------------------------------------
    # Exports: class that contains all the results and methods of interest
    # ---------------------------------------------------------------
    
    pipeline_result = pipeline_Vcn_result(passive_result, SA_results, tau_b_ref, tau_list)
    
    return(pipeline_result)    




if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn')#, force=True) 
    except:
        print("context already set")
        
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
    E_m0, E_c0 = [0.05, 0.01] # initial values for the SA
    
    E_m_list = [0.025, 0.05, 0.075, 0.1] # vector of changing values for matrix stiffness
    p_list = [[E_m, E_c0] for E_m in E_m_list] # first change only Em
    
    E_m_param = ParameterSpec(
                            key="Em",
                            label=r"E_m",
                            unit="MPa"
                        )
    #-----------------------------------------------------------------------------#
    # Run the SA : E_m
    #-----------------------------------------------------------------------------#
    E_m_SA_results = []
    
    for p in p_list:
        pipeline_result = pipeline_SA(p, name, folder_name, passive_simu_card, simu_card, media_card, adventitia_card, index_passive)
        
        pipeline_result.extract_results(index_relaxed, index_basal)
        pipeline_result.compute_functioning_range(delta=0.05)
        
        E_m_SA_results.append(pipeline_result)
    
    #%%
    plot_error_stress_cell_basal_from_pipelines(folder_name, name, E_m_param, E_m_SA_results, E_m_list, delta=0.05)
    #%%
    plot_tau_range_vs_parameter(folder_name, name, E_m_param, E_m_SA_results, E_m_list)
    #%%
    plot_radius_range_sensitivity(folder_name, name, E_m_param, E_m_SA_results, E_m_list)
    
    
    #%% 
    #-----------------------------------------------------------------------------#
    # Run the SA : E_c
    #-----------------------------------------------------------------------------#
    
    E_m0, E_c0 = [0.05, 0.01] # initial values for the SA
    
    E_c_list = [0.008, 0.01, 0.015, 0.02, 0.05] # vector of changing values for matrix stiffness
    p_c_list = [[E_m0, E_c] for E_c in E_c_list] # first change only Em
    
    E_c_param = ParameterSpec(
                            key="Em",
                            label=r"E_m",
                            unit="MPa"
                        )
    #-----------------------------------------------------------------------------#
    # Run the SA
    #-----------------------------------------------------------------------------#
    E_c_SA_results = []
    
    for p in p_c_list:
        pipeline_result = pipeline_SA(p, name, folder_name, passive_simu_card, simu_card, media_card, adventitia_card, index_passive)
        
        pipeline_result.extract_results(index_relaxed, index_basal)
        pipeline_result.compute_functioning_range(delta=0.05)
        
        E_c_SA_results.append(pipeline_result)
    
    #%%
    plot_error_stress_cell_basal_from_pipelines(folder_name, name, E_c_param, E_c_SA_results, E_c_list, delta=0.05)
    #%%
    plot_tau_range_vs_parameter(folder_name, name, E_c_param, E_c_SA_results, E_c_list)
    #%%
    plot_radius_range_sensitivity(folder_name, name, E_c_param, E_c_SA_results, E_c_list)