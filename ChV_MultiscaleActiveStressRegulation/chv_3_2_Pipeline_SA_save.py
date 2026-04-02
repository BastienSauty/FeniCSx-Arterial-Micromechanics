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
import time
import random
import pickle
import copy

import copy

import matplotlib as mpl

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
        Robustly computes the functioning range by expanding from tau_b_ref.
        """
        # 1. Find exact index for tau_b_ref
        idx_ref = np.where(np.isclose(self.tau_list, self.tau_b_ref))[0][0]
        self.r_b_ref = self.basal_radius[idx_ref]

        # 2. Search Left (from ref to 0)
        idx_left = idx_ref
        while idx_left > 0:
            # Check the next point to the left
            if self.error_cell_stress_basal[idx_left - 1] < delta:
                idx_left -= 1
            else:
                break
        
        # 3. Search Right (from ref to end)
        idx_right = idx_ref
        while idx_right < len(self.error_cell_stress_basal) - 1:
            # Check the next point to the right
            if self.error_cell_stress_basal[idx_right + 1] < delta:
                idx_right += 1
            else:
                break

        # 4. Store indices and physical values
        self.id_range = np.arange(idx_left, idx_right + 1)
        self.tau_range = [self.tau_list[idx_left], self.tau_list[idx_right]]
        self.r_range = [self.basal_radius[idx_left], self.basal_radius[idx_right]]
        
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
        loc='lower right',
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
        loc='upper right',
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
    
#%%
# Whole SA process

def calculate_tau_list(passive_result, index_passive):
    """
    Computes the reference cell stress and the stretched grid of tau values
    based on the results of a passive simulation.
    """
    # 1. Compute the tau_b_ref (average cell stress)
    r_pos = passive_result.dict_outputs['S_yy']['points']
    dr = np.gradient(r_pos)
    
    # Extract the cell stress at the specific pressure step (index_passive)
    s_cell_relaxed = passive_result.outputs['s_yy_cell'][index_passive, :]
    
    # Weighted average over the radial position
    tau_b_ref = np.sum(s_cell_relaxed * r_pos * dr) / np.sum(r_pos * dr)
    
    # 2. Define the range
    tau_min, tau_max = 0.1 * tau_b_ref, 3 * tau_b_ref
    
    # 3. Create the center-clustered tangent grid
    N, strength = 15, 1.2
    s = np.linspace(-1, 1, N)
    t = np.tan(strength * s)
    t /= np.max(np.abs(t))  # Normalize to [-1, 1]

    # Map to physical range symmetrically around tau_b_ref
    half_range = min(tau_b_ref - tau_min, tau_max - tau_b_ref)
    tau_list = tau_b_ref + t * half_range
    
    return tau_list, tau_b_ref

def run_single_tau(tau_val, name_simu_base, folder_name, simu_card, adventitia_card, media_card):

    from ChV_MultiscaleActiveStressRegulation.main_Vasoconstriction_25_11_24 import run_simulation

    # Ensure JIT has plenty of time for the first run
    os.environ["FENICS_JIT_TIMEOUT"] = "600"
    
    # Deep copy to ensure the modification of tau_b doesn't bleed into other runs
    local_media = copy.deepcopy(media_card)
    local_media["cells"]["basal stress"] = tau_val
    
    print(f"--> [SERIAL] Starting simulation for tau = {tau_val:.6f}")
    
    # Execute simulation
    return run_simulation(
        f"{name_simu_base}_{tau_val:.6f}", 
        folder_name, 
        simu_card, 
        adventitia_card, 
        local_media
    )

def pipeline_SA(p, name, folder_name, passive_simu_card, simu_card, media_card, adventitia_card, index_passive, executor):
    """
    Full pipeline for Sensitivity Analysis.
    Manages JIT warm-up, passive simulation, and parallel active simulations.
    """
    E_m, E_c = p
    name_simu_base = f"{name}_{E_m}_{E_c}"
    
    # --- STEP 0: Capture PRISTINE copies for the workers ---
    # These are 100% picklable because they haven't been touched by FEniCS yet.
    pristine_simu = copy.deepcopy(simu_card)
    pristine_media = copy.deepcopy(media_card)
    pristine_adv = copy.deepcopy(adventitia_card)

    # 1. EARLY EXIT CHECK: Total Pipeline Result
    final_output_path = f"./outputs/{folder_name}/{name_simu_base}_FINAL_PIPELINE.pkl"
    if os.path.exists(final_output_path):
        print(f"--> [SKIPPING PIPELINE] Result found for {name_simu_base}. Loading final object...")
        with open(final_output_path, "rb") as f:
            return pickle.load(f)
    
    from ChV_MultiscaleActiveStressRegulation.main_Vasoconstriction_25_11_24 import run_simulation
    
    # 2. CHECK INDIVIDUAL FILES: Passive and Active
    passive_namefile = f"{name_simu_base}_passive_scal.pkl"
    passive_filepath = os.path.join("./outputs", folder_name, passive_namefile)
    
    passive_exists = os.path.exists(passive_filepath)
    needs_simulation = not passive_exists
    tau_list = []
    passive_result = None
    tau_b_ref = None

    if passive_exists:
        print(f"    [OK] Passive file exists: {passive_namefile}")
        with open(passive_filepath, "rb") as f:
            passive_result = pickle.load(f)
        
        # Calculate the tau range required based on the passive result
        tau_list, tau_b_ref = calculate_tau_list(passive_result, index_passive)
        
        # Check if any active simulation files are missing
        missing_count = 0
        for tau in tau_list:
            tau_file = os.path.join("./outputs", folder_name, f"{name_simu_base}_{tau:.6f}_scal.pkl")
            if not os.path.exists(tau_file):
                missing_count += 1
        
        if missing_count > 0:
            print(f"    [!] {missing_count}/{len(tau_list)} active simulations missing.")
            needs_simulation = True
        else:
            print(f"    [OK] All {len(tau_list)} active simulation files found on disk.")

    # 3. EXECUTION PHASE
    if needs_simulation:
        if not passive_exists:
            print(f"--> [RUNNING] Passive Simulation...")
            warmup_media = copy.deepcopy(pristine_media)
            passive_result = run_simulation(f"{name_simu_base}_passive", folder_name, 
                                            passive_simu_card, pristine_adv, warmup_media)
            # Define these for the first time
            tau_list, tau_b_ref = calculate_tau_list(passive_result, index_passive)
        
        print(f"--> [PIPELINE] Running sweep in series...")
        SA_results = []
        for i, tau in enumerate(tau_list):
            tau_file = os.path.join("./outputs", folder_name, f"{name_simu_base}_{tau:.6f}_scal.pkl")
            
            if os.path.exists(tau_file):
                print(f"    [{i+1}/{len(tau_list)}] tau={tau:.4f} exists. Loading.")
                with open(tau_file, "rb") as f:
                    result = pickle.load(f)
            else:
                print(f"    [{i+1}/{len(tau_list)}] Processing tau={tau:.4f}")
                result = run_single_tau(tau, name_simu_base, folder_name, 
                                       pristine_simu, pristine_adv, pristine_media)
            SA_results.append(result)
    else:
        # 3. LOAD EXISTING DATA
        print(f"--> [LOADING] Reconstructing results from disk...")
        SA_results = []
        for tau in tau_list:
            tau_file = os.path.join("./outputs", folder_name, f"{name_simu_base}_{tau:.6f}_scal.pkl")
            with open(tau_file, "rb") as f:
                SA_results.append(pickle.load(f))

    # 4. EXPORT AND SAVE FINAL OBJECT
    # Ensure we have a valid tau_b_ref even if we just loaded everything
    if tau_b_ref is None and passive_result is not None:
         _, tau_b_ref = calculate_tau_list(passive_result, index_passive)

    print(f"--> [EXPORTING] Saving final pipeline object...")
    pipeline_result = pipeline_Vcn_result(passive_result, SA_results, tau_b_ref, tau_list)
    
    with open(final_output_path, "wb") as f:
        pickle.dump(pipeline_result, f)
        
    return pipeline_result


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    
    # Force environment settings for JIT and threading
    import os
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
    
    with ProcessPoolExecutor(max_workers=3) as shared_executor:
        
        # --- Run the SA : E_m ---
        E_m_SA_results = []
        for p in p_list:
            # Pass the shared_executor as an argument
            pipeline_result = pipeline_SA(p, name, folder_name, passive_simu_card, 
                                        simu_card, media_card, adventitia_card, 
                                        index_passive, executor=shared_executor)
            
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
    
    E_c_list = [0.008, 0.01, 0.013, 0.015, 0.017, 0.02, 0.05] # vector of changing values for matrix stiffness
    p_c_list = [[E_m0, E_c] for E_c in E_c_list] # first change only Em
    
    E_c_param = ParameterSpec(
                            key="Ec",
                            label=r"E_c",
                            unit="MPa"
                        )
    #-----------------------------------------------------------------------------#
    # Run the SA
    #-----------------------------------------------------------------------------#
    
    with ProcessPoolExecutor(max_workers=3) as shared_executor:
        E_c_SA_results = []
        for p in p_c_list:
            # Re-use the SAME executor
            pipeline_result = pipeline_SA(p, name, folder_name, passive_simu_card, 
                                        simu_card, media_card, adventitia_card, 
                                        index_passive, executor=shared_executor)
            
            pipeline_result.extract_results(index_relaxed, index_basal)
            pipeline_result.compute_functioning_range(delta=0.05)
            E_c_SA_results.append(pipeline_result)
            
    #%%
    plot_error_stress_cell_basal_from_pipelines(folder_name, name, E_c_param, E_c_SA_results, E_c_list, delta=0.05)
    #%%
    plot_tau_range_vs_parameter(folder_name, name, E_c_param, E_c_SA_results, E_c_list)    
    #%%
    plot_radius_range_sensitivity(folder_name, name, E_c_param, E_c_SA_results, E_c_list)