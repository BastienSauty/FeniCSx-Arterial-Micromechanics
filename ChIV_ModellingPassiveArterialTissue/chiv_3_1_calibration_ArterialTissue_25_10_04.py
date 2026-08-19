#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 15:55:46 2025

@author: bastien.sauty

Run the passive calibration of the arterial tissue on the experimental results of C. Cavinato

Use the main file 'main_ArterialTissue_25_06_04.py' to run one simulation. 

Plotting file for section 3.1 Result _ Calibration
- plots of (P, r) experimental / model comparison
- Young modulus constitutive law -> comparison with some litterature ?
- Cell, Fiber, kinematics. Adv vs Media

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import pickle
import glob
import json

from scipy.optimize import minimize
from scipy.interpolate import interp1d

import multiprocessing

import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.transforms import Bbox

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
#%%
#-----------------------------------------------------------------------------#
# Custom modules
#-----------------------------------------------------------------------------#

from Multiscale_Framework.class_modules.load_class import Artery_load
from ChIV_ModellingPassiveArterialTissue.main_ArterialTissue_25_06_04 import (
    run_simulation,
    load_JSON,
)

from Multiscale_Framework.function_modules.discretization_collagen import  (
    load_and_discretize_distribution,
    plot_discrete_vs_continuous,
    load_continuous_distribution,
    project_discrete_to_grid_centered_bins
)

def clean_fenics_cache():
    """Removes all cached FEniCS JIT compiled files."""
    cache_dir = os.path.expanduser('~/.cache/fenics/')
    if os.path.exists(cache_dir):
        # Match all JIT cache files
        patterns = ['libffcx_expressions_*', 'libffcx_forms_*']
        for pattern in patterns:
            files = glob.glob(os.path.join(cache_dir, pattern))
            for f in files:
                try:
                    os.remove(f)
                    print(f'Removed cache file: {f}')
                except Exception as e:
                    print(f'Could not remove {f}: {e}')
    else:
        print("Cache directory does not exist.")
        
def pack(params, bounds):
    """Normalize parameters to [0, 1] based on provided bounds."""
    params = np.array(params)
    mins = np.array([b[0] for b in bounds])
    maxs = np.array([b[1] for b in bounds])
    return (params - mins) / (maxs - mins)

def unpack(norm_params, bounds):
    """Denormalize parameters from [0, 1] back to real scale."""
    norm_params = np.array(norm_params)
    mins = np.array([b[0] for b in bounds])
    maxs = np.array([b[1] for b in bounds])
    return mins + norm_params * (maxs - mins)

def remove_nonserializable(d, keys_to_remove=("geometry","mu_0", "k_0")):
    """Recursively remove given keys from a nested dict."""
    if isinstance(d, dict):
        return {
            k: remove_nonserializable(v, keys_to_remove)
            for k, v in d.items()
            if k not in keys_to_remove
        }
    elif isinstance(d, list):
        return [remove_nonserializable(i, keys_to_remove) for i in d]
    else:
        return d
    
# Global variable to keep track of the best error
best_error = np.inf  # start with infinity

#%%
#-----------------------------------------------------------------------------#
# Cost function for optimization
#-----------------------------------------------------------------------------#
call_cost_func = 0

def cost_function(x, bounds,name, folder_name, press_exp, re_exp, ri_exp, media_card, adventitia_card, simu_card, collagen_keys_media, collagen_keys_adventitia, log_filename):
    """
    Cost function that compute the error between experimental and simulation 
    results with a given list of parameters

    Parameters
    ----------
    x : vector of unknown parameters -> E values

    Returns
    -------
    error : sum of squared errors between experimental and simu results
            simulation results are interpolated on expe observation points ?
            or vice versa

    """
    global call_cost_func, best_error
    
    call_cost_func += 1
    if call_cost_func%10==0:
        clean_fenics_cache()
    
    
    print(x)
    x_real = unpack(x, bounds)
    
    # # Change E values
    # media_card["matrix"]['young'] = x_real[0]
    # adventitia_card["matrix"]['young'] = x_real[0]
    
    for key_media in collagen_keys_media:
        media_card[key_media]['young'][0][0] = x_real[0]
        media_card[key_media]['young'][1][0] = x_real[1]
        media_card[key_media]['young'][2][0] = x_real[2]
    
    for key_adv in collagen_keys_adventitia:
        adventitia_card[key_adv]['young'][0][0] = x_real[0]
        adventitia_card[key_adv]['young'][1][0] = x_real[1]
        adventitia_card[key_adv]['young'][2][0] = x_real[2]
        
    # change radius
    ri, re = x_real[3], x_real[4]
    advTF =simu_card['advTF'] 
    simu_card['ri'] = ri
    simu_card['re'] = re
    simu_card['ri_adv'] = np.sqrt(re**2 - advTF*(re**2-ri**2))
    
    # Run the model simulation
    max_retries, attempt = 2, 0
    while attempt < max_retries:
        print(f'set of parameter is {x_real}')
        try:
            result, _ = run_simulation(name, folder_name, simu_card, adventitia_card, media_card)
            break
        except TimeoutError as e:
            print(f"TimeoutError occurred: {e}")
            print("Cleaning cache and retrying...")
            clean_fenics_cache()
            attempt += 1
        except Exception as e:
            print(f"Raised error: {e}")
            print(f"Assigning penalty cost")
            clean_fenics_cache()
            attempt += 1
            
    if attempt == max_retries:
        # If all retries fail, assign penalty
        print(f"Simulation failed after {max_retries} attempts. Assigning penalty cost")
        result = None

    if not result:
        cost = 1e10
    else:
        # Extract model results
        press_list = 7500.62*result.outputs['press'][:]
        ri_list = result.outputs['ri_d'] 
        re_list = result.outputs['re_d'] 
        
        # Interpolate the model results to experimental time points
        interp_func_ri = interp1d(press_list, ri_list, kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_func_re = interp1d(press_list, re_list, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        # get the index in the experimental pressure list of the maximum value of pressure in the model
        max_press = min(range(len(press_exp)), key=lambda i: abs(press_exp[i] - press_list[-1]))
        
        # lists of radius interpolated at experimental pressure points
        ri_interp = interp_func_ri(press_exp[:max_press])
        re_interp = interp_func_re(press_exp[:max_press])
    
        # Compute the sum of squared differences
        cost = np.sum((ri_exp[:max_press]- ri_interp) ** 2) + np.sum((re_exp[:max_press]- re_interp) ** 2) 
        
    print(f"Error measure is {cost} for x : {x_real}")
    # Log progress if it's a new best
    if cost < best_error:
        best_error = cost
        with open(log_filename, "a") as f:
            line = f"{cost}," + ",".join(map(str, x_real)) + "\n"
            f.write(line)
        print(f"New best cost: {cost:.6f} | Params: {x_real}")


    return(cost)    



#-----------------------------------------------------------------------------#
# Load general parameters
#-----------------------------------------------------------------------------#

folder_name = 'ChIV.3.1.Calibration'
name = '3.1.calib_passive_artery_vf'
namefile = name +'_scal.pkl'


simu_card_name = 'json_cards/simu_card_calib.json'
media_card_name ='json_cards/media_card_calib.json'
adventitia_card_name ='json_cards/adventitia_card_calib.json'
# Load material card        
simu_card = load_JSON(simu_card_name)
adventitia_card = load_JSON(adventitia_card_name)
media_card = load_JSON(media_card_name)

load_phase = simu_card['load_phase']
step_load = Artery_load(load_phase)

simu_card['XDMF_export'] = 0

keys = media_card.keys()
collagen_keys_media = [k for k in keys if k.startswith("collagen")]
keys = adventitia_card.keys()
collagen_keys_adventitia = [k for k in keys if k.startswith("collagen")]


#-----------------------------------------------------------------------------#
# Load Experimental Data
#-----------------------------------------------------------------------------#
# Load the CSV file
df = pd.read_csv("ChIV_ModellingPassiveArterialTissue/passive_calibration/DTA1_pressure_radius_combined.csv")

# Sort by 'pressure'
df_sorted = df.sort_values(by="pressure")

# Extract each column as NumPy arrays
press_exp = df_sorted["pressure"].to_numpy()
radius_internal = df_sorted["radius_internal"].to_numpy()
radius_external = df_sorted["radius_external"].to_numpy()

ri_exp = radius_internal/1000
re_exp = radius_external/1000

plt.figure(figsize=(4,3))
plt.plot(ri_exp, press_exp, label='$R_i^{{exp}}$')
plt.plot(re_exp, press_exp, label='$R_e^{{exp}}$')
plt.legend()
plt.grid()
plt.ylabel('Internal Pressure [mmHg]')
plt.xlabel('Radius [mm]')
plt.tight_layout()
plt.savefig(f'images_output/{folder_name}/exp_pressure_radius.pdf')
plt.show()

# load and discretize collagen fibers in the adventitia
N = len(collagen_keys_adventitia) # number of collagen families -> already initialized in the json, should be cleaned
filename = "ChIV_ModellingPassiveArterialTissue/passive_calibration/avg_density_Low.npz" # experimental continuous distribution
theta_coll_adv, weights_coll_adv = load_and_discretize_distribution(filename, N, name+'_init', folder_name, plot=True, verbose=False) # function in discretization_collagen.py
f_coll_adv= 0.5

for j, key in enumerate(collagen_keys_adventitia): # change the discrete distribution to correspond to the discretized experimental one
    adventitia_card[key]['theta'] = theta_coll_adv[j]
    adventitia_card[key]['volumic_fraction'] = f_coll_adv*weights_coll_adv[j]
    
with open(f'./json_cards/adventitia_card_calib.json', 'w') as fp:
    json.dump(adventitia_card, fp)
#-----------------------------------------------------------------------------#
# Run Calibration
#-----------------------------------------------------------------------------#

run_calib = False
if run_calib:
    """
    Calibrating 
       - Collagen Young modulus
       - Initial radius
       - Matrix young modulus
    """ # [0.04741369 0.17479321 2.04012616 1.09456744 0.33607003 0.42263801]
    # x0_real = np.array([0.02, 0.5, 3.4, 1.1]) #, 0.344, 0.423])   # Young modulus matrix,  collagen, ri, re
    x0_real = np.array([0.5, 3.4, 1.1, 0.344, 0.423])#  Young modulus collagen, ri, re
    
    bounds = [#[0.01, 0.04],# matrix young modulus
              [0.4, 1.0], # collagen young modulus e0
              [2.0, 4.0], # collagen k0 -> nonlinearity
              [1.05, 1.2],  # collagen lambda 0
              [0.32, 0.36],# ri
              [0.4, 0.44]] # re 
    
        
    log_filename = f"outputs/{folder_name}/{name}_optimization_logs.csv"
    
    # Initialize log file (optional: only once at the beginning)
    with open(log_filename, "w") as f:
        f.write("error," + ",".join([f"x0_real{i}" for i in range(len(x0_real))]) + "\n")
    
    x0 = pack(x0_real, bounds)
    cost_function_lambda = lambda x: cost_function(x, bounds,name, folder_name, press_exp, re_exp, ri_exp, media_card, adventitia_card, simu_card, collagen_keys_media, collagen_keys_adventitia, log_filename)
        
    # result_opti = minimize(cost_function_lambda, x0, bounds=[(0, 1) for _ in bounds], method='Powell')#L-BFGS-B 
    result_opti = minimize(cost_function_lambda,
                            x0,
                            method='trust-constr',
                            bounds=[(0, 1) for _ in bounds],  # Because we are now in normalized space
                            options={'xtol': 1e-4, 'gtol': 1e-4, 'maxiter': 1000}
                            )
    #print("Optimized parameters:", result_opti.x)
    
    x_calib = unpack(result_opti.x, bounds)

    # # Change E values
    # media_card["matrix"]['young'] = x_calib[0]
    # adventitia_card["matrix"]['young'] = x_calib[0]
    
    for key_media in collagen_keys_media:
        media_card[key_media]['young'][0][0] = x_calib[0]
        media_card[key_media]['young'][1][0] = x_calib[1]
        media_card[key_media]['young'][2][0] = x_calib[2]
    
    for key_adv in collagen_keys_adventitia:
        adventitia_card[key_adv]['young'][0][0] = x_calib[0]
        adventitia_card[key_adv]['young'][1][0] = x_calib[1]
        adventitia_card[key_adv]['young'][2][0] = x_calib[2]
    
    # change radius
    ri, re = x_calib[3], x_calib[4]
    advTF =simu_card['advTF'] 
    simu_card['ri'] = ri
    simu_card['re'] = re
    simu_card['ri_adv'] = np.sqrt(re**2 - advTF*(re**2-ri**2))
    
    
    media_card_export = remove_nonserializable(media_card)
    adventitia_card_export = remove_nonserializable(adventitia_card)
        
    with open(f'./outputs/{folder_name}/media_card_calib_export.json', 'w') as fp:
        json.dump(media_card_export, fp, indent=4, sort_keys=True, ensure_ascii=False)
    with open(f'./outputs/{folder_name}/adventitia_card_calib_export.json', 'w') as fp:
        json.dump(adventitia_card_export, fp, indent=4, sort_keys=True, ensure_ascii=False)
    with open(f'./outputs/{folder_name}/simu_card_calib_export.json', 'w') as fp:
        json.dump(simu_card, fp, indent=4, sort_keys=True, ensure_ascii=False)

#%%
#-----------------------------------------------------------------------------#
# Run Last Simulation
#-----------------------------------------------------------------------------#
simu_card['XDMF_export'] = 1

try:
    file = open(f'./outputs/{folder_name}/{namefile}', 'rb')

    result = pickle.load(file) # load and store pck file
    file.close()
except:
    print(f'Running simulation {namefile}')
    result, mech = run_simulation(name, folder_name, simu_card, adventitia_card, media_card)
    
#-----------------------------------------------------------------------------#
# Plot results -- Post Processing
#-----------------------------------------------------------------------------#
#%%
press_list = 7500.62*step_load.list_P
lambdaz_list = 1+step_load.list_uz/simu_card['lz']

# --- Compute phase lengths ---
n_phase1 = step_load.index_phase[0][1]          
n_phase2 = len(press_list) - step_load.index_phase[1][0]  
indices_1 = slice(step_load.index_phase[0][0], step_load.index_phase[0][1])
indices_2 = slice(step_load.index_phase[1][0], step_load.index_phase[1][1]+1)

# Full step indices
steps = np.arange(len(lambdaz_list))

fig, ax1 = plt.subplots(
    1, 1,
    figsize=(4,3),
    sharey=True,
    constrained_layout=True # tighten space
)

# --- Left y-axis: λ_z ---
ax1.plot(steps, lambdaz_list, color='tab:blue', label='$\lambda_z$')
ax1.set_ylabel('Axial stretch $\lambda_z$')

# --- Right y-axis: Pressure ---
ax2 = ax1.twinx()
ax2.plot(steps, press_list, color='tab:red', label='$P_{blood}$')
ax2.set_ylabel('Pressure [mmHg]')

# --- X-axis: step number / phase numbering ---
ax1.set_xlabel('Step number')
ax1.set_xlim(0, len(steps)-1)

# Optional: mark phase separation
x_sep = step_load.index_phase[0][1]  # end of phase 1
ax1.axvline(x=x_sep, color='k', linestyle='--', linewidth=1)

# Optional: phase background shading
ax1.axvspan(0, x_sep, facecolor='lightgray', alpha=0.4, zorder=-1)
ax1.axvspan(x_sep, len(steps), facecolor='lightyellow', alpha=0.7, zorder=-1)

# Optional: grid and legend
ax1.grid(True, linestyle=":", linewidth=0.5)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', frameon=True)

# phase titles
y_max = max(lambdaz_list) * 1.025  # small offset above max for title
ax1.text(x=x_sep/2, y=y_max, s='Phase 1', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.text(x=x_sep + (len(steps)-x_sep)/2, y=y_max, s='Phase 2', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.savefig(f'images_output/{folder_name}/{name}_load_phase.pdf')
plt.show()


#%%

ri_d = result.outputs['ri_d'][:]
re_d = result.outputs['re_d'][:]
area =  result.outputs['area'][:]

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(4,3),
    sharey=True,
    constrained_layout=True,
    gridspec_kw={'width_ratios': [n_phase1, n_phase2], 'wspace': 0.05}  # tighten space
)

# ---------------------------------------------------------------
# Phase 1: Axial stretch
# ---------------------------------------------------------------
ax1.plot(lambdaz_list[indices_1], ri_d[indices_1], color='tab:blue', label=r'$R_i$')
ax1.plot(lambdaz_list[indices_1], re_d[indices_1], color='tab:orange', label=r'$R_e$')

ax1.set_xlim([np.min(lambdaz_list), np.max(lambdaz_list)])
ax1.autoscale(enable=True, axis='x', tight=True)
ax1.set_xlabel(r"Axial stretch $\lambda_z$")
ax1.set_ylabel("Radius [mm]")
ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
ax1.set_title('Phase 1')
ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines

# Show y-axis ticks and labels only on the left subplot
ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)

# ---------------------------------------------------------------
# Phase 2: Inflation
# ---------------------------------------------------------------
ax2.plot(press_list[indices_2], ri_d[indices_2], color='tab:blue', label=r'$R_i$')
ax2.plot(press_list[indices_2], re_d[indices_2], color='tab:orange', label=r'$R_e$')
ax2.plot(press_exp, ri_exp, label=r'$R_i^{exp}$', linestyle='None', color='tab:blue', marker='+', markevery=10)
ax2.plot(press_exp, re_exp, label=r'$R_e^{exp}$', linestyle='None', color='tab:orange', marker='+', markevery=10)

ax2.set_xlabel("Pressure [mmHg]")
ax2.set_facecolor((1.0, 1.0, 0.88, 0.7))
ax2.set_xlim([np.min(press_list), np.max(press_list)])
ax2.autoscale(enable=True, axis='x', tight=True)
ax2.set_title('Phase 2')
ax2.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines

# Hide left y-axis ticks/labels on the right subplot (sharey=True keeps the scale)
ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)

# Add legend only to right subplot
ax2.legend(loc='lower right')

#plt.tight_layout()
plt.savefig(f'images_output/{folder_name}/{name}_pressure_radius.pdf')
plt.show()


#%%
# Collagen kinematics
def plotter_2_phases_Collagen(layer_key, key_suffix, fig_name):
    """
    input: 
        layer_key : 'media' or 'adv'
        key_suffix : 'theta' for orientation, 'stretch' for stretch and 'young' for young modulus
        fig_name : name of pdf output
    """
    # preconditionning of the figs based on the keys
    if layer_key=='adv':
        collagen_keys = collagen_keys_adventitia
    elif layer_key=='media':
        collagen_keys = collagen_keys_media
    
    if key_suffix=='theta':
        ylim = [-90, 90]
        ylabel = "Angle to axial direction [°]"
        y_ticks = ticker.MultipleLocator(30)
    elif key_suffix=='stretch':
        ylim = None #[0.5, 2.5]
        ylabel = "Fiber stretch"
        y_ticks = ticker.MultipleLocator(0.5)
    elif key_suffix=='young':
        ylim = None
        ylabel = "Fiber Young Modulus [MPa]"
        y_ticks = None
        
    
    # --- Create subplots with proportional widths and minimal spacing ---
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(5,3),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={'width_ratios': [n_phase1, n_phase2], 'wspace': 0.05}  # << tighten space
    )
    
        
    # ---------------------------------------------------------------
    # Phase 1: Axial stretch
    # ---------------------------------------------------------------
    for key in collagen_keys:
        ax1.plot(
            lambdaz_list[indices_1],
            result.outputs[f'{key}_{layer_key}_{key_suffix}'][indices_1],
            label=key
        )
        
    ax1.set_xlim([np.min(lambdaz_list), np.max(lambdaz_list)])
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel("Axial stretch $\lambda_z$")
    ax1.set_ylabel(ylabel)
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    ax1.grid(True, linestyle=":", linewidth=0.5)

    # ---------------------------------------------------------------
    # Phase 2: Inflation
    # ---------------------------------------------------------------
    for key in collagen_keys:
        theta_0 = float(result.outputs[f'{key}_{layer_key}_theta'][0])
        label_key = rf'$\theta_0 = {theta_0:.3g}$°'
        ax2.plot(
            press_list[indices_2],
            result.outputs[f'{key}_{layer_key}_{key_suffix}'][indices_2],
            label=label_key
        )

    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.7))
    ax2.set_xlim([np.min(press_list), np.max(press_list)])
    if ylim:
        ax2.set_ylim(ylim)
    ax2.autoscale(enable=True, axis='x', tight=True)
    if y_ticks:
        ax2.yaxis.set_major_locator(y_ticks)
    ax2.set_title('Phase 2')
    # # Align ticks nicely
    ax2.tick_params(left=False, labelleft=False)
    ax2.grid(True, linestyle=":", linewidth=0.5)
    # ax2.legend(loc='best')
    ax2.legend(loc='center left',          # anchor the left edge of the legend box
                bbox_to_anchor=(1.05, 0.5), # (x>1 moves it right outside; y=0.5 centers vertically)
                borderaxespad=0.0,
                frameon=False)
    
    
    
    plt.savefig(f"images_output/{folder_name}/{name}_{fig_name}.pdf")
    plt.show()

plotter_2_phases_Collagen('adv', 'theta', 'adventitia_collagen_orient')
plotter_2_phases_Collagen('media', 'theta', 'media_collagen_orient')
plotter_2_phases_Collagen('adv', 'stretch', 'adventitia_collagen_stretch')
plotter_2_phases_Collagen('media', 'stretch', 'media_collagen_stretch')
plotter_2_phases_Collagen('adv', 'young', 'adventitia_collagen_young')
plotter_2_phases_Collagen('media', 'young', 'media_collagen_young')


#%%
# Comparison with experimental values : comparison at specific pressure points

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


exp_pdf_Low = 'ChIV_ModellingPassiveArterialTissue/passive_calibration/avg_density_Low.npz'
exp_pdf_Dias = 'ChIV_ModellingPassiveArterialTissue/passive_calibration/avg_density_Dias.npz'
exp_pdf_Sys = 'ChIV_ModellingPassiveArterialTissue/passive_calibration/avg_density_Sys.npz'

angles_Low, exp_pdf_Low = load_continuous_distribution(exp_pdf_Low)
angles_Dias, exp_pdf_Dias = load_continuous_distribution(exp_pdf_Dias)
angles_Sys, exp_pdf_Sys = load_continuous_distribution(exp_pdf_Sys)

angles_model_Low = extract_angles(result, 10.)
angles_model_Dias = extract_angles(result, 80.)
angles_model_Sys = extract_angles(result, 120.)

projected_model_Low = project_discrete_to_grid_centered_bins(angles_model_Low, weights_coll_adv, angles_Low)
projected_model_Dias = project_discrete_to_grid_centered_bins(angles_model_Dias, weights_coll_adv, angles_Dias)
projected_model_Sys = project_discrete_to_grid_centered_bins(angles_model_Sys, weights_coll_adv, angles_Sys)

plot_discrete_vs_continuous(angles_Low, exp_pdf_Low, angles_model_Low, weights_coll_adv, projected_model_Low, name+'_Low', folder_name, savefig=True)
plot_discrete_vs_continuous(angles_Dias, exp_pdf_Dias, angles_model_Dias, weights_coll_adv, projected_model_Dias, name+'_Dias', folder_name, savefig=True)
plot_discrete_vs_continuous(angles_Sys, exp_pdf_Sys, angles_model_Sys, weights_coll_adv, projected_model_Sys, name+'_Sys', folder_name, savefig=True)

#%%
# Stress distributions between layers and components :

def plot_stresses(list_curves, curve_name, stress_name, figname, ylabel, ylim=None, hatched_phase1=False):
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(4,3),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={'width_ratios': [n_phase1, n_phase2], 'wspace': 0.05}  # tighten space
    )
    
    for i, curve in enumerate(list_curves):
        if not hatched_phase1:
            ax1.plot(lambdaz_list[indices_1], curve[indices_1], label=curve_name[i])
        ax2.plot(press_list[indices_2], curve[indices_2], label=curve_name[i])
    
    ax1.set_xlim([np.min(lambdaz_list), np.max(lambdaz_list)])
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel(ylabel)
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    
    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.7))
    ax2.set_xlim([np.min(press_list), np.max(press_list)])
    if ylim:
        ax2.set_ylim(ylim)
    else:
        ylim = ax2.get_ylim()
        
    if hatched_phase1:
        ax1.fill_between(
            lambdaz_list[indices_1],
            ylim[0], ylim[1],
            color='gray',
            alpha=0.3,
            hatch='//'
        )
        
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)  # horizontal grid lines
    
    
    # Hide left y-axis ticks/labels on the right subplot (sharey=True keeps the scale)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)
    
    # Add legend only to right subplot
    ax2.legend(loc='best')
    
    #plt.tight_layout()
    plt.savefig(f'images_output/{folder_name}/{name}_{figname}_{stress_name}_stress.pdf')
    plt.show()

#%%
# ---------------------------------------------------------------
# Stress in layers
# ---------------------------------------------------------------
S_yy_media = result.outputs['S_yy_avg_media']
S_zz_media = result.outputs['S_zz_avg_media']
S_yy_adv = result.outputs['S_yy_avg_adv']
S_zz_adv = result.outputs['S_zz_avg_adv']

plot_stresses([S_yy_media, S_yy_adv], ['Media', 'Adventitia'], 'Circumferential', 'layer', ylabel=f"Circumferential stress [MPa]")
plot_stresses([S_zz_media, S_zz_adv], ['Media', 'Adventitia'], 'Axial', 'layer', ylabel=f"Axial stress [MPa]")

#%%
# ---------------------------------------------------------------
# Stress in components
# ---------------------------------------------------------------
s_yy_matrix_avg = result.outputs['s_yy_matrix_avg']
s_yy_cell_avg = result.outputs['s_yy_cell_avg']
s_yy_collagen_media_avg = result.outputs['s_yy_collagen_media_avg']
s_yy_collagen_adv_avg = result.outputs['s_yy_collagen_adv_avg']

s_zz_matrix_avg = result.outputs['s_zz_matrix_avg']
s_zz_cell_avg = result.outputs['s_zz_cell_avg']
s_zz_collagen_media_avg = result.outputs['s_zz_collagen_media_avg']
s_zz_collagen_adv_avg = result.outputs['s_zz_collagen_adv_avg']


plot_stresses([S_yy_media, s_yy_matrix_avg, s_yy_cell_avg, s_yy_collagen_media_avg], ['Media', 'Matrix', 'SMC', 'Collagen'], 'Circumferential', 'RVE_media', ylabel=f"Circumferential stress [MPa]")
plot_stresses([S_zz_media, s_zz_matrix_avg, s_zz_cell_avg, s_zz_collagen_media_avg], ['Media', 'Matrix', 'SMC', 'Collagen'], 'Axial', 'RVE_media', ylabel=f"Axial stress [MPa]")

#%%
# Cell mechanical state


f_cells = media_card['cells']['volumic_fraction']
f_coll_media = 4*media_card['collagen0']['volumic_fraction']
f_matrix = 1 - f_cells - f_coll_media

s_yy_matrix_prop = f_matrix*s_yy_matrix_avg/S_yy_media
s_yy_cell_prop = f_cells*s_yy_cell_avg/S_yy_media
s_yy_collagen_media_prop = f_coll_media*s_yy_collagen_media_avg/S_yy_media

s_zz_matrix_prop = f_matrix*s_zz_matrix_avg/S_zz_media
s_zz_cell_prop = f_cells*s_zz_cell_avg/S_zz_media
s_zz_collagen_media_prop = f_coll_media*s_zz_collagen_media_avg/S_zz_media


plot_stresses([s_yy_matrix_prop, s_yy_cell_prop, s_yy_collagen_media_prop], ['Matrix', 'SMC', 'Collagen'], 'Circumferential', 'distribution', ylabel=f"Normalized Circumferential stress", ylim=[0,1], hatched_phase1=True)
plot_stresses([s_zz_matrix_prop, s_zz_cell_prop, s_zz_collagen_media_prop], ['Matrix', 'SMC', 'Collagen'], 'Axial', 'distribution', ylabel=f"Normalized Axial stress", ylim=[0,1])

#%%
# plot calibrated young modulus law

stretch_list = np.linspace(1, 2.5, 100)
e0 = 0.67
k = 3.64
l = 1.1
young_list = e0*np.exp(k*(stretch_list - l))

plt.figure()
plt.plot(stretch_list, young_list)
plt.grid()
plt.show()