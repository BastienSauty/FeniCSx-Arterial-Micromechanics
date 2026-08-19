#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting / post-processing for the CALIBRATED arterial tissue model.

Structure mirrors a single simulation script (like main_ArterialTissue /
3_1_calibration_ArterialTissue without the calibration loop) : load cards,
look for an existing cached result and load it, otherwise run the simulation
once, then plot. No calibration, no persistent-context machinery - this is a
one-off run, so it uses the plain run_simulation() from
main_ArterialTissue_26_07_21.py (also needed for XDMF export support, which
the persistent setup_simulation/solve_simulation path does not have).

Inputs (all produced by 3_1_calibration_ArterialTissue_persistent.py, except
simu_card_calib.json which is a separate, hand-maintained card - typically a
finer mesh than the one used during calibration, since speed no longer
matters for this single final run) :
    json_cards/simu_card_calib.json                          (mesh/loading resolution for this run)
    outputs/{folder_name}/media_card_calibrated.json          (calibrated media parameters)
    outputs/{folder_name}/adventitia_card_calibrated.json     (calibrated adventitia parameters)
    outputs/{folder_name}/calibrated_geometry.json             (calibrated ri/re/ri_adv/area/advTF)

The calibrated geometry is applied ON TOP of simu_card_calib.json's own
ri/re/ri_adv/area/advTF (mesh resolution nr/nz/n_int/n_NR/load_phase/
XDMF_export come from simu_card_calib.json itself, independent of what was
used for calibration).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle
import json

import matplotlib as mpl
import matplotlib.ticker as ticker

mpl.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc} \usepackage{lmodern}",
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9
})

#-----------------------------------------------------------------------------#
# Custom modules
#-----------------------------------------------------------------------------#
from Multiscale_Framework.class_modules.load_class import Artery_load
from Article_Calibration.main_ArterialTissue_26_07_21 import (
    run_simulation,
    load_JSON,
)

from Multiscale_Framework.function_modules.discretization_collagen import  (
    discretizing_distribution,
    plot_PDF_discrete,
    build_CDF,
    plot_CDF_discrete
)


#-----------------------------------------------------------------------------#
# Load general parameters
#-----------------------------------------------------------------------------#

folder_name = 'Article_Calibration'
name = '4_1_plot_ArterialTissue_calibrated'
namefile = name + '_scal.pkl'

simu_card_name = 'json_cards/simu_card_calib.json'
media_card_name = f'./outputs/{folder_name}/media_card_calibrated.json'
adventitia_card_name = f'./outputs/{folder_name}/adventitia_card_calibrated.json'
geometry_name = f'./outputs/{folder_name}/calibrated_geometry.json'

simu_card = load_JSON(simu_card_name)
media_card = load_JSON(media_card_name)
adventitia_card = load_JSON(adventitia_card_name)
calibrated_geometry = load_JSON(geometry_name)

# Apply the calibrated geometry on top of this (possibly finer-mesh) simu_card -
# mesh resolution (nr/nz), integration settings, and load_phase all come from
# simu_card_calib.json itself, only the physical geometry is overwritten.
simu_card['ri'] = calibrated_geometry['ri']
simu_card['re'] = calibrated_geometry['re']
simu_card['ri_adv'] = calibrated_geometry['ri_adv']
simu_card['area'] = calibrated_geometry['area']
simu_card['advTF'] = calibrated_geometry['advTF']

load_phase = simu_card['load_phase']
step_load = Artery_load(load_phase)

collagen_keys_media = [k for k in media_card.keys() if k.startswith("collagen")]
collagen_keys_adventitia = [k for k in adventitia_card.keys() if k.startswith("collagen")]

#-----------------------------------------------------------------------------#
# Load Experimental Data (for comparison plots only - not used to change any
# parameter here, the cards are already calibrated)
#-----------------------------------------------------------------------------#
filename = os.path.join(folder_name, "DTAavg.npz")
data = np.load(filename, allow_pickle=True)

# Orientation data for collagen fibers
orientation_angles = data["orientation_angles"]
orientation_Low = data["orientation_Low"]
orientation_Dias = data["orientation_Dias"]
orientation_Sys = data["orientation_Sys"]

Pressure_mmHg = data["Pressure_mmHg"]
InnerRadius_mm = data["InnerRadius_mm"]
OuterRadius_mm = data["OuterRadius_mm"]
Fzz_Sample_mN = data["Fzz_Sample_mN"]

filename_std = os.path.join(folder_name, "DTAsem.npz")
data_std = np.load(filename_std, allow_pickle=True)
F_zz_sd_exp = data_std['Fzz_Sample_mN']
re_sd_exp = data_std['OuterRadius_mm']

press_exp = Pressure_mmHg
ri_exp = InnerRadius_mm
re_exp = OuterRadius_mm
F_zz_exp = Fzz_Sample_mN

plt.figure(figsize=(4, 3))
plt.plot(press_exp, ri_exp, label='$R_i^{{exp}}$')
plt.plot(press_exp, re_exp, label='$R_e^{{exp}}$')
plt.legend()
plt.grid()
plt.xlabel('Internal Pressure [mmHg]')
plt.ylabel('Radius [mm]')
plt.tight_layout()
plt.savefig(f'images_output/{folder_name}/{name}_exp_pressure_radius.pdf')
plt.show()

plt.figure(figsize=(4, 3))
plt.plot(press_exp, F_zz_exp, label='$F_{{zz}}^{{exp}}$')
plt.legend()
plt.grid()
plt.xlabel('Internal Pressure [mmHg]')
plt.ylabel('Axial Force [mN]')
plt.tight_layout()
plt.savefig(f'images_output/{folder_name}/{name}_exp_force_pressure.pdf')
plt.show()

# weights_coll_adv is needed below for the orientation-PDF comparison plots.
# Recomputed deterministically from the same experimental distribution/N used
# during calibration (see 3_1_calibration_ArterialTissue_persistent.py) -
# NOT used to overwrite adventitia_card's theta/volumic_fraction, which are
# already the calibrated values loaded from adventitia_card_calibrated.json.
N = len(collagen_keys_adventitia)
theta_coll_adv, weights_coll_adv = discretizing_distribution(orientation_angles, orientation_Low, N)#, name + '_check', folder_name, plot=False, verbose=False)

if not os.path.exists(f"images_output/{folder_name}"):
    os.makedirs(f"images_output/{folder_name}")
if not os.path.exists(f"outputs/{folder_name}"):
    os.makedirs(f"outputs/{folder_name}")

#-----------------------------------------------------------------------------#
# Run (or load a cached) simulation with the calibrated cards
#-----------------------------------------------------------------------------#
try:
    with open(f'./outputs/{folder_name}/{namefile}', 'rb') as file:
        result = pickle.load(file)
    print(f"Loaded cached result from ./outputs/{folder_name}/{namefile}")
except FileNotFoundError:
    print(f"No cached result found - running simulation {name}")
    result, mech = run_simulation(name, folder_name, simu_card, adventitia_card, media_card)

#-----------------------------------------------------------------------------#
# Plot results -- Post Processing
#-----------------------------------------------------------------------------#
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
    figsize=(4, 3),
    sharey=True,
    constrained_layout=True
)

ax1.plot(steps, lambdaz_list, color='tab:blue', label=r'$\lambda_z$')
ax1.set_ylabel(r'Axial stretch $\lambda_z$')

ax2 = ax1.twinx()
ax2.plot(steps, press_list, color='tab:red', label=f'$P_{{blood}}$')
ax2.set_ylabel('Pressure [mmHg]')

ax1.set_xlabel('Step number')
ax1.set_xlim(0, len(steps)-1)

x_sep = step_load.index_phase[0][1]
ax1.axvline(x=x_sep, color='k', linestyle='--', linewidth=1)

ax1.axvspan(0, x_sep, facecolor='lightgray', alpha=0.4, zorder=-1)
ax1.axvspan(x_sep, len(steps), facecolor='lightyellow', alpha=0.7, zorder=-1)

ax1.grid(True, linestyle=":", linewidth=0.5)
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', frameon=True)

y_max = max(lambdaz_list) * 1.025
ax1.text(x=x_sep/2, y=y_max, s='Phase 1', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.text(x=x_sep + (len(steps)-x_sep)/2, y=y_max, s='Phase 2', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.savefig(f'images_output/{folder_name}/{name}_load_phase.pdf')
plt.show()

#%%
#-----------------------------------------------------------------------------#
# Pressure - outer radius & axial force, combined on a twin y-axis
# (Ri dropped, experimental points shown with error bars from DTAstd.npz)
#-----------------------------------------------------------------------------#
re_d = result.outputs['re_d'][:]
ri_d = result.outputs['ri_d'][:]
F_zz = result.outputs['F_zz'][:]*1000
area = simu_card['area']
ri_incompr = np.sqrt(re_d**2 - area/np.pi)

 
color_re = 'tab:blue'
color_ri = 'tab:orange'
color_ri_incompr = 'tab:red'
color_fz = 'tab:orange'

fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(4,3),
    constrained_layout=True,
    gridspec_kw={'width_ratios': [n_phase1, n_phase2], 'wspace': 0.05}
)

ax2.sharey(ax1)    # Re: same y-axis across Phase 1 & Phase 2


# --- Phase 1 (model only, no experimental data available) ---
ax1.plot(lambdaz_list[indices_1], re_d[indices_1], color=color_re, label=r'$R_e$')
 
ax2.plot(press_exp, re_exp, color=color_re, linestyle='', marker='+',markevery=15, label=r'$R_e^{exp}$')
ax1.plot(lambdaz_list[indices_1], ri_d[indices_1], color=color_ri, label=r'$R_i$')

ax1.plot(lambdaz_list[indices_1], ri_incompr[indices_1], color=color_ri_incompr, linestyle='-', label=r'$R_i^{incompr}$')
 
 
ax1.set_xlim([np.min(lambdaz_list), np.max(lambdaz_list)])
ax1.autoscale(enable=True, axis='x', tight=True)
ax1.set_xlabel(r"Axial stretch $\lambda_z$")
ax1.set_ylabel("Outer Radius [mm]")
ax1.tick_params(axis='y', left=True, labelleft=True)
ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
ax1.set_title('Phase 1')
ax1.grid(axis='both', linestyle=":", linewidth=0.5)
 
# --- Phase 2 (model + experimental data as a shaded std band) ---
ax2.plot(press_list[indices_2], re_d[indices_2], color=color_re, label=r'$R_e$')
ax2.plot(press_list[indices_2], ri_d[indices_2], color=color_ri, label=r'$R_i$')
ax2.fill_between(press_exp, re_exp -  re_sd_exp, re_exp +  re_sd_exp, color=color_re, alpha=0.2)

ax2.plot(press_list[indices_2], ri_incompr[indices_2], color=color_ri_incompr, linestyle='-', label=r'$R_i^{incompr}$')
 
ax2.set_xlabel("Pressure [mmHg]")
ax2.set_facecolor((1.0, 1.0, 0.88, 0.7))
ax2.set_xlim([np.min(press_list), np.max(press_list)])
ax2.autoscale(enable=True, axis='x', tight=True)
ax2.set_title('Phase 2')
ax2.grid(axis='both', linestyle=":", linewidth=0.5)
ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)

lines_1, labels_1 = ax2.get_legend_handles_labels()
ax2.legend(lines_1, labels_1, loc='lower right', frameon=True, fontsize=8)

plt.savefig(f'images_output/{folder_name}/{name}_pressure_radius.pdf')
plt.show()


fig, (ax1, ax2) = plt.subplots(
    1, 2,
    figsize=(4,3),
    constrained_layout=True,
    gridspec_kw={'width_ratios': [n_phase1, n_phase2], 'wspace': 0.05}
)

ax2.sharey(ax1)    # Re: same y-axis across Phase 1 & Phase 2


# --- Phase 1 (model only, no experimental data available) ---
ax1.plot(lambdaz_list[indices_1], F_zz[indices_1], color=color_fz, label=r'$F_{zz}$')
 
ax1.set_xlim([np.min(lambdaz_list), np.max(lambdaz_list)])
ax1.autoscale(enable=True, axis='x', tight=True)
ax1.set_xlabel(r"Axial stretch $\lambda_z$")
ax1.set_ylabel("Axial Force [mN]")
ax1.tick_params(axis='y', left=True, labelleft=True)
ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
ax1.set_title('Phase 1')
ax1.grid(axis='both', linestyle=":", linewidth=0.5)
 
# --- Phase 2 (model + experimental data as a shaded std band) ---
ax2.plot(press_list[indices_2], F_zz[indices_2], color=color_fz, label=r'$F_{zz}$')
 
ax2.plot(press_exp, F_zz_exp, color=color_fz, linestyle='', marker='+',markevery=15, label=r'$F_{zz}^{exp}$')
ax2.fill_between(press_exp, F_zz_exp -  np.sqrt(7)*F_zz_sd_exp, F_zz_exp +  np.sqrt(7)*F_zz_sd_exp, color=color_fz, alpha=0.2)
 
ax2.set_xlabel("Pressure [mmHg]")
ax2.set_facecolor((1.0, 1.0, 0.88, 0.7))
ax2.set_xlim([np.min(press_list), np.max(press_list)])
ax2.autoscale(enable=True, axis='x', tight=True)
ax2.set_title('Phase 2')
ax2.grid(axis='both', linestyle=":", linewidth=0.5)
ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)

lines_1, labels_1 = ax2.get_legend_handles_labels()
ax2.legend(lines_1, labels_1, loc='lower right', frameon=True, fontsize=8)

plt.savefig(f'images_output/{folder_name}/{name}_pressure_force.pdf')
plt.show()

#%%
#-----------------------------------------------------------------------------#
# Collagen kinematics
#-----------------------------------------------------------------------------#
def plotter_2_phases_Collagen(layer_key, key_suffix, fig_name):
    """
    input:
        layer_key : 'media' or 'adv'
        key_suffix : 'theta' for orientation, 'stretch' for stretch and 'young' for young modulus
        fig_name : name of pdf output
    """
    if layer_key == 'adv':
        collagen_keys = collagen_keys_adventitia
    elif layer_key == 'media':
        collagen_keys = collagen_keys_media

    if key_suffix == 'theta':
        ylim = [-90, 90]
        ylabel = "Angle to axial direction [°]"
        y_ticks = ticker.MultipleLocator(30)
    elif key_suffix == 'stretch':
        ylim = None
        ylabel = "Fiber stretch"
        y_ticks = ticker.MultipleLocator(0.5)
    elif key_suffix == 'young':
        ylim = None
        ylabel = "Fiber Young Modulus [MPa]"
        y_ticks = None

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(5, 3),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={'width_ratios': [n_phase1, n_phase2], 'wspace': 0.05}
    )

    for key in collagen_keys:
        ax1.plot(
            lambdaz_list[indices_1],
            result.outputs[f'{key}_{layer_key}_{key_suffix}'][indices_1],
            label=key
        )

    ax1.set_xlim([np.min(lambdaz_list), np.max(lambdaz_list)])
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel(ylabel)
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)
    ax1.grid(True, linestyle=":", linewidth=0.5)

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
    ax2.tick_params(left=False, labelleft=False)
    ax2.grid(True, linestyle=":", linewidth=0.5)
    ax2.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0, frameon=False)

    plt.savefig(f"images_output/{folder_name}/{name}_{fig_name}.pdf")
    plt.show()


plotter_2_phases_Collagen('adv', 'theta', 'adventitia_collagen_orient')
plotter_2_phases_Collagen('media', 'theta', 'media_collagen_orient')
plotter_2_phases_Collagen('adv', 'stretch', 'adventitia_collagen_stretch')
plotter_2_phases_Collagen('media', 'stretch', 'media_collagen_stretch')
plotter_2_phases_Collagen('adv', 'young', 'adventitia_collagen_young')
plotter_2_phases_Collagen('media', 'young', 'media_collagen_young')


#-----------------------------------------------------------------------------#
# Comparison with experimental values : orientation PDF at specific pressures
#-----------------------------------------------------------------------------#
def extract_angles(result, press_value):
    """extract the orientation angles of the collagen fibers"""
    keys = result.outputs.keys()
    theta_keys = [k for k in keys if k.startswith('collagen') and k.endswith('adv_theta')]
    pressure_list = result.outputs['press']

    target_pressure_mpa = press_value * 0.000133322
    index = np.argmin(np.abs(pressure_list - target_pressure_mpa))

    angles_model = [result.outputs[k][index] for k in theta_keys]
    return np.array(angles_model)

angles_model_Low = extract_angles(result, 10.)
angles_model_Dias = extract_angles(result, 80.)
angles_model_Sys = extract_angles(result, 120.)

# plot_PDF_discrete(discrete_weights, discrete_angles, continuous_pdf, continuous_angles, folder_name, name))
plot_PDF_discrete(weights_coll_adv, theta_coll_adv, orientation_Low, orientation_angles, folder_name, name+'_Init')
plot_PDF_discrete(weights_coll_adv, angles_model_Low, orientation_Low, orientation_angles, folder_name, name+'_Low')
plot_PDF_discrete(weights_coll_adv, angles_model_Dias, orientation_Dias, orientation_angles, folder_name, name+'_Dias')
plot_PDF_discrete(weights_coll_adv, angles_model_Sys, orientation_Sys, orientation_angles, folder_name, name+'_Sys')

cdf_Low_exp = build_CDF(orientation_angles, orientation_Low, orientation_angles, integral_type='Trapezoidal', normalize=True)
cdf_Dias_exp = build_CDF(orientation_angles, orientation_Dias, orientation_angles, integral_type='Trapezoidal', normalize=True)
cdf_Sys_exp = build_CDF(orientation_angles, orientation_Sys, orientation_angles, integral_type='Trapezoidal', normalize=True)

cdf_Init_model = build_CDF(theta_coll_adv, weights_coll_adv, orientation_angles) 
cdf_Low_model = build_CDF(angles_model_Low, weights_coll_adv, orientation_angles) 
cdf_Dias_model = build_CDF(angles_model_Dias, weights_coll_adv, orientation_angles)
cdf_Sys_model= build_CDF(angles_model_Sys, weights_coll_adv, orientation_angles)

plot_CDF_discrete(cdf_Init_model, cdf_Low_exp, orientation_angles, folder_name, name+'_Init')
plot_CDF_discrete(cdf_Low_model, cdf_Low_exp, orientation_angles, folder_name, name+'_Low')
plot_CDF_discrete(cdf_Dias_model, cdf_Dias_exp, orientation_angles, folder_name, name+'_Dias')
plot_CDF_discrete(cdf_Sys_model, cdf_Sys_exp, orientation_angles, folder_name, name+'_Sys')

#%%
#-----------------------------------------------------------------------------#
# Article Figure 20 - discretization quality, same-axis PDF + shaded CDF
#-----------------------------------------------------------------------------#
# The discrete fiber angles are only equispaced at t=0 (that's how
# discretizing_distribution builds them); at Diastolic/Systolic each family
# has kinematically rotated to a different angle, so a single fixed bin
# width can no longer convert weight -> density. local_bin_widths() gives
# each angle its own width via the midpoint to its neighbors (a 1D Voronoi
# partition of the domain, clipped at the edges), so weight/width stays a
# proper density - comparable to the continuous PDF - at every state.

def local_bin_widths(angles, domain):
    """
    Per-point bin widths for non-uniformly spaced 1D samples, via midpoints
    to nearest neighbors, clipped to `domain=(low, high)`.

    Returns
    -------
    widths : array, same order as `angles` (not sorted)
    edges  : array, ascending bin edges (sorted order) - for stairs()
    """
    angles = np.asarray(angles, dtype=float)
    order = np.argsort(angles)
    sorted_angles = angles[order]

    edges = np.empty(len(sorted_angles) + 1)
    edges[1:-1] = 0.5 * (sorted_angles[:-1] + sorted_angles[1:])
    edges[0] = domain[0]
    edges[-1] = domain[1]

    widths_sorted = np.diff(edges)
    if np.any(widths_sorted <= 0):
        raise ValueError(
            "local_bin_widths: duplicate/unsorted angles found -> zero or "
            "negative bin width. Merge coincident families before calling "
            "(can happen at highly-aligned states like systolic)."
        )

    widths = np.empty_like(widths_sorted)
    widths[order] = widths_sorted
    return widths, edges


def circular_std_deg(angles_deg, weights, period_deg=180.0):
    """
    Circular standard deviation (Mardia & Jupp, 1999) via the angle-doubling
    trick, for axial (period=180 deg) fiber-orientation data. `weights`
    need not be pre-normalized. Reimplemented locally (self-contained, no
    extra import) - swap for your own circular_dispersion_deg from the SA
    pipeline if you want bit-identical numbers to those scripts.
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    theta = np.deg2rad(np.asarray(angles_deg, dtype=float)) * (360.0 / period_deg)
    C = np.sum(w * np.cos(theta))
    S = np.sum(w * np.sin(theta))
    R = min(np.hypot(C, S), 1.0 - 1e-12)  # guard against log(0) from roundoff
    circ_std_doubled = np.sqrt(-2.0 * np.log(R))  # radians, doubled scale
    return np.rad2deg(circ_std_doubled) * (period_deg / 360.0)


def plot_PDF_shared_axis(discrete_weights, discrete_angles, continuous_pdf,
                          continuous_angles, folder_name, fig_prefix, state_label):
    """
    Converts each discrete weight to a density (weight / local bin width,
    via local_bin_widths) so the discrete distribution's SHAPE can be
    compared to the continuous PDF on one shared axis - mismatches are
    then visible directly rather than hidden behind an independently-
    rescaled second axis. The actual discrete weights (family volume
    fractions) are also plotted as red dots on a secondary axis, same
    convention as the original plot_PDF_discrete, so the real modeled
    Dirac masses are still visible on their own natural scale.
    """
    domain = (continuous_angles.min(), continuous_angles.max())
    widths, edges = local_bin_widths(discrete_angles, domain)
    order = np.argsort(discrete_angles)
    density_sorted = np.asarray(discrete_weights)[order] / widths

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.plot(continuous_angles, continuous_pdf, color='tab:blue',
            label='Experimental PDF', linewidth=1.5)
    ax.stairs(density_sorted, edges, color='tab:red', baseline=0, fill=False,
              linewidth=1.3, label='Discrete density (weight / bin width)')
    ax.set_xlabel("Angle to axial direction [°]")
    ax.set_ylabel("Probability density [1/°]")
    ax.set_xlim(domain)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle=":", linewidth=0.5)

    # Raw discrete weights on their own axis, as in the original
    # plot_PDF_discrete - the density stairs above show whether the shape
    # is well captured, this shows the actual modeled Dirac masses
    # (volume fractions) without going through the bin-width conversion.
    ax2 = ax.twinx()
    ax2.scatter(discrete_angles, discrete_weights, color='tab:red', marker='o',
                s=22, zorder=5, label='Discrete PDF (family weight)')
    ax2.set_ylabel("Discrete family weight [-]", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(bottom=0)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower center',
              fontsize=7, frameon=False)

    plt.savefig(f"images_output/{folder_name}/{fig_prefix}_{state_label}_pdf.pdf")
    plt.show()


def plot_CDF_shaded_wasserstein(cdf_model, cdf_exp, grid, folder_name,
                                 fig_prefix, state_label, circ_std=None):
    """
    CDF comparison with the region between the two curves shaded: that area
    is exactly the raw Wasserstein-1 distance (in degrees), since
    W1 = integral |F_model - F_exp| dtheta (same trapz definition as the
    corrected compute_wasserstein_distance). Annotates the raw value (and
    the normalized value, if circ_std is supplied) directly on the plot.
    """
    grid = np.asarray(grid)
    w1_raw = np.trapz(np.abs(np.asarray(cdf_model) - np.asarray(cdf_exp)), grid)

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.plot(grid, cdf_exp, color='tab:blue', label='Experimental CDF', linewidth=1.5)
    ax.plot(grid, cdf_model, color='tab:red', label='Discrete CDF', linewidth=1.5)
    ax.fill_between(grid, cdf_model, cdf_exp, color='tab:red', alpha=0.15)

    if circ_std is not None:
        annotation = rf'$W_1={w1_raw:.2f}$° (norm. {100.0*w1_raw/circ_std:.1f}\%)'
    else:
        annotation = rf'$W_1={w1_raw:.2f}$°'
    ax.text(0.03, 0.95, annotation, transform=ax.transAxes,
            fontsize=8, va='top', ha='left')

    ax.set_xlabel("Angle to axial direction [°]")
    ax.set_ylabel("Cumulative density function")
    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc='lower right', fontsize=7, frameon=False)

    plt.savefig(f"images_output/{folder_name}/{fig_prefix}_{state_label}_cdf.pdf")
    plt.show()

    return w1_raw

circ_std_Low = circular_std_deg(orientation_angles, orientation_Low)
circ_std_Dias = circular_std_deg(orientation_angles, orientation_Dias)
circ_std_Sys = circular_std_deg(orientation_angles, orientation_Sys)

plot_PDF_shared_axis(weights_coll_adv, theta_coll_adv, orientation_Low,
                      orientation_angles, folder_name, name, 'Init')
plot_PDF_shared_axis(weights_coll_adv, angles_model_Low, orientation_Low,
                      orientation_angles, folder_name, name, 'Low')
plot_PDF_shared_axis(weights_coll_adv, angles_model_Dias, orientation_Dias,
                      orientation_angles, folder_name, name, 'Dias')
plot_PDF_shared_axis(weights_coll_adv, angles_model_Sys, orientation_Sys,
                      orientation_angles, folder_name, name, 'Sys')

plot_CDF_shaded_wasserstein(cdf_Init_model, cdf_Low_exp, orientation_angles,
                             folder_name, name, 'Init', circ_std_Low)
plot_CDF_shaded_wasserstein(cdf_Low_model, cdf_Low_exp, orientation_angles,
                         folder_name, name, 'Init', circ_std_Low)
plot_CDF_shaded_wasserstein(cdf_Dias_model, cdf_Dias_exp, orientation_angles,
                             folder_name, name, 'Dias', circ_std_Dias)
plot_CDF_shaded_wasserstein(cdf_Sys_model, cdf_Sys_exp, orientation_angles,
                             folder_name, name, 'Sys', circ_std_Sys)

#%%
#-----------------------------------------------------------------------------#
# Stress distributions between layers and components
#-----------------------------------------------------------------------------#
def plot_stresses(list_curves, curve_name, stress_name, figname, ylabel, ylim=None, hatched_phase1=False):
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(4, 3),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={'width_ratios': [n_phase1, n_phase2], 'wspace': 0.05}
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
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)

    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.7))
    ax2.set_xlim([np.min(press_list), np.max(press_list)])
    if ylim:
        ax2.set_ylim(ylim)
    else:
        ylim = ax2.get_ylim()

    if hatched_phase1:
        ax1.fill_between(lambdaz_list[indices_1], ylim[0], ylim[1], color='gray', alpha=0.3, hatch='//')

    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)
    ax2.legend(loc='best')

    plt.savefig(f'images_output/{folder_name}/{name}_{figname}_{stress_name}_stress.pdf')
    plt.show()


# ---------------------------------------------------------------
# Stress in layers
# ---------------------------------------------------------------
S_yy_media = result.outputs['S_yy_avg_media']
S_zz_media = result.outputs['S_zz_avg_media']
S_yy_adv = result.outputs['S_yy_avg_adv']
S_zz_adv = result.outputs['S_zz_avg_adv']

plot_stresses([S_yy_media, S_yy_adv], ['Media', 'Adventitia'], 'Circumferential', 'layer', ylabel="Circumferential stress [MPa]")
plot_stresses([S_zz_media, S_zz_adv], ['Media', 'Adventitia'], 'Axial', 'layer', ylabel="Axial stress [MPa]")

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

plot_stresses([S_yy_media, s_yy_matrix_avg, s_yy_cell_avg, s_yy_collagen_media_avg], ['Media', 'Matrix', 'SMC', 'Collagen'], 'Circumferential', 'RVE_media', ylabel="Circumferential stress [MPa]")
plot_stresses([S_zz_media, s_zz_matrix_avg, s_zz_cell_avg, s_zz_collagen_media_avg], ['Media', 'Matrix', 'SMC', 'Collagen'], 'Axial', 'RVE_media', ylabel="Axial stress [MPa]")

# ---------------------------------------------------------------
# Cell mechanical state
# ---------------------------------------------------------------
f_cells = media_card['cells']['volumic_fraction']
f_coll_media = sum(media_card[collagen_keys_media[i]]['volumic_fraction'] for i in range(len(collagen_keys_media)))
f_matrix = 1 - f_cells - f_coll_media

s_yy_matrix_prop = f_matrix*s_yy_matrix_avg/S_yy_media
s_yy_cell_prop = f_cells*s_yy_cell_avg/S_yy_media
s_yy_collagen_media_prop = f_coll_media*s_yy_collagen_media_avg/S_yy_media

s_zz_matrix_prop = f_matrix*s_zz_matrix_avg/S_zz_media
s_zz_cell_prop = f_cells*s_zz_cell_avg/S_zz_media
s_zz_collagen_media_prop = f_coll_media*s_zz_collagen_media_avg/S_zz_media

plot_stresses([s_yy_matrix_prop, s_yy_cell_prop, s_yy_collagen_media_prop], ['Matrix', 'SMC', 'Collagen'], 'Circumferential', 'distribution', ylabel="Normalized Circumferential stress", ylim=[0, 1], hatched_phase1=True)
plot_stresses([s_zz_matrix_prop, s_zz_cell_prop, s_zz_collagen_media_prop], ['Matrix', 'SMC', 'Collagen'], 'Axial', 'distribution', ylabel="Normalized Axial stress", ylim=[0, 1])