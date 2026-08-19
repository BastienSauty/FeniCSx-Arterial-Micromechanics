#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting for the cell-mechanics sensitivity analysis (matrix / SMC / collagen
Young's moduli) produced by 3_3_cell_mechanics_sa_persistent.py.

Adapted from chiv_3_3_cell_mechanics_ArterialTissue_25_10_06.py's six plot
functions (plot_pressure_radius, plot_layer_stress, plot_layer_distribution,
plot_circ_stress, plot_normalized_stress, plot_fiber_orientation) -- logic
and normalizations are unchanged, only:
  - the source of results (manifest + per-combination pickles written by
    3_3_..._persistent.py, instead of the thesis's per-combination
    pipeline_SA namefile),
  - f_coll_media, generalized from the thesis's hardcoded `4*collagen0` (a
    4-family, equal-weight assumption) to `sum(volumic_fraction over every
    collagen_i in the calibrated media card)` -- correct for any N and any
    (possibly non-uniform) family weights,
  - plot_fiber_orientation's output key, generalized from the thesis's
    hardcoded 'collagen_4_adv_theta' to the manifest's
    'dominant_adventitia_collagen_key' + '_adv_theta' (see 3_3's module
    docstring),
were adapted.

plot_layer_distribution is ported but, exactly like the thesis __main__,
left uncalled below (defined for later use, not part of the default figure
set).

This script never re-runs a simulation -- every result it plots comes from
a pickle written by 3_3_cell_mechanics_sa_persistent.py; combinations marked
"failed" in the manifest (solve crashed or didn't converge) are skipped.
"""

import json
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D

from dataclasses import dataclass

from Multiscale_Framework.class_modules.load_class import Artery_load
from Article_Calibration.main_ArterialTissue_persistent import load_JSON

mpl.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage[T1]{fontenc} \usepackage{lmodern}",
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})


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
# Plotting functions (ported from chiv_3_3_cell_mechanics_ArterialTissue_25_10_06.py,
# unchanged apart from the module-level lambdaz_list/press_list globals they
# read, which this script defines below from the manifest's simu_card).
#-----------------------------------------------------------------------------#

def plot_pressure_radius(folder_name, name, param_name, list_result, param_list, config: PlotConfig):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5, 3), sharey=True, constrained_layout=True,
        gridspec_kw={'width_ratios': [config.n_phase1, config.n_phase2], 'wspace': 0.05}
    )

    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]

    for i, result in enumerate(list_result):
        color = colors[i]
        ri_d = result.outputs['ri_d'][:]
        re_d = result.outputs['re_d'][:]

        ax1.plot(config.lambdaz_list[config.indices_1], ri_d[config.indices_1], color=color, linestyle='-', linewidth=1.5)
        ax1.plot(config.lambdaz_list[config.indices_1], re_d[config.indices_1], color=color, linestyle='--', linewidth=1.5)

        ax2.plot(config.press_list[config.indices_2], ri_d[config.indices_2], color=color, linestyle='-', linewidth=1.5)
        ax2.plot(config.press_list[config.indices_2], re_d[config.indices_2], color=color, linestyle='--', linewidth=1.5)

    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel("Radius [mm]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)

    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.3))
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)

    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'$R_i$ (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'$R_e$ (dashed)'),
    ]
    style_legend_box = ax2.legend(
        handles=style_legend, loc='lower right', fontsize=8, frameon=True,
        framealpha=0.85, facecolor='white', edgecolor='gray', fancybox=True,
        borderpad=0.4, title_fontsize=9,
    )
    ax2.add_artist(style_legend_box)

    legend_elements = [
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2, label=rf'{param_list[i]:.3g}')
        for i in range(len(list_result))
    ]
    ax2.legend(
        handles=legend_elements, title=rf'${param_name}$ [MPa]', loc='center left',
        bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0, frameon=False, fontsize=9, title_fontsize=10,
    )

    plt.savefig(f'images_output/{folder_name}/{name}_{param_name}_pressure_radius.pdf')
    plt.show()


def plot_layer_stress(folder_name, name, param_name, list_result, param_list, stress_direction, config: PlotConfig):
    """stress_direction : 'Circumferential' or 'Axial'"""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5, 3), sharey=True, constrained_layout=True,
        gridspec_kw={'width_ratios': [config.n_phase1, config.n_phase2], 'wspace': 0.05}
    )

    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]

    for i, result in enumerate(list_result):
        color = colors[i]
        if stress_direction == 'Circumferential':
            S_yy_adv = result.outputs['S_yy_avg_adv'][:]
            S_yy_media = result.outputs['S_yy_avg_media'][:]
        elif stress_direction == 'Axial':
            S_yy_adv = result.outputs['S_zz_avg_adv'][:]
            S_yy_media = result.outputs['S_zz_avg_media'][:]

        ax1.plot(config.lambdaz_list[config.indices_1], S_yy_media[config.indices_1], color=color, linestyle='-', linewidth=1.5)
        ax1.plot(config.lambdaz_list[config.indices_1], S_yy_adv[config.indices_1], color=color, linestyle='--', linewidth=1.5)

        ax2.plot(config.press_list[config.indices_2], S_yy_media[config.indices_2], color=color, linestyle='-', linewidth=1.5)
        ax2.plot(config.press_list[config.indices_2], S_yy_adv[config.indices_2], color=color, linestyle='--', linewidth=1.5)

    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel(f"{stress_direction} stress [MPa]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)

    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.3))
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)

    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'Media (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'Adventitia (dashed)'),
    ]
    style_legend_box = ax2.legend(
        handles=style_legend, loc='best', fontsize=8, frameon=True, framealpha=0.85,
        facecolor='white', edgecolor='gray', fancybox=True, borderpad=0.4, title_fontsize=9,
    )
    ax2.add_artist(style_legend_box)

    legend_elements = [
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2, label=rf'{param_list[i]:.3g}')
        for i in range(len(list_result))
    ]
    ax2.legend(
        handles=legend_elements, title=rf'${param_name}$ [MPa]', loc='center left',
        bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0, frameon=False, fontsize=9, title_fontsize=10,
    )

    plt.savefig(f'images_output/{folder_name}/{name}_{param_name}_layer_{stress_direction}_stress.pdf')
    plt.show()


def plot_layer_distribution(folder_name, name, param_name, list_result, param_list, config: PlotConfig):
    """
    Ratio of stresses between layers (media/adventitia). Ported but left
    UNCALLED below, exactly like the thesis __main__ (defined, unused).
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5, 3), sharey=True, constrained_layout=True,
        gridspec_kw={'width_ratios': [config.n_phase1, config.n_phase2], 'wspace': 0.05}
    )

    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]

    for i, result in enumerate(list_result):
        color = colors[i]
        S_yy_adv = result.outputs['S_yy_avg_adv'][:]
        S_yy_media = result.outputs['S_yy_avg_media'][:]
        S_zz_adv = result.outputs['S_zz_avg_adv'][:]
        S_zz_media = result.outputs['S_zz_avg_media'][:]

        S_zz_props = S_zz_media / S_zz_adv
        S_yy_props = S_yy_media / S_yy_adv

        ax1.plot(config.lambdaz_list[config.indices_1], S_yy_props[config.indices_1], color=color, linestyle='-', linewidth=1.5)
        ax1.plot(config.lambdaz_list[config.indices_1], S_zz_props[config.indices_1], color=color, linestyle='--', linewidth=1.5)

        ax2.plot(config.press_list[config.indices_2], S_yy_props[config.indices_2], color=color, linestyle='-', linewidth=1.5)
        ax2.plot(config.press_list[config.indices_2], S_zz_props[config.indices_2], color=color, linestyle='--', linewidth=1.5)

    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel("Ratio of the stress component in the media over the adventitia")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)
    ax1.set_ylim([-1, 2])
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)

    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.3))
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)

    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'Circumferential (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'Axial (dashed)'),
    ]
    style_legend_box = ax2.legend(
        handles=style_legend, loc='best', fontsize=8, frameon=True, framealpha=0.85,
        facecolor='white', edgecolor='gray', fancybox=True, borderpad=0.4, title_fontsize=9,
    )
    ax2.add_artist(style_legend_box)

    legend_elements = [
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2, label=rf'{param_list[i]:.3g}')
        for i in range(len(list_result))
    ]
    ax2.legend(
        handles=legend_elements, title=rf'${param_name}$ [MPa]', loc='center left',
        bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0, frameon=False, fontsize=9, title_fontsize=10,
    )

    plt.savefig(f'images_output/{folder_name}/{name}_{param_name}_layer_distribution_stress.pdf')
    plt.show()


def plot_circ_stress(folder_name, name, param_name, list_result, param_list, config: PlotConfig):
    """Media micromechanics stress decomposition (matrix/SMC/collagen/tissue), circumferential."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5, 3), sharey=True, constrained_layout=True,
        gridspec_kw={'width_ratios': [config.n_phase1, config.n_phase2], 'wspace': 0.05}
    )

    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]

    for i, result in enumerate(list_result):
        color = colors[i]
        stress_matrix = result.outputs['s_yy_matrix_avg']
        stress_cell = result.outputs['s_yy_cell_avg']
        stress_collagen = result.outputs['s_yy_collagen_media_avg']
        stress_hom = result.outputs['S_yy_avg_media']

        ax1.plot(config.lambdaz_list[config.indices_1], stress_matrix[config.indices_1], color=color, linestyle=':', linewidth=1.5)
        ax1.plot(config.lambdaz_list[config.indices_1], stress_cell[config.indices_1], color=color, linestyle='-', linewidth=1.5)
        ax1.plot(config.lambdaz_list[config.indices_1], stress_collagen[config.indices_1], color=color, linestyle='--', linewidth=1.5)
        ax1.plot(config.lambdaz_list[config.indices_1], stress_hom[config.indices_1], color=color, linestyle='-.', linewidth=1.5)

        ax2.plot(config.press_list[config.indices_2], stress_matrix[config.indices_2], color=color, linestyle=':', linewidth=1.5)
        ax2.plot(config.press_list[config.indices_2], stress_cell[config.indices_2], color=color, linestyle='-', linewidth=1.5)
        ax2.plot(config.press_list[config.indices_2], stress_collagen[config.indices_2], color=color, linestyle='--', linewidth=1.5)
        ax2.plot(config.press_list[config.indices_2], stress_hom[config.indices_2], color=color, linestyle='-.', linewidth=1.5)

    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel("Circumferential stress [MPa]")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)
    ax2.set_yscale('symlog', linthresh=1e-3)
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)

    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.3))
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)

    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle=':', label=r'$\sigma_{matrix}$ (dot)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'$\sigma_{smc}$ (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'$\sigma_{coll}$ (dashed)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='-.', label=r'$\sigma_{tissue}$ (dash-dot)'),
    ]
    style_legend_box = ax2.legend(
        handles=style_legend, loc='best', fontsize=8, frameon=True, framealpha=0.85,
        facecolor='white', edgecolor='gray', fancybox=True, borderpad=0.4, title_fontsize=9,
    )
    ax2.add_artist(style_legend_box)

    legend_elements = [
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2, label=rf'{param_list[i]:.3g}')
        for i in range(len(list_result))
    ]
    ax2.legend(
        handles=legend_elements, title=rf'${param_name}$ [MPa]', loc='center left',
        bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0, frameon=False, fontsize=9, title_fontsize=10,
    )

    plt.savefig(f'images_output/{folder_name}/{name}_{param_name}_stress_distribution.pdf')
    plt.show()


def plot_normalized_stress(folder_name, name, param_name, list_result, param_list, stress_direction, config: PlotConfig):
    """stress_direction : 'Circumferential' or 'Axial'. Media-only (no adventitia matrix/cell output exists)."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5, 3), sharey=True, constrained_layout=True,
        gridspec_kw={'width_ratios': [config.n_phase1, config.n_phase2], 'wspace': 0.05}
    )

    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]

    f_matrix = 1 - config.f_cells - config.f_coll_media

    for i, result in enumerate(list_result):
        color = colors[i]
        if stress_direction == 'Axial':
            stress_hom = result.outputs['S_zz_avg_media']
            stress_matrix_prop = f_matrix * result.outputs['s_zz_matrix_avg'] / stress_hom
            stress_cell_prop = config.f_cells * result.outputs['s_zz_cell_avg'] / stress_hom
            stress_collagen_prop = config.f_coll_media * result.outputs['s_zz_collagen_media_avg'] / stress_hom

            ax1.plot(config.lambdaz_list[config.indices_1], stress_cell_prop[config.indices_1], color=color, linestyle='-', linewidth=1.5)
            ax1.plot(config.lambdaz_list[config.indices_1], stress_collagen_prop[config.indices_1], color=color, linestyle='--', linewidth=1.5)
            ax1.plot(config.lambdaz_list[config.indices_1], stress_matrix_prop[config.indices_1], color=color, linestyle=':', linewidth=1.5)

        elif stress_direction == 'Circumferential':
            stress_hom = result.outputs['S_yy_avg_media']
            stress_matrix_prop = f_matrix * result.outputs['s_yy_matrix_avg'] / stress_hom
            stress_cell_prop = config.f_cells * result.outputs['s_yy_cell_avg'] / stress_hom
            stress_collagen_prop = config.f_coll_media * result.outputs['s_yy_collagen_media_avg'] / stress_hom

        ax2.plot(config.press_list[config.indices_2], stress_cell_prop[config.indices_2], color=color, linestyle='-', linewidth=1.5)
        ax2.plot(config.press_list[config.indices_2], stress_collagen_prop[config.indices_2], color=color, linestyle='--', linewidth=1.5)
        ax2.plot(config.press_list[config.indices_2], stress_matrix_prop[config.indices_2], color=color, linestyle=':', linewidth=1.5)

    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel(f"Normalized {stress_direction} stress")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)
    ax1.set_ylim([0, 1])
    if stress_direction == 'Circumferential':
        ax1.fill_between(
            config.lambdaz_list[config.indices_1], 0, 1, color='gray', alpha=0.3, hatch='//',
        )
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)

    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.3))
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)

    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'$\sigma_{smc}$ (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'$\sigma_{coll}$ (dashed)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle=':', label=r'$\sigma_{matrix}$ (dot)'),
    ]
    style_legend_box = ax2.legend(
        handles=style_legend, loc='best', fontsize=8, frameon=True, framealpha=0.85,
        facecolor='white', edgecolor='gray', fancybox=True, borderpad=0.4, title_fontsize=9,
    )
    ax2.add_artist(style_legend_box)

    legend_elements = [
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2, label=rf'{param_list[i]:.3g}')
        for i in range(len(list_result))
    ]
    ax2.legend(
        handles=legend_elements, title=rf'${param_name}$ [MPa]', loc='center left',
        bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0, frameon=False, fontsize=9, title_fontsize=10,
    )

    plt.savefig(f'images_output/{folder_name}/{name}_{param_name}_stress_{stress_direction}_distribution.pdf')
    plt.show()


def plot_fiber_orientation(folder_name, name, param_name, list_result, param_list, fiber_key, config: PlotConfig):
    """
    fiber_key : output key for the representative adventitia collagen family's
    angle, e.g. 'collagen_3_adv_theta' -- see manifest['dominant_adventitia_
    collagen_key'] (replaces the thesis's hardcoded 'collagen_4_adv_theta').
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5, 3), sharey=True, constrained_layout=True,
        gridspec_kw={'width_ratios': [config.n_phase1, config.n_phase2], 'wspace': 0.05}
    )

    cmap = plt.cm.get_cmap('viridis', len(list_result))
    colors = [cmap(i / (len(list_result) - 1)) for i in range(len(list_result))]

    for i, result in enumerate(list_result):
        color = colors[i]
        theta_coll = result.outputs[fiber_key]

        ax1.plot(config.lambdaz_list[config.indices_1], theta_coll[config.indices_1], color=color, linestyle='-', linewidth=1.5)
        ax1.plot(config.lambdaz_list[config.indices_1], theta_coll[config.indices_1], color=color, linestyle='--', linewidth=1.5)

        ax2.plot(config.press_list[config.indices_2], theta_coll[config.indices_2], color=color, linestyle='-', linewidth=1.5)
        ax2.plot(config.press_list[config.indices_2], theta_coll[config.indices_2], color=color, linestyle='--', linewidth=1.5)

    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.set_xlabel(r"Axial stretch $\lambda_z$")
    ax1.set_ylabel("Fiber Angle")
    ax1.set_facecolor((0.83, 0.83, 0.83, 0.4))
    ax1.set_title('Phase 1')
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)

    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.3))
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)

    legend_elements = [
        Line2D([0], [0], color=cmap(i / (len(list_result) - 1)), lw=2, label=rf'{param_list[i]:.3g}')
        for i in range(len(list_result))
    ]
    ax2.legend(
        handles=legend_elements, title=rf'${param_name}$ [MPa]', loc='center left',
        bbox_to_anchor=(1.05, 0.5), borderaxespad=0.0, frameon=False, fontsize=9, title_fontsize=10,
    )

    plt.savefig(f'images_output/{folder_name}/{name}_{param_name}_fiber_angle.pdf')
    plt.show()


#-----------------------------------------------------------------------------#
# Load manifest + cached results, build PlotConfig, produce every figure.
#-----------------------------------------------------------------------------#

if __name__ == "__main__":

    folder_name = 'Article_Calibration_SA'
    name = '3_3_cell_mechanics_SA'

    with open(f'./outputs/{folder_name}/{name}_manifest.json', 'r') as fp:
        manifest = json.load(fp)

    folder_name_calib = manifest['folder_name_calib']
    simu_card = load_JSON(manifest['simu_card_name'])
    media_card_baseline = load_JSON(manifest['media_card_name'])
    calibrated_geometry = load_JSON(manifest['geometry_name'])

    simu_card['ri'] = calibrated_geometry['ri']
    simu_card['re'] = calibrated_geometry['re']
    simu_card['ri_adv'] = calibrated_geometry['ri_adv']
    simu_card['area'] = calibrated_geometry['area']
    simu_card['advTF'] = calibrated_geometry['advTF']

    dominant_fiber_key = manifest['dominant_adventitia_collagen_key'] + '_adv_theta'

    #-------------------------------------------------------------------------#
    # Loading protocol -> press_list / lambdaz_list / phase indices, same
    # construction as 3_1/3_2.
    #-------------------------------------------------------------------------#
    load_phase = simu_card['load_phase']
    step_load = Artery_load(load_phase)

    press_list = 7500.62 * step_load.list_P
    lambdaz_list = 1 + step_load.list_uz / simu_card['lz']

    n_phase1 = step_load.index_phase[0][1]
    n_phase2 = len(press_list) - step_load.index_phase[1][0]
    indices_1 = slice(step_load.index_phase[0][0], step_load.index_phase[0][1])
    indices_2 = slice(step_load.index_phase[1][0], step_load.index_phase[1][1] + 1)

    #-------------------------------------------------------------------------#
    # f_cells / f_coll_media from the CALIBRATED media card -- constant across
    # every combination in every sweep here (only moduli are swept, never
    # volumic fractions). f_coll_media is the sum over every collagen_i
    # (generalizes the thesis's hardcoded `4*collagen0['volumic_fraction']`,
    # which assumed exactly 4 equal-weight media families).
    #-------------------------------------------------------------------------#
    collagen_keys_media = [k for k in media_card_baseline if k.startswith("collagen")]
    f_cells = media_card_baseline['cells']['volumic_fraction']
    f_coll_media = sum(media_card_baseline[k]['volumic_fraction'] for k in collagen_keys_media)

    config = PlotConfig(
        lambdaz_list=lambdaz_list, press_list=press_list,
        f_cells=f_cells, f_coll_media=f_coll_media,
        n_phase1=n_phase1, n_phase2=n_phase2,
        indices_1=indices_1, indices_2=indices_2,
    )

    #-------------------------------------------------------------------------#
    # Load cached results per sweep, in manifest order, skipping failed combos.
    #-------------------------------------------------------------------------#
    def load_sweep_results(label):
        values, results = [], []
        for entry in manifest['entries']:
            if entry['param'] != label or entry['status'] == 'failed':
                continue
            with open(entry['result_path'], 'rb') as fh:
                results.append(pickle.load(fh))
            values.append(entry['value'])
        return values, results

    sweep_axis_labels = {s['label']: s['axis_label'] for s in manifest['sweeps']}

    os.makedirs(f'images_output/{folder_name}', exist_ok=True)

    for label in ['E_m', 'E_smc', 'E_coll']:
        param_list, all_results = load_sweep_results(label)
        if not all_results:
            print(f"No cached (non-failed) results for sweep '{label}' - skipping its plots.")
            continue
        axis_label = sweep_axis_labels[label]

        #---------------------------------------------------------------#
        # Pressure-Radius
        #---------------------------------------------------------------#
        plot_pressure_radius(folder_name, name, axis_label, all_results, param_list, config)

        #---------------------------------------------------------------#
        # Layer stress
        #---------------------------------------------------------------#
        plot_layer_stress(folder_name, name, axis_label, all_results, param_list, 'Circumferential', config)
        plot_layer_stress(folder_name, name, axis_label, all_results, param_list, 'Axial', config)

        #---------------------------------------------------------------#
        # Stress distribution (media micromechanics)
        #---------------------------------------------------------------#
        plot_circ_stress(folder_name, name, axis_label, all_results, param_list, config)

        #---------------------------------------------------------------#
        # Normalized stress
        #---------------------------------------------------------------#
        plot_normalized_stress(folder_name, name, axis_label, all_results, param_list, 'Axial', config)
        plot_normalized_stress(folder_name, name, axis_label, all_results, param_list, 'Circumferential', config)

        #---------------------------------------------------------------#
        # Fiber angle (dominant adventitia collagen family)
        #---------------------------------------------------------------#
        plot_fiber_orientation(folder_name, name, axis_label, all_results, param_list, dominant_fiber_key, config)

        # plot_layer_distribution left uncalled here, matching the thesis
        # __main__ (defined above, not part of the default figure set) --
        # uncomment to include it:
        # plot_layer_distribution(folder_name, name, axis_label, all_results, param_list, config)