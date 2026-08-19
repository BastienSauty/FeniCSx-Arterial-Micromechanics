#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting / post-processing for the fiber-family-count sensitivity analysis
(3_2_fiberdiscretization_sa_persistent.py).

Loads the manifest + per-N cached result pickles that script writes -- no
simulation is ever (re-)run here. To refresh a given N, delete its
outputs/<folder>/<name>_N<N>_scal.pkl and re-run
3_2_fiberdiscretization_sa_persistent.py (it will only recompute the missing
N's, see its docstring).
"""

import json
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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

from Multiscale_Framework.class_modules.load_class import Artery_load
from Multiscale_Framework.function_modules.discretization_collagen import (
    discretizing_distribution,
    build_CDF,
    compute_wasserstein_distance,
    circular_dispersion_deg,
    plot_PDF_discrete,
    plot_CDF_discrete,
)

# Set to False to skip the per-N, per-state PDF/CDF comparison figures (13
# N's x 4 states x 2 plots = a lot of files with the default N_discrete_list)
# and only produce the summary plots (pressure-radius, Wasserstein vs N,
# last radius vs N).
PLOT_PER_N_DISTRIBUTIONS = True

# Additional per-N, per-state pair on top of the above: same-axis PDF
# (density stairs + continuous, with the raw discrete weights on their own
# secondary axis) and CDF with the Wasserstein-distance area shaded. Doubles
# the file count of PLOT_PER_N_DISTRIBUTIONS again when both are True - set
# independently since these are for the article figure, not routine SA runs.
PLOT_SHARED_AXIS_VERSIONS = True


def load_JSON(card_name):
    with open(card_name, 'r') as file:
        return json.load(file)


def extract_angles(result, press_value):
    """
    Extract the orientation angles of the ADVENTITIA collagen fibers at the
    step closest to press_value [mmHg]. Matches the extraction Bastien is
    now using (filters on 'collagen' ... 'adv_theta', excluding the media's
    collagen thetas, which also start with 'collagen').
    """
    keys = result.outputs.keys()
    theta_keys = [k for k in keys if k.startswith('collagen') and k.endswith('adv_theta')]
    pressure_list = result.outputs['press']
    target_pressure_mpa = press_value * 0.000133322
    index = np.argmin(np.abs(pressure_list - target_pressure_mpa))
    angles_model = [result.outputs[k][index] for k in theta_keys]
    return np.array(angles_model)


def wasserstein_errors(orientation_angles, state_pdf, angles_model, weights_model):
    """
    Raw and circular-dispersion-normalized Wasserstein-1 distance between the
    experimental continuous orientation PDF (state_pdf, on orientation_angles)
    and the model's discrete orientation distribution (angles_model,
    weights_model) -- replaces the old L2/moment-error metrics.

    Normalization uses circular_dispersion_deg's circ_std_deg computed on the
    EXPERIMENTAL distribution for that state : divides the raw W1 (in
    degrees) by the natural angular spread of the reference distribution, so
    the normalized error is comparable across states/datasets with different
    intrinsic dispersion.
    """
    continuous_cdf = build_CDF(orientation_angles, state_pdf, orientation_angles,
                                integral_type='Trapezoidal', normalize=True)
    raw = compute_wasserstein_distance(weights_model, angles_model, continuous_cdf, orientation_angles)
    _, circ_std_deg, _ = circular_dispersion_deg(orientation_angles, state_pdf)
    normalized = raw / circ_std_deg
    return raw, normalized


def local_bin_widths(angles, domain):
    """
    Per-point bin widths for non-uniformly spaced 1D samples, via midpoints
    to nearest neighbors, clipped to `domain=(low, high)`. Needed because
    the discrete family angles are only equispaced at N-family construction
    time (Init) - at Low/Dias/Sys each family has kinematically rotated to
    a different angle, so a single fixed bin width can't convert weight ->
    density anymore.

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
            "negative bin width (can happen at highly-aligned states, or "
            "at high N, where two families land on the same angle)."
        )

    widths = np.empty_like(widths_sorted)
    widths[order] = widths_sorted
    return widths, edges


def plot_PDF_shared_axis(discrete_weights, discrete_angles, continuous_pdf,
                          continuous_angles, folder_name, fig_name):
    """
    Converts each discrete weight to a density (weight / local bin width)
    so the discrete distribution's SHAPE can be compared to the continuous
    PDF on one shared axis - mismatches are then visible directly rather
    than hidden behind an independently-rescaled second axis. The actual
    discrete weights (family volume fractions) are also plotted as red
    dots on a secondary axis, same convention as plot_PDF_discrete, so the
    real modeled Dirac masses are still visible on their own natural scale.
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

    plt.savefig(f"images_output/{folder_name}/{fig_name}_pdf_shared.pdf")
    plt.show()


def plot_CDF_shaded_wasserstein(cdf_model, cdf_exp, grid, folder_name, fig_name,
                                 w1_raw, w1_normalized=None):
    """
    CDF comparison with the region between the two curves shaded: that area
    is exactly the raw Wasserstein-1 distance, since
    W1 = integral |F_model - F_exp| dtheta. w1_raw/w1_normalized are taken
    as already computed by wasserstein_errors() (same call already made in
    the N/state loop below) rather than recomputed here, so the annotated
    number is guaranteed to match the one driving the summary Wasserstein-
    vs-N plots elsewhere in this script.
    """
    grid = np.asarray(grid)

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    ax.plot(grid, cdf_exp, color='tab:blue', label='Experimental CDF', linewidth=1.5)
    ax.plot(grid, cdf_model, color='tab:red', label='Discrete CDF', linewidth=1.5)
    ax.fill_between(grid, cdf_model, cdf_exp, color='tab:red', alpha=0.15)

    if w1_normalized is not None:
        annotation = rf'$W_1={w1_raw:.2f}$° (norm. {100.0*w1_normalized:.1f}\%)'
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

    plt.savefig(f"images_output/{folder_name}/{fig_name}_cdf_shaded.pdf")
    plt.show()


if __name__ == "__main__":
    #-----------------------------------------------------------------------------#
    # Load manifest written by 3_2_fiberdiscretization_sa_persistent.py
    #-----------------------------------------------------------------------------#
    folder_name = 'Article_Calibration_SA'
    folder_name_calib = 'Article_Calibration'
    name = '3_2_fiberdiscretization_SA'

    manifest_path = f'./outputs/{folder_name}/{name}_manifest.json'
    with open(manifest_path, 'r') as fp:
        manifest = json.load(fp)

    N_discrete_list = np.array(manifest['N_discrete_list'])
    simu_card_name = manifest['simu_card_name']
    geometry_name = manifest['geometry_name']
    failed_N = {int(k): v for k, v in manifest.get('failed_N', {}).items()}
    if failed_N:
        print(f"Manifest reports {len(failed_N)} N value(s) that did not complete last run: "
              f"{failed_N} (exitcode -9 == killed by the OS, almost certainly OOM). Skipping those.")

    simu_card = load_JSON(simu_card_name)
    calibrated_geometry = load_JSON(geometry_name)
    simu_card['ri'] = calibrated_geometry['ri']
    simu_card['re'] = calibrated_geometry['re']
    simu_card['ri_adv'] = calibrated_geometry['ri_adv']
    simu_card['area'] = calibrated_geometry['area']
    simu_card['advTF'] = calibrated_geometry['advTF']

    load_phase = simu_card['load_phase']
    step_load = Artery_load(load_phase)

    #-----------------------------------------------------------------------------#
    # Experimental orientation distributions
    #-----------------------------------------------------------------------------#
    filename = os.path.join(folder_name_calib, "DTAavg.npz")
    data = np.load(filename, allow_pickle=True)
    orientation_angles = data["orientation_angles"]
    orientation_Low = data["orientation_Low"]
    orientation_Dias = data["orientation_Dias"]
    orientation_Sys = data["orientation_Sys"]

    #-----------------------------------------------------------------------------#
    # Load per-N cached results. theta/weights are recomputed here (cheap,
    # deterministic given N) rather than re-pickled by
    # 3_2_fiberdiscretization_sa_persistent.py -- only the FE result itself
    # (expensive) is loaded from disk.
    #-----------------------------------------------------------------------------#
    all_results = []
    valid = np.ones(len(N_discrete_list), dtype=bool)
    for i, N in enumerate(N_discrete_list):
        name_simu = f"{name}_N{int(N)}"
        result_path = f'./outputs/{folder_name}/{name_simu}_scal.pkl'
        try:
            with open(result_path, 'rb') as f:
                result = pickle.load(f)
        except FileNotFoundError:
            if int(N) in failed_N:
                print(f"WARNING: N={N} did not complete last run (exitcode={failed_N[int(N)]}). Skipping.")
            else:
                print(f"WARNING: no cached result for N={N} at {result_path} -- "
                      f"run 3_2_fiberdiscretization_sa_persistent.py first. Skipping.")
            all_results.append((None, None, False))
            valid[i] = False
            continue
        theta_coll_adv, weights_coll_adv = discretizing_distribution(orientation_angles, orientation_Low, int(N))
        if not result:
            print(f"WARNING: cached run for N={N} did not converge (result=False). Skipping.")
            valid[i] = False
        all_results.append((theta_coll_adv, weights_coll_adv, result))

    if not valid.all():
        N_discrete_list = N_discrete_list[valid]
        all_results = [r for r, v in zip(all_results, valid) if v]

    if not os.path.exists(f"images_output/{folder_name}"):
        os.makedirs(f"images_output/{folder_name}")

    #-----------------------------------------------------------------------------#
    # Errors vs N (Init/Low/Dias/Sys) -- raw + normalized Wasserstein-1
    # distance between the experimental continuous PDF and the model's
    # discrete distribution. "Init" (unloaded configuration, before any
    # solve) is compared against the low-pressure experimental PDF, same
    # convention as before, since there is no true zero-pressure dataset.
    # Also produces the per-N, per-state PDF/CDF comparison figures (see
    # PLOT_PER_N_DISTRIBUTIONS above).
    #-----------------------------------------------------------------------------#
    states = [
        ('Init', orientation_Low),   # angles taken at theta_coll_adv (pre-solve), compared to Low
        ('Low', orientation_Low),
        ('Dias', orientation_Dias),
        ('Sys', orientation_Sys),
    ]

    errors_wasserstein_raw = np.zeros((len(N_discrete_list), 4))
    errors_wasserstein_norm = np.zeros((len(N_discrete_list), 4))

    for i, N in enumerate(N_discrete_list):
        theta_coll_adv, weights_coll_adv, result = all_results[i]
        angles_by_state = {
            'Init': theta_coll_adv,
            'Low': extract_angles(result, 10.),
            'Dias': extract_angles(result, 80.),
            'Sys': extract_angles(result, 120.),
        }

        for j, (keyword, state_pdf) in enumerate(states):
            angles_model = angles_by_state[keyword]

            raw, normalized = wasserstein_errors(orientation_angles, state_pdf, angles_model, weights_coll_adv)
            errors_wasserstein_raw[i, j] = raw
            errors_wasserstein_norm[i, j] = normalized

            if PLOT_PER_N_DISTRIBUTIONS:
                fig_name = f'{name}_N{int(N)}_{keyword}'
                plot_PDF_discrete(weights_coll_adv, angles_model, state_pdf, orientation_angles, folder_name, fig_name)

                cdf_model = build_CDF(angles_model, weights_coll_adv, orientation_angles)
                cdf_exp = build_CDF(orientation_angles, state_pdf, orientation_angles,
                                     integral_type='Trapezoidal', normalize=True)
                plot_CDF_discrete(cdf_model, cdf_exp, orientation_angles, folder_name, fig_name)

                if PLOT_SHARED_AXIS_VERSIONS:
                    try:
                        plot_PDF_shared_axis(weights_coll_adv, angles_model, state_pdf,
                                              orientation_angles, folder_name, fig_name)
                        plot_CDF_shaded_wasserstein(cdf_model, cdf_exp, orientation_angles,
                                                     folder_name, fig_name, raw, normalized)
                    except ValueError as e:
                        print(f"WARNING: skipped shared-axis plot for N={N}, {keyword} ({e})")

    #-----------------------------------------------------------------------------#
    # Pressure-Radius curves, colored by N
    #-----------------------------------------------------------------------------#
    press_list = 7500.62 * step_load.list_P
    lambdaz_list = 1 + step_load.list_uz / simu_card['lz']

    n_phase1 = step_load.index_phase[0][1]
    n_phase2 = len(press_list) - step_load.index_phase[1][0]
    indices_1 = slice(step_load.index_phase[0][0], step_load.index_phase[0][1])
    indices_2 = slice(step_load.index_phase[1][0], step_load.index_phase[1][1] + 1)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5, 3), sharey=True, constrained_layout=True,
        gridspec_kw={'width_ratios': [n_phase1, n_phase2], 'wspace': 0.05}
    )

    cmap = plt.cm.get_cmap('viridis', len(N_discrete_list))
    colors = [cmap(i / max(len(N_discrete_list) - 1, 1)) for i in range(len(N_discrete_list))]

    for i, N in enumerate(N_discrete_list):
        color = colors[i]
        _, _, result = all_results[i]
        ri_d = result.outputs['ri_d'][:]
        re_d = result.outputs['re_d'][:]

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
    ax1.grid(axis='both', linestyle=":", linewidth=0.5)
    ax1.tick_params(left=True, labelleft=True, right=False, labelright=False)

    ax2.set_xlabel("Pressure [mmHg]")
    ax2.set_facecolor((1.0, 1.0, 0.88, 0.7))
    ax2.set_xlim([np.min(press_list), np.max(press_list)])
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax2.set_title('Phase 2')
    ax2.grid(axis='both', linestyle=":", linewidth=0.5)
    ax2.tick_params(left=False, labelleft=False, right=False, labelright=False)

    style_legend = [
        Line2D([0], [0], color='k', lw=1.8, linestyle='-', label=r'$R_i$ (solid)'),
        Line2D([0], [0], color='k', lw=1.8, linestyle='--', label=r'$R_e$ (dashed)')
    ]
    style_legend_box = ax2.legend(handles=style_legend, loc='lower right', fontsize=8, frameon=True,
                                   framealpha=0.85, facecolor='white', edgecolor='gray', fancybox=True,
                                   borderpad=0.4, title_fontsize=9)
    ax2.add_artist(style_legend_box)

    legend_elements = [
        Line2D([0], [0], color=colors[i], lw=2, label=rf'N={N_discrete_list[i]}')
        for i in range(len(N_discrete_list))
    ]
    ax2.legend(handles=legend_elements, title='N values', loc='center left', bbox_to_anchor=(1.05, 0.5),
               borderaxespad=0.0, frameon=False, fontsize=9, title_fontsize=10)

    plt.savefig(f'images_output/{folder_name}/{name}_pressure_radius.pdf', bbox_inches='tight')
    plt.show()

    #-----------------------------------------------------------------------------#
    # Wasserstein-1 distance vs N -- raw (degrees) and normalized (by the
    # experimental distribution's circular standard deviation, dimensionless)
    #-----------------------------------------------------------------------------#
    plt.figure(figsize=(4, 3))
    plt.semilogy(N_discrete_list, errors_wasserstein_raw[:, 0], label='Unloaded configuration',
                 marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.semilogy(N_discrete_list, errors_wasserstein_raw[:, 2], label='Diastolic pressure',
                 marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.semilogy(N_discrete_list, errors_wasserstein_raw[:, 3], label='Systolic pressure',
                 marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.xlabel('Number of families')
    plt.ylabel(r'Wasserstein-1 distance [$^\circ$]')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right', frameon=True, framealpha=0.8, fancybox=True, ncol=2, fontsize=8,
               handlelength=2.5, columnspacing=0.8)
    plt.tight_layout()
    plt.savefig(f'images_output/{folder_name}/{name}_Wasserstein_raw_Rotation.pdf')
    plt.show()

    plt.figure(figsize=(4, 3))
    plt.semilogy(N_discrete_list, errors_wasserstein_norm[:, 0], label='Unloaded configuration',
                 marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.semilogy(N_discrete_list, errors_wasserstein_norm[:, 2], label='Diastolic pressure',
                 marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.semilogy(N_discrete_list, errors_wasserstein_norm[:, 3], label='Systolic pressure',
                 marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.xlabel('Number of families')
    plt.ylabel('Normalized Wasserstein-1 distance [-]')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='lower right', frameon=True, framealpha=0.8, fancybox=True, ncol=2, fontsize=8,
               handlelength=2.5, columnspacing=0.8)
    plt.tight_layout()
    plt.savefig(f'images_output/{folder_name}/{name}_Wasserstein_normalized_Rotation.pdf')
    plt.show()

    #-----------------------------------------------------------------------------#
    # Radius at max pressure vs N
    #-----------------------------------------------------------------------------#
    list_lastradii = np.zeros(len(N_discrete_list))
    for j, N in enumerate(N_discrete_list):
        _, _, result = all_results[j]
        list_lastradii[j] = result.outputs['ri_d'][-1]

    plt.figure()
    plt.plot(N_discrete_list, list_lastradii, marker='+')
    plt.grid()
    plt.xlabel('Number of families')
    plt.ylabel(r'$R_i$ at $P=140$mmHg [mm]')
    plt.tight_layout()
    plt.savefig(f'images_output/{folder_name}/{name}_lastradius_vs_N.pdf')
    plt.show()