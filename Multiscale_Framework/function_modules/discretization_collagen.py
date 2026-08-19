#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 09:21:55 2026

@author: bastien.sauty

New function file for discretizing collagen fiber. 
The goal is to provide a function discretizing_distribution that is usable in
lieu and place of the old one. 

The goal fo this module is to discretize the experimental continuous probability 
density function of fibers into a dirac defined pdf.
The key difference is the approach to discretization. The earlier version used 
a projection of the dirac distrib onto the continuous space using centered bins.
Once rotated, these centered bins lose sense. 
The new version uses the Wasserstein distance applied on the Cumulative Density functions.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize

#%%
###############################################################################
# Functions for distribution managing : discretization, cumulative density...
# 
###############################################################################

def build_CDF(discrete_angles, discrete_weights, continuous_angles, integral_type='Heavyside', normalize=True):
    """
    Build the cumulative density function of a distribution onto the
    continuous_angles space. For a discrete distribution, integral_type='Heavyside' 
    is adapted for Dirac. 
    In the case of a continuous pdf, this function can still be used. Provide equal
    discrete_angles and continuous_angles and specify integral_type='Trapezoidal' NOT IMPLEMENTED

    Parametersbuild_CDF
    ----------
    discrete_angles : n array
    discrete_weights : n array
    continuous_angles : m array, m>=n

    Returns
    -------
    cdf : m array
    """
    if integral_type == 'Heavyside':
        cdf = np.zeros(continuous_angles.shape)
        for weights_i, theta_i in zip(discrete_weights, discrete_angles):
            zeroval = np.ones(continuous_angles.shape)
            cdf += weights_i * np.heaviside(continuous_angles - theta_i, zeroval)

    elif integral_type == 'Trapezoidal':
        if not np.allclose(discrete_angles, continuous_angles):
            raise ValueError("Trapezoidal mode requires discrete_angles == continuous_angles "
                              "(discrete_weights is treated as the pdf sampled on that grid).")
        pdf = discrete_weights
        cdf = cumulative_trapezoid(pdf, continuous_angles, initial=0.0)

    else:
        raise ValueError(f"Unknown integral_type: {integral_type}")

    if normalize and cdf[-1] > 0:
        cdf = cdf / cdf[-1]

    return cdf

def compute_wasserstein_distance(discrete_weights, discrete_angles, continuous_cdf, continuous_angles):
    """
    Return the W1 wasserstein distance between the cumulative density functions
    of the experimental/continuous PDF and the discrete PDF. 

    Parameters
    ----------
    discrete_weights : N array
    discrete_angles : N array
    continuous_cdf : m array m > N
    continuous_angles : m array

    Returns
    -------
    W1 : float
    """
    discrete_cdf = build_CDF(discrete_angles, discrete_weights, continuous_angles)
    # For two 1D distributions with CDFs F, G on the same support, the
    # Wasserstein-1 distance has the closed form W1(F,G) = integral |F-G| dx.
    # (scipy.stats.wasserstein_distance expects sample VALUES, not CDF
    # arrays -- calling it directly on discrete_cdf/continuous_cdf does not
    # compute the distance between the two distributions.)
    try:
        _trapz = np.trapezoid  # numpy >= 2.0
    except AttributeError:
        _trapz = np.trapz      # numpy < 2.0
    distance = _trapz(np.abs(discrete_cdf - continuous_cdf), continuous_angles)
    return(distance)
    
def discretizing_distribution(continuous_angles, continuous_pdf, N):
    """
    Build the discrete distribution defined by a set of Dirac peak equispaced. 
    The weights are found by minimizing the error between the cumulative density function
    defined by the wasserstein_distance

    Parameters
    ----------
    continuous_angles : m array
    continuous_pdf : m array
    N : int number of families

    Returns
    -------
    discrete_angles : N array
    discrete_weights : N array
    """
    
    # equally space angles
    discrete_angles = np.linspace(min(continuous_angles), max(continuous_angles), N+1)
    discrete_angles = 1/2*(discrete_angles[1:] + discrete_angles[:-1])
    
    
    continuous_cdf = build_CDF(continuous_angles, continuous_pdf, continuous_angles, integral_type='Trapezoidal')
    w0 = np.ones(N) / N
    bounds = [(0, 1) for _ in range(N)]
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    result = minimize(compute_wasserstein_distance, w0, args=(discrete_angles, continuous_cdf, continuous_angles), 
                      bounds=bounds, constraints=cons)
    discrete_weights = result.x
    return(discrete_angles, discrete_weights)


def circular_dispersion_deg(theta_axis, pdf_axis):
    """
    Circular mean and circular standard deviation (in degrees) of a
    pi-periodic (axial) orientation distribution, e.g. collagen fiber
    orientation where +90 deg and -90 deg are the same physical direction.

    Uses the standard "doubling" trick for axial data: angles are doubled
    so that the pi-periodicity maps onto a full 2*pi circle, on which
    ordinary circular statistics (mean resultant length R, circular std
    sqrt(-2 ln R)) apply; the result is then halved back to the original
    (un-doubled) angle scale.

    Returns
    -------
    mean_angle_deg : circular mean orientation, in degrees
    circ_std_deg : circular standard deviation, in degrees (dispersion
        measure used to normalize the Wasserstein error)
    R : mean resultant length in [0,1], 1 = fully concentrated (single
        direction), 0 = uniformly spread over all orientations
    """
    theta_rad = np.deg2rad(theta_axis)
    delta = theta_axis[1] - theta_axis[0]
    weights = pdf_axis * delta
    weights = weights / weights.sum()

    C = np.sum(weights * np.cos(2 * theta_rad))
    S = np.sum(weights * np.sin(2 * theta_rad))
    R = np.hypot(C, S)

    mean_angle_rad = 0.5 * np.arctan2(S, C)
    circ_std_rad = 0.5 * np.sqrt(-2.0 * np.log(R))  # halved back to original scale

    return np.rad2deg(mean_angle_rad), np.rad2deg(circ_std_rad), R





#%%
###############################################################################
# Functions for plots and checkup
# 
###############################################################################

def plot_PDF_CDF(continuous_pdf, continuous_cdf, continuous_angles, folder_name, name):
    
    fig, ax1 = plt.subplots(figsize=(6,4))

    ax1.plot(continuous_angles, continuous_pdf, label="Experimental PDF", color='tab:blue')
    ax1.set_xlabel("Angle to axial direction [°]")
    ax1.set_ylabel("Probability Density Function")
    ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)
    
    
    # Create second y-axis for pdf
    ax2 = ax1.twinx()
    ax2.plot(continuous_angles, continuous_cdf, color='tab:red', label="Experimental CDF")
    ax2.set_ylabel("Cumulative Density Function", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add legend for scatter
    lines_labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_labels[0] + lines2, lines_labels[1] + labels2,loc='lower center',bbox_to_anchor=(0.7, 1.0), bbox_transform=ax1.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'images_output/{folder_name}/{name}_pdf_cdf.pdf')
    plt.show()
    

def plot_CDF_discrete(discrete_cdf, continuous_cdf, continuous_angles, folder_name, name):
    
    fig, ax1 = plt.subplots(figsize=(5,4))

    ax1.plot(continuous_angles, continuous_cdf, label="Experimental CDF", color='tab:blue')
    ax1.plot(continuous_angles, discrete_cdf, label="Discrete CDF", color='tab:red')
    ax1.set_xlabel("Angle to axial direction [°]")
    ax1.set_ylabel("Cumulative Density Function")
    ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)
    
    
    # Add legend for scatter
    lines_labels = ax1.get_legend_handles_labels()
    ax1.legend(lines_labels[0], lines_labels[1], loc='lower right')#,bbox_to_anchor=(0.7, 1.0), bbox_transform=ax1.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'images_output/{folder_name}/{name}_cdf.pdf')
    plt.show()
    
def plot_PDF_discrete(discrete_weights, discrete_angles, continuous_pdf, continuous_angles, folder_name, name):
    
    fig, ax1 = plt.subplots(figsize=(5,4))

    ax1.plot(continuous_angles, continuous_pdf, label="Experimental PDF", color='tab:blue')
    ax1.set_xlabel("Angle to axial direction [°]")
    ax1.set_ylabel("Probability Density Function")
    ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)
    ax1.set_ylim([0, None])
    
    # Create second y-axis for pdf
    ax2 = ax1.twinx()
    ax2.scatter(discrete_angles, discrete_weights, color='tab:red', label="Discrete PDF")
    ax2.set_ylabel("Discrete Probability Density Function", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim([0, None])
    
    # Add legend for scatter
    lines_labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_labels[0] + lines2, lines_labels[1] + labels2,loc='lower center')# ,bbox_to_anchor=(0.7, 1.0), bbox_transform=ax1.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'images_output/{folder_name}/{name}_pdf.pdf')
    plt.show()
    


#%%
###############################################################################
# Main for standalone file
# 
###############################################################################

if __name__=='__main__':
    #-----------------------------------------------------------------------------#
    # Load the experimental continuous distribution
    #-----------------------------------------------------------------------------#
    
    data = np.load('collagen_media_orientation_OConnell2008.npz')
    
    continuous_angles = data['angle_deg']
    continuous_pdf = data['pdf']
    
    
    # re-center: circumferential (90 deg) -> theta = 0
    # continuous_angles = continuous_angles - 90.0

    # switch density units from 1/rad to 1/deg
    continuous_pdf = continuous_pdf * np.pi / 180.0

    
    folder_name = ''
    name = 'media'
    
    continuous_cdf = build_CDF(continuous_angles, continuous_pdf, continuous_angles, integral_type='Trapezoidal')
    
    plot_PDF_CDF(continuous_pdf, continuous_cdf, continuous_angles, folder_name, name)
    
    N = 8
    discrete_angles, discrete_weights = discretizing_distribution(continuous_angles, continuous_pdf, N)
    discrete_cdf = build_CDF(discrete_angles, discrete_weights, continuous_angles)
    
    plot_CDF_discrete(discrete_cdf, continuous_cdf, continuous_angles, folder_name, name)
    plot_PDF_discrete(discrete_weights, discrete_angles, continuous_pdf, continuous_angles, folder_name, name)
    