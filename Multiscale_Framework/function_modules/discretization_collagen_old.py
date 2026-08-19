#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:49:28 2025

@author: bastien.sauty

Written with the help of Chatgpt
Based on the work of Placido Andrea Grillo
Based on the old version of the code - modified to have a stronger theoretical basis

Applying the principle of maximum entropy to build a statistical estimator of
a continuous distribution into a discrete distribution

BASED ON THE ASSUMPTION THAT THE CONTINUOUS DISTRIBUTION IS BASED ON EQUIDISTANT ANGLE ARRAY -> -89.5 , 89.5 with 180 points
"""

import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------#
# Functions to do the optimization for maximum entropy under constraints
#-----------------------------------------------------------------------------#

def compute_continuous_moments(theta, pdf, max_order):
    """
    Compute the moments of a continuous distribution using the trapezoidal integration
    inputs :
        theta : continuous range for the pdf, array with this range : [-pi/2, pi/2]
        pdf : probability density function; continuous; array of size len(theta)
        max_order : maximal order where the moments are computed
    outputs:
        moments : array of size max_order containing the central moments from 1 to max_order
                don't contain moment 0 as this is a distribution, hence is 1.    
    """
    delta_theta = theta[1] - theta[0]  # Should be 1.0 for your data
    pdf = pdf / np.sum(pdf * delta_theta)  # Normalize using rectangle integration

    # Mean (1st moment)
    mu = np.sum(theta * pdf) * delta_theta

    # Central differences
    diffs = theta - mu

    # Higher-order central moments
    powers = np.power.outer(diffs, np.arange(1, max_order + 1))  # shape (N, max_order)
    weighted = pdf[:, None] * powers                            # shape (N, max_order)
    moments = np.sum(weighted, axis=0) * delta_theta            # shape (max_order,)

    return np.concatenate(([mu], moments[1:]))  # [mu, central_moment_2, ..., central_moment_K]

def moment_error_vectorized(discrete_theta, weights, target_moments, k, K):
    """
    Compute RMSE between discrete and target central moments (orders k+1 to K), vectorized.
    
    Parameters:
    - discrete_theta: array of Dirac support points
    - weights: corresponding weights (should sum to 1)
    - target_moments: array of target central moments [mu, mu2, mu3, ..., mu_K]
    - k: number of matched moments
    - K: maximum moment order to include in the error
    
    Returns:
    - RMSE of higher-order moment errors (orders k+1 to K)
    """
    mu = np.sum(weights * discrete_theta)
    diffs = discrete_theta - mu  # shape (N,)
    
    # Compute powers for orders k+1 to K (shape: N × (K - k))
    orders = np.arange(k + 1, K + 1)
    powers = np.power.outer(diffs, orders)  # shape (N, K-k)
    
    # Weighted moments
    m_discrete = np.sum(weights[:, None] * powers, axis=0)  # shape (K-k,)
    m_target = target_moments[orders - 1]                   # shape (K-k,)
    
    errors = (m_discrete - m_target) ** 2
    return np.sqrt(np.mean(errors))

def RMSE_L2(angle_axis, distribution, projected):
    """
    Root Mean Square Error -> L2 error between the continuous and discrete projected distribution
    """
    delta_theta = angle_axis[1]-angle_axis[0]
    return(np.sqrt(np.sum((distribution - projected)**2) * delta_theta))

#-----------------------------------------------------------------------------#
# Comparing discrete and continuous distribution : on the continuous angle space
#-----------------------------------------------------------------------------#

def project_discrete_to_grid_centered_bins(support_angles, weights, angle_axis):
    """
    build a continuous distribution based on the support_angles. These should be 
    in ascending order. 
    the distribution is constant piecewise. Basically it's a redistribution of the 
    weight of one support angle to a continuous bin.
    inputs:        
        support_angles : dirac angle space
        weights : dirac weights
        angle_axis : continuous angle space
    output :
        projected distribution of the dirac distrib onto the continuous angle space
    """
    projected = np.zeros_like(angle_axis)
    N = len(support_angles)

    # Step 1: Compute bin edges based on midpoints
    edges = np.zeros(N + 1)
    edges[1:-1] = 0.5 * (support_angles[1:] + support_angles[:-1])
    edges[0] = support_angles[0] - 0.5 * (support_angles[1] - support_angles[0])
    edges[-1] = support_angles[-1] + 0.5 * (support_angles[-1] - support_angles[-2])

    # Step 2: Assign weights to bins
    for i in range(N):
        left, right = edges[i], edges[i + 1]
        if i==N-1:
            mask = (angle_axis >= left) & (angle_axis <= right) # takes also the value at the end
        else:            
            mask = (angle_axis >= left) & (angle_axis < right)
        count = np.sum(mask)
        if count > 0:
            projected[mask] = weights[i] / count

    # Step 3: Normalize (just in case)
    projected /= np.sum(projected)
    return projected


def normalized_moment_error(discrete_angles, discrete_weights, projected_pdf, angle_axis, max_order=6, eps=1e-12):
    """
    Computes the RMSE of the relative error between central moments of the discrete and projected distributions.

    Each moment error is normalized by the moment from the discrete distribution.
    """
    # Discrete (Dirac) central moments
    mu_d = np.sum(discrete_weights * discrete_angles)
    moments_d = []
    for n in range(1, max_order + 1):
        moment = np.sum(discrete_weights * (discrete_angles - mu_d)**n)
        moments_d.append(moment)

    # Projected central moments
    delta_theta = angle_axis[1] - angle_axis[0]
    projected_pdf = projected_pdf / np.sum(projected_pdf * delta_theta)
    mu_p = np.sum(angle_axis * projected_pdf) * delta_theta
    moments_p = []
    for n in range(1, max_order + 1):
        moment = np.sum(((angle_axis - mu_p) ** n) * projected_pdf) * delta_theta
        moments_p.append(moment)

    # Relative squared error with small epsilon to avoid division by 0
    rel_sq_errors = [((mp - md) / (abs(mp) + eps))**2 for md, mp in zip(moments_d, moments_p)]

    return np.sqrt(np.mean(rel_sq_errors))

#-----------------------------------------------------------------------------#
# KL divergence
#-----------------------------------------------------------------------------#

def kl_loss(w, theta_dirac, angle_axis, pdf_target, eps=1e-12):
    w = np.clip(w, eps, 1.0)
    w /= np.sum(w)
    
    q = project_discrete_to_grid_centered_bins(theta_dirac, w, angle_axis)
    q = np.clip(q, eps, 1.0)
    
    mask = (pdf_target > eps) & (q > eps)
    kl = np.sum(q[mask] * np.log(q[mask] / pdf_target[mask])) * (angle_axis[1] - angle_axis[0])
    return kl

def optimize_kl(theta_dirac, angle_axis, pdf_target):
    N = len(theta_dirac)
    w0 = np.ones(N) / N
    bounds = [(0, 1) for _ in range(N)]
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    result = minimize(kl_loss, w0, args=(theta_dirac, angle_axis, pdf_target), 
                      bounds=bounds, constraints=cons)
    if not result.success:
        raise RuntimeError("KL optimization failed: " + result.message)
    return result.x


#-----------------------------------------------------------------------------#
# Plot the distributions
#-----------------------------------------------------------------------------#

def plot_discrete_vs_continuous(continuous_angles, continuous_density, theta_dirac, weights, projected, name, folder_name, savefig=False, verbose=True):
    """
    plot the continuous experimental distribution; the discretized distribution and the constant piece wise projected discretized distribution
    inputs :
        - continuous_angles : continuous angle space
        - continuous_density : continuous (expe) density function
        - theta_dirac : discrete angle space
        - weights : weights of the discrete distribution
        - projected : continuous density, interpolation of the discrete one onto the continuous space
    """

    # Create the main plot
    fig, ax1 = plt.subplots(figsize=(6,4))

    # Plot continuous and projected PDFs
    ax1.plot(continuous_angles, continuous_density, label="Experimental Distribution", color='blue')
    ax1.plot(continuous_angles, projected, label="Projected Discrete Distribution", linestyle='--', color='green')
    ax1.set_xlabel("Angle to axial direction [°]")
    ax1.set_ylabel("Continuous Probability Density")
    ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.5)

    # Create second y-axis for scatter
    ax2 = ax1.twinx()
    ax2.scatter(theta_dirac, weights, color='red', label="Discrete distribution")
    ax2.set_ylabel("Discrete Probability Density", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add legend for scatter
    lines_labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_labels[0] + lines2, lines_labels[1] + labels2,loc='lower center',bbox_to_anchor=(0.7, 1.0), bbox_transform=ax1.transAxes)
    
    plt.tight_layout()
    if savefig:
        plt.savefig(f'images_output/{folder_name}/{name}_orientation_distribution.pdf')
        
    if verbose:
        plt.show()
    else:
        plt.close(fig)



#-----------------------------------------------------------------------------#
# Discretizing process
#-----------------------------------------------------------------------------#
def load_continuous_distribution(filename):
    try:
        data = np.load(filename)
        avg_density = data['avg_density'] # experimental continuous PDF
        angles = data['angles'] # continuous angle space
    except:
        print(f"error loading file {filename}, either it does not exist or the keywords avg_density and angles are not defined correctly")
    return(angles, avg_density)

def discretizing_distribution(continuous_angles, continuous_density, N, name, folder_name, plot=False, verbose=False):
    """
    Function to discretize a continuous distribution 
    Inputs : 
        continuous_angles, continuous_density : array of same size
        N : number of discrete samples for the discretization
        error_metric : in ['L2', 'TV', 'KL', 'JS', 'EMD']
        plot and verbose : flags for export and print some results
        
    outputs : support_angles, weights and error metric
    """    
    theta_dirac = np.linspace(-89.5, 89.5, N+1)
    theta_dirac = 1/2*(theta_dirac[1:] + theta_dirac[:-1])
    
    weights_KL = optimize_kl(theta_dirac, continuous_angles, continuous_density)

    projected = project_discrete_to_grid_centered_bins(theta_dirac, weights_KL, continuous_angles)
    l2_error = RMSE_L2(continuous_angles, continuous_density, projected)
    if verbose:
        for angle, weight in zip(theta_dirac, weights_KL):
            print(f"Angle: {angle:.1f}°, Weight: {weight:.3f}")
            print(f"L2 Error: {l2_error:.4f}")
    if plot:
        plot_discrete_vs_continuous(continuous_angles, continuous_density, theta_dirac, weights_KL, projected, name, folder_name, savefig=True, verbose=False)
    return([theta_dirac, weights_KL])

def load_and_discretize_distribution(filename, N, name, folder_name, plot=False, verbose=False):
    """
    Function to discretize a continuous distribution 
    Inputs : 
        filename : npz file containing the continuous distribution named "avg_density" with an continuous space named "angles"
        N : number of discrete samples for the discretization
        error_metric : in ['L2', 'TV', 'KL', 'JS', 'EMD']
        plot and verbose : flags for export and print some results
        
    outputs : support_angles, weights and error metric
    """
    continuous_angles, continuous_density = load_continuous_distribution(filename)
    
    return(discretizing_distribution(continuous_angles, continuous_density, N, name, folder_name, plot, verbose))

# %% 
# Main code

if __name__=='__main__':
    #-----------------------------------------------------------------------------#
    # Load the experimental continuous distribution
    #-----------------------------------------------------------------------------#
    
    # Load Low continuous distribution
    keyword = 'Low'
    data = np.load('avg_density_Low.npz')
    avg_Low_density = data['avg_density'] # experimental continuous PDF
    angles_Low = data['angles'] # continuous angle space
    avg_Low_density /= np.trapz(avg_Low_density, angles_Low) # normalization
    
    # Support angles for the discrete distrib
    list_k = np.arange(3, 41)
    errors_representation = np.zeros(list_k.shape)
    errors_projection = np.zeros(list_k.shape)
    errors_discretization = np.zeros(list_k.shape)
    discretization = {}
    for i, k in enumerate(list_k):
        print(k)
        theta_dirac = np.linspace(-89.5, 89.5, k+1)
        theta_dirac = 1/2*(theta_dirac[1:] + theta_dirac[:-1])
        
        weights_KL = optimize_kl(theta_dirac, angles_Low, avg_Low_density)
        projected_KL = project_discrete_to_grid_centered_bins(theta_dirac, weights_KL, angles_Low)
        #plot_discrete_vs_continuous('Low', angles_Low, avg_Low_density, theta_dirac, weights_KL, projected_KL)
        
        error_L2 = RMSE_L2(angles_Low, avg_Low_density, projected_KL)
        error_moments = normalized_moment_error(theta_dirac, weights_KL, projected_KL, angles_Low, max_order=4)
        error_total = normalized_moment_error(theta_dirac, weights_KL, avg_Low_density, angles_Low, max_order=4)
        
        discretization[k] = [theta_dirac, weights_KL, error_L2, error_moments, error_total]
        errors_representation[i] = error_L2
        errors_projection[i] = error_moments
        errors_discretization[i] = error_total
    
    #%%
    # Plot the l2 error between projection and experimental
    plt.figure(figsize=(4,3))
    plt.plot(list_k, errors_representation, marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.xlabel('Number of families')
    plt.ylabel('L2 Error')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/L2error_discretization_Low.pdf')
    plt.show()
    #%%
    # Plot the moments error between the dirac and its projection
    plt.figure(figsize=(4,3))
    plt.semilogy(list_k, errors_projection, marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.xlabel('Number of families')
    plt.ylabel('Projection Error - Moments RMSE')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/Projection_error_Low.pdf')
    plt.show()
    
    plt.figure(figsize=(4,3))
    plt.semilogy(list_k, errors_discretization, marker='+', linestyle='-', linewidth=1, alpha=0.8)
    plt.xlabel('Number of families')
    plt.ylabel('Projection Error - Moments RMSE')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/Discretization_error_Low.pdf')
    plt.show()
    #%%
    # Plot one example for 8 families of fibers
    k = 4
    projected_KL = project_discrete_to_grid_centered_bins(discretization[k][0], discretization[k][1], angles_Low)
    plot_discrete_vs_continuous(f'Low_{k}', angles_Low, avg_Low_density, discretization[k][0], discretization[k][1], projected_KL)
    
    k = 8
    projected_KL = project_discrete_to_grid_centered_bins(discretization[k][0], discretization[k][1], angles_Low)
    plot_discrete_vs_continuous(f'Low_{k}', angles_Low, avg_Low_density, discretization[k][0], discretization[k][1], projected_KL)
    
    k = 14
    projected_KL = project_discrete_to_grid_centered_bins(discretization[k][0], discretization[k][1], angles_Low)
    plot_discrete_vs_continuous(f'Low_{k}', angles_Low, avg_Low_density, discretization[k][0], discretization[k][1], projected_KL)
    
    
    k = 20
    projected_KL = project_discrete_to_grid_centered_bins(discretization[k][0], discretization[k][1], angles_Low)
    plot_discrete_vs_continuous(f'Low_{k}', angles_Low, avg_Low_density, discretization[k][0], discretization[k][1], projected_KL)
    

    k = 40
    projected_KL = project_discrete_to_grid_centered_bins(discretization[k][0], discretization[k][1], angles_Low)
    plot_discrete_vs_continuous(f'Low_{k}', angles_Low, avg_Low_density, discretization[k][0], discretization[k][1], projected_KL)
    