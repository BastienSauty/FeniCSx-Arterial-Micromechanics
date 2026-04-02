#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 15:49:28 2025

@author: bastien.sauty

Written with the help of Chatgpt
Based on the work of Placido Andrea Grillo

Applying the principle of maximum entropy to build a statistical estimator of
a continuous distribution into a discrete distribution
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import moment

from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------#
# Functions to do the optimization for maximum entropy under constraints
#-----------------------------------------------------------------------------#

def compute_moments(angle_axis, distribution):
    """
    Compute mean, std, skewness of a continuous distribution over angles.
    """
    mu = np.sum(angle_axis * distribution)
    sigma = np.sqrt(np.sum((angle_axis - mu)**2 * distribution))
    skew = np.sum(((angle_axis - mu)/sigma)**3 * distribution)
    return mu, sigma, skew

def max_entropy_discretization(angle_axis, distribution, support_angles):
    """
    Discretize the continuous distribution using maximum entropy, constrained
    to match mean, std, and skewness.

    Parameters:
        angle_axis: np.array, angle domain for the continuous distribution
        distribution: np.array, normalized continuous distribution (sum=1)
        support_angles: np.array, selected support angles for discretization

    Returns:
        optimal_weights: np.array of weights (probabilities) at support_angles
    """
    N = len(support_angles)

    # Compute target moments from the continuous distribution
    mu_c, sigma_c, skew_c = compute_moments(angle_axis, distribution)

    def constraint_sum(q):
        return np.sum(q) - 1

    def constraint_mean(q):
        return np.dot(q, support_angles) - mu_c

    def constraint_std(q):
        mu_q = np.dot(q, support_angles)
        return np.dot(q, (support_angles - mu_q)**2) - sigma_c**2

    def constraint_skew(q):
        mu_q = np.dot(q, support_angles)
        sigma_q = np.sqrt(np.dot(q, (support_angles - mu_q)**2))
        if sigma_q == 0:
            return -skew_c  # or return large penalty
        skew_q = np.dot(q, ((support_angles - mu_q)/sigma_q)**3)
        return skew_q - skew_c

    def negative_entropy(q):
        q_safe = np.where(q > 0, q, 1e-12)
        return np.sum(q_safe * np.log(q_safe))

    constraints = [
        {'type': 'eq', 'fun': constraint_sum},
        {'type': 'eq', 'fun': constraint_mean},
        {'type': 'eq', 'fun': constraint_std},
        {'type': 'eq', 'fun': constraint_skew}
    ]

    bounds = [(1e-10, 1.0) for _ in range(N)]
    q0 = np.ones(N) / N

    result = minimize(negative_entropy, q0, method='SLSQP', bounds=bounds, constraints=constraints)

    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    return result.x


#-----------------------------------------------------------------------------#
# Comparing discrete and continuous distribution : on the continuous angle space
#-----------------------------------------------------------------------------#

def project_discrete_to_grid_centered_bins(support_angles, weights, angle_axis):
    """
    build a continuous distribution based on the support_angles. These should be 
    in ascending order. 
    the distribution is constant piecewise. Basically it's a redistribution of the 
    weight of one support angle to a continuous bin.
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
        mask = (angle_axis >= left) & (angle_axis < right)
        count = np.sum(mask)
        if count > 0:
            projected[mask] = weights[i] / count

    # Step 3: Normalize (just in case)
    projected /= np.sum(projected)
    return projected


def compare_distributions(p_cont, q_proj, method="L2"):
    """
    Compare two distributions over the same grid using various metrics.

    Parameters:
        p_cont: np.array (continuous PDF over angle axis)
        q_proj: np.array (discrete PDF projected onto same axis)
        method: str, one of ['L2', 'TV', 'KL', 'JS', 'EMD']

    Returns:
        Scalar difference between distributions
    """
    p = np.clip(p_cont, 1e-12, None)
    q = np.clip(q_proj, 1e-12, None)
    
    if method == "L2":
        return np.sqrt(np.sum((p - q) ** 2))
    
    elif method == "TV":
        return 0.5 * np.sum(np.abs(p - q))
    
    elif method == "KL":
        return entropy(p, q)
    
    elif method == "JS":
        return jensenshannon(p, q)
    
    elif method == "EMD":
        cdf_p = np.cumsum(p) / np.sum(p)
        cdf_q = np.cumsum(q) / np.sum(q)
        return np.sum(np.abs(cdf_p - cdf_q))
    
    else:
        raise ValueError("Unknown method. Choose from 'L2', 'TV', 'KL', 'JS', 'EMD'.")

#-----------------------------------------------------------------------------#
# Plot the distributions
#-----------------------------------------------------------------------------#

def plot_discrete_vs_continuous(keyword, angle_axis, distribution, support_angles, weights, projected):
    """
    plot the continuous experimental distribution; the discretized distribution and the constant piece wise projected discretized distribution
    """

    # Create the main plot
    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot continuous and projected PDFs
    ax1.plot(angle_axis, distribution, label="Continuous PDF", color='blue')
    ax1.plot(angle_axis, projected, label="Projected Discrete PDF", linestyle='--', color='green')
    ax1.set_xlabel("Angle (degrees)")
    ax1.set_ylabel("Continuous Probability Density")
    ax1.legend(loc="upper left")

    # Create second y-axis for scatter
    ax2 = ax1.twinx()
    ax2.scatter(support_angles, weights, color='red', label="Discrete Weights")
    ax2.set_ylabel("Discrete Probability Density", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Add legend for scatter
    lines_labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_labels[0] + lines2, lines_labels[1] + labels2, loc="upper right")
    
    
    #plt.title("Maximum Entropy Discretization")
    plt.grid()
    plt.tight_layout()
    plt.savefig('images_output/ArterialTissue/discretized_collagen_'+keyword+'.pdf')
    plt.show()

def load_continuous_distribution(filename):
    try:
        data = np.load(filename)
        avg_density = data['avg_density'] # experimental continuous PDF
        angles = data['angles'] # continuous angle space
    except:
        print(f"error loading file {filename}, either it does not exist or the keywords avg_density and angles are not defined correctly")
    return(avg_density, angles)

def discretizing_distribution(filename, N, error_metric="L2", plot=False, verbose=False):
    """
    Function to discretize a continuous distribution using the principle of maximum entropy
    Inputs : 
        filename : npz file containing the continuous distribution named "avg_density" with an continuous space named "angles"
        N : number of discrete samples for the discretization
        error_metric : in ['L2', 'TV', 'KL', 'JS', 'EMD']
        plot and verbose : flags for export and print some results
        
    outputs : support_angles, weights and error metric
    """
    avg_density, angles = load_continuous_distribution(filename)
    
    support_angles = np.linspace(-90, 90, N) # discrete angle space
    
    # discrete probability on the support angle space
    weights = max_entropy_discretization(angles, avg_density, support_angles) 
    
    q_projected = project_discrete_to_grid_centered_bins(support_angles, weights, angles)
    l2_error = compare_distributions(avg_density, q_projected, method=error_metric)
    
    if verbose:
        for angle, weight in zip(support_angles, weights):
            print(f"Angle: {angle:.1f}°, Weight: {weight:.3f}")
            print(f"L2 Error: {l2_error:.4f}")
    
    return(support_angles, weights, l2_error)
    