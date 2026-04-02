#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation worker for chv_3_2: Isolated cache for parallel computing.
"""
import os
import sys
import pickle
import copy
import numpy as np
import shutil


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
        """
        Extracts scalar results, radial averages, and inhomogeneity metrics
        from the simulation series.
        """
        # 1. Initialize arrays for scalar/average results
        self.relaxed_radius = np.zeros(self.tau_list.shape)
        self.basal_radius = np.zeros(self.tau_list.shape)
        self.error_cell_stress_basal = np.zeros(self.tau_list.shape)
        
        # Average stress/stretch components
        self.lambda_cell_in_avg = np.zeros(self.tau_list.shape)
        self.S_yy_avg_media = np.zeros(self.tau_list.shape)
        self.s_yy_collagen = np.zeros(self.tau_list.shape)
        self.s_yy_matrix = np.zeros(self.tau_list.shape)
        self.s_yy_cell = np.zeros(self.tau_list.shape)
        
        # 2. Initialize arrays for Inhomogeneity metrics
        # Difference between innermost and outermost media points
        self.all_lambda_in_grads = np.zeros(self.tau_list.shape)
        
        for i, result in enumerate(self.SA_results):
            # Radius extraction
            self.relaxed_radius[i] = result.outputs['ri_d'][index_relaxed]
            self.basal_radius[i] = result.outputs['ri_d'][index_basal]
            
            # Position and Cell Stress profile for error calculation
            # Note: r_pos corresponds to the radial points of the media
            r_pos = result.dict_outputs['S_yy']['points']
            s_cell_basal_profile = result.outputs['s_yy_cell'][index_basal, :]
            
            # L2 Relative Error to target tau
            self.error_cell_stress_basal[i] = axisymmetric_relative_error(
                r_pos, s_cell_basal_profile, self.tau_list[i]
            )
            
            # 3. Extract Axisymmetric Averages at the basal state
            # lambda_cell_in and S_yy (total) are often stored as local fields
            self.lambda_cell_in_avg[i] = axisymmetric_average(r_pos, result.outputs['lambda_cell_in'][index_basal, :])
            self.S_yy_avg_media[i] = axisymmetric_average(r_pos, result.outputs['S_yy'][index_basal, :])
            self.s_yy_collagen[i] = axisymmetric_average(r_pos, result.outputs['s_yy_collagen'][index_basal, :])
            self.s_yy_matrix[i] = axisymmetric_average(r_pos, result.outputs['s_yy_matrix'][index_basal, :])
            self.s_yy_cell[i] = axisymmetric_average(r_pos, result.outputs['s_yy_cell'][index_basal, :])
            
            # 4. Characterize Inhomogeneity (Intima vs Externa)
            # l_in_profile is the radial distribution of inelastic stretch
            l_in_profile = result.outputs['lambda_cell_in'][index_basal, :]
            
            # Positive value means higher inelastic stretch at the inner wall
            self.all_lambda_in_grads[i] = l_in_profile[0] - l_in_profile[-1]
        
    def compute_functioning_range(self, delta):
        """
        Robustly computes the functioning range by expanding from tau_b_ref.
        Captures radii, stresses, and stretches at the range boundaries.
        """
        # 1. Find exact index for tau_b_ref
        idx_ref = np.where(np.isclose(self.tau_list, self.tau_b_ref))[0][0]
        
        # Store Reference values at tau_b_ref
        self.r_b_ref = self.basal_radius[idx_ref]
        self.lambda_cell_in_ref = self.lambda_cell_in_avg[idx_ref]
        self.S_yy_media_ref = self.S_yy_avg_media[idx_ref]

        # 2. Search Left (from ref to 0)
        idx_left = idx_ref
        while idx_left > 0:
            if self.error_cell_stress_basal[idx_left - 1] < delta:
                idx_left -= 1
            else:
                break
        
        # 3. Search Right (from ref to end)
        idx_right = idx_ref
        while idx_right < len(self.error_cell_stress_basal) - 1:
            if self.error_cell_stress_basal[idx_right + 1] < delta:
                idx_right += 1
            else:
                break

        # 4. Store indices and physical value ranges [min_at_delta, max_at_delta]
        self.id_range = np.arange(idx_left, idx_right + 1)
        self.tau_range = [self.tau_list[idx_left], self.tau_list[idx_right]]
        self.r_range = [self.basal_radius[idx_left], self.basal_radius[idx_right]]
        
        # New functioning ranges for stresses and stretches
        self.lambda_range = [self.lambda_cell_in_avg[idx_left], self.lambda_cell_in_avg[idx_right]]
        self.S_yy_media_range = [self.S_yy_avg_media[idx_left], self.S_yy_avg_media[idx_right]]
        self.s_yy_collagen_range = [self.s_yy_collagen[idx_left], self.s_yy_collagen[idx_right]]
        self.s_yy_matrix_range = [self.s_yy_matrix[idx_left], self.s_yy_matrix[idx_right]]
        self.s_yy_cell_range = [self.s_yy_cell[idx_left], self.s_yy_cell[idx_right]]
        
        # Store the specific gradients for plotting
        self.grad_lambda_ref = self.all_lambda_in_grads[idx_ref]
        self.grad_lambda_range = [self.all_lambda_in_grads[idx_left], 
                                  self.all_lambda_in_grads[idx_right]]
        
        
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
    # Import inside the function to respect the environment variables set in the caller
    from ChV_MultiscaleActiveStressRegulation.main_Vasoconstriction_25_11_24 import run_simulation

    os.environ["FENICS_JIT_TIMEOUT"] = "600"
    
    local_media = copy.deepcopy(media_card)
    local_media["cells"]["basal stress"] = tau_val
    
    print(f"--> [WORKER PID {os.getpid()}] Starting simulation: tau = {tau_val:.6f}")
    
    return run_simulation(
        f"{name_simu_base}_{tau_val:.6f}", 
        folder_name, 
        simu_card, 
        adventitia_card, 
        local_media
    )

def run_isolated_subprocess_simulation(p_vals, name, folder_name, passive_simu_card, 
                                       simu_card, media_card, adventitia_card, index_passive):
    
    # 1. IMMEDIATE DEEPCOPY
    # Since you are sure these contain no FEniCSx objects yet, this is safe.
    # This protects the 'originals' from being modified by the passive run.
    pristine_simu = copy.deepcopy(simu_card)
    pristine_media = copy.deepcopy(media_card)
    pristine_adv = copy.deepcopy(adventitia_card)

    # 2. Environment & JIT Isolation
    # ---------------------------------------------------------------
    worker_id = os.getpid()
    E_m, E_c, f_c = p_vals
    name_simu_base = f"{name}_{E_m}_{E_c}_{f_c}"
    
    unique_cache = os.path.abspath(f"./jit_cache/run_{name_simu_base}_{worker_id}")
    os.makedirs(unique_cache, exist_ok=True)
    
    os.environ["XDG_CACHE_HOME"] = unique_cache
    os.environ["FFCX_CACHE_DIR"] = unique_cache
    os.environ["PYTHONPYCACHEPREFIX"] = os.path.join(unique_cache, "pycache")

    # 3. Early Exit Check
    # ---------------------------------------------------------------
    final_output_path = f"./outputs/{folder_name}/{name_simu_base}_FINAL_PIPELINE.pkl"
    if os.path.exists(final_output_path):
        if os.path.exists(unique_cache):
            shutil.rmtree(unique_cache)
        with open(final_output_path, "rb") as f:
            return pickle.load(f)

    # 4. Local Import & Passive Run
    # ---------------------------------------------------------------
    from ChV_MultiscaleActiveStressRegulation.main_Vasoconstriction_25_11_24 import run_simulation
    
    passive_namefile = f"{name_simu_base}_passive_scal.pkl"
    passive_filepath = os.path.join("./outputs", folder_name, passive_namefile)
    
    if not os.path.exists(passive_filepath):
        print(f"--> [RUNNING] Passive Simulation for {name_simu_base}...")
        # Note: run_simulation will modify these cards (adding Meshes/Functions)
        # but it only modifies the copies we pass to it.
        passive_result = run_simulation(f"{name_simu_base}_passive", folder_name, 
                                        passive_simu_card, 
                                        copy.deepcopy(pristine_adv), 
                                        copy.deepcopy(pristine_media))
    else:
        with open(passive_filepath, "rb") as f:
            passive_result = pickle.load(f)

    # 5. Active Loop (The "Series" part)
    # ---------------------------------------------------------------
    tau_list, tau_b_ref = calculate_tau_list(passive_result, index_passive)
    SA_results = []
    
    print(f"--> [PIPELINE] Running {len(tau_list)} simulations in series...")
    for tau in tau_list:
        tau_file = os.path.join("./outputs", folder_name, f"{name_simu_base}_{tau:.6f}_scal.pkl")
        
        if os.path.exists(tau_file):
            with open(tau_file, "rb") as f:
                SA_results.append(pickle.load(f))
        else:
            # We pass a FRESH deepcopy of the pristine cards to every iteration.
            # This ensures each 'tau' simulation starts with NO FEniCSx objects.
            result = run_single_tau(tau, name_simu_base, folder_name, 
                                   copy.deepcopy(pristine_simu), 
                                   copy.deepcopy(pristine_adv), 
                                   copy.deepcopy(pristine_media))
            SA_results.append(result)

    # 6. Finalization
    # ---------------------------------------------------------------
    pipeline_result = pipeline_Vcn_result(passive_result, SA_results, tau_b_ref, tau_list)
    with open(final_output_path, "wb") as f:
        pickle.dump(pipeline_result, f)
    
    shutil.rmtree(unique_cache, ignore_errors=True)
    return pipeline_result


if __name__ == "__main__":
    arg_file = sys.argv[1]
    with open(arg_file, "rb") as f:
        args_dict = pickle.load(f)
    
    # Run simulation
    _ = run_isolated_subprocess_simulation(**args_dict)
    
    sys.exit(0)