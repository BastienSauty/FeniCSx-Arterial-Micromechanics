#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistent-context calibration of the arterial tissue model.

Same physics/cost-function as 3_1_calibration_ArterialTissue_26_07_20.py, but :
  - the FE problem (mesh, function spaces, weak form, solver) is built ONCE via
    main_ArterialTissue_persistent.setup_simulation(), instead of being rebuilt
    from scratch (main_ArterialTissue_26_07_21.run_simulation()) on every cost
    function call. Every subsequent evaluation only pushes new parameter values
    in place (mech.update_subdomain_parameters / mech.update_geometry) and
    re-solves (mech.reset_state() + the loading loop) - no UFL form or
    fem.Expression is ever rebuilt, so FFCx never recompiles after the first
    call. See parameter_class.py / inclusions_class.py / material_class.py /
    mech_problem_class.py for where the underlying set_value()/set_parameters()/
    update_parameters()/update_geometry() machinery lives.

  - the free parameter vector is no longer hardcoded to [E0, k0, lambda0, re].
    Any physical entry in the media/adventitia json cards - young modulus
    (Constant or the [E0,k0,lambda0] Exponential law), poisson ratio, volumic
    fraction, fiber orientation (theta/phi) - can be added to PARAMETER_SPECS
    below and it becomes a free parameter for calibration (or a sweep axis for
    a sensitivity analysis), without touching the FE code at all.

The plotting/post-processing step is entirely separate now - this script only
calibrates and exports the resulting cards. Run
4_1_plot_ArterialTissue_calibrated.py afterward: it loads the exported cards,
applies them on top of simu_card_calib.json (a separate, typically finer-mesh
simulation card - calibration uses simu_card_calib_gross.json for speed), and
either loads a cached run or re-runs the model once at full resolution before
producing every plot.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import json
from dataclasses import dataclass, field
from typing import List, Optional

from scipy.optimize import minimize
# from scipy.optimize import least_squares
from scipy.interpolate import interp1d

import matplotlib as mpl
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
from Article_Calibration.main_ArterialTissue_persistent import (
    setup_simulation,
    solve_simulation,
    load_JSON,
)
from Multiscale_Framework.function_modules.discretization_collagen import (
    discretizing_distribution,
)

#-----------------------------------------------------------------------------#
# Generic parameter selection : add/remove entries here to change what's
# calibrated (or swept for a sensitivity analysis) - no FE code involved.
#-----------------------------------------------------------------------------#

@dataclass
class ParamSpec:
    """
    Describes one free parameter, possibly shared across several json-card
    entries AND across layers (e.g. the same E0/k0/lambda0 applied to every
    collagen fiber key in BOTH the media and the adventitia).

    label     : name for logging/plots
    targets   : list of (layer, keys) pairs that all share this ONE value.
                layer is "media" or "adventitia" ; keys is a list of json-card
                entry names within that layer ("matrix", or an inclusion key
                like "collagen_0"). E.g. to share a value across every
                collagen fiber in both layers :
                    targets=[("media", collagen_keys_media),
                             ("adventitia", collagen_keys_adventitia)]
                To make media and adventitia independent instead, use two
                separate ParamSpec entries, each with a single-layer target.
                Ignored when is_geometry=True.
    field_    : "young", "poisson", "volumic_fraction", "theta", "phi",
                "shape_ratio". Ignored when is_geometry=True.
    subindex  : for field_="young" with the Exponential law, which coefficient
                this targets: 0 (E0), 1 (k0), or 2 (lambda0). None for a
                Constant young modulus or any other field.
    is_geometry : special-cased free parameter : re (drives ri/ri_adv too,
                see cost_function). targets/field_/subindex are unused.
    bounds    : (lo, hi) physical bounds used to normalize for the optimizer.
    """
    label: str
    targets: List[tuple] = field(default_factory=list)
    field_: str = ""
    subindex: Optional[int] = None
    is_geometry: bool = False
    bounds: tuple = (0.0, 1.0)


def get_param_value(cards, spec):
    if spec.is_geometry:
        return cards["simu"]["re"]
    layer, keys = spec.targets[0]  # all targets share the same value - read the first
    entry = cards[layer][keys[0]]
    if spec.field_ == "young" and spec.subindex is not None:
        return entry["young"][spec.subindex][0]
    return entry[spec.field_]


def set_param_value(cards, spec, value):
    if spec.is_geometry:
        # handled separately in cost_function (re also drives ri, ri_adv)
        return
    for layer, keys in spec.targets:
        card = cards[layer]
        for key in keys:
            entry = card[key]
            if spec.field_ == "young" and spec.subindex is not None:
                entry["young"][spec.subindex][0] = float(value)
            else:
                entry[spec.field_] = float(value)


def pack(params, bounds):
    params = np.array(params)
    mins = np.array([b[0] for b in bounds])
    maxs = np.array([b[1] for b in bounds])
    return (params - mins) / (maxs - mins)


def unpack(norm_params, bounds):
    norm_params = np.array(norm_params)
    mins = np.array([b[0] for b in bounds])
    maxs = np.array([b[1] for b in bounds])
    return mins + norm_params * (maxs - mins)

def gradient_sem(values_sem, x):
    """
    Propagate per-point SEM through np.gradient(., x) using the *exact*
    finite-difference stencil numpy uses (handles non-uniform spacing and
    edge points correctly, matches whatever edge_order np.gradient applies),
    assuming independent per-point measurement errors.
    """
    n = len(x)
    G = np.gradient(np.eye(n), x, axis=0)      # G[i, j] = d(gradient_i)/d(value_j)
    var_deriv = (G**2) @ (np.asarray(values_sem)**2)
    return np.sqrt(var_deriv)

#-----------------------------------------------------------------------------#
# Load general parameters
#-----------------------------------------------------------------------------#

folder_name = 'Article_Calibration'
name = '3_1_calibration_ArterialTissue_persistent'

simu_card_name = 'json_cards/simu_card_calib_gross.json'
media_card_name = 'json_cards/media_card_calib_N5.json'
adventitia_card_name = 'json_cards/adventitia_card_calib_N8.json'

simu_card = load_JSON(simu_card_name)
adventitia_card = load_JSON(adventitia_card_name)
media_card = load_JSON(media_card_name)
simu_card['XDMF_export'] = 0

collagen_keys_media = [k for k in media_card.keys() if k.startswith("collagen")]
collagen_keys_adventitia = [k for k in adventitia_card.keys() if k.startswith("collagen")]

#-----------------------------------------------------------------------------#
# Load & process experimental data (unchanged from the original script)
#-----------------------------------------------------------------------------#
filename = os.path.join(folder_name, "DTAavg.npz")
data = np.load(filename, allow_pickle=True)

orientation_angles = data["orientation_angles"]
orientation_Low = data["orientation_Low"]

lambdaz_exp = data["lambdaz"]
press_exp = data["Pressure_mmHg"]
re_exp = data["OuterRadius_mm"]
F_zz_exp = data["Fzz_Sample_mN"]

filename_std = os.path.join(folder_name, "DTAsem.npz")
data_std = np.load(filename_std, allow_pickle=True)
F_zz_sd_exp = data_std['Fzz_Sample_mN']
re_sd_exp = data_std['OuterRadius_mm']

dF_zz_exp_dp = np.gradient(F_zz_exp, press_exp)
dre_exp_dp   = np.gradient(re_exp, press_exp)

dF_zz_sem_dp = gradient_sem(F_zz_sd_exp, press_exp)
dre_sem_dp   = gradient_sem(re_sd_exp, press_exp)

N = len(collagen_keys_adventitia)
theta_coll_adv, weights_coll_adv = discretizing_distribution(orientation_angles, orientation_Low, N) #, name+'_init', folder_name, plot=True, verbose=False)

f_coll_adv = 0.5
for j, key in enumerate(collagen_keys_adventitia):
    adventitia_card[key]['theta'] = theta_coll_adv[j]
    adventitia_card[key]['volumic_fraction'] = f_coll_adv*weights_coll_adv[j]

area = np.pi*(re_exp[0]**2 - data["InnerRadius_mm"][0]**2)
simu_card['area'] = area

#-----------------------------------------------------------------------------#
# Build the persistent FE context ONCE - this is the only call that pays the
# FFCx compilation cost for the whole calibration run.
#-----------------------------------------------------------------------------#
if not os.path.exists(f"outputs/{folder_name}"):
    os.makedirs(f"outputs/{folder_name}")

ctx = setup_simulation(name, folder_name, simu_card, adventitia_card, media_card)


def rebuild_ctx():
    """
    Rebuild the persistent context from scratch, using whatever values are
    CURRENTLY in simu_card/adventitia_card/media_card - i.e. the trial point
    the optimizer is presently evaluating, since set_param_value() has
    already been applied to them by the time cost_function calls this.

    Robustness note (2026-08-17): a failed/non-converged solve_simulation()
    call can leave stale derived state behind on ctx.mech that reset_state()
    + update_local_quantities() don't fully clean up (see
    main_ArterialTissue_persistent.py's solve_simulation() docstring) -
    confirmed here by an isolated single-solve test: a parameter point that
    fails immediately after a PRIOR crash on the same ctx converges fine on
    a fresh one. Since trust-constr can and does wander into locally
    unstable regions of parameter space while exploring, one bad trial can
    otherwise silently poison every evaluation after it for the rest of the
    optimization (each returning the 1e5 penalty regardless of how good the
    trial point actually is) - see cost_function below, which rebuilds and
    retries once whenever solve_simulation() fails, rather than only
    penalizing and moving on.

    Reassigns the MODULE-LEVEL ctx (via `global`) rather than returning a
    new one, because cost_function_lambda re-reads the module-level `ctx`
    fresh on every minimize() call - a purely local reassignment inside
    cost_function would not be visible to the next call.
    """
    global ctx
    t0 = time.time()
    ctx = setup_simulation(name, folder_name, simu_card, adventitia_card, media_card)
    print(f"[setup] Persistent context (re)built in {time.time()-t0:.1f}s.", flush=True)

#-----------------------------------------------------------------------------#
# Free parameters - equivalent to the original [E0, k0, lambda0, re] calibration,
# expressed generically. Add/remove ParamSpec entries to calibrate (or later
# run a sensitivity analysis on) any other physical entry in the cards.
#-----------------------------------------------------------------------------#
PARAMETER_SPECS = [
    ParamSpec("E0 (collagen, media)",
              targets=[("media", collagen_keys_media), ("adventitia", collagen_keys_adventitia)], field_="young", subindex=0, bounds=(0.6, 1.5)),
    # ParamSpec("E0 (collagen, adventitia)",
    #           targets=[("adventitia", collagen_keys_adventitia)], field_="young", subindex=0, bounds=(0.6, 1.0)),
    ParamSpec("k0 (collagen, media)",
              targets=[("media", collagen_keys_media), ("adventitia", collagen_keys_adventitia)], field_="young", subindex=1, bounds=(3.0, 6)),
    # ParamSpec("k0 (collagen, adventitia)",
    #           targets=[("adventitia", collagen_keys_adventitia)], field_="young", subindex=1, bounds=(3.0, 4.5)),
    ParamSpec("lambda0 (collagen, media)",
              targets=[("media", collagen_keys_media)], field_="young", subindex=2, bounds=(1.0, 1.5)),
    # ParamSpec("lambda0 (collagen, adventitia)",
    #           targets=[("adventitia", collagen_keys_adventitia)], field_="young", subindex=2, bounds=(1.05, 1.3)),
    # ParamSpec("cells young (SMC)",  
    #       targets=[("media", ["cells"])], field_="young", bounds=(0.001, 0.05)),
    ParamSpec("re", is_geometry=True, bounds=(0.43, 0.48)),
]
# Example of extending the calibration to more physical parameters, left
# commented out - just uncomment to add them as free parameters. A single
# target list ([("media", [...])] only) makes a parameter specific to one
# layer ; listing both layers (as above) shares it between them :
#   ParamSpec("matrix young (media)", targets=[("media", ["matrix"])], field_="young", bounds=(0.01, 0.08)),
#   ParamSpec("cells poisson", targets=[("media", ["cells"])], field_="poisson", bounds=(0.3, 0.49)),
#   ParamSpec("collagen_0 volumic fraction (adv)", targets=[("adventitia", ["collagen_0"])],
#             field_="volumic_fraction", bounds=(0.002, 0.02)),
#   ParamSpec("E0 (media collagen only, independent from adv)", targets=[("media", collagen_keys_media)],
#             field_="young", subindex=0, bounds=(0.1, 1.0)),


BOUNDS = [spec.bounds for spec in PARAMETER_SPECS]
X0_REAL = np.array([get_param_value({"media": media_card, "adventitia": adventitia_card, "simu": simu_card}, spec)
                     for spec in PARAMETER_SPECS])

print(X0_REAL)

#-----------------------------------------------------------------------------#
# Cost function
#-----------------------------------------------------------------------------#
best_error = np.inf
call_cost_func = 0


def cost_function(x, bounds, specs, press_exp, re_exp, re_sd_exp, F_zz_exp, F_zz_sd_exp, fit_mode,
                   media_card, adventitia_card, simu_card, log_filename):
    global call_cost_func, best_error, ctx
    call_cost_func += 1

    x_real = unpack(x, bounds)
    print(f"call {call_cost_func} : {dict(zip([s.label for s in specs], x_real))}")

    cards = {"media": media_card, "adventitia": adventitia_card, "simu": simu_card}
    for spec, value in zip(specs, x_real):
        if spec.is_geometry:
            re = float(value)
            area = simu_card['area']
            ri = np.sqrt(re**2 - area/np.pi)
            advTF = simu_card['advTF']
            simu_card['ri'] = ri
            simu_card['re'] = re
            simu_card['ri_adv'] = np.sqrt(re**2 - advTF*(re**2 - ri**2))
        else:
            set_param_value(cards, spec, value)

    result, _ = solve_simulation(ctx, simu_card, adventitia_card, media_card)

    if not result:
        # Don't trust this ctx for anything downstream (including the NEXT
        # cost_function call): rebuild from scratch and retry ONCE, on the
        # SAME trial values (already baked into media_card/adventitia_card/
        # simu_card above), before falling back to the penalty cost - see
        # rebuild_ctx()'s docstring.
        print(f"call {call_cost_func}: solve FAILED - rebuilding context and "
              f"retrying once (in case a previous evaluation left stale "
              f"state behind)...", flush=True)
        rebuild_ctx()
        result, _ = solve_simulation(ctx, simu_card, adventitia_card, media_card)
        if not result:
            print(f"call {call_cost_func}: solve FAILED again on a fresh "
                  f"context - genuine non-convergence at this trial point.",
                  flush=True)

    if not result:
        cost = 1e5
    else:
        press_list = 7500.62*result.outputs['press'][:]
        F_zz_list = result.outputs['F_zz']*1000
        re_list = result.outputs['re_d']
    
        interp_func_re = interp1d(press_list, re_list, kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_func_F_zz = interp1d(press_list, F_zz_list, kind='linear', bounds_error=False, fill_value="extrapolate")
    
        F_zz_interp = interp_func_F_zz(press_exp)
        re_interp = interp_func_re(press_exp)
    
        chi_square_force = np.sum(((F_zz_exp - F_zz_interp)/(F_zz_sd_exp))**2)
        chi_square_rad = np.sum(((re_exp - re_interp)/(re_sd_exp))**2)
        cost = 1/(2*len(re_interp))*chi_square_rad + 1/(2*len(F_zz_interp))*chi_square_force
    
        
        press_list = 7500.62*result.outputs['press'][:]
        F_zz_list = result.outputs['F_zz']*1000
        re_list = result.outputs['re_d']
        
        interp_func_re = interp1d(press_list, re_list, kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_func_F_zz = interp1d(press_list, F_zz_list, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        F_zz_interp = interp_func_F_zz(press_exp)
        re_interp = interp_func_re(press_exp)
        
        # # --- derivative branch ---
        # dF_zz_list_dp = np.gradient(F_zz_list, press_list)
        # dre_list_dp = np.gradient(re_list, press_list)
        
        # interp_func_dF_zz_dp = interp1d(press_list, dF_zz_list_dp, kind='linear', bounds_error=False, fill_value="extrapolate")
        # interp_func_dre_dp = interp1d(press_list, dre_list_dp, kind='linear', bounds_error=False, fill_value="extrapolate")
        
        # dF_zz_interp_dp = interp_func_dF_zz_dp(press_exp)
        # dre_interp_dp = interp_func_dre_dp(press_exp)
        
        dF_zz_interp_dp = np.gradient(F_zz_interp, press_exp)
        dre_interp_dp = np.gradient(re_interp, press_exp)
        
        if fit_mode == 'value':
            chi_square_force = np.sum(((F_zz_exp - F_zz_interp)/(F_zz_sd_exp))**2)
            chi_square_rad   = np.sum(((re_exp - re_interp)/(re_sd_exp))**2)
        elif fit_mode == 'derivative':
            chi_square_force = np.sum(((dF_zz_exp_dp - dF_zz_interp_dp)/(dF_zz_sem_dp))**2)
            chi_square_rad   = np.sum(((dre_exp_dp - dre_interp_dp)/(dre_sem_dp))**2)
        
        cost = 1/(2*len(re_interp))*chi_square_rad + 1/(2*len(F_zz_interp))*chi_square_force
    print(f"Error measure is {cost} for x : {x_real}")
    if cost < best_error:
        best_error = cost
        with open(log_filename, "a") as f:
            f.write(f"{cost}," + ",".join(map(str, x_real)) + "\n")
        print(f"New best cost: {cost:.6f}")
    
    return cost


#-----------------------------------------------------------------------------#
# Run calibration
#-----------------------------------------------------------------------------#
log_filename = f"outputs/{folder_name}/{name}_optimization_logs.csv"
with open(log_filename, "w") as f:
    f.write("error," + ",".join([s.label for s in PARAMETER_SPECS]) + "\n")

fit_mode = 'value'
x0 = pack(X0_REAL, BOUNDS)
cost_function_lambda = lambda x: cost_function(
    x, BOUNDS, PARAMETER_SPECS, press_exp, re_exp, re_sd_exp, F_zz_exp, F_zz_sd_exp, fit_mode,
    media_card, adventitia_card, simu_card, log_filename)

t0 = time.time()
# result_opti = minimize(
#     cost_function_lambda,
#     x0,
#     method='Powell',
#     bounds=[(0, 1) for _ in BOUNDS],
#     options={'xtol': 1e-3, 'ftol': 1e-4, 'maxiter': 200, 'maxfev': 2000}
# )

result_opti = minimize(
    cost_function_lambda,
    x0,
    method="trust-constr", #'L-BFGS-B',
    bounds=[(0, 1) for _ in BOUNDS],
    options={
        'xtol': 1e-4,
        'gtol': 1e-3,              # loosened vs. before - don't chase FE noise
        'maxiter': 300,
        # 'initial_tr_radius': 0.05,  # caps the very first step - solves point 2
        # 'finite_diff_rel_step': 1e-4,  # explicit, noise-aware FD step
    }
)

print(f"Calibration finished in {time.time()-t0:.1f}s over {call_cost_func} evaluations "
      f"({(time.time()-t0)/max(call_cost_func,1):.2f}s/eval on average, vs ~150-300s/eval before)")

x_calib = unpack(result_opti.x, BOUNDS)
print("Calibrated parameters:")
for spec, value in zip(PARAMETER_SPECS, x_calib):
    where = "geometry" if spec.is_geometry else ",".join(layer for layer, _ in spec.targets)
    print(f"  {spec.label} ({where}) = {value}")

# Final solve at the calibrated point, to get `result` for the plotting cells
cards = {"media": media_card, "adventitia": adventitia_card, "simu": simu_card}
for spec, value in zip(PARAMETER_SPECS, x_calib):
    if spec.is_geometry:
        re = float(value)
        area = simu_card['area']
        ri = np.sqrt(re**2 - area/np.pi)
        advTF = simu_card['advTF']
        simu_card['ri'] = ri
        simu_card['re'] = re
        simu_card['ri_adv'] = np.sqrt(re**2 - advTF*(re**2 - ri**2))
    else:
        set_param_value(cards, spec, value)
result, mech = solve_simulation(ctx, simu_card, adventitia_card, media_card)

#-----------------------------------------------------------------------------#
# Export the calibrated cards - this is the single source of truth consumed
# by 4_1_plot_ArterialTissue_calibrated.py, which re-runs (or loads a cached
# run of) the calibrated model at full resolution and does all the plotting /
# post-processing. Nothing plotting-related happens in this script anymore.
#-----------------------------------------------------------------------------#

def remove_nonserializable(d, keys_to_remove=("geometry", "mu_0", "k_0")):
    """
    Recursively strip keys holding live FEniCSx/UFL objects (added in place
    into media_card/adventitia_card during solves - see setup_simulation/
    material_class.py) so the card can be JSON-dumped.
    """
    if isinstance(d, dict):
        return {k: remove_nonserializable(v, keys_to_remove) for k, v in d.items() if k not in keys_to_remove}
    elif isinstance(d, list):
        return [remove_nonserializable(i, keys_to_remove) for i in d]
    else:
        return d


media_card_export = remove_nonserializable(media_card)
adventitia_card_export = remove_nonserializable(adventitia_card)

with open(f'./outputs/{folder_name}/media_card_calibrated.json', 'w') as fp:
    json.dump(media_card_export, fp, indent=4, sort_keys=True, ensure_ascii=False)
with open(f'./outputs/{folder_name}/adventitia_card_calibrated.json', 'w') as fp:
    json.dump(adventitia_card_export, fp, indent=4, sort_keys=True, ensure_ascii=False)

# Geometry actually used at the calibrated point (ri/re/ri_adv derived from
# the calibrated re + area/advTF - see cost_function above). Kept separate
# from simu_card_calib_gross.json's own nr/nz/etc (mesh resolution used only
# for calibration speed) so the plotting script can apply this exact geometry
# on top of a different, finer simu_card_calib.json.
calibrated_geometry = {"ri": simu_card['ri'], "re": simu_card['re'], "ri_adv": simu_card['ri_adv'],
                        "area": simu_card['area'], "advTF": simu_card['advTF']}
with open(f'./outputs/{folder_name}/calibrated_geometry.json', 'w') as fp:
    json.dump(calibrated_geometry, fp, indent=4)

# Human-readable summary of the raw calibrated values (one entry per ParamSpec)
calibrated_params = {spec.label: float(value) for spec, value in zip(PARAMETER_SPECS, x_calib)}
with open(f'./outputs/{folder_name}/calibrated_params.json', 'w') as fp:
    json.dump(calibrated_params, fp, indent=4)

print(f"Exported calibrated cards to ./outputs/{folder_name}/ "
      f"(media_card_calibrated.json, adventitia_card_calibrated.json, "
      f"calibrated_geometry.json, calibrated_params.json)")
print("Run 4_1_plot_ArterialTissue_calibrated.py to re-run at full resolution and plot.")



#-----------------------------------------------------------------------------#
# Pressure - outer radius & axial force, combined on a twin y-axis
# (Ri dropped, experimental points shown with error bars from DTAstd.npz)
# Preview of the results
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# Load Experimental Data (for comparison plots only - not used to change any
# parameter here, the cards are already calibrated)
#-----------------------------------------------------------------------------#
filename = os.path.join(folder_name, "DTAavg.npz")
data = np.load(filename, allow_pickle=True)

orientation_angles = data["orientation_angles"]
orientation_Low = data["orientation_Low"]

Pressure_mmHg = data["Pressure_mmHg"]
InnerRadius_mm = data["InnerRadius_mm"]
OuterRadius_mm = data["OuterRadius_mm"]
Fzz_Sample_mN = data["Fzz_Sample_mN"]

filename_std = os.path.join(folder_name, "DTAsem.npz")
data_std = np.load(filename_std, allow_pickle=True)
F_zz_sd_exp = data_std['Fzz_Sample_mN']
re_sd_exp = data_std['OuterRadius_mm']

# --- Compute phase lengths ---

load_phase = simu_card['load_phase']
step_load = Artery_load(load_phase)

press_list = 7500.62*step_load.list_P
lambdaz_list = 1+step_load.list_uz/simu_card['lz']

n_phase1 = step_load.index_phase[0][1]
n_phase2 = len(press_list) - step_load.index_phase[1][0]
indices_1 = slice(step_load.index_phase[0][0], step_load.index_phase[0][1])
indices_2 = slice(step_load.index_phase[1][0], step_load.index_phase[1][1]+1)


press_exp = Pressure_mmHg
ri_exp = InnerRadius_mm
re_exp = OuterRadius_mm
F_zz_exp = Fzz_Sample_mN
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