#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity analysis: impact of the number of discrete collagen fiber
families (N) used to represent the adventitia's continuous orientation
distribution -- run through the persistent-context FE workflow
(main_ArterialTissue_persistent.py) instead of the old
main_ArterialTissue_25_06_04.run_simulation().

Adapted from chiv_3_2_fiberdiscretization_ArterialTissue_25_10_06.py. Run
this file first, then 4_2_plot_fiberdiscretization_sa.py (loads the manifest
+ per-N cached results this script writes; never re-runs a simulation).

Why one setup_simulation() call per N, not a single persistent context reused
across every N
------------------------------------------------------------------------------
Changing N changes the NUMBER of collagen_i entries in the adventitia card,
which changes the number of terms summed into the weak form (see
setup_simulation() in main_ArterialTissue_persistent.py: the sum over
collagen_adv_keys is baked into the UFL form at build time, e.g.
s_tensor_collagen_adv = sum(... for key in collagen_adv_keys)). So the
persistent-context trick (build once, solve many times without recompiling)
only pays off WITHIN a fixed N -- each distinct N unavoidably requires its
own FFCx compilation, exactly like the old "one process per N" approach.
What using the persistent main buys here is staying on the current, actively
maintained code path (current Results/card schema, current
update_geometry/update_subdomain_parameters machinery) instead of the frozen
main_ArterialTissue_25_06_04.run_simulation(), plus a cleaner state-reset if
this script is later extended to also try several parameter values at a
fixed N (those extra evaluations would then reuse ctx cheaply).

Memory : per-N compile/solve cost grows with N, and can be OOM-killed
------------------------------------------------------------------------------
The generated C kernel for the weak form's residual/Jacobian grows with N
(each collagen family contributes its own additive term), and BOTH (a) the
one-time FFCx/gcc compilation of that kernel (during setup_simulation()) and
(b) the PETSc/MUMPS direct-solve factorization during the loading loop
(during solve_simulation()) can become memory-heavy at high N -- gcc's
optimization passes (-O2, the dolfinx default) scale badly with generated
function size, independent of anything to do with multiprocessing. This
script mitigates the compile-time half automatically (see
_write_low_memory_jit_config() / LOW_MEMORY_JIT below: caps the C compiler
at -O0, which keeps compile memory roughly flat in N at the cost of a
slower, unoptimized kernel -- an easy trade since each N is only solved
ONCE here). If you're still getting killed after that, check the last
printed line before the kill : anything before "[setup] Setup complete" ->
still the compile step (try lowering mesh resolution, see simu_card_name
below, or ask about restructuring the per-inclusion sum in the framework
itself); "[solve] Time step ..." lines or a Newton iteration -> the
MUMPS direct-solve factorization, for which the levers are mesh resolution
(nr/nz in the simu_card) and, in petsc4py options below, `mat_mumps_icntl_14`
(currently 40, i.e. MUMPS pre-allocates 40% extra workspace as a safety
margin -- lowering it trades a higher chance of a mid-factorization restart
for a lower peak allocation).

Parallelization, caching & crash resilience
------------------------------------------------------------------------------
One OS process per N, at most n_workers concurrent (see run_batch() below) --
NOT a multiprocessing.Pool. A Pool's map() has no clean way to detect a
worker being killed outright by the OS OOM killer (it can just hang the rest
of the run), which is exactly the failure mode reported when running this at
high N even with n_workers=1. Here every N gets its own fresh
multiprocessing.Process (so memory is always fully reclaimed on exit, same
guarantee a Pool would need maxtasksperchild=1 for) AND its exitcode is
checked after join(): a killed N (exitcode<0, e.g. -9 for SIGKILL) is
logged and skipped, and the batch continues with the remaining N's instead
of losing everything. Already-cached N's (result pickle already on disk,
see below) never spawn a process at all. Each N's solve_simulation() call
exports its own result pickle (outputs/<folder>/<name>_N<N>_scal.pkl, via
the Results class -- see main_ArterialTissue_persistent.py's
result.export()); run_batch() checks for that file up front, so re-running
this script (e.g. after editing N_discrete_list, or after fixing whatever
made a high N OOM) only pays for the N's that are still missing.

Material template & total collagen fraction
------------------------------------------------------------------------------
Every swept collagen family (for a given N) shares the SAME calibrated
material law (young modulus Exponential [E0,k0,lambda0], poisson) -- taken
from outputs/Article_Calibration/adventitia_card_calibrated.json's
collagen_0 entry; only theta/volumic_fraction differ per family. Unlike the
old script (which hardcoded f_tot = 0.5 for the total adventitia collagen
volume fraction), this version sums volumic_fraction across every
collagen_i in the CALIBRATED card and redistributes that total across the N
new families -- preserves whatever total fiber content the calibration
actually converged to, instead of assuming a fixed split. Media stays fixed
at its calibrated card -- only the adventitia is swept.

Mesh/loading resolution
------------------------------------------------------------------------------
simu_card_calib.json (fine mesh, same as the final calibrated plots) with
the calibrated geometry (ri/re/ri_adv/area/advTF from
outputs/Article_Calibration/calibrated_geometry.json) applied on top --
chosen over the coarse calibration mesh since geometry/mechanics (not only
the fiber PDF) are compared across N here. If high-N solves are OOM-killed
during solve_simulation() (not setup_simulation()) specifically, switching
to simu_card_calib_gross.json (or a dedicated coarser card) for just the
high-N end of the sweep would directly cut MUMPS's peak memory too.
"""

import copy
import json
import os
import pickle
import time

import multiprocessing as mp

import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from Multiscale_Framework.function_modules.discretization_collagen import (
    discretizing_distribution,
)


def load_JSON(card_name):
    print('Opening card : ' + card_name)
    try:
        with open(card_name, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found: {card_name}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON: {e}")


def _write_low_memory_jit_config():
    """
    Cap the C compiler's optimization level (-O0 instead of dolfinx's
    default -O2) for EVERY fem.form()/fem.Expression() compiled by any code
    in this run, including deep inside Multiscale_Framework -- with zero
    framework code changes needed.

    How : dolfinx.jit.get_options() merges, in increasing priority: built-in
    defaults < ~/.config/dolfinx/dolfinx_jit_parameters.json <
    ./dolfinx_jit_parameters.json (current working directory) < jit_options
    passed explicitly to an individual form()/Expression() call. We don't
    have (and don't need) access to the framework's call sites -- writing
    both config-file locations covers dolfinx's lookup regardless of the
    exact working directory this script is launched from.

    Why -O0 helps : gcc's memory/time cost at -O2 is dominated by
    optimization passes (global value numbering, instruction scheduling,
    register allocation) that scale badly with generated function size --
    and the generated kernel's size grows with N (each collagen family
    contributes its own additive term to the weak form/Jacobian). -O0 skips
    those passes entirely, keeping compile-time memory roughly flat in N,
    at the cost of a slower (unoptimized) kernel being evaluated during
    assembly. Since this script solves each N exactly once, that per-N
    slowdown is a one-time cost -- a good trade against being OOM-killed.
    """
    paths = [
        os.path.join(os.getcwd(), "dolfinx_jit_parameters.json"),
        os.path.expanduser("~/.config/dolfinx/dolfinx_jit_parameters.json"),
    ]
    for path in paths:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        existing = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = {}
        existing["cffi_extra_compile_args"] = ["-O0"]
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
    print(f"[jit] cffi_extra_compile_args=['-O0'] written to:\n  {paths[0]}\n  {paths[1]}\n"
          f"(caps C-compiler memory during FFCx form compilation, at the cost of slower kernels; "
          f"set LOW_MEMORY_JIT=False in this script to skip).", flush=True)


def build_adventitia_card(template, theta_coll_adv, weights_coll_adv):
    """
    Build an N-family adventitia card from the calibrated template : matrix
    unchanged, N collagen families sharing the calibrated collagen_0
    material law (young/poisson/type/phi), only theta/volumic_fraction swept.

    weights_coll_adv is the raw discretization output (sums to 1) -- it is
    rescaled here by the calibrated total collagen fraction f_tot, computed
    from the template itself (sum of volumic_fraction over every collagen_i
    already in the calibrated card), not a hardcoded constant.
    """
    collagen_keys_template = [k for k in template.keys() if k.startswith("collagen")]
    f_tot = sum(template[k]["volumic_fraction"] for k in collagen_keys_template)
    collagen_template = template[collagen_keys_template[0]]

    adventitia_card = {"matrix": copy.deepcopy(template["matrix"])}
    for n, (theta, w) in enumerate(zip(theta_coll_adv, weights_coll_adv)):
        entry = copy.deepcopy(collagen_template)
        entry["theta"] = float(theta)
        entry["volumic_fraction"] = float(f_tot * w)
        adventitia_card[f"collagen_{n}"] = entry
    return adventitia_card


#-----------------------------------------------------------------------------#
# Specific sensitivity analysis : number of collagen families (adventitia)
#-----------------------------------------------------------------------------#

def run_one_N(N, name, folder_name, simu_card, media_card, adventitia_template,
              orientation_angles, orientation_Low):
    """
    Discretize the experimental adventitia orientation PDF into N families,
    then build + setup + solve ONE persistent-context simulation for it
    (own FEniCS JIT cache dir, so different N's compilations never collide).

    Runs as the target of its own multiprocessing.Process (see run_batch()),
    not inside a Pool worker -- if the OS OOM-kills this process, that's
    visible to the parent as a nonzero/negative exitcode after join(), and
    only this one N is lost. Nothing needs to be returned : the result is
    already on disk via solve_simulation()'s own result.export() (see
    main_ArterialTissue_persistent.py), which run_batch() checks for
    up front to skip already-cached N's without spawning a process at all.
    """
    N = int(N)
    name_simu = f"{name}_N{N}"

    cache_path = os.path.expanduser(f"~/.cache/fenics/{name_simu}/")
    os.makedirs(cache_path, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = cache_path

    # Import everything AFTER setting the per-N cache dir env var, and inside
    # the child process (spawn context) so each process gets its own FFCx JIT.
    from Article_Calibration.main_ArterialTissue_persistent import (
        setup_simulation,
        solve_simulation,
    )
    from petsc4py import PETSc
    opts = PETSc.Options()
    opts["mat_mkl_cpardiso_omp_num_threads"] = 1
    opts["mat_mumps_icntl_14"] = 40  # MUMPS extra workspace margin -- lower this
                                      # (e.g. 20, the MUMPS default) to reduce
                                      # peak factorization memory if OOM happens
                                      # during solve_simulation(), not setup.
    opts["num_threads"] = 1

    theta_coll_adv, weights_coll_adv = discretizing_distribution(orientation_angles, orientation_Low, N)

    print(f"[N={N}] Running simulation {name_simu} (pid={os.getpid()})", flush=True)
    adventitia_card = build_adventitia_card(adventitia_template, theta_coll_adv, weights_coll_adv)
    ctx = setup_simulation(name_simu, folder_name, simu_card, adventitia_card, media_card)
    solve_simulation(ctx, simu_card, adventitia_card, media_card)


def run_batch(N_discrete_list, n_workers, pipeline_kwargs, folder_name, name):
    """
    Launch at most n_workers concurrent Process(es), one per N, each a
    brand-new OS process (memory is always fully reclaimed when it exits).
    Unlike multiprocessing.Pool.map(), a Process's .exitcode after join()
    tells us if it was killed (negative -> killed by a signal, e.g. -9 for
    SIGKILL from the OS OOM killer) -- that N is logged and skipped, and the
    batch keeps going with the rest instead of hanging/dying entirely.

    Already-cached N's (result pickle already on disk) are detected here in
    the PARENT process and never spawn a child.

    Returns {N: status} where status is 'cached', 'ok', or ('failed', exitcode).
    """
    status = {}
    pending = []
    for N in N_discrete_list:
        N = int(N)
        result_path = f"./outputs/{folder_name}/{name}_N{N}_scal.pkl"
        if os.path.exists(result_path):
            status[N] = 'cached'
        else:
            pending.append(N)

    n_to_run = len(pending)
    n_cached = len(status)
    print(f"{n_cached} N value(s) already cached, {n_to_run} to run "
          f"(n_workers={n_workers} concurrent process(es) at a time).", flush=True)

    running = {}  # N -> Process

    def launch(N):
        p = mp.Process(target=run_one_N, args=(N,), kwargs=pipeline_kwargs, name=f"SA-N{N}")
        p.start()
        running[N] = p
        print(f"[scheduler] Launched N={N} (pid={p.pid}) -- "
              f"{len(running)}/{n_workers} slot(s) in use, {len(pending)} N's still queued.", flush=True)

    while pending or running:
        while pending and len(running) < n_workers:
            launch(pending.pop(0))
        time.sleep(1.0)
        for N in list(running.keys()):
            p = running[N]
            if not p.is_alive():
                p.join()
                if p.exitcode == 0:
                    status[N] = 'ok'
                    print(f"[scheduler] N={N} finished OK.", flush=True)
                else:
                    status[N] = ('failed', p.exitcode)
                    reason = "likely OOM-killed by the OS (SIGKILL)" if p.exitcode == -9 \
                        else f"exitcode={p.exitcode}"
                    print(f"[scheduler] N={N} DID NOT complete ({reason}) -- "
                          f"skipping, continuing with remaining N's.", flush=True)
                del running[N]

    return status


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        print("context already set")

    #-----------------------------------------------------------------------------#
    # General parameters -- edit N_discrete_list / n_workers here
    #-----------------------------------------------------------------------------#
    folder_name = 'Article_Calibration_SA'
    folder_name_calib = 'Article_Calibration'
    name = '3_2_fiberdiscretization_SA'

    N_discrete_list = np.array([4, 5, 6, 7, 8, 9, 10, 13, 15, 18, 20, 23, 25])
    n_workers = 1  # at most n_workers concurrent processes (each a fresh OS
                    # process per N -- see run_batch()); tune down (even to 1)
                    # if memory is still tight, and see LOW_MEMORY_JIT below
    LOW_MEMORY_JIT = True  # cap the C compiler at -O0 (see _write_low_memory_jit_config)

    simu_card_name = 'json_cards/simu_card_calib.json'
    media_card_name = f'./outputs/{folder_name_calib}/media_card_calibrated.json'
    adventitia_card_name = f'./outputs/{folder_name_calib}/adventitia_card_calibrated.json'
    geometry_name = f'./outputs/{folder_name_calib}/calibrated_geometry.json'

    simu_card = load_JSON(simu_card_name)
    media_card = load_JSON(media_card_name)
    adventitia_template = load_JSON(adventitia_card_name)
    calibrated_geometry = load_JSON(geometry_name)

    simu_card['XDMF_export'] = 0
    simu_card['ri'] = calibrated_geometry['ri']
    simu_card['re'] = calibrated_geometry['re']
    simu_card['ri_adv'] = calibrated_geometry['ri_adv']
    simu_card['area'] = calibrated_geometry['area']
    simu_card['advTF'] = calibrated_geometry['advTF']

    if not os.path.exists(f"outputs/{folder_name}"):
        os.makedirs(f"outputs/{folder_name}")

    if LOW_MEMORY_JIT:
        _write_low_memory_jit_config()

    #-----------------------------------------------------------------------------#
    # Experimental orientation distribution (adventitia) -- same source used by
    # 3_2_calibration_ArterialTissue_persistent.py / 4_1_plot_ArterialTissue_calibrated.py
    #-----------------------------------------------------------------------------#
    filename = os.path.join(folder_name_calib, "DTAavg.npz")
    data = np.load(filename, allow_pickle=True)
    orientation_angles = data["orientation_angles"]
    orientation_Low = data["orientation_Low"]

    #-----------------------------------------------------------------------------#
    # Run -- one setup_simulation()+solve_simulation() per N, each its own
    # process, at most n_workers concurrent, crash-resilient (see run_batch()).
    #-----------------------------------------------------------------------------#
    pipeline_kwargs = dict(
        name=name,
        folder_name=folder_name,
        simu_card=simu_card,
        media_card=media_card,
        adventitia_template=adventitia_template,
        orientation_angles=orientation_angles,
        orientation_Low=orientation_Low,
    )

    t0 = time.time()
    status = run_batch(N_discrete_list, n_workers, pipeline_kwargs, folder_name, name)
    print(f"Sensitivity analysis over {len(N_discrete_list)} N values finished in {time.time()-t0:.1f}s", flush=True)

    ok_N = sorted(N for N, s in status.items() if s in ('ok', 'cached'))
    failed_N = {N: s[1] for N, s in status.items() if isinstance(s, tuple)}
    print(f"Succeeded/cached: {len(ok_N)}/{len(N_discrete_list)} -> N={ok_N}", flush=True)
    if failed_N:
        print(f"WARNING: {len(failed_N)} N value(s) did not complete: {failed_N} "
              f"(exitcode -9 == killed by the OS, almost certainly OOM -- see this script's "
              f"docstring for mitigations). 4_2_plot_fiberdiscretization_sa.py will skip these.",
              flush=True)

    #-----------------------------------------------------------------------------#
    # Manifest for the plotting script : single source of truth for N_discrete_list
    # and which cards/geometry were used, so 4_2 doesn't hardcode a second copy.
    #-----------------------------------------------------------------------------#
    manifest = {
        'folder_name': folder_name,
        'name': name,
        'N_discrete_list': [int(n) for n in N_discrete_list],
        'simu_card_name': simu_card_name,
        'geometry_name': geometry_name,
        'failed_N': failed_N,
    }
    with open(f'./outputs/{folder_name}/{name}_manifest.json', 'w') as fp:
        json.dump(manifest, fp, indent=4)

    print(f"Saved manifest to ./outputs/{folder_name}/{name}_manifest.json")
    print("Run 4_2_plot_fiberdiscretization_sa.py to produce the figures.")