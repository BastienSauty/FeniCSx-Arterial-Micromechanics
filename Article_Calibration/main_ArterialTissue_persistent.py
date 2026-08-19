#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistent-problem variant of main_ArterialTissue_26_07_21.py.

The original run_simulation() rebuilds the mesh, function spaces, materials,
weak form and solver from scratch on every call. That is what forces FFCx to
JIT-recompile every single form/expression again: each call creates brand new
UFL objects (mesh, Functions, Constants...) whose internal, process-global
"count" numbering never matches anything seen before within the same Python
process, so the JIT cache key never hits - independently of whether any
*value* actually changed (confirmed empirically: an unchanged-parameter
repeat call still produced ~90 new compiled files).

This file splits the work into two phases:

    setup_simulation(...)  - build the mesh, function spaces, subdomains,
                              materials, weak form, boundary conditions,
                              solver, and every output fem.form/fem.Expression
                              ONCE. Returns a SimulationContext holding
                              everything needed to solve repeatedly.

    solve_simulation(ctx, simu_card, adventitia_card=None, media_card=None)
                            - resets the mechanical state to the reference
                              configuration, pushes any new parameter values
                              (young modulus, poisson ratio, volumic fraction,
                              orientation, geometry re/ri...) into the SAME
                              persistent objects created by setup_simulation
                              (via update_subdomain_parameters/update_geometry,
                              see mech_problem_class.py / parameter_class.py),
                              and re-runs the loading loop. This never touches
                              a UFL form or fem.Expression, so it never
                              triggers a JIT recompilation: only the very
                              first call to setup_simulation pays the FE
                              compilation cost.

Use setup_simulation() once per optimization run, and solve_simulation() once
per cost function evaluation.

Limitations (see docstrings for details) :
    - assumes UNIFORM radial mesh spacing at setup time (mesh.create_rectangle).
      The media/adventitia split itself is then represented EXACTLY for any
      subsequent (ri, re, ri_adv), via a piecewise-affine radial remap fixed
      once at setup (see Mechanical_Problem_axi.init_radial_layering/
      update_geometry) - no more "snap to nearest element boundary"
      approximation, and no restriction on how far ri_adv can move.
    - XDMF export is not supported in solve_simulation (exporting many re-solved
      time series would need one file per call, defeating the point of reuse).
      Use the original main_ArterialTissue_26_07_21.py for that.
"""

import json, sys, time, os
import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

# Fenicsx modules in the main
from dolfinx import mesh, fem, io
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import ufl

# Homemade Libraries
from Multiscale_Framework.class_modules.mech_problem_class import Mechanical_Problem_axi
from Multiscale_Framework.class_modules.result_class import Results
from Multiscale_Framework.class_modules.load_class import Artery_load

from Multiscale_Framework.function_modules.auxiliary_functions import (
    Tensor2Voigt,
    Voigt2Tensor,
)


@dataclass
class SimulationContext:
    """
    Holds every persistent FEniCSx object built once by setup_simulation() :
    the mechanical problem itself, boundary condition constants, and every
    output fem.form/fem.Expression. solve_simulation() only ever reads/writes
    into these existing objects.
    """
    mech: Any
    name: str
    folder_name: str
    T_press: Any
    disp_z: Any
    step_load: Any
    n_NR: int
    tag_adv: int
    tag_media: int
    ri0: float
    re0: float
    ri_adv0: float
    nr: int
    nz: int
    lz: float
    volume_media_form: Any
    volume_adv_form: Any
    ur: Any
    ur_expr: Any
    collagen_adv_keys: List[str]
    collagen_media_keys: List[str]
    collagen_adv_expr: Dict[str, Any]
    collagen_media_expr: Dict[str, Any]
    collagen_adv_keys_theta: List[str]
    collagen_adv_keys_stretch: List[str]
    collagen_adv_keys_young: List[str]
    collagen_media_keys_theta: List[str]
    collagen_media_keys_stretch: List[str]
    collagen_media_keys_young: List[str]
    r_pos: Any
    r_pos_expr: Any
    bottom_line_cells: Any
    S_yy: Any
    S_yy_expr: Any
    S_yy_avg_media_expr: Any
    S_yy_avg_adv_expr: Any
    S_zz_avg_media_expr: Any
    S_zz_avg_adv_expr: Any
    F_zz_form: Any
    s_yy_matrix_avg_form: Any
    s_yy_cell_avg_form: Any
    s_yy_collagen_media_avg_form: Any
    s_yy_collagen_adv_avg_form: Any
    s_zz_matrix_avg_form: Any
    s_zz_cell_avg_form: Any
    s_zz_collagen_media_avg_form: Any
    s_zz_collagen_adv_avg_form: Any
    sig_er_export: Any
    sig_er_export_expr: Any
    list_outputs_avg: List[str]
    list_outputs_local: List[str]


def setup_simulation(name, folder_name, simu_card, adventitia_card, media_card):
    """
    Build the mesh, function spaces, subdomains, materials, weak form,
    boundary conditions, solver, and every output form/expression ONCE.
    This is the only call in the whole workflow that pays the FE/FFCx
    compilation cost (comparable to run_simulation()'s first call).

    Returns
    -------
    SimulationContext, to be passed to solve_simulation() as many times as needed.
    """
    t_setup0 = time.time()
    objective_derivative = simu_card["objective_derivative"]
    n_int = simu_card["n_int"]
    n_NR = simu_card["n_NR"]
    XDMF_export = simu_card['XDMF_export']
    if XDMF_export:
        raise NotImplementedError(
            "XDMF export is not supported in the persistent setup/solve workflow "
            "(it would need a distinct file per solve). Use "
            "main_ArterialTissue_26_07_21.py::run_simulation for that."
        )

    # Loads (fixed loading protocol - list_dP/list_duz/list_dt are read by
    # index every solve, not mutated, so the object is safely reusable)
    load_phase = simu_card['load_phase']
    step_load = Artery_load(load_phase)

    # geom
    ri = simu_card['ri']
    re = simu_card['re']
    lz = simu_card['lz']
    ri_adv = simu_card['ri_adv']
    nr = simu_card['nr']
    nz = simu_card['nz']

    #-------------------------------------------------------------------------#
    ### Initiate Mech object
    mech = Mechanical_Problem_axi(name, objective_derivative, n_int)

    #-------------------------------------------------------------------------#
    ### Geometry : create Mesh (only time this is ever called for this context)
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [[ri, 0], [re, lz]], [nr, nz], mesh.CellType.quadrilateral)
    mech.build_space_functions(domain)

    # Place the media/adventitia interface EXACTLY at ri_adv (piecewise-affine
    # radial remap, area-conserving for both layers), instead of snapping to
    # the nearest of the nr uniform element boundaries. See
    # Mechanical_Problem_axi.init_radial_layering/update_geometry. This also
    # fixes, once and for all, how many of the nr radial elements are
    # allocated to each layer (mech.nmed) - update_geometry() can then move
    # ri/re/ri_adv freely afterwards without any restriction.
    mech.init_radial_layering(ri, re, ri_adv, nr)
    print(f"[setup] Adventitia interface placed exactly at ri_adv={ri_adv} "
          f"({mech.nmed}/{nr} radial elements allocated to the media)", flush=True)

    #-------------------------------------------------------------------------#
    ### Material properties
    def Omega_adv(x):
        return (x[0] >= ri_adv) | (np.isclose(x[0], ri_adv))
    tag_adv = 1
    cells_adv = mesh.locate_entities(mech.domain, mech.domain.topology.dim, Omega_adv)

    adventitia_card["geometry"] = {"type": "geometry",
                               "cells": cells_adv,
                               "tag": tag_adv,
                               "domain": mech.domain,
                               "stiff spacefunction": mech.V_stiff,
                               "mandel spacefunction": mech.V_mandel,
                               "scalar spacefunction": mech.V_scalar,
                               "vector spacefunction": mech.V_vec,
                               "matrix spacefunction": mech.V_mat,
                               "objective derivative": mech.objective_derivative}
    mech.add_subdomain("adventitia", adventitia_card, 'MT')

    def Omega_media(x):
        return (x[0] <= ri_adv) | (np.isclose(x[0], ri_adv))
    tag_media = 2
    cells_media = mesh.locate_entities(mech.domain, mech.domain.topology.dim, Omega_media)

    media_card["geometry"] = {"type": "geometry",
                               "cells": cells_media,
                               "tag": tag_media,
                               "domain": mech.domain,
                               "stiff spacefunction": mech.V_stiff,
                               "mandel spacefunction": mech.V_mandel,
                               "scalar spacefunction": mech.V_scalar,
                               "vector spacefunction": mech.V_vec,
                               "matrix spacefunction": mech.V_mat,
                               "objective derivative": mech.objective_derivative}
    mech.add_subdomain("media", media_card, 'MT')

    mech.build_meshtags()

    #-------------------------------------------------------------------------#
    ### Building Weak form (this + microscopic_mech() inside it is where every
    ### fem.Expression/fem.form gets JIT-compiled, exactly once)
    mech.build_weak_form()
    mech.update_local_quantities()

    #-------------------------------------------------------------------------#
    ### Boundary conditions
    boundaries = [(1, lambda x: np.isclose(x[0], ri)),
                  (2, lambda x: np.isclose(x[0], re)),
                  (3, lambda x: np.isclose(x[1], 0)),
                  (4, lambda x: np.isclose(x[1], lz))]

    T_press = fem.Constant(mech.domain, np.array([0, 0], dtype=np.float64))
    disp_z = fem.Constant(mech.domain, 0.)

    boundary_conditions = [["Dirichlet", 3, ("clamped", 1)],
                           ["Dirichlet", 4, (disp_z, 1)],
                           ["Neumann_follower", 1, T_press]]
    mech.build_BCs(boundaries, boundary_conditions)

    #-------------------------------------------------------------------------#
    mech.build_solver()

    #-------------------------------------------------------------------------#
    ### Output forms/expressions - built once, reused (re-assembled/re-interpolated,
    ### never recompiled) by every solve_simulation() call.
    volume_media_form = fem.form(mech.r*mech.dx(tag_media))
    volume_adv_form = fem.form(mech.r*mech.dx(tag_adv))

    V_u_exp = fem.functionspace(domain, ("P", 1, (1, )))
    ur = fem.Function(V_u_exp)
    ur_expr = fem.Expression(ufl.dot(ufl.as_vector([1, 0]), mech.un), V_u_exp.element.interpolation_points())

    # Collagen fiber angle/stretch/young expressions - adventitia
    keys = mech.subdomain['adventitia'].inclusions.keys()
    collagen_adv_keys = [k for k in keys if k.startswith("collagen")]
    collagen_adv_expr = {}
    for key in collagen_adv_keys:
        eer = mech.subdomain["adventitia"].inclusions[key].e_r
        eet = mech.subdomain["adventitia"].inclusions[key].e_theta
        costp = ufl.dot(eer, ufl.as_vector([0, 0, 1]))
        sintp = ufl.dot(eet, ufl.as_vector([0, 0, -1]))
        angle = ufl.atan2(sintp, costp)
        abs_angle = ufl.conditional(angle >= 0, angle, -angle)
        folded_angle = ufl.conditional(abs_angle <= np.pi/2, angle, angle - np.pi*ufl.sign(angle))

        collagen_adv_theta = 180/np.pi*folded_angle*mech.r*mech.dx(tag_adv)
        collagen_adv_young = mech.subdomain["adventitia"].inclusions[key].E.func[0]*mech.r*mech.dx(tag_adv)
        collagen_adv_stretch = mech.subdomain["adventitia"].inclusions[key].lambda_er[0]*mech.r*mech.dx(tag_adv)

        collagen_adv_expr[key] = {'theta': fem.form(collagen_adv_theta),
                                   'young': fem.form(collagen_adv_young),
                                   'stretch': fem.form(collagen_adv_stretch)}

    collagen_adv_keys_theta = [key + '_adv_theta' for key in collagen_adv_keys]
    collagen_adv_keys_stretch = [key + '_adv_stretch' for key in collagen_adv_keys]
    collagen_adv_keys_young = [key + '_adv_young' for key in collagen_adv_keys]

    # Collagen fiber angle/stretch/young expressions - media
    keys = mech.subdomain['media'].inclusions.keys()
    collagen_media_keys = [k for k in keys if k.startswith("collagen")]
    collagen_media_expr = {}
    for key in collagen_media_keys:
        eer = mech.subdomain["media"].inclusions[key].e_r
        eet = mech.subdomain["media"].inclusions[key].e_theta
        costp = ufl.dot(eer, ufl.as_vector([0, 0, 1]))
        sintp = ufl.dot(eet, ufl.as_vector([0, 0, -1]))
        angle = ufl.atan2(sintp, costp)
        abs_angle = ufl.conditional(angle >= 0, angle, -angle)
        folded_angle = ufl.conditional(abs_angle <= np.pi/2, angle, angle - np.pi*ufl.sign(angle))

        collagen_media_theta = 180/np.pi*folded_angle*mech.r*mech.dx(tag_media)
        collagen_media_young = mech.subdomain["media"].inclusions[key].E.func[0]*mech.r*mech.dx(tag_media)
        collagen_media_stretch = mech.subdomain["media"].inclusions[key].lambda_er[0]*mech.r*mech.dx(tag_media)

        collagen_media_expr[key] = {'theta': fem.form(collagen_media_theta),
                                     'young': fem.form(collagen_media_young),
                                     'stretch': fem.form(collagen_media_stretch)}

    collagen_media_keys_theta = [key + '_media_theta' for key in collagen_media_keys]
    collagen_media_keys_stretch = [key + '_media_stretch' for key in collagen_media_keys]
    collagen_media_keys_young = [key + '_media_young' for key in collagen_media_keys]

    # Radial position of element centroids, and the (topological) bottom-line
    # cell selection - fixed forever now that the media/adventitia interface
    # sits exactly at ri_adv (see init_radial_layering); VALUES of r_pos are
    # geometry-dependent and get refreshed every solve.
    r_pos = fem.Function(mech.V_scalar)
    r_pos_expr = fem.Expression(mech.x[0], mech.V_scalar.element.interpolation_points())
    r_pos.interpolate(r_pos_expr)
    bottom_line_cells = mesh.locate_entities(mech.domain, mech.domain.topology.dim,
                                              lambda x: (x[1] <= lz/nz) & (x[0] <= ri_adv))

    # Stress fields / forms
    S_yy = fem.Function(mech.V_scalar)
    tau_tissue = ufl.dot(mech.Fn, ufl.dot(Voigt2Tensor(mech.Sn), mech.Fn.T))
    S_yy_form = ufl.dot(ufl.as_vector([0, 1, 0]), ufl.dot(tau_tissue, ufl.as_vector([0, 1, 0])))
    S_yy_expr = fem.Expression(S_yy_form, mech.V_scalar.element.interpolation_points())
    S_yy.interpolate(S_yy_expr)
    S_zz_form = ufl.dot(ufl.as_vector([0, 0, 1]), ufl.dot(tau_tissue, ufl.as_vector([0, 0, 1])))

    S_yy_avg_media_expr = fem.form(S_yy_form*mech.r*mech.dx(tag_media))
    S_yy_avg_adv_expr = fem.form(S_yy_form*mech.r*mech.dx(tag_adv))
    S_zz_avg_media_expr = fem.form(S_zz_form*mech.r*mech.dx(tag_media))
    S_zz_avg_adv_expr = fem.form(S_zz_form*mech.r*mech.dx(tag_adv))

    F_zz_form = fem.form(2*np.pi*S_zz_form*mech.r*mech.ds(4))

    s_tensor_matrix = mech.subdomain['media'].matrix.taun
    s_tensor_cell = mech.subdomain['media'].inclusions['cells'].taun
    f_collagen_media = sum(mech.subdomain['media'].inclusions[key].f.func[0] for key in collagen_media_keys)
    s_tensor_collagen_media = sum(mech.subdomain['media'].inclusions[key].f.func[0]*mech.subdomain['media'].inclusions[key].taun for key in collagen_media_keys)/f_collagen_media

    f_collagen_adv = sum(mech.subdomain['adventitia'].inclusions[key].f.func[0] for key in collagen_adv_keys)
    s_tensor_collagen_adv = sum(mech.subdomain['adventitia'].inclusions[key].f.func[0]*mech.subdomain['adventitia'].inclusions[key].taun for key in collagen_adv_keys)/f_collagen_adv

    s_yy_matrix = ufl.dot(s_tensor_matrix, ufl.as_vector([0, 1, 0, 0, 0, 0]))
    s_yy_cell = ufl.dot(s_tensor_cell, ufl.as_vector([0, 1, 0, 0, 0, 0]))
    s_yy_collagen_media = ufl.dot(s_tensor_collagen_media, ufl.as_vector([0, 1, 0, 0, 0, 0]))
    s_yy_collagen_adv = ufl.dot(s_tensor_collagen_adv, ufl.as_vector([0, 1, 0, 0, 0, 0]))

    s_zz_matrix = ufl.dot(s_tensor_matrix, ufl.as_vector([0, 0, 1, 0, 0, 0]))
    s_zz_cell = ufl.dot(s_tensor_cell, ufl.as_vector([0, 0, 1, 0, 0, 0]))
    s_zz_collagen_media = ufl.dot(s_tensor_collagen_media, ufl.as_vector([0, 0, 1, 0, 0, 0]))
    s_zz_collagen_adv = ufl.dot(s_tensor_collagen_adv, ufl.as_vector([0, 0, 1, 0, 0, 0]))

    s_yy_matrix_avg_form = fem.form(s_yy_matrix*mech.r*mech.dx(tag_media))
    s_yy_cell_avg_form = fem.form(s_yy_cell*mech.r*mech.dx(tag_media))
    s_yy_collagen_media_avg_form = fem.form(s_yy_collagen_media*mech.r*mech.dx(tag_media))
    s_yy_collagen_adv_avg_form = fem.form(s_yy_collagen_adv*mech.r*mech.dx(tag_adv))

    s_zz_matrix_avg_form = fem.form(s_zz_matrix*mech.r*mech.dx(tag_media))
    s_zz_cell_avg_form = fem.form(s_zz_cell*mech.r*mech.dx(tag_media))
    s_zz_collagen_media_avg_form = fem.form(s_zz_collagen_media*mech.r*mech.dx(tag_media))
    s_zz_collagen_adv_avg_form = fem.form(s_zz_collagen_adv*mech.r*mech.dx(tag_adv))

    cell_incl = mech.subdomain['media'].inclusions['cells']
    sig_er_export = fem.Function(mech.V_scalar)
    sig_er_form = ufl.dot(cell_incl.e_r, ufl.dot(Voigt2Tensor(cell_incl.taun), cell_incl.e_r))
    sig_er_export_expr = fem.Expression(sig_er_form, mech.V_scalar.element.interpolation_points())
    sig_er_export.interpolate(sig_er_export_expr, mech.subdomain['media'].cells)

    list_outputs_avg = ['time', 'ri_d', 're_d', 'press', 'area', 'F_zz',
                        'S_yy_avg_adv', 'S_yy_avg_media', 'S_zz_avg_adv', 'S_zz_avg_media',
                        's_yy_matrix_avg', 's_yy_cell_avg', 's_yy_collagen_media_avg', 's_yy_collagen_adv_avg',
                        's_zz_matrix_avg', 's_zz_cell_avg', 's_zz_collagen_media_avg', 's_zz_collagen_adv_avg']
    list_outputs_avg += collagen_adv_keys_theta + collagen_adv_keys_stretch + collagen_adv_keys_young
    list_outputs_avg += collagen_media_keys_theta + collagen_media_keys_stretch + collagen_media_keys_young
    list_outputs_local = ['S_yy', 'sig_er']

    print(f"[setup] Setup complete in {time.time()-t_setup0:.2f}s "
          f"(this is where the one-time FFCx compilation cost is paid)", flush=True)

    return SimulationContext(
        mech=mech, name=name, folder_name=folder_name,
        T_press=T_press, disp_z=disp_z, step_load=step_load, n_NR=n_NR,
        tag_adv=tag_adv, tag_media=tag_media,
        ri0=ri, re0=re, ri_adv0=ri_adv, nr=nr, nz=nz, lz=lz,
        volume_media_form=volume_media_form, volume_adv_form=volume_adv_form,
        ur=ur, ur_expr=ur_expr,
        collagen_adv_keys=collagen_adv_keys, collagen_media_keys=collagen_media_keys,
        collagen_adv_expr=collagen_adv_expr, collagen_media_expr=collagen_media_expr,
        collagen_adv_keys_theta=collagen_adv_keys_theta, collagen_adv_keys_stretch=collagen_adv_keys_stretch,
        collagen_adv_keys_young=collagen_adv_keys_young,
        collagen_media_keys_theta=collagen_media_keys_theta, collagen_media_keys_stretch=collagen_media_keys_stretch,
        collagen_media_keys_young=collagen_media_keys_young,
        r_pos=r_pos, r_pos_expr=r_pos_expr, bottom_line_cells=bottom_line_cells,
        S_yy=S_yy, S_yy_expr=S_yy_expr,
        S_yy_avg_media_expr=S_yy_avg_media_expr, S_yy_avg_adv_expr=S_yy_avg_adv_expr,
        S_zz_avg_media_expr=S_zz_avg_media_expr, S_zz_avg_adv_expr=S_zz_avg_adv_expr,
        F_zz_form=F_zz_form,
        s_yy_matrix_avg_form=s_yy_matrix_avg_form, s_yy_cell_avg_form=s_yy_cell_avg_form,
        s_yy_collagen_media_avg_form=s_yy_collagen_media_avg_form, s_yy_collagen_adv_avg_form=s_yy_collagen_adv_avg_form,
        s_zz_matrix_avg_form=s_zz_matrix_avg_form, s_zz_cell_avg_form=s_zz_cell_avg_form,
        s_zz_collagen_media_avg_form=s_zz_collagen_media_avg_form, s_zz_collagen_adv_avg_form=s_zz_collagen_adv_avg_form,
        sig_er_export=sig_er_export, sig_er_export_expr=sig_er_export_expr,
        list_outputs_avg=list_outputs_avg, list_outputs_local=list_outputs_local,
    )


def solve_simulation(ctx, simu_card, adventitia_card=None, media_card=None):
    """
    Re-solve the persistent problem built by setup_simulation() for a new set
    of parameters, without rebuilding/recompiling anything.

    Parameters
    ----------
    ctx : SimulationContext returned by setup_simulation()
    simu_card : current simulation card. Only 're'/'ri'/'ri_adv' (geometry) are
        read here for a potential geometry update; the loading protocol itself
        (load_phase, n_NR...) is assumed unchanged from setup (rebuild ctx via
        setup_simulation() again if it changes).
    adventitia_card, media_card : optional, possibly partial, parameter dicts
        (same shape as the json cards) with the NEW values to push into the
        persistent materials before solving. Pass None (or an empty dict) to
        keep the current values (e.g. a first exploratory solve).

    Returns
    -------
    (Result, mech) - same contract as the original run_simulation().
    """
    t_solve0 = time.time()
    mech = ctx.mech

    #-------------------------------------------------------------------------#
    # 1) Reset the mechanical state to the reference configuration
    mech.reset_state()
    ctx.T_press.value[0] = 0.0
    ctx.T_press.value[1] = 0.0
    ctx.disp_z.value = 0.0

    #-------------------------------------------------------------------------#
    # 2) Push new material parameters (young modulus, poisson ratio, volumic
    #    fraction, orientation, ... any subset of the json card) in place
    if adventitia_card:
        mech.update_subdomain_parameters("adventitia", adventitia_card)
    if media_card:
        mech.update_subdomain_parameters("media", media_card)

    #-------------------------------------------------------------------------#
    # 3) Update geometry (ri/re/ri_adv) in place - exact for any target values,
    #    no restriction (see Mechanical_Problem_axi.update_geometry): the
    #    media/adventitia interface always lands exactly at ri_adv, area of
    #    the tissue and of the adventitia (through advTF) are conserved by
    #    construction since ri/ri_adv are themselves derived from area/advTF
    #    by the caller (see cost_function in the calibration script).
    ri = simu_card['ri']
    re = simu_card['re']
    ri_adv = simu_card['ri_adv']
    mech.update_geometry(ri, re, ri_adv)

    #-------------------------------------------------------------------------#
    # 4) Refresh every DERIVED per-inclusion/matrix quantity (young modulus
    #    field, axial stretch lambda_er, microscopic stress increment
    #    expressions, the Mori-Tanaka H2inv/M matrices...) from the state we
    #    just reset/updated above. reset_state() only zeroes/re-identities the
    #    PRIMARY fields (Fn, taun, un, Sn) ; update_micro_mech()/H2inv/M are
    #    only ever recomputed as a side effect of compute_increment() during
    #    the loading loop below, so without this call they would still hold
    #    whatever they were left at by the END of the PREVIOUS solve_simulation
    #    call - including a partially-diverged/crashed one, since a failed
    #    Newton-Raphson step still calls compute_increment() before detecting
    #    non-convergence. That stale derived state (not the physical
    #    parameters themselves) is what corrupts the very first Newton step of
    #    the next solve : this call is what actually makes reset_state() reset
    #    everything, not just the primary fields. Cheap (re-interpolation of
    #    already-compiled expressions only), never triggers a recompilation.
    mech.update_local_quantities()

    #-------------------------------------------------------------------------#
    # 5) Refresh geometry-dependent quantities (numeric only - re-assembly /
    #    re-interpolation of already-compiled forms/expressions, no JIT)
    volume_media = fem.assemble_scalar(ctx.volume_media_form)
    volume_adv = fem.assemble_scalar(ctx.volume_adv_form)
    ctx.r_pos.interpolate(ctx.r_pos_expr)
    r_pos_bottom = ctx.r_pos.x.array[ctx.bottom_line_cells]

    #-------------------------------------------------------------------------#
    # 6) Build a fresh (cheap, no FE) Results object for this solve
    load_phase = simu_card['load_phase']
    n_steps = len(ctx.step_load.list_dt)
    dict_outputs = {}
    for key in ctx.list_outputs_avg:
        dict_outputs[key] = {'points': None}
    for key in ctx.list_outputs_local:
        dict_outputs[key] = {'points': r_pos_bottom}
    result = Results(ctx.name, ctx.folder_name, dict_outputs, n_steps)

    #-------------------------------------------------------------------------#
    # 7) t=0 outputs
    ctx.ur.interpolate(ctx.ur_expr)
    result.outputs['time'][0] = 0
    result.outputs['ri_d'][0] = ri + np.max(ctx.ur.x.array[:])
    result.outputs['re_d'][0] = re + np.min(ctx.ur.x.array[:])
    result.outputs['press'][0] = ctx.T_press.value[0]
    result.outputs['area'][0] = np.pi*(result.outputs['re_d'][0]**2 - result.outputs['ri_d'][0]**2)
    result.outputs['F_zz'][0] = fem.assemble_scalar(ctx.F_zz_form)

    for key in ctx.collagen_adv_keys:
        result.outputs[key+'_adv_theta'][0] = fem.assemble_scalar(ctx.collagen_adv_expr[key]['theta'])/volume_adv
        result.outputs[key+'_adv_stretch'][0] = fem.assemble_scalar(ctx.collagen_adv_expr[key]['stretch'])/volume_adv
        result.outputs[key+'_adv_young'][0] = fem.assemble_scalar(ctx.collagen_adv_expr[key]['young'])/volume_adv

    for key in ctx.collagen_media_keys:
        result.outputs[key+'_media_theta'][0] = fem.assemble_scalar(ctx.collagen_media_expr[key]['theta'])/volume_media
        result.outputs[key+'_media_stretch'][0] = fem.assemble_scalar(ctx.collagen_media_expr[key]['stretch'])/volume_media
        result.outputs[key+'_media_young'][0] = fem.assemble_scalar(ctx.collagen_media_expr[key]['young'])/volume_media

    result.outputs["s_yy_matrix_avg"][0] = fem.assemble_scalar(ctx.s_yy_matrix_avg_form)/volume_media
    result.outputs["s_yy_cell_avg"][0] = fem.assemble_scalar(ctx.s_yy_cell_avg_form)/volume_media
    result.outputs["s_yy_collagen_media_avg"][0] = fem.assemble_scalar(ctx.s_yy_collagen_media_avg_form)/volume_media
    result.outputs["s_yy_collagen_adv_avg"][0] = fem.assemble_scalar(ctx.s_yy_collagen_adv_avg_form)/volume_adv

    result.outputs["s_zz_matrix_avg"][0] = fem.assemble_scalar(ctx.s_zz_matrix_avg_form)/volume_media
    result.outputs["s_zz_cell_avg"][0] = fem.assemble_scalar(ctx.s_zz_cell_avg_form)/volume_media
    result.outputs["s_zz_collagen_media_avg"][0] = fem.assemble_scalar(ctx.s_zz_collagen_media_avg_form)/volume_media
    result.outputs["s_zz_collagen_adv_avg"][0] = fem.assemble_scalar(ctx.s_zz_collagen_adv_avg_form)/volume_adv

    result.outputs["S_yy_avg_adv"][0] = fem.assemble_scalar(ctx.S_yy_avg_adv_expr)/volume_adv
    result.outputs["S_yy_avg_media"][0] = fem.assemble_scalar(ctx.S_yy_avg_adv_expr)/volume_media
    result.outputs["S_zz_avg_adv"][0] = fem.assemble_scalar(ctx.S_zz_avg_adv_expr)/volume_adv
    result.outputs["S_zz_avg_media"][0] = fem.assemble_scalar(ctx.S_zz_avg_adv_expr)/volume_media

    result.outputs["S_yy"][0, :] = ctx.S_yy.x.array[ctx.bottom_line_cells]
    result.outputs["sig_er"][0, :] = ctx.sig_er_export.x.array[ctx.bottom_line_cells]

    #-------------------------------------------------------------------------#
    # 8) Run the loading loop (identical logic to the original run_simulation)
    for n in range(1, ctx.n_NR+1):
        ctx.T_press.value[0] += ctx.step_load.list_dP[n-1]
        ctx.disp_z.value += ctx.step_load.list_duz[n-1]
        delta_t = ctx.step_load.list_dt[n-1]

        try:
            num_its, conv = mech.solve_1_step(delta_t)
        except Exception:
            print('Step crashed ; simulation stopped', flush=True)
            result.runtime = False
            result.export()
            return (False, mech)

        if num_its == mech.max_iter:
            print('Step not converged ; simulation stopped', flush=True)
            result.runtime = False
            result.export()
            return (False, mech)

        ctx.ur.interpolate(ctx.ur_expr)
        ctx.S_yy.interpolate(ctx.S_yy_expr)
        ctx.sig_er_export.interpolate(ctx.sig_er_export_expr)

        result.outputs['time'][n] = n
        result.outputs['ri_d'][n] = ri + np.max(ctx.ur.x.array[:])
        result.outputs['re_d'][n] = re + np.min(ctx.ur.x.array[:])
        result.outputs['press'][n] = ctx.T_press.value[0]
        result.outputs['area'][n] = np.pi*(result.outputs['re_d'][n]**2 - result.outputs['ri_d'][n]**2)
        result.outputs['F_zz'][n] = fem.assemble_scalar(ctx.F_zz_form)

        for key in ctx.collagen_adv_keys:
            result.outputs[key+'_adv_theta'][n] = fem.assemble_scalar(ctx.collagen_adv_expr[key]['theta'])/volume_adv
            result.outputs[key+'_adv_stretch'][n] = fem.assemble_scalar(ctx.collagen_adv_expr[key]['stretch'])/volume_adv
            result.outputs[key+'_adv_young'][n] = fem.assemble_scalar(ctx.collagen_adv_expr[key]['young'])/volume_adv

        for key in ctx.collagen_media_keys:
            result.outputs[key+'_media_theta'][n] = fem.assemble_scalar(ctx.collagen_media_expr[key]['theta'])/volume_media
            result.outputs[key+'_media_stretch'][n] = fem.assemble_scalar(ctx.collagen_media_expr[key]['stretch'])/volume_media
            result.outputs[key+'_media_young'][n] = fem.assemble_scalar(ctx.collagen_media_expr[key]['young'])/volume_media

        result.outputs["S_yy_avg_adv"][n] = fem.assemble_scalar(ctx.S_yy_avg_adv_expr)/volume_adv
        result.outputs["S_yy_avg_media"][n] = fem.assemble_scalar(ctx.S_yy_avg_media_expr)/volume_media
        result.outputs["S_zz_avg_adv"][n] = fem.assemble_scalar(ctx.S_zz_avg_adv_expr)/volume_adv
        result.outputs["S_zz_avg_media"][n] = fem.assemble_scalar(ctx.S_zz_avg_media_expr)/volume_media

        result.outputs["s_yy_matrix_avg"][n] = fem.assemble_scalar(ctx.s_yy_matrix_avg_form)/volume_media
        result.outputs["s_yy_cell_avg"][n] = fem.assemble_scalar(ctx.s_yy_cell_avg_form)/volume_media
        result.outputs["s_yy_collagen_media_avg"][n] = fem.assemble_scalar(ctx.s_yy_collagen_media_avg_form)/volume_media
        result.outputs["s_yy_collagen_adv_avg"][n] = fem.assemble_scalar(ctx.s_yy_collagen_adv_avg_form)/volume_adv

        result.outputs["s_zz_matrix_avg"][n] = fem.assemble_scalar(ctx.s_zz_matrix_avg_form)/volume_media
        result.outputs["s_zz_cell_avg"][n] = fem.assemble_scalar(ctx.s_zz_cell_avg_form)/volume_media
        result.outputs["s_zz_collagen_media_avg"][n] = fem.assemble_scalar(ctx.s_zz_collagen_media_avg_form)/volume_media
        result.outputs["s_zz_collagen_adv_avg"][n] = fem.assemble_scalar(ctx.s_zz_collagen_adv_avg_form)/volume_adv

        result.outputs["S_yy"][n, :] = ctx.S_yy.x.array[ctx.bottom_line_cells]
        result.outputs["sig_er"][n, :] = ctx.sig_er_export.x.array[ctx.bottom_line_cells]

        print(f"[solve] Time step {n}, iterations {num_its}, Press {result.outputs['press'][n]} MPa, "
              f"RI {result.outputs['ri_d'][n]}, area {result.outputs['area'][n]}, Residuals {conv}", flush=True)

    result.runtime = time.time() - t_solve0
    result.export()
    print(f'[solve] Runtime for {ctx.name} is {result.runtime:.4f} sec', flush=True)
    return (result, mech)


def load_JSON(card_name):
    print('Opening card : '+card_name)
    try:
        with open(card_name, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found: {card_name}")
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON: {e}")


if __name__ == '__main__':
    # Standalone smoke test : setup once, solve twice with the SAME parameters
    # to verify the second solve reproduces the first (sanity check against
    # main_ArterialTissue_26_07_21.py::run_simulation before wiring this into
    # the calibration loop).
    argv = sys.argv[1:]
    if len(argv) >= 3 and argv[0] == "python" and argv[1] == "-m":
        argv = argv[3:]

    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("foldername")
    parser.add_argument("simu_card")
    parser.add_argument("media_card")
    parser.add_argument("adventitia_card")
    args = parser.parse_args(argv)

    simu_card_name = f"json_cards/{args.simu_card}.json"
    media_card_name = f"json_cards/{args.media_card}.json"
    adventitia_card_name = f"json_cards/{args.adventitia_card}.json"

    simu_card = load_JSON(simu_card_name)
    adventitia_card = load_JSON(adventitia_card_name)
    media_card = load_JSON(media_card_name)

    if not os.path.exists("./outputs/"+args.foldername):
        os.makedirs("./outputs/"+args.foldername)

    ctx = setup_simulation(args.name, args.foldername, simu_card, adventitia_card, media_card)
    result1, mech = solve_simulation(ctx, simu_card)
    result2, mech = solve_simulation(ctx, simu_card)  # should be ~4s and match result1

    plt.figure()
    plt.plot(result1.outputs['press'][:], result1.outputs['re_d'][:], 'o-', label='solve 1')
    plt.plot(result2.outputs['press'][:], result2.outputs['re_d'][:], 'x--', label='solve 2 (repeat)')
    plt.xlabel("Pressure")
    plt.ylabel(r"$R_e$")
    plt.legend()
    plt.grid()
    plt.savefig(f'images_output/{args.foldername}/pressure_radius_persistent_check.pdf')
    plt.show()
