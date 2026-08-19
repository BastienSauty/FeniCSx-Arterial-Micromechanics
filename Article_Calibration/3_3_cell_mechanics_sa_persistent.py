#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity analysis: impact of matrix / smooth-muscle-cell (SMC) / collagen
Young's moduli on cellular mechanics -- run through the persistent-context FE
workflow (main_ArterialTissue_persistent.py) instead of the old
main_ArterialTissue_25_06_04.run_simulation().

Adapted from chiv_3_3_cell_mechanics_ArterialTissue_25_10_06.py. Run this
file first, then 4_3_plot_cell_mechanics_sa.py (loads the manifest + the
per-combination cached results this script writes; never re-runs a
simulation).

Why ONE persistent context for the WHOLE sweep (unlike 3_2's one-per-N)
------------------------------------------------------------------------------
The N-sweep in 3_2_fiberdiscretization_sa_persistent.py needs a fresh
setup_simulation() (and therefore a fresh FFCx compile) per N because N
changes the NUMBER of collagen_i terms baked into the weak form. This sweep
is different in kind: E_m, E_smc and E_coll are plain VALUES inside the
material cards -- the card STRUCTURE (number of inclusions in each layer)
never changes across any combination below. That is exactly the situation
main_ArterialTissue_persistent.py was built for (see its module docstring
and 3_1_calibration_ArterialTissue_persistent.py, which already reuses one
ctx across every cost-function evaluation of an optimizer): setup_simulation()
is called ONCE here, and every combination in every sweep below is just
another solve_simulation(ctx, ...) call with new parameter values pushed in
via mech.update_subdomain_parameters(...) -- no recompilation, no
multiprocessing, no OOM story. This should be far cheaper than the thesis
script's one-process-per-combination Pool.

Design decisions made without re-asking (confirmed with Bastien 2026-08-17)
------------------------------------------------------------------------------
- Baseline cards = the ACTUAL calibrated cards
  (outputs/Article_Calibration/{media,adventitia}_card_calibrated.json) +
  calibrated_geometry.json, not the thesis's hardcoded
  [E_m0, E_c0, E_coll0, k_coll0, lambda_coll0] = [0.05, 0.01, 0.67, 3.64, 1.1]
  read off json_cards/*_calib.json. Mesh/loading resolution =
  simu_card_calib.json (fine mesh), matching 3_2's choice, not the coarse
  calibration mesh.
- Swept values are expressed as RATIOS of whatever the calibrated baseline
  turns out to be (read at runtime from the calibrated cards -- see
  get_param_value below), not absolute MPa values. The ratio lists below
  were chosen to reproduce the THESIS sweep's relative/fold-change spread
  around ITS baseline (e.g. thesis E_m: [0.025,0.04,0.05,0.06,0.08] around
  0.05 -> fold-changes [0.5,0.8,1.0,1.2,1.6], reused verbatim as ratios;
  E_coll's odd thesis fold-changes [~0.75,1.0,~1.12,~1.49] were rounded to
  a clean [0.75,1.0,1.25,1.5]) -- edit SWEEP_SPECS below directly if a
  different spread is wanted.
- k_coll / lambda0_coll : NOT swept here, same as the thesis __main__ (which
  carried them in the pipeline_SA parameter vector but never actually varied
  them in any of its three sweeps) -- held fixed at the calibrated value.
  Trivial to add as another SweepSpec (field_="young", subindex=1 or 2) if
  you want that later.
- plot_fiber_orientation's thesis-hardcoded output key 'collagen_4_adv_theta'
  (implicitly assumed >=5 adventitia families and picked index 4 with no
  stated reason) is replaced by picking, ONCE here, the adventitia collagen
  family with the LARGEST calibrated volumic_fraction -- deterministic,
  N-agnostic (works whether the calibrated adventitia card has 8 families or
  any other number), and physically = the dominant fiber direction. Recorded
  in the manifest as 'dominant_adventitia_collagen_key' so 4_3 doesn't
  re-derive it. Flag if you'd rather plot every family, or a specific one.
- Per-combination result caching is done BY HAND: solve_simulation()'s own
  automatic result.export() (see main_ArterialTissue_persistent.py) always
  writes to a path built from ctx.name/folder_name -- fixed once at
  setup_simulation() time. With a single shared ctx (unlike 3_2, where every
  N got its own ctx.name), every solve_simulation() call here would export
  to the SAME path and overwrite it. So this script pickles the `result`
  object it gets back to a combination-specific filename itself (mirroring
  what the framework's own export does per-N in 3_2), and checks for that
  file up front to skip combinations that are already cached.
"""

import copy
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from Article_Calibration.main_ArterialTissue_persistent import (
    setup_simulation,
    solve_simulation,
    load_JSON,
)


#-----------------------------------------------------------------------------#
# Generic parameter get/set -- same field_/subindex/targets convention as
# 3_1_calibration_ArterialTissue_persistent.py's ParamSpec/get_param_value/
# set_param_value. Redefined here (not imported) since 3_1 runs a full
# calibration as top-level module code and is not safely importable.
#-----------------------------------------------------------------------------#

@dataclass
class SweepSpec:
    """
    One swept parameter, expressed as a list of RATIOS applied to whatever
    baseline value is currently in the calibrated cards (see get_param_value/
    set_param_value below).

    label      : short tag used in filenames/manifest, e.g. "E_m".
    axis_label : LaTeX-ish label used in plot legends/titles, e.g. r"E_m".
    targets    : list of (layer, keys) pairs that all share this ONE value,
                 same convention as 3_1's ParamSpec.targets -- layer is
                 "media" or "adventitia", keys is a list of json-card entry
                 names ("matrix", "cells", or a collagen_i key).
    field_     : "young", "poisson", "volumic_fraction", ...
    subindex   : for field_="young" with the Exponential law [E0,k0,lambda0],
                 which coefficient this targets (0/1/2). None for a Constant
                 young modulus (matrix, cells).
    ratios     : multiplicative factors applied to the baseline value.
    """
    label: str
    axis_label: str
    targets: List[tuple] = field(default_factory=list)
    field_: str = ""
    subindex: Optional[int] = None
    ratios: List[float] = field(default_factory=list)


def get_param_value(cards, spec):
    layer, keys = spec.targets[0]  # all targets share the same value - read the first
    entry = cards[layer][keys[0]]
    if spec.field_ == "young" and spec.subindex is not None:
        return entry["young"][spec.subindex][0]
    return entry[spec.field_]


def set_param_value(cards, spec, value):
    for layer, keys in spec.targets:
        card = cards[layer]
        for key in keys:
            entry = card[key]
            if spec.field_ == "young" and spec.subindex is not None:
                entry["young"][spec.subindex][0] = float(value)
            else:
                entry[spec.field_] = float(value)


def tag_value(value):
    """Filesystem/manifest-safe string for a float parameter value."""
    return f"{value:.6g}"


if __name__ == "__main__":

    #-------------------------------------------------------------------------#
    # General parameters
    #-------------------------------------------------------------------------#
    folder_name = 'Article_Calibration_SA'
    folder_name_calib = 'Article_Calibration'
    name = '3_3_cell_mechanics_SA'

    simu_card_name = 'json_cards/simu_card_calib.json'
    media_card_name = f'./outputs/{folder_name_calib}/media_card_calibrated.json'
    adventitia_card_name = f'./outputs/{folder_name_calib}/adventitia_card_calibrated.json'
    geometry_name = f'./outputs/{folder_name_calib}/calibrated_geometry.json'

    simu_card = load_JSON(simu_card_name)
    media_card_baseline = load_JSON(media_card_name)
    adventitia_card_baseline = load_JSON(adventitia_card_name)
    calibrated_geometry = load_JSON(geometry_name)

    simu_card['XDMF_export'] = 0
    simu_card['ri'] = calibrated_geometry['ri']
    simu_card['re'] = calibrated_geometry['re']
    simu_card['ri_adv'] = calibrated_geometry['ri_adv']
    simu_card['area'] = calibrated_geometry['area']
    simu_card['advTF'] = calibrated_geometry['advTF']

    collagen_keys_media = [k for k in media_card_baseline if k.startswith("collagen")]
    collagen_keys_adventitia = [k for k in adventitia_card_baseline if k.startswith("collagen")]

    if not os.path.exists(f"outputs/{folder_name}"):
        os.makedirs(f"outputs/{folder_name}")

    #-------------------------------------------------------------------------#
    # Build the persistent FE context ONCE -- reused for every combination in
    # every sweep below (see module docstring). setup_simulation() mutates its
    # card arguments in place (adds a "geometry" entry per subdomain), so pass
    # it deep copies and keep media_card_baseline/adventitia_card_baseline
    # pristine for later use as the per-combination reset point.
    #
    # Robustness note (2026-08-17): a failed/non-converged solve_simulation()
    # call can leave stale derived state behind on ctx.mech that reset_state()
    # + update_local_quantities() don't fully clean up (see
    # main_ArterialTissue_persistent.py's solve_simulation() docstring) -
    # observed as every combination AFTER a crash also failing, instantly,
    # with zero converged load steps (vs. the crashed combo itself, which
    # typically fails partway through a real loading path). rebuild_ctx()
    # below pays a fresh FFCx-free setup (no recompilation - same card
    # structure) to reset ctx from scratch whenever a solve fails, and the
    # run loop retries once on the fresh context before giving up on that
    # combination - see run_batch loop below.
    #-------------------------------------------------------------------------#
    def rebuild_ctx():
        t0 = time.time()
        new_ctx = setup_simulation(
            name, folder_name, simu_card,
            copy.deepcopy(adventitia_card_baseline),
            copy.deepcopy(media_card_baseline),
        )
        print(f"[setup] Persistent context (re)built in {time.time()-t0:.1f}s.", flush=True)
        return new_ctx

    ctx = rebuild_ctx()

    #-------------------------------------------------------------------------#
    # Sweep definitions (see module docstring for the ratio choices). Each
    # sweep varies ONE parameter at a time, holding every other parameter at
    # its calibrated baseline -- same one-at-a-time logic as the thesis
    # __main__'s three p_list blocks.
    #-------------------------------------------------------------------------#
    SWEEP_SPECS = [
        SweepSpec(
            "E_m", r"E_m",
            targets=[("media", ["matrix"]), ("adventitia", ["matrix"])],
            field_="young",
            ratios=[0.8, 1.0, 1.2, 1.6],
        ),
        SweepSpec(
            "E_smc", r"E_{smc}",
            targets=[("media", ["cells"])],
            field_="young",
            ratios=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
        ),
        SweepSpec(
            "E_coll", r"E_{coll}",
            targets=[("media", collagen_keys_media), ("adventitia", collagen_keys_adventitia)],
            field_="young", subindex=0,
            ratios=[0.75, 1.0, 1.25, 1.5],
        ),
    ]

    cards_baseline = {"media": media_card_baseline, "adventitia": adventitia_card_baseline}
    baseline_values = {spec.label: get_param_value(cards_baseline, spec) for spec in SWEEP_SPECS}
    print("Baseline (calibrated) values:", baseline_values, flush=True)

    #-------------------------------------------------------------------------#
    # Representative adventitia collagen family for plot_fiber_orientation --
    # see module docstring (replaces the thesis's hardcoded 'collagen_4').
    #-------------------------------------------------------------------------#
    dominant_key = max(
        collagen_keys_adventitia,
        key=lambda k: adventitia_card_baseline[k]["volumic_fraction"],
    )
    print(f"Dominant adventitia collagen family (largest calibrated "
          f"volumic_fraction): {dominant_key}", flush=True)

    #-------------------------------------------------------------------------#
    # Run every combination. Sequential -- every call reuses ctx, no
    # recompilation (see module docstring). Each combination resets BOTH
    # cards to a fresh deep copy of the calibrated baseline before overriding
    # the one swept value, so sweeps never accumulate state into each other.
    #-------------------------------------------------------------------------#
    manifest_entries = []

    t0 = time.time()
    for spec in SWEEP_SPECS:
        baseline_value = baseline_values[spec.label]
        for ratio in spec.ratios:
            value = baseline_value * ratio

            media_card = copy.deepcopy(media_card_baseline)
            adventitia_card = copy.deepcopy(adventitia_card_baseline)
            cards = {"media": media_card, "adventitia": adventitia_card}
            set_param_value(cards, spec, value)

            result_path = (f"./outputs/{folder_name}/{name}_{spec.label}_"
                            f"{tag_value(value)}_scal.pkl")

            if os.path.exists(result_path):
                print(f"[{spec.label}={value:.4g} MPa (x{ratio})] cached -> "
                      f"{result_path}", flush=True)
                status = "cached"
            else:
                print(f"[{spec.label}={value:.4g} MPa (x{ratio})] running "
                      f"(pid={os.getpid()})...", flush=True)
                result, _ = solve_simulation(ctx, simu_card, adventitia_card, media_card)

                if result is False:
                    # Don't trust this ctx for anything downstream: rebuild
                    # from scratch and retry ONCE before giving up on this
                    # combination (see rebuild_ctx() note above).
                    print(f"[{spec.label}={value:.4g} MPa (x{ratio})] solve "
                          f"FAILED - rebuilding context and retrying once "
                          f"(in case the previous combination left stale "
                          f"state behind)...", flush=True)
                    ctx = rebuild_ctx()
                    result, _ = solve_simulation(ctx, simu_card, adventitia_card, media_card)

                if result is False:
                    print(f"[{spec.label}={value:.4g} MPa (x{ratio})] solve "
                          f"FAILED again on a fresh context - genuine "
                          f"non-convergence for this parameter value, "
                          f"skipping.", flush=True)
                    manifest_entries.append({
                        "param": spec.label, "ratio": ratio, "value": value,
                        "status": "failed", "result_path": None,
                    })
                    continue

                with open(result_path, 'wb') as fh:
                    pickle.dump(result, fh)
                status = "ok"

            manifest_entries.append({
                "param": spec.label, "ratio": ratio, "value": value,
                "status": status, "result_path": result_path,
            })

    n_total = sum(len(s.ratios) for s in SWEEP_SPECS)
    n_failed = sum(1 for e in manifest_entries if e["status"] == "failed")
    print(f"Sensitivity analysis over {n_total} combinations finished in "
          f"{time.time()-t0:.1f}s ({n_failed} failed).", flush=True)

    #-------------------------------------------------------------------------#
    # Manifest for the plotting script -- single source of truth for the
    # sweep definitions, baseline, dominant fiber family, card/geometry paths
    # and per-combination result locations, so 4_3 never hardcodes a second
    # copy of any of it.
    #-------------------------------------------------------------------------#
    manifest = {
        'folder_name': folder_name,
        'folder_name_calib': folder_name_calib,
        'name': name,
        'simu_card_name': simu_card_name,
        'media_card_name': media_card_name,
        'adventitia_card_name': adventitia_card_name,
        'geometry_name': geometry_name,
        'baseline_values': baseline_values,
        'sweeps': [
            {"label": s.label, "axis_label": s.axis_label, "ratios": s.ratios}
            for s in SWEEP_SPECS
        ],
        'dominant_adventitia_collagen_key': dominant_key,
        'entries': manifest_entries,
    }
    with open(f'./outputs/{folder_name}/{name}_manifest.json', 'w') as fp:
        json.dump(manifest, fp, indent=4)

    print(f"Saved manifest to ./outputs/{folder_name}/{name}_manifest.json")
    print("Run 4_3_plot_cell_mechanics_sa.py to produce the figures.")