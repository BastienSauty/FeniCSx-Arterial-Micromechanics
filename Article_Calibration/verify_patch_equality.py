#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Before/after equality check for the inclusions_class.py compile-count patch
(bundling e_r/e_theta/e_phi into one fem.Expression, and dF/dF_inel into
another, per inclusion -- see the accompanying patch notes). This does NOT
check compile time or memory: it checks that the physics is bit-for-bit
(up to floating point noise) unchanged.

How to use
----------
1. With the ORIGINAL (unpatched) inclusions_class.py still in place:

       python3 verify_patch_equality.py --save before.npz

2. Swap in the PATCHED inclusions_class.py (the one delivered alongside this
   script), then:

       python3 verify_patch_equality.py --compare before.npz

Run this from inside Article_Calibration/ (same working directory you'd use
to run 3_2_fiberdiscretization_sa_persistent.py), so the relative paths to
the json_cards/ and outputs/ directories resolve the same way.

What it does
------------
Runs ONE small setup_simulation()+solve_simulation() (N=5 collagen families,
same cards your real sensitivity analysis uses) and, for every inclusion in
every subdomain (media, adventitia), extracts the FINAL post-solve state:
taun (cumulative stress), Fn (cumulative deformation gradient), and -- where
the inclusion type has them -- e_r/e_theta/e_phi (orientation) and F_inel
(inelastic deformation gradient). These are the actual physical outputs any
calibration/plot depends on, so if they match before vs after, the patch is
behavior-preserving for this scenario.

N=5 is deliberately small/fast; if you want extra confidence, rerun with a
larger --N (e.g. 15 or 25) as a second check, since some patched classes
(Active_Spheroidal_inclusion, Prestretched_Cylinder_inclusion) are only used
for a subset of inclusion types and you want at least one of each type
exercised. Check the printed "inclusion types seen" line to confirm.
"""

import argparse
import json
import os

import numpy as np


def load_JSON(card_name):
    with open(card_name, "r") as f:
        return json.load(f)


def build_adventitia_card(template, N):
    """Same discretization used by 3_2_fiberdiscretization_sa_persistent.py."""
    from Multiscale_Framework.function_modules.discretization_collagen import (
        discretizing_distribution,
    )

    filename = os.path.join("Article_Calibration", "DTAavg.npz")
    data = np.load(filename, allow_pickle=True)
    orientation_angles = data["orientation_angles"]
    orientation_Low = data["orientation_Low"]
    theta_coll_adv, weights_coll_adv = discretizing_distribution(orientation_angles, orientation_Low, N)

    import copy
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


def extract_state(mech):
    """
    Walk mech.subdomain -> {material_name: material} -> material.inclusions
    (+ material.matrix) and pull out the final post-solve numeric state of
    every inclusion into a flat dict of numpy arrays, plus a record of which
    inclusion classes were actually exercised (for the docstring's sanity
    check).
    """
    state = {}
    types_seen = set()

    def grab(prefix, obj):
        types_seen.add(type(obj).__name__)
        for attr in ("taun", "Fn", "e_r", "e_theta", "e_phi", "F_inel"):
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if hasattr(val, "x"):  # fem.Function
                    state[f"{prefix}.{attr}"] = np.array(val.x.array, copy=True)

    for mat_name, mat in mech.subdomain.items():
        grab(f"{mat_name}.matrix", mat.matrix)
        for incl_name, incl in mat.inclusions.items():
            grab(f"{mat_name}.{incl_name}", incl)

    return state, types_seen


def run(N):
    from Article_Calibration.main_ArterialTissue_persistent import (
        setup_simulation,
        solve_simulation,
    )

    simu_card = load_JSON("json_cards/simu_card_calib.json")
    media_card = load_JSON("./outputs/Article_Calibration/media_card_calibrated.json")
    adventitia_template = load_JSON("./outputs/Article_Calibration/adventitia_card_calibrated.json")
    calibrated_geometry = load_JSON("./outputs/Article_Calibration/calibrated_geometry.json")

    simu_card["XDMF_export"] = 0
    simu_card["ri"] = calibrated_geometry["ri"]
    simu_card["re"] = calibrated_geometry["re"]
    simu_card["ri_adv"] = calibrated_geometry["ri_adv"]
    simu_card["area"] = calibrated_geometry["area"]
    simu_card["advTF"] = calibrated_geometry["advTF"]

    adventitia_card = build_adventitia_card(adventitia_template, N)

    print(f"[verify] setup_simulation(N={N}) ...", flush=True)
    ctx = setup_simulation(f"verify_patch_N{N}", "Article_Calibration", simu_card, adventitia_card, media_card)
    print("[verify] solve_simulation() ...", flush=True)
    solve_simulation(ctx, simu_card, adventitia_card, media_card)

    state, types_seen = extract_state(ctx.mech)
    print(f"[verify] inclusion/material types exercised: {sorted(types_seen)}")
    return state


def compare(before, after, atol=1e-10, rtol=1e-8):
    keys_before = set(before.keys())
    keys_after = set(after.keys())
    if keys_before != keys_after:
        print("MISMATCH IN KEYS (this alone is a FAIL - the patch changed which "
              "state is exposed, not just how it's computed):")
        print("  only in before:", sorted(keys_before - keys_after))
        print("  only in after :", sorted(keys_after - keys_before))
        return False

    all_ok = True
    print(f"{'quantity':45s} {'max_abs_diff':>14s} {'max_rel_diff':>14s}  result")
    for key in sorted(keys_before):
        a = before[key]
        b = after[key]
        if a.shape != b.shape:
            print(f"{key:45s} SHAPE MISMATCH {a.shape} vs {b.shape}  FAIL")
            all_ok = False
            continue
        diff = np.abs(a - b)
        max_abs = float(np.max(diff)) if diff.size else 0.0
        denom = np.maximum(np.abs(a), 1e-300)
        max_rel = float(np.max(diff / denom)) if diff.size else 0.0
        ok = np.allclose(a, b, atol=atol, rtol=rtol)
        all_ok &= ok
        print(f"{key:45s} {max_abs:14.3e} {max_rel:14.3e}  {'PASS' if ok else 'FAIL'}")

    print()
    print("ALL PASS - patch is behavior-preserving for this scenario." if all_ok
          else "AT LEAST ONE FAIL - do not trust the patch yet, investigate the "
               "flagged quantity/quantities above before using it.")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=5, help="number of collagen families (small = fast)")
    parser.add_argument("--save", type=str, default=None, help="run and save the state snapshot to this .npz path")
    parser.add_argument("--compare", type=str, default=None, help="run and compare the state against this saved .npz path")
    args = parser.parse_args()

    if not args.save and not args.compare:
        parser.error("pass either --save <path> (before, with original code) "
                      "or --compare <path> (after, with patched code)")

    state = run(args.N)

    if args.save:
        np.savez(args.save, **state)
        print(f"[verify] saved {len(state)} arrays to {args.save}")
    else:
        loaded = dict(np.load(args.compare, allow_pickle=True))
        compare(loaded, state)