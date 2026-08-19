#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Before/after equality check for the mech_problem_class.py MATERIALIZE_KINEMATICS
patch (Fn/J/Finv/B/tau materialized into numeric fem.Function snapshots,
refreshed via update_kinematics(), instead of being left as live symbolic
expressions of self.un/self.Sn - see the MATERIALIZE_KINEMATICS docstring at
the top of mech_problem_class.py for the full rationale).

Unlike verify_patch_equality.py (which only checks per-inclusion state), this
script also checks the actual SIMULATION OUTPUTS (pressure-radius curve,
stress curves, F_zz...) and the Newton-Raphson iteration count per load step,
since those are what a convergence/correctness regression would actually show
up in, and what calibration/plots ultimately depend on.

How to use
----------
1. With MATERIALIZE_KINEMATICS = False (the ORIGINAL, purely symbolic,
   behavior) set at the top of mech_problem_class.py:

       python3 verify_mech_patch_equality.py --save before.npz

2. Set MATERIALIZE_KINEMATICS = True (the patched behavior), then:

       python3 verify_mech_patch_equality.py --compare before.npz

Run from inside Article_Calibration/ (same working directory used for
3_2_fiberdiscretization_sa_persistent.py), so json_cards/ and outputs/
resolve the same way.

Also prints/records this process's peak resident memory (ru_maxrss) so you
can directly compare RAM usage of the two runs (each run is necessarily a
separate process, which is required for a fair peak-RSS comparison anyway).
This is informational only - it is NOT part of the pass/fail verdict.
"""

import argparse
import json
import os
import resource

import numpy as np


def load_JSON(card_name):
    with open(card_name, "r") as f:
        return json.load(f)


def build_adventitia_card(template, N):
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


def run(N):
    import Article_Calibration.main_ArterialTissue_persistent as main_mod
    from Multiscale_Framework.class_modules.mech_problem_class import (
        MATERIALIZE_KINEMATICS,
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

    print(f"[verify] MATERIALIZE_KINEMATICS = {MATERIALIZE_KINEMATICS}", flush=True)
    print(f"[verify] setup_simulation(N={N}) ...", flush=True)
    ctx = main_mod.setup_simulation(f"verify_mech_N{N}", "Article_Calibration", simu_card, adventitia_card, media_card)

    # Record Newton iteration counts per load step without touching
    # solve_simulation() itself: wrap mech.solve_1_step.
    num_its_log = []
    mech = ctx.mech
    orig_solve_1_step = mech.solve_1_step

    def wrapped(delta_t):
        num_its, conv = orig_solve_1_step(delta_t)
        num_its_log.append(int(num_its))
        return num_its, conv

    mech.solve_1_step = wrapped

    print("[verify] solve_simulation() ...", flush=True)
    # solve_simulation() returns (result, mech) on a completed run, or
    # (False, mech) if a step crashed / failed to converge (see
    # main_ArterialTissue_persistent.py lines ~546-558).
    result, mech = main_mod.solve_simulation(ctx, simu_card, adventitia_card, media_card)

    return mech, num_its_log, result


def extract_state(mech, num_its_log, result):
    state = {}
    state["_converged"] = np.array([result is not False])
    state["_num_its_per_step"] = np.array(num_its_log, dtype=np.int64)

    # Global primary fields, final state
    state["mech.un"] = np.array(mech.un.x.array, copy=True)
    state["mech.Sn"] = np.array(mech.Sn.x.array, copy=True)

    # Every scalar/local output curve produced by solve_simulation, if the
    # run completed (result is a Results instance with an .outputs dict of
    # {key: np.ndarray}); if a step crashed/diverged, result is False and
    # there is nothing further to compare beyond convergence + iteration
    # counts + whatever state un/Sn were left at.
    if result is not False:
        for key, val in result.outputs.items():
            arr = np.asarray(val)
            if arr.dtype == object:
                continue
            state[f"result.{key}"] = arr

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"[verify] peak RSS (this process): {peak_rss_kb/1e6:.3f} GB "
          f"({peak_rss_kb} kB)", flush=True)
    state["_peak_rss_kb"] = np.array([peak_rss_kb], dtype=np.int64)

    return state


def compare(before, after, atol=1e-8, rtol=1e-6):
    keys_before = set(before.keys())
    keys_after = set(after.keys())
    if keys_before != keys_after:
        print("MISMATCH IN KEYS (this alone is a FAIL - the patch changed what "
              "the run produced, e.g. one converged and the other didn't, or "
              "the output-curve set differs):")
        print("  only in before:", sorted(keys_before - keys_after))
        print("  only in after :", sorted(keys_after - keys_before))
        return False

    all_ok = True
    print(f"{'quantity':40s} {'max_abs_diff':>14s} {'max_rel_diff':>14s}  result")
    for key in sorted(keys_before):
        if key == "_peak_rss_kb":
            before_gb = before[key][0] / 1e6
            after_gb = after[key][0] / 1e6
            print(f"{key:40s} before={before_gb:.3f} GB  after={after_gb:.3f} GB  "
                  f"(informational, not pass/fail)")
            continue
        a = np.asarray(before[key])
        b = np.asarray(after[key])
        if a.shape != b.shape:
            print(f"{key:40s} SHAPE MISMATCH {a.shape} vs {b.shape}  FAIL")
            all_ok = False
            continue
        if a.dtype == bool or np.issubdtype(a.dtype, np.integer):
            ok = np.array_equal(a, b)
            extra = "" if ok else f"  ({a.tolist()} vs {b.tolist()})"
            print(f"{key:40s} {'-':>14s} {'-':>14s}  {'PASS' if ok else 'FAIL'}{extra}")
            all_ok &= ok
            continue
        diff = np.abs(a - b)
        max_abs = float(np.max(diff)) if diff.size else 0.0
        denom = np.maximum(np.abs(a), 1e-300)
        max_rel = float(np.max(diff / denom)) if diff.size else 0.0
        ok = np.allclose(a, b, atol=atol, rtol=rtol)
        all_ok &= ok
        print(f"{key:40s} {max_abs:14.3e} {max_rel:14.3e}  {'PASS' if ok else 'FAIL'}")

    print()
    print("ALL PASS - MATERIALIZE_KINEMATICS patch is behavior-preserving for this scenario."
          if all_ok else
          "AT LEAST ONE FAIL - do not trust the patch yet, investigate the flagged "
          "quantity/quantities above before using it.")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=8, help="number of collagen families")
    parser.add_argument("--save", type=str, default=None,
                         help="run and save the state snapshot to this .npz path "
                              "(use with MATERIALIZE_KINEMATICS=False)")
    parser.add_argument("--compare", type=str, default=None,
                         help="run and compare the state against this saved .npz path "
                              "(use with MATERIALIZE_KINEMATICS=True)")
    args = parser.parse_args()

    if not args.save and not args.compare:
        parser.error("pass either --save <path> (before, MATERIALIZE_KINEMATICS=False) "
                      "or --compare <path> (after, MATERIALIZE_KINEMATICS=True)")

    mech, num_its_log, result = run(args.N)
    state = extract_state(mech, num_its_log, result)

    if args.save:
        np.savez(args.save, **state)
        print(f"[verify] saved {len(state)} arrays to {args.save}")
    else:
        loaded = dict(np.load(args.compare, allow_pickle=True))
        compare(loaded, state)