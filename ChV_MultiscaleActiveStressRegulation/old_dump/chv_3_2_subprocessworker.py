#!/usr/bin/env python3
import os
import sys
import pickle
import json
    
# --------------------------------------------------
# 1. SET CACHE FIRST
# --------------------------------------------------
pid = os.getpid()
cache_root = f"/tmp/fenics_cache_{pid}"

os.environ["XDG_CACHE_HOME"] = cache_root
os.environ["FFCX_CACHE_DIR"] = f"{cache_root}/ffcx"
os.environ["DIJITSO_CACHE_DIR"] = f"{cache_root}/dijitso"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --------------------------------------------------
# 2. READ INPUTS
# --------------------------------------------------
tau_b = None if sys.argv[1] == "None" else float(sys.argv[1])
name = sys.argv[2]
folder_name = sys.argv[3]

# Card files
simu_card_file = sys.argv[4]
media_card_file = sys.argv[5]
adventitia_card_file = sys.argv[6]

# --------------------------------------------------
# 3. LOAD CARDS
# --------------------------------------------------
with open(simu_card_file, "r") as f:
    simu_card = json.load(f)

with open(media_card_file, "r") as f:
    media_card = json.load(f)

with open(adventitia_card_file, "r") as f:
    adventitia_card = json.load(f)

# --------------------------------------------------
# 4. IMPORT FENICS CODE
# --------------------------------------------------
from ChV_MultiscaleActiveStressRegulation.main_Vasoconstriction_25_11_24 import run_simulation
from petsc4py import PETSc

opts = PETSc.Options()
opts["mat_mkl_cpardiso_omp_num_threads"] = 1
opts["mat_mumps_icntl_14"] = 40
opts["num_threads"] = 1

# --------------------------------------------------
# 5. DETERMINE NAME AND MODIFY CARDS
# --------------------------------------------------
if tau_b is None:
    namefile = f"{name}_passive_scal.pkl"
    name_simu = f"{name}_passive"
    media_card["cells"]["basal stress"] = 0.1
else:
    namefile = f"{name}_{tau_b:.6f}_scal.pkl"
    name_simu = f"{name}_{tau_b:.6f}"
    media_card["cells"]["basal stress"] = tau_b

os.makedirs(f"./outputs/{folder_name}", exist_ok=True)
filepath = f"./outputs/{folder_name}/{namefile}"

# --------------------------------------------------
# 6. LOAD OR RUN
# --------------------------------------------------
try:
    with open(filepath, "rb") as f:
        result = pickle.load(f)
    print(f"[Worker {os.getpid()}] Loaded {namefile} from disk.")

except FileNotFoundError:
    print(f"[Worker {os.getpid()}] File not found: {namefile}. Running simulation...")
    result = run_simulation(name_simu, folder_name, simu_card, adventitia_card, media_card)
    print(f"[Worker {os.getpid()}] Finished {namefile}.")