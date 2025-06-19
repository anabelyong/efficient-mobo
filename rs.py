#!/usr/bin/env python3
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.utils_final import evaluate_fex_objectives 
from acquisition_funcs.pareto import pareto_front
from acquisition_funcs.hypervolume import Hypervolume, infer_reference_point

# === Config ===
DATA_PATH = "guacamol_dataset/guacamol_v1_train.smiles"
OUT_CSV = "evaluated_rs/random_sampling_fex_trial3.csv"
N_INIT = 10
N_SELECT = 200

# === Load & Shuffle Dataset ===
df = pd.read_csv(DATA_PATH, header=None, names=["smiles"])
all_smiles = df["smiles"].tolist()
random.shuffle(all_smiles)

init_smiles = all_smiles[:N_INIT]
selected_smiles = all_smiles[N_INIT:N_INIT + N_SELECT]

# === Evaluate initial SMILES (not recorded)
init_obj = evaluate_fex_objectives(init_smiles)
archive = init_obj.tolist()
ref_point = infer_reference_point(np.array(archive))

# === Evaluate selected SMILES
selected_obj = evaluate_fex_objectives(selected_smiles)

# === Iterative Hypervolume Tracking
records = []
for i, (s, o) in tqdm(enumerate(zip(selected_smiles, selected_obj)), total=N_SELECT, desc="Random Sampling"):
    archive.append(o.tolist())
    Y = np.array(archive)
    front = Y[pareto_front(Y, maximize=True)]
    hv = Hypervolume(ref_point).compute(front)

    row = {
        "BO Iteration": i,
        "SMILES": s,
        "Hypervolume": hv
    }
    for j, val in enumerate(o):
        row[f"f{j+1}"] = val
    records.append(row)

# === Save CSV
df_out = pd.DataFrame(records)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df_out.to_csv(OUT_CSV, index=False)
print(f"Saved fair random sampling results to: {OUT_CSV}")
