#!/usr/bin/env python3
import os
import random
import pandas as pd
import numpy as np

from utils.utils_final import evaluate_perin_MPO  # <-- ensure this is accessible

# === Config ===
DATA_PATH = "guacamol_dataset/guacamol_v1_train.smiles"
OUT_CSV = "random_sampling/random_sampling_perin_results_3.csv"
N_INIT = 10         # how many to start with (for BO warm-up, but will not be recorded)
N_SELECT = 200      # how many random selections to make (recorded in CSV)

# === Load Data ===
df = pd.read_csv(DATA_PATH, header=None, names=["smiles"])
all_smiles = df["smiles"].tolist()
random.shuffle(all_smiles)

init_smiles = all_smiles[:N_INIT]
pool_smiles = all_smiles[N_INIT:]

# === Evaluate initial SMILES (not stored in CSV) ===
_ = evaluate_perin_MPO(init_smiles)  # evaluated but not saved

# === Random sampling loop ===
selected_smiles = []
selected_objectives = []

for _ in range(N_SELECT):
    cand = random.choice(pool_smiles)
    pool_smiles.remove(cand)
    obj = evaluate_perin_MPO([cand])[0]
    selected_smiles.append(cand)
    selected_objectives.append(obj.tolist())

# === Save as CSV ===
fcols = [f"f{i+1}" for i in range(len(selected_objectives[0]))]
df_out = pd.DataFrame(selected_objectives, columns=fcols)
df_out.insert(0, "SMILES", selected_smiles)
df_out.insert(0, "BO Iteration", list(range(0, N_SELECT)))  

df_out.to_csv(OUT_CSV, index=False)
print(f"Saved random sampling results to: {OUT_CSV}")
