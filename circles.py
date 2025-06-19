import os
import re
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from tqdm import tqdm
from typing import List, Any

from acquisition_funcs.measures import NCircles
from acquisition_funcs.pareto import pareto_front
from kernel_only_GP.tanimoto_gp import get_fingerprint

def vectorizer(smiles_list: List[str]) -> List[Any]:
    return [get_fingerprint(smi) for smi in tqdm(smiles_list, desc="Vectorizing")]

def sim_matrix(fps_a: List[Any], fps_b: List[Any]) -> List[List[float]]:
    return [DataStructs.BulkTanimotoSimilarity(fp, fps_b) for fp in fps_a]

def compute_circles(smiles_list: List[str], threshold: float) -> int:
    ncircle = NCircles(vectorizer=vectorizer, sim_mat_func=sim_matrix, threshold=threshold)
    n_circles, _ = ncircle.measure(smiles_list)
    return n_circles

def extract_initial_smiles(log_text: str) -> list:
    match = re.findall(r"Initial SMILES:\s*\[((?:\s*'[^']+',?\s*)+)\]", log_text, re.DOTALL)
    if not match:
        raise ValueError("No initial SMILES found.")
    return re.findall(r"'([^']+)'", match[0])

def extract_initial_objectives(log_text: str) -> np.ndarray:
    """
    Parses the multi-line initial objective array from the log text.
    Handles formats like:
        Initial objectives:
        [[... ... ...]
         [... ... ...]
         ...
        ]
    """
    lines = log_text.splitlines()
    start_idx = None

    for i, line in enumerate(lines):
        if "Initial Y:" in line:
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError("Initial objectives block not found in log.")

    objective_lines = []
    for line in lines[start_idx:]:
        if line.strip().startswith("["):
            objective_lines.append(line.strip().lstrip("[").rstrip("]"))
        else:
            break  # end of matrix block

    if not objective_lines:
        raise ValueError("Failed to parse objectives block.")

    # Convert lines into floats
    parsed = []
    for line in objective_lines:
        row = list(map(float, line.split()))
        parsed.append(row)

    return np.array(parsed)


if __name__ == "__main__":
    method = "ehvi"        # or "ei", "rs"
    target = "fex"         # or "perin", etc.
    column_name = "Selected SMILES"  # common column name after parsing

    for trial in [1, 2, 3]:
        print(f"\n== Trial {trial} ==")

        csv_path = f"evaluated_{method}/r2/logs_trial{trial}_{method}_evaluated_{target}_r2.csv"
        log_path = f"logs_trial{trial}/terminal_output_jax_{target}_{method}_.log"

        if not os.path.exists(csv_path) or not os.path.exists(log_path):
            print(f"[WARN] Missing data for trial {trial}: {csv_path} or {log_path}")
            continue

        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} BO entries from {csv_path}")

        with open(log_path, "r") as f:
            log_text = f.read()

        try:
            initial_smiles = extract_initial_smiles(log_text)
            initial_Y = extract_initial_objectives(log_text)
        except Exception as e:
            print(f"[ERROR] Failed to extract from {log_path}: {e}")
            continue

        print(f"Extracted {len(initial_smiles)} initial points")

        # Combine BO data and initial data
        all_smiles = df[column_name].dropna().tolist() + initial_smiles
        all_objectives = np.vstack([df[["f1", "f2", "f3"]].values, initial_Y])

        # Pareto filtering
        mask = pareto_front(all_objectives, maximize=True)
        pareto_smiles = [all_smiles[i] for i in range(len(all_smiles)) if mask[i]]
        print(f"→ {len(pareto_smiles)} Pareto-optimal SMILES")

        # Compute #Circles
        thresholds = np.arange(0.1, 0.91, 0.05)
        results = []

        for t in thresholds:
            print(f"Computing #Circles for threshold t={t:.2f}...")
            try:
                n = compute_circles(pareto_smiles, threshold=t)
                print(f"  → #Circles = {n}")
                results.append({"Threshold": t, "NumCircles": n})
            except Exception as e:
                print(f"  → Failed at t={t:.2f}: {e}")
                results.append({"Threshold": t, "NumCircles": np.nan})

        out_path = f"circles/circles_{method}_trial{trial}_{target}.csv"
        pd.DataFrame(results).to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
