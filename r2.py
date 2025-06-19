import os
import re
import numpy as np
import pandas as pd
import logging
from acquisition_funcs.r2 import r2_indicator_set, uniform_reference_points
from acquisition_funcs.pareto import pareto_front
from utils.utils_final import evaluate_amlo_objectives

# --- Logger setup ---
logger = logging.getLogger("BO")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    fh = logging.FileHandler("bo_eval.log", mode="w")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(ch)
    logger.addHandler(fh)

# --- Helper functions ---
def extract_initial_smiles(log_text: str) -> list:
    match = re.findall(r"Initial SMILES:\s*\[((?:\s*'[^']+',?\s*)+)\]", log_text, re.DOTALL)
    if not match:
        raise ValueError("No initial SMILES found.")
    return re.findall(r"'([^']+)'", match[0])

def extract_bo_smiles_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[["BO Iteration", "Selected SMILES by EI"]]

# --- Main execution ---
def main():
    log_path = "logs_trial3/terminal_output_jax_amlo_ehvi.log"
    csv_path = "evaluated_ei/logs_trial1_ei_evaluated_amlo.csv"
    output_csv_path = "evaluated_ei/r2/logs_trial1_ei_evaluated_amlo_r2.csv"

    with open(log_path, "r") as f:
        log_text = f.read()

    # Initial SMILES and evaluation
    initial_smiles = extract_initial_smiles(log_text)
    logger.info(f"Initial SMILES ({len(initial_smiles)}): {initial_smiles}")
    initial_Y = evaluate_amlo_objectives(initial_smiles)
    logger.info(f"Initial objectives:\n{initial_Y}")

    archive = initial_Y.tolist()
    bo_df = extract_bo_smiles_from_csv(csv_path)
    results = []

    # Setup R2 evaluation
    nobj = initial_Y.shape[1]
    ref_points = uniform_reference_points(nobj, p=10)

    for i, row in bo_df.iterrows():
        idx = int(row["BO Iteration"])
        smile = row["Selected SMILES by EI"]
        logger.info(f"\n--- Iter {idx} ---")

        f_vec = evaluate_amlo_objectives([smile])[0]
        logger.info(f"SMILES: {smile} → {f_vec}")

        archive.append(f_vec.tolist())
        Y_array = np.array(archive)
        pareto_Y = Y_array[pareto_front(Y_array, maximize=True)]
        utopian_point = np.max(Y_array, axis=0) + 0.05

        r2_val = r2_indicator_set(ref_points, pareto_Y, utopian_point)
        logger.info(f"R2 Indicator: {r2_val:.4f}")

        results.append({
            "BO Iteration": idx,
            "Selected SMILES by EI": smile,
            "f1": f_vec[0],
            "f2": f_vec[1],
            "R2 Indicator": round(r2_val, 4)
        })

    pd.DataFrame(results).to_csv(output_csv_path, index=False)
    logger.info(f"Saved to: {output_csv_path}")

if __name__ == "__main__":
    main()
