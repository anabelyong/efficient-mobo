import os
import re
import numpy as np
import pandas as pd
import logging
from acquisition_funcs.hypervolume import Hypervolume, infer_reference_point
from acquisition_funcs.pareto import pareto_front
from utils.utils_final import evaluate_fex_objectives

logger = logging.getLogger("BO")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # File
    fh = logging.FileHandler("bo_eval_fex_3d.log", mode="w")
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger.addHandler(ch)
    logger.addHandler(fh)

def extract_initial_smiles(log_text: str) -> list:
    match = re.findall(r"Initial SMILES:\s*\[((?:\s*'[^']+',?\s*)+)\]", log_text, re.DOTALL)
    if not match:
        raise ValueError("No initial SMILES found.")
    return re.findall(r"'([^']+)'", match[0])

def extract_bo_smiles_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[["BO Iteration", "Selected SMILES"]]

def main():
    log_path = "logs_trial1/terminal_output_jax_ei_amlo.log"
    csv_path = "rs/random_sampling_fex_results_1.csv"
    output_csv_path = "evaluated_rs/logs_trial1_rs_evaluated_fex.csv"

    with open(log_path, "r") as f:
        log_text = f.read()

    # Extract and evaluate initial SMILES
    initial_smiles = extract_initial_smiles(log_text)
    logger.info(f"Initial SMILES ({len(initial_smiles)}): {initial_smiles}")
    initial_Y = evaluate_fex_objectives(initial_smiles)
    logger.info(f"f1 shape: {initial_Y[:, 0].shape}")
    logger.info(f"f2 shape: {initial_Y[:, 1].shape}")
    logger.info(f"f3 shape: {initial_Y[:, 2].shape}")
    logger.info(f"Initial objectives:\n{initial_Y}")

    # Initial reference point
    ref_point = infer_reference_point(initial_Y)
    logger.info(f"Inferred reference point at iter 0: {ref_point}")

    # Start BO loop
    archive = initial_Y.tolist()
    bo_df = extract_bo_smiles_from_csv(csv_path)
    results = []

    for i, row in bo_df.iterrows():
        idx = int(row["BO Iteration"])
        smile = row["Selected SMILES"]

        logger.info(f"\n--- Iter {idx} (train size {np.array(archive).shape}) ---")

        # Evaluate new SMILES
        f_vec = evaluate_fex_objectives([smile])[0]
        logger.info(f"f1 shape: {np.array([f_vec[0]]).shape}")
        logger.info(f"f2 shape: {np.array([f_vec[1]]).shape}")
        logger.info(f"f3 shape: {np.array([f_vec[2]]).shape}")
        logger.info(f"Selected SMILES: {smile} → {f_vec}")

        # Append and update
        archive.append(f_vec.tolist())
        Y_array = np.array(archive)

        # Infer reference point again (for analysis/debug, optional)
        ref_point_iter = infer_reference_point(Y_array)
        logger.info(f"Inferred reference point at iter {idx+1}: {ref_point_iter}")

        mask = pareto_front(Y_array, maximize=True)
        hv = Hypervolume(ref_point_iter).compute(Y_array[mask])
        logger.info(f"Hypervolume: {hv:.4f}")

        results.append({
            "BO Iteration": idx,
            "Selected SMILES by EI": smile,
            "f1": f_vec[0],
            "f2": f_vec[1],
            "f3": f_vec[2],
            "Hypervolume": hv
        })

    # Save final CSV
    pd.DataFrame(results).to_csv(output_csv_path, index=False)
    logger.info(f"\nFinal results saved to: {output_csv_path}")

if __name__ == "__main__":
    main()
