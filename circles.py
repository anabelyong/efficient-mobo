import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from tqdm import tqdm
from typing import List, Any

from acquisition_funcs.measures import NCircles
from kernel_only_GP.tanimoto_gp import get_fingerprint

def vectorizer(smiles_list: List[str]) -> List[Any]:
    return [get_fingerprint(smi) for smi in tqdm(smiles_list, desc="Vectorizing")]

def sim_matrix(fps_a: List[Any], fps_b: List[Any]) -> List[List[float]]:
    return [DataStructs.BulkTanimotoSimilarity(fp, fps_b) for fp in fps_a]

def compute_circles(smiles_list: List[str], threshold: float) -> int:
    ncircle = NCircles(vectorizer=vectorizer, sim_mat_func=sim_matrix, threshold=threshold)
    n_circles, _ = ncircle.measure(smiles_list)
    return n_circles

if __name__ == "__main__":
    # Load CSV and SMILES column
    df = pd.read_csv("evaluated_ei/logs_trial2_ei_evaluated_perin.csv")
    smiles = df["Selected SMILES by EI"].dropna().tolist()
    print(f"Loaded {len(smiles)} SMILES")

    # Loop through thresholds
    thresholds = np.arange(0.1, 0.91, 0.05)
    results = []

    for t in thresholds:
        print(f"Computing #Circles for threshold t={t:.2f}...")
        try:
            n = compute_circles(smiles, threshold=t)
            print(f"  → #Circles = {n}")
            results.append({"Threshold": t, "NumCircles": n})
        except Exception as e:
            print(f"  → Failed at t={t:.2f}: {e}")
            results.append({"Threshold": t, "NumCircles": np.nan})

    # Save results
    out_df = pd.DataFrame(results)
    out_df.to_csv("circles/circles_ei_trial2_perin.csv", index=False)
    print("Saved to circles_ei_trial2_perin.csv")
