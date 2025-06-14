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

if __name__ == "__main__":
    # === Load evaluated CSV ===
    df = pd.read_csv("evaluated_ehvi/logs_trial1_ehvi_evaluated_fex_r2.csv")
    print(f"Loaded {len(df)} total entries")

    # === Extract objectives and pareto filter ===
    objectives = df[["f1", "f2", "f3"]].values  # or ["f1", "f2", "f3"] if needed
    mask = pareto_front(objectives, maximize=True)
    pareto_df = df[mask]

    pareto_smiles = pareto_df["Selected SMILES"].dropna().tolist()
    print(f"→ Retained {len(pareto_smiles)} Pareto-optimal SMILES")

    # === Loop through thresholds ===
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

    # === Save results ===
    out_df = pd.DataFrame(results)
    out_path = "circles/circles_ehvi_trial1_fex.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")