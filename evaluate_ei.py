#!/usr/bin/env python3
import os
import pandas as pd
from utils.utils_final import (
    evaluate_amlo_objectives,
    evaluate_perin_objectives,
    evaluate_fex_objectives,
)

# input directory of parsed EI CSVs
IN_DIR  = "parsed_csvs"
# output directory
OUT_DIR = "evaluated_ei"

def choose_evaluator(fn):
    """Pick the right evaluator based on file name."""
    if "amlo" in fn:
        return evaluate_amlo_objectives
    if "perin" in fn:
        return evaluate_perin_objectives
    if "fex" in fn:
        return evaluate_fex_objectives
    raise ValueError(f"Could not infer evaluator from {fn!r}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for fn in sorted(os.listdir(IN_DIR)):
        if "_ei_" not in fn or not fn.endswith(".csv"):
            continue

        path_in = os.path.join(IN_DIR, fn)
        df = pd.read_csv(path_in)

        # some sanity checks
        if "Selected SMILES" not in df.columns or "BO Iteration" not in df.columns:
            print(f"[WARN] {fn} missing required columns, skipping")
            continue

        smiles = df["Selected SMILES"].tolist()
        evaluator = choose_evaluator(fn)

        # call into utils_final to get objective arrays
        # evaluate_*_objectives expects a list[str] and returns np.ndarray of shape (N, D)
        Y = evaluator(smiles)  
        # assemble result DataFrame
        out = df[["BO Iteration", "Selected SMILES"]].copy()
        # objective columns f1, f2, (f3)
        for i in range(Y.shape[1]):
            out[f"f{i+1}"] = Y[:, i]

        # write out
        base = fn.replace(".csv","")
        out_path = os.path.join(OUT_DIR, f"{base}_evaluated.csv")
        out.to_csv(out_path, index=False)
        print(f"Wrote evaluated EI → {out_path}")

if __name__ == "__main__":
    main()
