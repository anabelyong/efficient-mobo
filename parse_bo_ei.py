#!/usr/bin/env python3
import os
import re
import csv

# directories to scan (relative to where this script lives)
LOG_DIRS = ["logs_trial1", "logs_trial2", "logs_trial3"]
# where to put the CSVs
OUT_DIR = "parsed_csvs"

# Regexes for EI logs
EI_ITER_RE    = re.compile(r"--- Iter\s+(\d+)")
EI_SELECT_RE  = re.compile(r"Selected\s+(.+?)\s+→\s*EI\s*=\s*([0-9\.eE\-\+]+)")
EI_TRUE_F_RE  = re.compile(r"True f\s*=\s*([0-9\.eE\-\+]+)")


def parse_ei_log(path):
    rows = []
    with open(path) as f:
        iter_num = None
        sel_smiles = None
        ei_val = None

        for line in f:
            # iteration header
            m = EI_ITER_RE.search(line)
            if m:
                iter_num = int(m.group(1))
                sel_smiles = ei_val = None
                continue

            # selection line with SMILES + EI value
            if iter_num is not None:
                m = EI_SELECT_RE.search(line)
                if m:
                    sel_smiles = m.group(1).strip()
                    ei_val = float(m.group(2))
                    continue

            # true f line completes the record
            if iter_num is not None and sel_smiles is not None and ei_val is not None:
                m = EI_TRUE_F_RE.search(line)
                if m:
                    true_f = float(m.group(1))
                    rows.append({
                        "iteration": iter_num,
                        "smiles": sel_smiles,
                        "EI": ei_val,
                        "f": true_f
                    })
                    iter_num = None
    return rows

def write_csv(outpath, rows, headers):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", newline="") as outf:
        writer = csv.DictWriter(outf, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for log_dir in LOG_DIRS:
        if not os.path.isdir(log_dir):
            continue

        for fn in sorted(os.listdir(log_dir)):
            path = os.path.join(log_dir, fn)

            # EI logs
            if "_ei_" in fn and fn.endswith(".log"):
                rows = parse_ei_log(path)
                if not rows:
                    print(f"[WARN] no EI rows in {path}")
                    continue

                headers = ["BO Iteration", "Selected SMILES", "EI", "Objective Value"]

                # remap for EI
                mapped = []
                for r in rows:
                    mapped.append({
                        "BO Iteration":    r["iteration"],
                        "Selected SMILES": r["smiles"],
                        "EI":              r["EI"],
                        "Objective Value": r["f"],
                    })

                base   = fn.replace(".log", "")
                outname= f"{log_dir}_{base}.csv"
                outpath= os.path.join(OUT_DIR, outname)
                write_csv(outpath, mapped, headers)
                print(f"Wrote EI  → {outpath}")

if __name__ == "__main__":
    main()