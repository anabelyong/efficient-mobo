import os
import re
import csv

LOG_DIRS = ["logs_trial1", "logs_trial2", "logs_trial3"]
OUT_DIR = "parsed_csvs"

# Regex patterns
EHVI_ITER_RE    = re.compile(r"--- Iter\s+(\d+)")
EHVI_REF_RE     = re.compile(r"Inferred reference point at iter \d+: \[([0-9eE\.\,\-\s]+)\]")
EHVI_SELECT_RE  = re.compile(r"Selected idx=\d+\s*\((.+?)\)\s*→\s*\[([0-9eE\.\-\s]+)\]")
EHVI_HV_RE      = re.compile(r"Hypervolume:\s*([0-9\.]+)")

def parse_ehvi_log(path):
    rows = []
    with open(path) as f:
        iter_num = None
        sel_smiles = sel_vals = ref_point = None

        for line in f:
            # Iteration header
            m = EHVI_ITER_RE.search(line)
            if m:
                iter_num = int(m.group(1))
                sel_smiles = sel_vals = ref_point = None
                continue

            # RefPoint line
            if iter_num is not None:
                m = EHVI_REF_RE.search(line)
                if m:
                    ref_point = [float(x.strip()) for x in m.group(1).split(",")]
                    continue

            # Selected SMILES + f-values
            if iter_num is not None:
                m = EHVI_SELECT_RE.search(line)
                if m:
                    sel_smiles = m.group(1)
                    sel_vals = [float(x) for x in m.group(2).split()]
                    continue

            # Hypervolume completes the row
            if iter_num is not None and sel_smiles and sel_vals and ref_point:
                m = EHVI_HV_RE.search(line)
                if m:
                    hv = float(m.group(1))
                    row = {
                        "iteration": iter_num,
                        "smiles": sel_smiles,
                        "hypervolume": hv,
                        "RefPoint": ref_point,
                        **{f"f{i+1}": v for i, v in enumerate(sel_vals)}
                    }
                    rows.append(row)
                    iter_num = None
    return rows

def write_csv(outpath, rows, headers):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w", newline="") as outf:
        writer = csv.DictWriter(outf, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            r_out = r.copy()
            if "RefPoint" in r:
                r_out["RefPoint"] = "[" + ", ".join(f"{x:.6f}" for x in r["RefPoint"]) + "]"
            writer.writerow(r_out)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for log_dir in LOG_DIRS:
        if not os.path.isdir(log_dir):
            continue

        for fn in sorted(os.listdir(log_dir)):
            path = os.path.join(log_dir, fn)

            if fn.endswith("_ehvi.log"):
                rows = parse_ehvi_log(path)
                if not rows:
                    print(f"[WARN] no EHVI rows in {path}")
                    continue

                fcols = sorted([c for c in rows[0] if c.startswith("f")], key=lambda x: int(x[1:]))
                headers = ["BO Iteration", "Selected SMILES", "Hypervolume", "RefPoint"] + fcols

                mapped = []
                for r in rows:
                    m = {
                        "BO Iteration":    r["iteration"],
                        "Selected SMILES": r["smiles"],
                        "Hypervolume":     r["hypervolume"],
                        "RefPoint":        r["RefPoint"],
                    }
                    for f in fcols:
                        m[f] = r[f]
                    mapped.append(m)

                base = fn.replace(".log", "")
                outname = f"{log_dir}_{base}.csv"
                outpath = os.path.join(OUT_DIR, outname)
                write_csv(outpath, mapped, headers)
                print(f"Wrote EHVI → {outpath}")

if __name__ == "__main__":
    main()
