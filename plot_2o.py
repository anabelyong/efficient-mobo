#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TRIAL_DIRS = ["logs_trial1", "logs_trial2", "logs_trial3"]
TRIAL_TITLES = ["Trial 1" , "Trial 2" , "Trial 3"]

EHVI_CSV_TPL = "parsed_csvs/{trial}_terminal_output_jax_amlo_ehvi.csv"
EI_CSV_TPL   = "evaluated_ei/{trial}_terminal_output_jax_ei_amlo_evaluated.csv"
EHVI_LOG_TPL = "{trial}/terminal_output_jax_amlo_ehvi.log"

OUT_DIR  = "plots"
OUT_FILE = "amlo_bo_comparison.pdf"
# ──────────────────────────────────────────────────────────────────────────────

PARETO_RE = re.compile(r"Final Pareto front points:\s*(\[\[.*?\]\])", re.S)

def load_two_obj_data(trial):
    df_ehvi = pd.read_csv(EHVI_CSV_TPL.format(trial=trial))
    df_ei   = pd.read_csv(EI_CSV_TPL.format(trial=trial))

    # pick out f1,f2
    fcols = sorted([c for c in df_ehvi.columns if c.startswith("f")])
    if len(fcols) != 2:
        raise RuntimeError(f"{trial}: expected exactly 2 objectives, got {fcols}")
    pts_ehvi = df_ehvi[fcols].values
    pts_ei   = df_ei[fcols].values

    # parse final Pareto
    txt = open(EHVI_LOG_TPL.format(trial=trial)).read()
    m = PARETO_RE.search(txt)
    if not m:
        raise RuntimeError(f"{trial}: Pareto block not found")
    pareto = np.array(eval(m.group(1)))  # (M,2)

    return fcols, pts_ehvi, pts_ei, pareto

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    for ax, trial, title in zip(axes, TRIAL_DIRS, TRIAL_TITLES):
        try:
            fcols, pts_ehvi, pts_ei, pareto = load_two_obj_data(trial)
        except Exception as e:
            ax.text(0.5, 0.5, str(e), ha="center", va="center")
            ax.set_title(title)
            continue

        ax.scatter(pts_ehvi[:,0], pts_ehvi[:,1],
                c="C0", marker="o", label="EHVI", alpha=0.7)
        ax.scatter(pts_ei[:,0], pts_ei[:,1],
                c="C1", marker="^", label="EI", alpha=0.9)
        ax.scatter(pareto[:,0], pareto[:,1],
                c="red", marker="s", s=80, label="Pareto")

        ax.set_title(title)
        ax.set_xlabel(fcols[0])
        ax.set_ylabel(fcols[1])

    # single legend for all
    axes[0].legend(loc="upper left")
    plt.suptitle("Amlodipine MPO: EHVI vs EI across Trials", y=1.02)
    plt.tight_layout()

    # show interactively
    plt.show()

    outpath = os.path.join(OUT_DIR, OUT_FILE)
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    print(f"Saved combined BO comparison → {outpath}")

if __name__=="__main__":
    main()
