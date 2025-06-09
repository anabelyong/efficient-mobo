#!/usr/bin/env python3
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ehvi_csv = "parsed_csvs/logs_trial1_terminal_output_jax_fex_ehvi.csv"
ei_csv   = "evaluated_ei/logs_trial1_terminal_output_jax_ei_fex_evaluated.csv"
ehvi_log = "logs_trial1/terminal_output_jax_fex_ehvi.log"
out_pdf  = "plots/fex_trial1_plots.pdf"

def load_data():
    # EHVI
    df_ehvi = pd.read_csv(ehvi_csv)
    # EI
    df_ei   = pd.read_csv(ei_csv)

    # pick out objective columns from EHVI
    fcols = sorted(c for c in df_ehvi.columns if c.startswith("f"))
    pts_ehvi = df_ehvi[fcols].values
    missing = set(fcols) - set(df_ei.columns)
    if missing:
        raise RuntimeError(f"EI CSV missing columns: {missing}")
    pts_ei = df_ei[fcols].values

    # pareto
    with open(ehvi_log) as f:
        txt = f.read()
    m = re.search(r"Final Pareto front points:\s*(\[\[.*?\]\])", txt, re.S)
    if not m:
        raise RuntimeError(f"Could not find Pareto block in {ehvi_log}")
    pareto = np.array(eval(m.group(1)))  # M×3 array

    return fcols, pts_ehvi, pts_ei, pareto

def plot_2d(fcols, pts_ehvi, pts_ei, pareto):
    pairs = [(0,1),(1,2),(0,2)]
    titles = [f"{fcols[i]} vs {fcols[j]}" for i,j in pairs]
    fig, axes = plt.subplots(1,3,figsize=(15,5))

    for ax,(i,j),title in zip(axes,pairs,titles):
        ax.scatter(pts_ehvi[:,i], pts_ehvi[:,j], c="C0", marker="o", label="EHVI")
        ax.scatter(pts_ei[:,i],   pts_ei[:,j],   c="C1", marker="^", label="EI")
        ax.scatter(pareto[:,i],   pareto[:,j],   c="C3", marker="s", label="Pareto")
        ax.set_xlabel(fcols[i]); ax.set_ylabel(fcols[j])
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(out_pdf, format="pdf")

def plot_3d(fcols, pts_ehvi, pts_ei, pareto):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111,projection="3d")
    ax.scatter(pts_ehvi[:,0], pts_ehvi[:,1], pts_ehvi[:,2],
               c="C0", marker="o", s=40, label="EHVI")
    ax.scatter(pts_ei[:,0],   pts_ei[:,1],   pts_ei[:,2],
               c="C1", marker="^", s=60, label="EI")
    ax.scatter(pareto[:,0],   pareto[:,1],   pareto[:,2],
               c="C3", marker="s", s=80, label="Pareto")
    ax.set_xlabel(fcols[0]); ax.set_ylabel(fcols[1]); ax.set_zlabel(fcols[2])
    ax.set_title("3D Scatter of Selected & Pareto")
    ax.legend()

    print("3D axis limits:")
    print(" x:", ax.get_xlim())
    print(" y:", ax.get_ylim())
    print(" z:", ax.get_zlim())

    plt.show()
    fig.savefig(out_pdf.replace(".pdf","_3D.pdf"), format="pdf")

if __name__=="__main__":
    fcols, pts_ehvi, pts_ei, pareto = load_data()
    plot_2d(fcols, pts_ehvi, pts_ei, pareto)
    plot_3d(fcols, pts_ehvi, pts_ei, pareto)
