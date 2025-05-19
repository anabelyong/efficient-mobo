#!/usr/bin/env python3
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.backends.backend_pdf import PdfPages

# paths
ehvi_csv = "parsed_csvs/logs_trial1_terminal_output_jax_fex_ehvi.csv"
ei_csv   = "evaluated_ei/logs_trial1_terminal_output_jax_ei_fex_evaluated.csv"
ehvi_log = "logs_trial1/terminal_output_jax_fex_ehvi.log"
out_pdf   = "fex_trial1_plots.pdf"

# 1) load EHVI & EI selections
df_ehvi = pd.read_csv(ehvi_csv)
df_ei   = pd.read_csv(ei_csv)

# expect f1,f2,f3 columns
fcols = [c for c in df_ehvi.columns if c.startswith("f")]
pts_ehvi = df_ehvi[fcols].values
pts_ei   = df_ei[["f"]].values if "f" in df_ei.columns else df_ei[fcols].values

# 2) parse true Pareto out of log
with open(ehvi_log) as f:
    txt = f.read()
m = re.search(r"Final Pareto front points:\s*(\[\[.*\]\])", txt, re.S)
if not m:
    raise RuntimeError("Could not find Pareto block in log")
pareto = np.array(eval(m.group(1)))  # shape (M,3)

# 3) Begin PDF and interactive plots
with PdfPages(out_pdf) as pdf:
    fig = plt.figure(figsize=(18,4))
    axes = [
        fig.add_subplot(1,4,1),
        fig.add_subplot(1,4,2),
        fig.add_subplot(1,4,3),
    ]
    pairs = [(0,1),(1,2),(0,2)]
    titles = ["f1 vs f2","f2 vs f3","f1 vs f3"]
    colors = dict(ehvi="C0", ei="gold", pareto="red")

    for ax, (i,j), title in zip(axes, pairs, titles):
        ax.scatter(pts_ehvi[:,i], pts_ehvi[:,j], c=colors["ehvi"], label="EHVI", alpha=0.7)
        ax.scatter(pts_ei[:,i],   pts_ei[:,j],   c=colors["ei"],   label="EI",   alpha=0.7)
        ax.scatter(pareto[:,i],   pareto[:,j],   c=colors["pareto"], s=50, label="Pareto")
        ax.set_xlabel(f"f{i+1}"); ax.set_ylabel(f"f{j+1}")
        ax.set_title(title)
    axes[0].legend(loc="upper left")
    # 4th slot: 3D
    ax3 = fig.add_subplot(1,4,4, projection="3d")
    ax3.scatter(pts_ehvi[:,0], pts_ehvi[:,1], pts_ehvi[:,2], c=colors["ehvi"], alpha=0.7, label="EHVI")
    ax3.scatter(pts_ei[:,0],   pts_ei[:,1],   pts_ei[:,2],   c=colors["ei"],   alpha=0.7, label="EI")
    ax3.scatter(pareto[:,0],   pareto[:,1],   pareto[:,2],   c=colors["pareto"], s=50, label="Pareto")
    ax3.set_xlabel("f1"); ax3.set_ylabel("f2"); ax3.set_zlabel("f3")
    ax3.set_title("3D scatter")
    ax3.legend()

    fig.suptitle("Fexofenadine MPO — Trial 1", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])

    # save and show
    pdf.savefig(fig)
    plt.show()
    plt.close(fig)

print(f"Saved plots to {out_pdf}")
