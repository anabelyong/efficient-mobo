import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === Paths to each method's 3 trial results ===
methods = {
    "EHVI": [
        "circles/circles_ehvi_trial1_fex.csv",
        "circles/circles_ehvi_trial2_fex.csv",
        "circles/circles_ehvi_trial3_fex.csv"
    ],
    "EI": [
        "circles/circles_ei_trial1_fex.csv",
        "circles/circles_ei_trial2_fex.csv",
        "circles/circles_ei_trial3_fex.csv"
    ]
}

# === Plot aesthetics ===
colors = {
    "EHVI": "#5e3c99",  # purple
    "EI": "#e377c2",    # pink
}
markers = {
    "EHVI": "D",   # diamond
    "EI": "o",     # circle
}

# === Load and compute mean/std across thresholds ===
data = {}
for method, files in methods.items():
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df = df[["Threshold", "NumCircles"]].dropna()
        df = df.set_index("Threshold")
        dfs.append(df)

    df_all = pd.concat(dfs, axis=1)
    df_all.columns = [f"trial_{i+1}" for i in range(len(dfs))]
    data[method] = {
        "mean": df_all.mean(axis=1).sort_index(),
        "std": df_all.std(axis=1).sort_index()
    }

# === Plot ===
fig, ax = plt.subplots(figsize=(7.2, 5))

for method, stats in data.items():
    thresholds = stats["mean"].index.values
    mean = stats["mean"].values
    std = stats["std"].values

    ax.errorbar(
        thresholds, mean,
        yerr=std,
        label=method,
        color=colors[method],
        marker=markers[method],
        linewidth=2,
        capsize=3,
        markersize=6
    )

ax.set_xlabel("Threshold $t$", fontsize=13)
ax.set_ylabel("#Circles", fontsize=13)
ax.set_xticks(np.arange(0.4, 0.91, 0.05))
ax.set_xlim(0.39, 0.91)
ax.set_yscale("log")
ax.tick_params(labelsize=11)
ax.grid(True, which="both", linestyle="--", alpha=0.3)
ax.legend(title="Method", fontsize=11)
ax.set_title("Fexofenadine: #Circles across thresholds", fontsize=14)

plt.tight_layout()
plt.savefig("fex_circles_markerplot.pdf", dpi=300)
plt.show()
