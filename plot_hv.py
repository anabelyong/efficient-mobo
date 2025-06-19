import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import MultipleLocator

def load_trials(prefix: str, suffix: str, trials: list[int]) -> list:
    """Load multiple trial CSVs into a list of DataFrames."""
    dfs = []
    for t in trials:
        path = f"{prefix}{t}{suffix}"
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
        else:
            print(f"[WARN] Missing: {path}")
    return dfs

def plot_refined_hv_comparison(dfs: dict, title: str, out_file: str):
    plt.figure(figsize=(8, 5))
    colors = {
        "EHVI": "#5e3c99",             # purple
        "EI": "#e377c2",               # pink
        "Random Sampling": "#e6ac00"   # gold
    }

    for method, method_dfs in dfs.items():
        df_all = pd.concat(method_dfs)
        grouped = df_all.groupby("BO Iteration")["Hypervolume"]

        mean = grouped.mean()
        std = grouped.std()

        xticks = sorted(grouped.mean().index)
        mean_vals = mean.loc[xticks]
        std_vals = std.loc[xticks]

        # Step plot
        plt.step(
            xticks, mean_vals,
            label=method,
            color=colors[method],
            linewidth=2,
            where="post"
        )

        # Fill between mean ± std
        plt.fill_between(
            xticks,
            (mean_vals - std_vals).clip(lower=0.0),
            (mean_vals + std_vals),
            step="post",
            color=colors[method],
            alpha=0.2,
            linewidth=0
        )

    plt.xlabel("BO Iteration", fontsize=13)
    plt.ylabel("Hypervolume", fontsize=13)
    plt.xticks(np.arange(0, 201, 10), fontsize=11)
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))  # tick every 0.05 (or 0.025 for finer)
    plt.yticks(fontsize=11)
    plt.title(title, fontsize=14)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(loc="lower right", fontsize=11, frameon=False)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

# === Example usage ===
trials = [1, 2, 3]

ei_dfs = load_trials(
    prefix="evaluated_ei/logs_trial",
    suffix="_ei_evaluated_perin.csv",
    trials=trials
)

ehvi_dfs = load_trials(
    prefix="parsed_csvs/logs_trial",
    suffix="_terminal_output_jax_perin_ehvi.csv",
    trials=trials
)

rs_dfs = load_trials(
    prefix="evaluated_rs/random_sampling_perin_trial",
    suffix=".csv",
    trials=trials
)

dfs = {
    "EHVI": ehvi_dfs,
    "EI": ei_dfs,
    "Random Sampling": rs_dfs
}

plot_refined_hv_comparison(
    dfs,
    "Perindopril MPO: Hypervolume over 200 BO Evaluations",
    "perin_hv3.pdf"
)
