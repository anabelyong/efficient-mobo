import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d

def load_trials(prefix: str, suffix: str, trials: list[int]) -> list:
    dfs = []
    for t in trials:
        path = f"{prefix}{t}{suffix}"
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
        else:
            print(f"[WARN] Missing: {path}")
    return dfs

def plot_r2_comparison(dfs: dict, title: str, out_file: str, smooth_sigma: float = 1.5):
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "EHVI": "#5e3c99",             # deep purple
        "EI": "#e377c2",               # pink
        "Random Sampling": "#e6ac00"   # gold
    }

    for method, method_dfs in dfs.items():
        df_all = pd.concat(method_dfs)
        grouped = df_all.groupby("BO Iteration")["R2 Indicator"]

        mean = grouped.mean()
        std = grouped.std()

        xticks = sorted(mean.index)
        mean_vals = mean.loc[xticks].values
        std_vals = std.loc[xticks].values

        # Smooth if desired
        mean_smooth = gaussian_filter1d(mean_vals, sigma=smooth_sigma)
        std_smooth = gaussian_filter1d(std_vals, sigma=smooth_sigma)

        ax.plot(
            xticks, mean_smooth,
            label=method,
            color=colors[method],
            linewidth=2,
        )

        ax.fill_between(
            xticks,
            mean_smooth - std_smooth,
            mean_smooth + std_smooth,
            color=colors[method],
            alpha=0.2,
        )

    # === Manual formatting ===
    ax.set_xlabel("BO Iteration", fontsize=13)
    ax.set_ylabel("R2 Indicator", fontsize=13)
    ax.set_xticks(np.arange(0, 201, 10))
    ax.tick_params(axis='both', labelsize=11)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=11, frameon=False)

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()

# === Example usage ===
if __name__ == "__main__":
    trials = [1, 2, 3]

    ei_dfs = load_trials(
        prefix="evaluated_ei/r2/logs_trial",
        suffix="_ei_evaluated_amlo_r2.csv",
        trials=trials
    )

    ehvi_dfs = load_trials(
        prefix="evaluated_ehvi/logs_trial",
        suffix="_ehvi_evaluated_amlo_r2.csv",
        trials=trials
    )

    rs_dfs = load_trials(
        prefix="evaluated_rs/r2/logs_trial",
        suffix="_rs_evaluated_amlo_r2.csv",
        trials=trials
    )

    dfs = {
        "EHVI": ehvi_dfs,
        "EI": ei_dfs,
        "Random Sampling": rs_dfs
    }

    plot_r2_comparison(
        dfs,
        "Amlodipine MPO: R2 Indicator over 200 BO Evaluations",
        "amlo_r2_comparison.pdf"
    )
