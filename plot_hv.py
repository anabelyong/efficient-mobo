import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Trials and file pattern
TRIAL_IDS = [1, 2, 3]
CSV_PATTERN = "parsed_csvs/logs_trial{tid}_terminal_output_jax_fex_ehvi.csv"  # change `fex` if needed
OUTPUT_PDF = "plots/pareto_hypervolume_fex_.pdf"

def load_hypervolume(csv_path):
    df = pd.read_csv(csv_path)
    if "hypervolume" not in df.columns:
        raise ValueError(f"'hypervolume' column not found in {csv_path}")
    return df["hypervolume"].tolist()

def aggregate_hypervolumes(trial_ids):
    all_hv = []
    for tid in trial_ids:
        csv_path = CSV_PATTERN.format(tid=tid)
        hv = load_hypervolume(csv_path)
        all_hv.append(hv)
    min_len = min(len(hv) for hv in all_hv)
    return np.array([hv[:min_len] for hv in all_hv])  # shape (n_trials, min_len)

def plot_hypervolume(hv_matrix, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    iters = np.arange(1, hv_matrix.shape[1] + 1)
    mean = np.mean(hv_matrix, axis=0)
    std  = np.std(hv_matrix, axis=0)

    def smooth(y, window_size=5):
        return np.convolve(y, np.ones(window_size)/window_size, mode='same')

    smoothed_mean = smooth(mean)
    smoothed_std  = smooth(std)

    plt.figure(figsize=(8, 5))
    plt.plot(iters, smoothed_mean, label="EHVI Hypervolume", color="purple", marker="o", markersize=4)
    plt.fill_between(iters, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std,
                     color="purple", alpha=0.2)

    plt.xlabel("BO Iteration")
    plt.ylabel("Total Pareto Hypervolume")
    plt.title("Pareto Hypervolume (avg ± std) over 3 Trials")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    print(f"Saved hypervolume plot to {out_path}")

if __name__ == "__main__":
    hv_matrix = aggregate_hypervolumes(TRIAL_IDS)
    plot_hypervolume(hv_matrix, OUTPUT_PDF)
