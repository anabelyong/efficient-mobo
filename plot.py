from utils.utils_final import evaluate_fex_MPO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_ehvi_smiles(csv_path):
    df = pd.read_csv(csv_path)
    return df["Selected SMILES"].tolist()

def evaluate_mpo(smiles_list):
    scores = evaluate_fex_MPO(smiles_list)
    return np.array(scores).flatten().tolist()

def load_ei_scores(csv_path):
    df = pd.read_csv(csv_path)
    return df["Objective Value"].tolist()

def load_rand_scores(csv_path):
    df = pd.read_csv(csv_path)
    return df["f1"].tolist()

def aggregate_trials(trial_ids):
    ehvi_all, ei_all, rand_all = [], [], []
    for tid in trial_ids:
        ehvi_csv = f"parsed_csvs/logs_trial{tid}_terminal_output_jax_fex_ehvi.csv"
        ei_csv   = f"parsed_csvs/logs_trial{tid}_terminal_output_jax_ei_fex.csv"
        rand_csv = f"random_sampling/random_sampling_fex_results_trial{tid}.csv"

        ehvi_smiles = load_ehvi_smiles(ehvi_csv)
        ehvi_scores = evaluate_mpo(ehvi_smiles)
        ei_scores   = load_ei_scores(ei_csv)
        rand_scores = load_rand_scores(rand_csv)

        min_len = min(len(ehvi_scores), len(ei_scores), len(rand_scores))
        ehvi_all.append(ehvi_scores[:min_len])
        ei_all.append(ei_scores[:min_len])
        rand_all.append(rand_scores[:min_len])

    return np.array(ehvi_all), np.array(ei_all), np.array(rand_all)

def get_dataset_best_average(n=1):
    df = pd.read_csv("guacamol_dataset/guacamol_v1_train.smiles", header=None, names=["smiles"])
    all_sm = df["smiles"].tolist()[:10000]
    scores = evaluate_fex_MPO(all_sm)
    top_n = sorted(scores, reverse=True)[:n]
    return np.mean(top_n)

def plot_mpo_avg_std(ehvi_all, ei_all, rand_all, dataset_best, out_path="plots/fex_mpo_trials.pdf"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    iters = np.arange(1, ehvi_all.shape[1] + 1)

    def smooth(y, window_size=5):
        return np.convolve(y, np.ones(window_size)/window_size, mode='same')

    def plot_with_error(data, label, color, marker):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        plt.plot(iters, smooth(mean), label=label, color=color, marker=marker, markersize=4)
        plt.fill_between(iters, smooth(mean - std), smooth(mean + std), color=color, alpha=0.2)

    plt.figure(figsize=(8, 5))
    plot_with_error(ehvi_all, "EHVI", "blue", "o")
    plot_with_error(ei_all, "EI", "orange", "^")
    plot_with_error(rand_all, "Random Sampling", "green", "s")
    plt.axhline(dataset_best, linestyle="--", color="black", label="Dataset Best")

    plt.xlabel("BO Iteration")
    plt.ylabel("Fexofenadine MPO")
    plt.title("Fexofenadine MPO (avg ± std) over 3 Experiment Runs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(out_path, format="pdf")
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    ehvi_all, ei_all, rand_all = aggregate_trials([1, 2, 3])
    dataset_best = get_dataset_best_average(n=1)
    plot_mpo_avg_std(ehvi_all, ei_all, rand_all, dataset_best)
