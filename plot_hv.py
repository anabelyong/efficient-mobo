import pandas as pd
import matplotlib.pyplot as plt

def plot_hv_for_trial(trial_id):
    csv_path = f"parsed_csvs/logs_trial{trial_id}_terminal_output_jax_perin_ehvi.csv"
    df = pd.read_csv(csv_path)

    # Sort by BO Iteration
    if "BO Iteration" in df.columns:
        df = df.sort_values("BO Iteration")
    else:
        raise ValueError(f"'BO Iteration' not found in {csv_path}")

    iters = df["BO Iteration"].values
    hv = df["Hypervolume"].values

    plt.figure(figsize=(8, 5))
    plt.plot(iters, hv, color="purple", marker="o", label=f"Trial {trial_id} HV")
    plt.xlabel("BO Iteration")
    plt.ylabel("Total Pareto Hypervolume")
    plt.title(f"Pareto Hypervolume over BO Iterations for Perindopril MPO (Trial {trial_id})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = f"plots/trial{trial_id}_perin_hypervolume.pdf"
    plt.savefig(out_path, format="pdf")
    print(f"Saved: {out_path}")
    plt.close()

if __name__ == "__main__":
    for trial in [1, 2, 3]:
        plot_hv_for_trial(trial)
