import pandas as pd
import matplotlib.pyplot as plt

def load_trials(prefix, suffix, trials, column="Hypervolume", path_template="{prefix}{n}{suffix}"):
    dfs = []
    for n in trials:
        path = path_template.format(prefix=prefix, n=n, suffix=suffix)
        df = pd.read_csv(path)
        df['BO Iteration'] = df['BO Iteration'] + 1
        dfs.append(df)
    return dfs

def combine_trials(dfs):
    df_comb = pd.DataFrame({"BO Iteration": dfs[0]["BO Iteration"]})
    for i, df in enumerate(dfs, 1):
        df_comb[f"trial{i}"] = df["Hypervolume"]
    df_comb["mean"] = df_comb[[f"trial{i}" for i in range(1, 4)]].mean(axis=1)
    df_comb["std"] = df_comb[[f"trial{i}" for i in range(1, 4)]].std(axis=1)
    return df_comb

# === Load all trials ===
trials = [1, 2, 3]

# EHVI
ehvi_dfs = load_trials(
    prefix="parsed_csvs/logs_trial",
    suffix="_terminal_output_jax_perin_ehvi.csv",
    trials=trials
)
ehvi_hv = combine_trials(ehvi_dfs)

# EI
ei_dfs = load_trials(
    prefix="evaluated_ei/logs_trial",
    suffix="_ei_evaluated_perin.csv",
    trials=trials
)
ei_hv = combine_trials(ei_dfs)

# Random Sampling
rs_dfs = load_trials(
    prefix="evaluated_rs/random_sampling_perin_trial",
    suffix=".csv",
    trials=trials
)
rs_hv = combine_trials(rs_dfs)

# === Plotting ===
plt.figure(figsize=(10, 6))

# EHVI
plt.plot(ehvi_hv["BO Iteration"], ehvi_hv["mean"], label="EHVI (mean)", color="tab:blue")
plt.fill_between(ehvi_hv["BO Iteration"], ehvi_hv["mean"] - ehvi_hv["std"], ehvi_hv["mean"] + ehvi_hv["std"],
                alpha=0.25, color="tab:blue", label="EHVI ± std")

# EI
plt.plot(ei_hv["BO Iteration"], ei_hv["mean"], label="EI (mean)", color="tab:orange")
plt.fill_between(ei_hv["BO Iteration"], ei_hv["mean"] - ei_hv["std"], ei_hv["mean"] + ei_hv["std"],
                 alpha=0.25, color="tab:orange", label="EI ± std")

# RS
plt.plot(rs_hv["BO Iteration"], rs_hv["mean"], label="Random Sampling (mean)", color="tab:green")
#plt.fill_between(rs_hv["BO Iteration"], rs_hv["mean"] - rs_hv["std"], rs_hv["mean"] + rs_hv["std"],
                 #alpha=0.25, color="tab:green", label="RS ± std")

plt.xlabel("BO Iteration")
plt.ylabel("Hypervolume")
plt.title("Perindopril MPO: Hypervolume over BO Iterations")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save as PDF
plt.savefig("perin_hypervolume_comparison.pdf", format="pdf")
plt.close()
