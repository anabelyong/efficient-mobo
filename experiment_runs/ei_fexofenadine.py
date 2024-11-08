import csv
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from kern_gp.gp_model import independent_tanimoto_gp_predict
from utils.utils_final import evaluate_fex_MPO

def expected_improvement(pred_means: np.array, pred_vars: np.array, y_best: float) -> np.array:
    std = np.sqrt(pred_vars)
    z = (pred_means - y_best) / std
    ei = (pred_means - y_best) * stats.norm.cdf(z) + std * stats.norm.pdf(z)
    return np.maximum(ei, 1e-30)

def ei_acquisition(query_smiles, known_smiles, known_Y, gp_means, gp_amplitudes, gp_noises):
    pred_means, pred_vars = independent_tanimoto_gp_predict(
        query_smiles=query_smiles,
        known_smiles=known_smiles,
        known_Y=known_Y,
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
    )
    y_best = np.max(known_Y)
    ei_values = expected_improvement(pred_means, pred_vars, y_best)
    return ei_values, pred_means

def bayesian_optimization_loop(
    known_smiles,
    query_smiles,
    known_Y,
    gp_means,
    gp_amplitudes,
    gp_noises,
    n_iterations=20,
):
    S_chosen = set(known_smiles)
    results = []
    ei_values_list = []

    for iteration in range(n_iterations):
        print(f"Start BO iteration {iteration}. Dataset size={known_Y.shape}")

        max_acq = -np.inf
        best_smiles = None

        for smiles in query_smiles:
            if smiles in S_chosen:
                continue
            ei_values, _ = ei_acquisition(
                query_smiles=[smiles],
                known_smiles=known_smiles,
                known_Y=known_Y,
                gp_means=gp_means,
                gp_amplitudes=gp_amplitudes,
                gp_noises=gp_noises,
            )
            ei_value = ei_values[0]
            if ei_value > max_acq:
                max_acq = ei_value
                best_smiles = smiles

        print(f"Max acquisition value (EI): {max_acq}")
        ei_values_list.append(max_acq)

        if best_smiles:
            S_chosen.add(best_smiles)
            new_Y = evaluate_fex_MPO([best_smiles])
            known_smiles.append(best_smiles)
            known_Y = np.vstack([known_Y, new_Y])
            chosen_value = new_Y[0][0]  # Extract the actual value of the chosen SMILES
            print(f"Chosen SMILES: {best_smiles} with EI value = {max_acq}")
            print(f"Value of chosen SMILES: {new_Y}")
            print(f"Updated dataset size: {known_Y.shape}")
            # Append only the chosen value to the results
            results.append([iteration + 1, chosen_value])
        else:
            print("No new SMILES selected.")

    return known_smiles, known_Y, results, ei_values_list

def random_sampling_loop(
    known_smiles,
    query_smiles,
    known_Y,
    n_iterations=20,
):
    S_chosen = set(known_smiles)
    results = []

    for iteration in range(n_iterations):
        print(f"Start Random Sampling iteration {iteration}. Dataset size={known_Y.shape}")

        random_smiles = random.choice([s for s in query_smiles if s not in S_chosen])
        S_chosen.add(random_smiles)
        new_Y = evaluate_fex_MPO([random_smiles])
        known_smiles.append(random_smiles)
        known_Y = np.vstack([known_Y, new_Y])
        chosen_value = new_Y[0][0]  # Extract the actual value of the chosen SMILES
        print(f"Chosen SMILES (Random Sampling): {random_smiles}")
        print(f"Value of chosen SMILES: {new_Y}")
        print(f"Updated dataset size: {known_Y.shape}")
        results.append([iteration + 1, chosen_value])

    return known_smiles, known_Y, results

def write_to_csv(results_bo, results_rs, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Experiment", "Iteration", "Chosen Smiles for Fex-MPO", "RS-MPO"])
        for bo, rs in zip(results_bo, results_rs):
            writer.writerow(bo + rs[1:])

def run_experiment(repeats, n_iterations):
    all_results_bo = []
    all_results_rs = []

    for experiment_num in range(1, repeats + 1):
        print(f"\nStarting Experiment {experiment_num}...\n")

        guacamol_dataset_path = "guacamol_dataset/guacamol_v1_train.smiles"
        guacamol_dataset = pd.read_csv(guacamol_dataset_path, header=None, names=["smiles"])
        ALL_SMILES = guacamol_dataset["smiles"].tolist()[:10_000]

        random.shuffle(ALL_SMILES)
        training_smiles = ALL_SMILES[:10]
        query_smiles = ALL_SMILES[10:10_000]

        # Calculate objectives for training smiles
        training_Y = evaluate_fex_MPO(training_smiles)

        # Calculate GP hyperparameters from the training set
        gp_means = np.asarray([0.0])
        gp_amplitudes = np.asarray([1.0])
        gp_noises = np.asarray([1e-4])

        print("Calculated GP Hyperparameters:")
        print(f"GP Means: {gp_means}")
        print(f"GP Amplitudes: {gp_amplitudes}")
        print(f"GP Noises: {gp_noises}\n")

        known_smiles_bo = training_smiles.copy()
        known_Y_bo = training_Y.copy()

        known_smiles_rs = training_smiles.copy()
        known_Y_rs = training_Y.copy()

        # Bayesian Optimization Loop
        _, _, experiment_results_bo, ei_values = bayesian_optimization_loop(
            known_smiles=known_smiles_bo,
            query_smiles=query_smiles,
            known_Y=known_Y_bo,
            gp_means=gp_means,
            gp_amplitudes=gp_amplitudes,
            gp_noises=gp_noises,
            n_iterations=n_iterations,
        )

        # Random Sampling Loop
        _, _, experiment_results_rs = random_sampling_loop(
            known_smiles=known_smiles_rs,
            query_smiles=query_smiles,
            known_Y=known_Y_rs,
            n_iterations=n_iterations,
        )

        for iteration in range(n_iterations):
            all_results_bo.append([f"Experiment {experiment_num}"] + experiment_results_bo[iteration])
            all_results_rs.append([f"Experiment {experiment_num}"] + experiment_results_rs[iteration])

        # Plot the acquisition function values over BO iterations
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_iterations + 1), ei_values, marker='o', color='blue', label='EI over BO Iterations')
        plt.xlabel('BO Iteration')
        plt.ylabel('Expected Improvement (EI)')
        plt.title(f'Acquisition Function (EI) over BO Iterations for Experiment {experiment_num}')
        plt.grid(True)
        plt.legend()
        plt.show()

    return all_results_bo, all_results_rs

if __name__ == "__main__":
    repeats = 3  # Number of experiments
    n_iterations = 20  # Number of BO iterations per experiment

    # Run the experiment and collect results
    results_bo, results_rs = run_experiment(repeats, n_iterations)

    # Write results to a CSV file
    write_to_csv(results_bo, results_rs, "ei_bo_rs_experiments_fexofenadine_results.csv")
