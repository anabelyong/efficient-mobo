import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

from acquisition_funcs.hypervolume import Hypervolume, infer_reference_point
from acquisition_funcs.pareto import pareto_front
from kern_gp.gp_model import independent_tanimoto_gp_predict
from utils.utils_final import evaluate_perin_objectives

def expected_hypervolume_improvement(pred_means, pred_vars, reference_point, pareto_front, N=1000):
    num_points, num_objectives = pred_means.shape
    ehvi_values = np.zeros(num_points)

    hv = Hypervolume(reference_point)
    current_hv = hv.compute(pareto_front)

    for i in range(num_points):
        mean = pred_means[i]
        var = pred_vars[i]
        cov = np.diag(var)

        # Monte Carlo integration
        samples = np.random.multivariate_normal(mean, cov, size=N)

        hvi = 0.0
        for sample in samples:
            augmented_pareto_front = np.vstack([pareto_front, sample])
            hv_sample = hv.compute(augmented_pareto_front)
            hvi += max(0, hv_sample - current_hv)

        ehvi_values[i] = hvi / N

    return ehvi_values

def ehvi_acquisition(query_smiles, known_smiles, known_Y, gp_means, gp_amplitudes, gp_noises, reference_point):
    pred_means, pred_vars = independent_tanimoto_gp_predict(
        query_smiles=query_smiles,
        known_smiles=known_smiles,
        known_Y=known_Y,
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
    )

    pareto_mask = pareto_front(known_Y)
    pareto_Y = known_Y[pareto_mask]

    ehvi_values = expected_hypervolume_improvement(pred_means, pred_vars, reference_point, pareto_Y)

    return ehvi_values, pred_means

def bayesian_optimization_loop(
    known_smiles,
    query_smiles,
    known_Y,
    gp_means,
    gp_amplitudes,
    gp_noises,
    max_ref_point=None,
    scale=0.1,
    scale_max_ref_point=False,
    n_iterations=20,
):
    S_chosen = set()
    hypervolumes_bo = []
    acquisition_values = []
    results = []

    print(f"GP Means: {gp_means}")
    print(f"GP Amplitudes: {gp_amplitudes}")
    print(f"GP Noises: {gp_noises}\n")

    for iteration in range(n_iterations):
        print(f"Start BO iteration {iteration}. Dataset size={known_Y.shape}")

        reference_point = infer_reference_point(
            known_Y, max_ref_point=max_ref_point, scale=scale, scale_max_ref_point=scale_max_ref_point
        )

        max_acq = -np.inf
        best_smiles = None
        best_means = None

        for smiles in query_smiles:
            if smiles in S_chosen:
                continue
            ehvi_values, pred_means = ehvi_acquisition(
                query_smiles=[smiles],
                known_smiles=known_smiles,
                known_Y=known_Y,
                gp_means=gp_means,
                gp_amplitudes=gp_amplitudes,
                gp_noises=gp_noises,
                reference_point=reference_point,
            )
            ehvi_value = ehvi_values[0]
            if ehvi_value > max_acq:
                max_acq = ehvi_value
                best_smiles = smiles
                best_means = pred_means[0]

        acquisition_values.append(max_acq)
        print(f"Max acquisition value: {max_acq}")
        if best_smiles:
            S_chosen.add(best_smiles)
            new_Y = evaluate_perin_objectives([best_smiles])
            known_smiles.append(best_smiles)
            known_Y = np.vstack([known_Y, new_Y])
            print(f"Chosen SMILES: {best_smiles} with acquisition function value = {max_acq}")
            print(f"Value of chosen SMILES: {new_Y}")
            print(f"Updated dataset size: {known_Y.shape}")
            # Store the chosen SMILES EHVI-MPO values
            f1, f2 = new_Y[0]  # Extract f1, f2 from the actual evaluated values
            results.append([f1, f2])
        else:
            print("No new SMILES selected.")

        hv = Hypervolume(reference_point)
        current_hypervolume = hv.compute(known_Y)
        hypervolumes_bo.append(current_hypervolume)
        print(f"Hypervolume: {current_hypervolume}")

    return known_smiles, known_Y, hypervolumes_bo, acquisition_values, results

def random_sampling_loop(query_smiles, known_Y, n_iterations=20):
    results = []
    for iteration in range(n_iterations):
        print(f"Start RS iteration {iteration}. Dataset size={known_Y.shape}")
        
        random_smiles = random.choice(query_smiles)
        new_Y = evaluate_perin_objectives([random_smiles])
        known_Y = np.vstack([known_Y, new_Y])
        
        # Store the chosen SMILES RS-MPO values
        f1, f2 = new_Y[0]  # Extract f1, f2 from the actual evaluated values
        results.append([f1, f2])

    return known_Y, results

def plot_pairwise(experiment_results_bo, experiment_results_rs, experiment_num):
    experiment_results_bo = np.array(experiment_results_bo)
    experiment_results_rs = np.array(experiment_results_rs)
    
    # Pareto Front
    pareto_mask_bo = pareto_front(experiment_results_bo)
    pareto_optimal_bo = experiment_results_bo[pareto_mask_bo]

    # Pairwise Plot for f1 vs f2
    plt.figure(figsize=(8, 6))
    plt.scatter(experiment_results_bo[:, 0], experiment_results_bo[:, 1], color='blue', label='BO Samples')
    plt.scatter(experiment_results_rs[:, 0], experiment_results_rs[:, 1], color='green', label='RS Samples')
    plt.scatter(pareto_optimal_bo[:, 0], pareto_optimal_bo[:, 1], color='red', label='Pareto Optimal Front')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.title(f'Pairwise Plot of Pareto Front with Samples - Experiment {experiment_num}')
    plt.legend()
    plt.show()

def run_experiment(repeats, n_iterations):
    all_results = []

    for experiment_num in range(1, repeats + 1):
        print(f"\nStarting Experiment {experiment_num}...\n")

        guacamol_dataset_path = "guacamol_dataset/guacamol_v1_train.smiles"
        guacamol_dataset = pd.read_csv(guacamol_dataset_path, header=None, names=["smiles"])
        ALL_SMILES = guacamol_dataset["smiles"].tolist()[:10_000]

        ALL_SMILES = ALL_SMILES[:10_000]
        random.shuffle(ALL_SMILES)
        training_smiles = ALL_SMILES[:10]
        query_smiles = ALL_SMILES[10:10_000]

        # Calculate objectives for training smiles
        training_Y = evaluate_perin_objectives(training_smiles)

        # Calculate GP hyperparameters from the training set
        gp_means = np.asarray([0.0, 0.0])
        gp_amplitudes = np.asarray([1.0, 1.0])
        gp_noises = np.asarray([1e-4, 1e-4])

        # Run Bayesian Optimization Loop
        known_smiles_bo, known_Y_bo, hypervolumes_bo, acquisition_values_bo, experiment_results_bo = bayesian_optimization_loop(
            known_smiles=training_smiles.copy(),
            query_smiles=query_smiles,
            known_Y=training_Y.copy(),
            gp_means=gp_means,
            gp_amplitudes=gp_amplitudes,
            gp_noises=gp_noises,
            n_iterations=n_iterations,
        )

        # Run Random Sampling Loop
        known_Y_rs, experiment_results_rs = random_sampling_loop(
            query_smiles=query_smiles,
            known_Y=training_Y.copy(),
            n_iterations=n_iterations,
        )

        # Plotting Pairwise plot for BO and RS results
        plot_pairwise(experiment_results_bo, experiment_results_rs, experiment_num)

        # Append results with experiment number for BO
        for iteration in range(n_iterations):
            f1, f2 = experiment_results_bo[iteration]
            all_results.append([f"Experiment {experiment_num}", iteration + 1, f1, f2])

    return all_results, acquisition_values_bo

def write_results_to_csv(results, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Experiment", "BO Iteration", "EHVI-MPO-F1", "EHVI-MPO-F2"])
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    repeats = 3  # Number of experiments
    n_iterations = 20  # Number of BO iterations per experiment

    # Run the experiment and collect results
    results, acquisition_values_bo = run_experiment(repeats, n_iterations)

    # Write results to a CSV file
    write_results_to_csv(results, 'ehvi_bo_perindopril_results.csv')

    # Plotting the EHVI acquisition function values over BO iterations
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, n_iterations + 1), acquisition_values_bo, marker='o', color='blue', label='EHVI MPO')
    plt.xlabel('BO Iteration')
    plt.ylabel('EHVI Acquisition Function Value')
    plt.title('EHVI Acquisition Function Value over BO Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()
