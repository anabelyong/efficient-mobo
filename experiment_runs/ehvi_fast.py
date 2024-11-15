import logging
import time
import numpy as np
import random
import pandas as pd
from acquisition_funcs.hypervolume import Hypervolume, infer_reference_point
from acquisition_funcs.pareto import pareto_front
from kern_gp.optimized_gp_model import independent_tanimoto_gp_predict, get_fingerprint
from utils.utils_final import evaluate_fex_objectives

# Setup logging
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
stream_handler.setFormatter(formatter)

bo_loop_logger = logging.getLogger("bo_loop_logger")
bo_loop_logger.setLevel(logging.DEBUG)
bo_loop_logger.addHandler(stream_handler)


def expected_hypervolume_improvement(pred_means, pred_vars, reference_point, pareto_front, N=1000): 
    num_points, num_objectives = pred_means.shape
    ehvi_values = np.zeros(num_points)

    hv = Hypervolume(reference_point)
    current_hv = hv.compute(pareto_front)

    for i in range(num_points):
        mean = pred_means[i]
        var = pred_vars[i]
        cov = np.diag(var)

        # Monte Carlo integration (sampling 1000 points)
        samples = np.random.multivariate_normal(mean, cov, size=N)

        hvi = 0.0
        for sample in samples:
            augmented_pareto_front = np.vstack([pareto_front, sample])
            hv_sample = hv.compute(augmented_pareto_front)
            hvi += max(0, hv_sample - current_hv)

        ehvi_values[i] = hvi / N

    return ehvi_values


def ehvi_acquisition(query_smiles, known_smiles, known_Y, gp_means, gp_amplitudes, gp_noises, reference_point, known_fp):
    """Calculate the Expected Hypervolume Improvement (EHVI) for given SMILES."""
    pred_means, pred_vars = independent_tanimoto_gp_predict(
        query_smiles=query_smiles,
        known_smiles=known_smiles,
        known_Y=known_Y,
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
        known_fp=known_fp
    )

    pareto_mask = pareto_front(known_Y)
    pareto_Y = known_Y[pareto_mask]

    ehvi_values = expected_hypervolume_improvement(pred_means, pred_vars, reference_point, pareto_Y)

    return ehvi_values, pred_means

def bayesian_optimization_loop(
    known_smiles, query_smiles, known_Y, gp_means, gp_amplitudes, gp_noises, max_ref_point=None, scale=0.1, scale_max_ref_point=False, n_iterations=20
):
    """Main loop for Bayesian Optimization (BO) with EHVI acquisition function."""
    bo_loop_logger.info("Starting BO loop...")

    # Precompute fingerprints for known SMILES
    start_time = time.time()
    known_fp = [get_fingerprint(s) for s in known_smiles]
    bo_loop_logger.info(f"Fingerprint precomputation time: {time.time() - start_time:.2f} seconds")

    S_chosen = set()
    hypervolumes_bo = []
    acquisition_values = []

    for iteration in range(n_iterations):
        iter_start_time = time.time()
        bo_loop_logger.info(f"BO iteration {iteration}. Dataset size={len(known_Y)}")

        # Infer reference point for hypervolume calculation
        reference_point = infer_reference_point(
            known_Y, max_ref_point=max_ref_point, scale=scale, scale_max_ref_point=scale_max_ref_point
        )

        max_acq = -np.inf
        best_smiles = None

        # Log the start of MC sampling generation
        bo_loop_logger.debug(f"Starting Monte Carlo sampling for BO iteration {iteration}")
        mc_total_start_time = time.time()

        for smiles in query_smiles:
            if smiles in S_chosen:
                continue
            # Compute EHVI acquisition value
            ehvi_values, pred_means = ehvi_acquisition(
                query_smiles=[smiles],
                known_smiles=known_smiles,
                known_Y=known_Y,
                gp_means=gp_means,
                gp_amplitudes=gp_amplitudes,
                gp_noises=gp_noises,
                reference_point=reference_point,
                known_fp=known_fp
            )
            ehvi_value = ehvi_values[0]

            if ehvi_value > max_acq:
                max_acq = ehvi_value
                best_smiles = smiles

        # Log the total time taken for MC sampling across all calls in this iteration
        mc_total_time = time.time() - mc_total_start_time
        bo_loop_logger.debug(f"Total time for generating all Monte Carlo samples in BO iteration {iteration}: {mc_total_time:.4f} seconds")

        acquisition_values.append(max_acq)
        if best_smiles:
            S_chosen.add(best_smiles)
            new_Y = evaluate_fex_objectives([best_smiles])
            known_smiles.append(best_smiles)
            known_Y = np.vstack([known_Y, new_Y])

            # Update fingerprints with new SMILES
            known_fp.append(get_fingerprint(best_smiles))
            bo_loop_logger.info(f"Selected SMILES: {best_smiles} with acquisition value = {max_acq}")
            bo_loop_logger.info(f"Updated dataset size: {len(known_Y)}")

        # Compute hypervolume
        hv = Hypervolume(reference_point)
        current_hypervolume = hv.compute(known_Y)
        hypervolumes_bo.append(current_hypervolume)
        bo_loop_logger.info(f"Iteration {iteration} hypervolume: {current_hypervolume}")
        bo_loop_logger.info(f"BO iteration {iteration} time: {time.time() - iter_start_time:.2f} seconds")

    bo_loop_logger.info("Completed BO loop.")
    return known_smiles, known_Y, hypervolumes_bo, acquisition_values



if __name__ == "__main__":
    # Load the Guacamol dataset and select the top 10,000 SMILES
    guacamol_dataset_path = "guacamol_dataset/guacamol_v1_train.smiles"  # Ensure path is correct
    guacamol_dataset = pd.read_csv(guacamol_dataset_path, header=None, names=["smiles"])
    all_smiles = guacamol_dataset["smiles"].tolist()[:10_000]

    # Shuffle the dataset to introduce randomness
    random.shuffle(all_smiles)

    # Select the first 10 SMILES for known_smiles, the rest for query_smiles
    known_smiles = all_smiles[:10]
    query_smiles = all_smiles[10:]

    # Calculate initial objective values for known SMILES (replace with actual objective values if available)
    known_Y = evaluate_fex_objectives(known_smiles)  # Evaluates objectives for the initial SMILES

    # Define GP parameters (these are placeholders and can be tuned based on your application)
    gp_means = np.array([0.0, 0.0, 0.0])
    gp_amplitudes = np.array([1.0, 1.0, 1.0])
    gp_noises = np.array([1e-4, 1e-4, 1e-4])

    # Run Bayesian Optimization
    bayesian_optimization_loop(
        known_smiles=known_smiles,
        query_smiles=query_smiles,
        known_Y=known_Y,
        gp_means=gp_means,
        gp_amplitudes=gp_amplitudes,
        gp_noises=gp_noises,
        n_iterations=10
    )
